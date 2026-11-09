"""Search-based constructor for size-efficient sorting networks on 18 wires."""
import time
import random
import numpy as np

N = 18
DEADLINE = 82.0  # safety margin before 90s kill

# EVOLVE-BLOCK-START

def _apply_network(net, state):
    """Apply comparator network to a state (list of 0/1). Returns sorted state if correct."""
    for i, j in net:
        if state[i] > state[j]:
            state[i], state[j] = state[j], state[i]
    return state

def _is_sorted_net_fast(net):
    """
    Check if network sorts all 2^18 boolean inputs using the 0-1 principle.
    Uses bitmask representation for speed.
    Returns True if the network is a valid sorting network.
    """
    # Pre-extract comparator pairs into separate arrays for speed
    if len(net) == 0:
        return N <= 1
    
    # For each mask from 0 to 2^N - 1, simulate the network
    # We need to check: after processing all comparators, the result is sorted
    # (all 0s come before all 1s)
    
    # Optimization: we only need to check masks with exactly k ones for each k,
    # but actually the 0-1 principle says we need ALL 2^N masks.
    # However, we can use a clever trick: if we process masks in Gray code order,
    # we can incrementally update the state. But for simplicity and correctness,
    # let's just iterate all masks.
    
    # Even faster: use the fact that a comparator (i,j) with min-to-i, max-to-j
    # only changes the state if bit i is 1 and bit j is 0.
    
    for mask in range(1 << N):
        state = mask
        for i, j in net:
            # Check if bit i is 1 and bit j is 0
            if (state & (1 << i)) and not (state & (1 << j)):
                # Swap: clear bit i, set bit j
                state = (state & ~(1 << i)) | (1 << j)
        
        # Check if sorted: all 0s at low indices, all 1s at high indices
        # Count number of 1s
        ones = bin(state).count('1')
        # Expected: bits [N-ones, N-1] are 1, rest are 0
        if ones == 0:
            expected = 0
        elif ones == N:
            expected = (1 << N) - 1
        else:
            expected = ((1 << ones) - 1) << (N - ones)
        
        if state != expected:
            return False
    
    return True

def _is_sorted_net_sampled(net, num_samples=10000, rng=None):
    """
    Sampled check for quick filtering during search.
    Returns True if all sampled inputs are sorted correctly.
    """
    if rng is None:
        rng = random.Random(42)
    
    for _ in range(num_samples):
        mask = rng.getrandbits(N)
        state = mask
        for i, j in net:
            if (state & (1 << i)) and not (state & (1 << j)):
                state = (state & ~(1 << i)) | (1 << j)
        
        ones = bin(state).count('1')
        if ones == 0:
            expected = 0
        elif ones == N:
            expected = (1 << N) - 1
        else:
            expected = ((1 << ones) - 1) << (N - ones)
        
        if state != expected:
            return False
    
    return True

def _odd_even_transposition():
    """Generate the seed odd-even transposition network."""
    net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            net.append((i, i + 1))
    return net

def _bubble_sort_network():
    """Generate a bubble sort network (adjacent comparators, multiple passes)."""
    net = []
    for _ in range(N):
        for i in range(N - 1):
            net.append((i, i + 1))
    return net

def _improve_by_removal(net, deadline):
    """
    Try to improve a network by removing redundant comparators.
    Uses a randomized approach to find removable comparators.
    """
    current = list(net)
    best = list(current)
    best_len = len(current)
    
    # First, do a full validation to make sure we start valid
    if not _is_sorted_net_fast(current):
        return current
    
    iteration = 0
    while time.time() < deadline:
        if not current:
            break
        
        # Try removing each comparator in random order
        rng = random.Random(iteration + 42)
        indices = list(range(len(current)))
        rng.shuffle(indices)
        
        improved = False
        for idx in indices:
            if time.time() > deadline:
                break
            
            candidate = current[:idx] + current[idx+1:]
            
            # Quick sampled check first
            if not _is_sorted_net_sampled(candidate, num_samples=5000, rng=rng):
                continue
            
            # Full check
            if _is_sorted_net_fast(candidate):
                current = candidate
                best = list(current)
                best_len = len(current)
                improved = True
                break
        
        if not improved:
            # Try some random perturbations to escape local optima
            for _ in range(10):
                if time.time() > deadline:
                    break
                new_net = list(current)
                if len(new_net) >= 2:
                    i, j = rng.sample(range(len(new_net)), 2)
                    new_net[i], new_net[j] = new_net[j], new_net[i]
                    if _is_sorted_net_sampled(new_net, num_samples=5000, rng=rng):
                        if _is_sorted_net_fast(new_net):
                            current = new_net
                            if len(current) < best_len:
                                best = list(current)
                                best_len = len(current)
                            improved = True
                            break
                elif len(new_net) == 1:
                    # Try moving the single comparator
                    pass
        
        iteration += 1
    
    return best

def _add_comparators(net, count, rng):
    """Add random comparators to a network."""
    new_net = list(net)
    for _ in range(count):
        i = rng.randint(0, N - 2)
        j = rng.randint(i + 1, N - 1)
        new_net.append((i, j))
    return new_net

def construct():
    start_time = time.time()
    
    # Start with the seed network
    seed_net = _odd_even_transposition()
    best_net = list(seed_net)
    best_len = len(seed_net)
    
    # Verify seed
    if not _is_sorted_net_fast(seed_net):
        # Shouldn't happen, but fallback
        raise ValueError("Seed network is invalid")
    
    # Strategy 1: Improve the seed network by removal
    if time.time() < DEADLINE - 30:
        improved = _improve_by_removal(seed_net, deadline=time.time() + min(30, DEADLINE - time.time() - 5))
        if len(improved) < best_len:
            best_net = improved
            best_len = len(improved)
    
    # Strategy 2: Try bubble sort network and improve
    if time.time() < DEADLINE - 20:
        bubble_net = _bubble_sort_network()
        if _is_sorted_net_fast(bubble_net):
            if len(bubble_net) < best_len:
                best_net = list(bubble_net)
                best_len = len(bubble_net)
            
            if time.time() < DEADLINE - 10:
                improved = _improve_by_removal(bubble_net, deadline=time.time() + min(10, DEADLINE - time.time() - 5))
                if len(improved) < best_len:
                    best_net = improved
                    best_len = len(improved)
    
    # Strategy 3: Random search with guided improvement
    rng = random.Random(12345)
    
    # Try various sizes
    for target_size in [best_len - 2, best_len - 1, best_len, best_len + 5, best_len + 10]:
        if target_size <= 0 or time.time() > DEADLINE - 10:
            break
        
        # Generate random networks of this size and try to improve
        for attempt in range(5):
            if time.time() > DEADLINE - 10:
                break
            
            rand_net = []
            for _ in range(target_size):
                i = rng.randint(0, N - 2)
                j = rng.randint(i + 1, N - 1)
                rand_net.append((i, j))
            
            # Quick check
            if not _is_sorted_net_sampled(rand_net, num_samples=2000, rng=rng):
                continue
            
            # Full check
            if _is_sorted_net_fast(rand_net):
                if len(rand_net) < best_len:
                    best_net = rand_net
                    best_len = len(rand_net)
                
                # Try to improve
                if time.time() < DEADLINE - 5:
                    improved = _improve_by_removal(rand_net, deadline=time.time() + min(5, DEADLINE - time.time() - 2))
                    if len(improved) < best_len:
                        best_net = improved
                        best_len = len(improved)
    
    # Final verification
    if not _is_sorted_net_fast(best_net):
        # Fallback to seed
        best_net = seed_net
    
    if len(best_net) > 4096:
        # Truncate (shouldn't happen)
        best_net = best_net[:4096]
    
    return best_net

# EVOLVE-BLOCK-END
