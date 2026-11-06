import time
import random
import numpy as np

N = 18
MAX_COMPARATORS = 4096
DEADLINE_SECONDS = 85  # Safety margin before 90s kill

# Precompute full matrix of all 2^N boolean inputs
all_masks = np.arange(1 << N, dtype=np.int32)
W0_full = np.zeros((1 << N, N), dtype=np.int8)
for i in range(N):
    W0_full[:, i] = (all_masks >> i) & 1

# Precompute subset matrix for fast rejection
# Use a mix of structured and random inputs for better coverage
subset_indices = np.concatenate([
    np.arange(2048),
    np.random.choice(1 << N, 2048, replace=False)
])
W0_subset = W0_full[subset_indices]

def _depth(net):
    """Compute the parallel depth of a comparator network."""
    if not net:
        return 0
    layer = [0] * N
    for i, j in net:
        layer[i] = max(layer[i], layer[j]) + 1
    return max(layer) if layer else 0

def _is_sorted_network(net):
    """
    Verify that the comparator network sorts all 2^N boolean inputs.
    Uses the 0-1 principle with bitwise operations for speed.
    """
    n = len(net)
    if n > MAX_COMPARATORS:
        return False
    if n == 0:
        return True
    
    # Fast subset check
    W_sub = W0_subset.copy()
    for i, j in net:
        col_i = W_sub[:, i]
        col_j = W_sub[:, j]
        # min = AND, max = OR for bits
        tmp = col_i | col_j
        col_i[:] = col_i & col_j
        col_j[:] = tmp
        
    if np.any(W_sub[:, :-1] > W_sub[:, 1:]):
        return False
        
    # Full check
    W_full = W0_full.copy()
    for i, j in net:
        col_i = W_full[:, i]
        col_j = W_full[:, j]
        tmp = col_i | col_j
        col_i[:] = col_i & col_j
        col_j[:] = tmp
        
    if np.any(W_full[:, :-1] > W_full[:, 1:]):
        return False
        
    return True

def _odd_even_transposition():
    """Generate the odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

def _bitonic_sort_network():
    """Generate a bitonic sorting network for N wires."""
    net = []
    def bitonic_merge(lo, hi, direction):
        """Merge a bitonic sequence from lo to hi (exclusive)."""
        length = hi - lo
        if length <= 1:
            return
        half = length // 2
        bitonic_merge(lo, lo + half, 1)  # ascending
        bitonic_merge(lo + half, hi, 0)  # descending
        for i in range(half):
            a = lo + i
            b = lo + half + i
            if direction == 1:  # ascending
                if a < b:
                    net.append((a, b))
            else:  # descending
                if a > b:
                    net.append((b, a))
    
    def bitonic_sort(lo, hi, direction):
        """Sort a sequence from lo to hi (exclusive)."""
        length = hi - lo
        if length <= 1:
            return
        half = length // 2
        bitonic_sort(lo, lo + half, 1)  # ascending
        bitonic_sort(lo + half, hi, 0)  # descending
        bitonic_merge(lo, hi, direction)
    
    # For 18 wires, we need to handle non-power-of-2
    # Pad to 32 wires conceptually, but only use 18
    # Actually, let's use a different approach: recursive bitonic sort
    # For simplicity, use odd-even merge sort which works for any N
    
    return _odd_even_merge_sort()

def _odd_even_merge_sort():
    """Generate an odd-even merge sort network."""
    net = []
    def odd_even_merge(a, b, k):
        """Merge two sorted sequences of length k starting at a and b."""
        if k == 1:
            net.append((a, b))
            return
        half = k // 2
        # Recursively merge odd and even elements
        odd_even_merge(a + 1, b + 1, half)
        odd_even_merge(a, b, half)
        # Final compare-exchanges
        for i in range(half):
            net.append((a + 2 * i + 1, b + 2 * i))
    
    def odd_even_merge_sort(a, k):
        """Sort sequence starting at a of length k."""
        if k <= 1:
            return
        half = k // 2
        odd_even_merge_sort(a, half)
        odd_even_merge_sort(a + half, k - half)
        odd_even_merge(a, a + half, min(half, k - half))
    
    odd_even_merge_sort(0, N)
    return net

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Start with the best initial network
    net_odd = _odd_even_transposition()
    net_merge = _odd_even_merge_sort()
    
    # Choose the smaller one as starting point
    if len(net_odd) <= len(net_merge):
        net = net_odd[:]
    else:
        net = net_merge[:]
    
    best_net = net[:]
    best_count = len(net)
    
    # Phase 1: Aggressive greedy removal with multiple passes
    changed = True
    passes = 0
    while changed and time.time() < deadline - 10:
        changed = False
        passes += 1
        
        # Try removing from end to start
        for idx in range(len(net) - 1, -1, -1):
            if time.time() > deadline - 10:
                break
            
            new_net = net[:idx] + net[idx+1:]
            
            if _is_sorted_network(new_net):
                net = new_net
                best_net = net[:]
                best_count = len(net)
                changed = True
                break
        
        if time.time() > deadline - 10:
            break
        
        # If no change from end-to-start, try start-to-end
        if not changed:
            for idx in range(len(net)):
                if time.time() > deadline - 10:
                    break
                
                new_net = net[:idx] + net[idx+1:]
                
                if _is_sorted_network(new_net):
                    net = new_net
                    best_net = net[:]
                    best_count = len(net)
                    changed = True
                    break
    
    # Phase 2: Simulated Annealing with improved moves
    if time.time() < deadline - 5:
        T = 2.0
        min_T = 0.01
        cooling = 0.999
        current_count = len(net)
        stuck_counter = 0
        max_stuck = 500
        
        while time.time() < deadline - 3:
            if T < min_T:
                # Restart from best
                net = best_net[:]
                current_count = best_count
                T = 2.0
                stuck_counter = 0
            
            # Generate move
            r = random.random()
            if r < 0.35:
                # Remove
                if len(net) == 0:
                    break
                idx = random.randint(0, len(net) - 1)
                new_net = net[:idx] + net[idx+1:]
            elif r < 0.65:
                # Insert
                idx = random.randint(0, len(net))
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                new_net = net[:idx] + [(i, j)] + net[idx:]
            else:
                # Mutate - try swapping adjacent comparators or changing one
                if len(net) == 0:
                    continue
                if random.random() < 0.5 and len(net) > 1:
                    # Swap adjacent
                    idx = random.randint(0, len(net) - 2)
                    new_net = net[:]
                    new_net[idx], new_net[idx+1] = new_net[idx+1], new_net[idx]
                else:
                    # Change one comparator
                    idx = random.randint(0, len(net) - 1)
                    i, j = net[idx]
                    ni = random.randint(0, N - 2)
                    nj = random.randint(ni + 1, N - 1)
                    if (ni, nj) == (i, j):
                        continue
                    new_net = net[:]
                    new_net[idx] = (ni, nj)
            
            if not _is_sorted_network(new_net):
                stuck_counter += 1
                continue
                
            new_count = len(new_net)
            delta = new_count - current_count
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                net = new_net
                current_count = new_count
                stuck_counter = 0
                if current_count < best_count:
                    best_net = net[:]
                    best_count = current_count
                    
            T *= cooling
            
            # Early exit if stuck too long
            if stuck_counter > max_stuck:
                net = best_net[:]
                current_count = best_count
                T = max(T, 0.5)
                stuck_counter = 0
    
    # Phase 3: Final polish - try more aggressive greedy removal
    if time.time() < deadline - 2:
        net = best_net[:]
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            for idx in range(len(net) - 1, -1, -1):
                if time.time() > deadline - 2:
                    break
                new_net = net[:idx] + net[idx+1:]
                if _is_sorted_network(new_net):
                    net = new_net
                    best_net = net[:]
                    best_count = len(net)
                    changed = True
                    break
    
    # Final validation
    if not _is_sorted_network(best_net):
        best_net = _odd_even_transposition()
        
    return best_net
