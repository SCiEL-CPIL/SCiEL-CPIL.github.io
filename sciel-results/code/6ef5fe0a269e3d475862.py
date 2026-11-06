import time
import random
import numpy as np

N = 18
MAX_COMPARATORS = 4096
DEADLINE_SECONDS = 55  # Safety margin before 60s kill

# Precompute full matrix of all 2^N boolean inputs
# Using int32 to allow bitwise operations, though int8 is enough for values 0/1
# But bitwise ops on int8 might be slower or require casting. 
# Let's stick to the efficient approach: store as uint8, use bitwise.
all_masks = np.arange(1 << N, dtype=np.int32)
W0_full = np.zeros((1 << N, N), dtype=np.uint8)
for i in range(N):
    W0_full[:, i] = (all_masks >> i) & 1

# Precompute a subset for fast verification
# Include inputs with low and high weights (boundary cases) and random samples
subset_indices = np.unique(np.concatenate([
    np.arange(0, 100),       # Low weights
    np.arange((1 << N) - 100, 1 << N), # High weights
    np.random.choice(1 << N, 2000, replace=False)
]))
W0_subset = W0_full[subset_indices]

def _depth(net):
    """Compute the parallel depth of a comparator network."""
    if not net:
        return 0
    layer = [0] * N
    for i, j in net:
        layer[i] = max(layer[i], layer[j]) + 1
    return max(layer) if layer else 0

def _verify_full(net):
    """
    Verify that the comparator network sorts all 2^N boolean inputs.
    Uses the 0-1 principle with bitwise operations.
    Returns True if sorted, False otherwise.
    """
    n = len(net)
    if n > MAX_COMPARATORS:
        return False
    if n == 0:
        return True
    
    W = W0_full.copy()
    for i, j in net:
        # min = AND, max = OR for bits
        # Using in-place operations to avoid temporary arrays where possible
        # Note: W[:, i] and W[:, j] are views
        col_i = W[:, i]
        col_j = W[:, j]
        
        # We need to be careful with in-place bitwise ops on overlapping views? 
        # No, i != j.
        # tmp = col_i | col_j
        # col_i[:] = col_i & col_j
        # col_j[:] = tmp
        
        # To optimize, we can do:
        # min_val = col_i & col_j
        # max_val = col_i | col_j
        # W[:, i] = min_val
        # W[:, j] = max_val
        
        # NumPy bitwise ops on uint8 are fast.
        tmp = col_i | col_j
        col_i[:] = col_i & col_j
        col_j[:] = tmp
        
    # Check if sorted: W[:, k] <= W[:, k+1] for all k
    # Equivalent to: no 1 followed by 0
    # W[:, :-1] > W[:, 1:]
    if np.any(W[:, :-1] > W[:, 1:]):
        return False
    return True

def _verify_fast(net):
    """
    Fast verification using a subset of inputs.
    Returns True if the subset is sorted. Note: This is a necessary but not sufficient condition.
    """
    n = len(net)
    if n > MAX_COMPARATORS:
        return False
    if n == 0:
        return True
    
    W = W0_subset.copy()
    for i, j in net:
        col_i = W[:, i]
        col_j = W[:, j]
        tmp = col_i | col_j
        col_i[:] = col_i & col_j
        col_j[:] = tmp
        
    if np.any(W[:, :-1] > W[:, 1:]):
        return False
    return True

def _batcher_odd_merge(n):
    """
    Generate Batcher's Odd-Merge sorting network for n wires.
    Returns a list of (i, j) tuples.
    """
    if n == 1:
        return []
    
    # Split into two parts
    n1 = n // 2
    n2 = n - n1
    
    # Recursively sort the two parts
    net1 = _batcher_odd_merge(n1)
    net2 = _batcher_odd_merge(n2)
    
    # Offset net2 by n1
    net2_offset = [(i + n1, j + n1) for i, j in net2]
    
    # Odd-even merge
    net_merge = _odd_even_merge(n1, n2)
    
    return net1 + net2_offset + net_merge

def _odd_even_merge(n1, n2):
    """
    Perform the odd-even merge step for two sorted lists of sizes n1 and n2.
    Assumes the first list is on wires 0..n1-1 and the second on n1..n1+n2-1.
    """
    comparators = []
    
    # The odd-even merge algorithm involves a series of comparator stages.
    # For simplicity and correctness, we can implement the recursive definition.
    # However, a direct iterative implementation is complex.
    # Let's use a known property: Batcher's merge is defined recursively.
    
    # If n1 == 0 or n2 == 0, no merge needed
    if n1 == 0 or n2 == 0:
        return []
    
    # Base case
    if n1 == 1 and n2 == 1:
        return [(0, 1)]
    
    # Split the combined array into odd and even positions
    # The odd-even merge splits the input into two sub-merges:
    # One for odd indices and one for even indices.
    
    # This implementation is tricky to get right without a reference.
    # Alternative: Use a standard library or a simpler seed if Batcher is too complex to implement correctly quickly.
    # Let's fall back to Odd-Even Transposition if Batcher implementation is risky, 
    # BUT the prompt suggests Batcher is a good baseline.
    
    # Let's implement a simpler, known-good sorting network generator: 
    # Batcher's Bitonic Sort is easier to implement correctly?
    # Bitonic Sort for N=18 (not power of 2) is also tricky.
    
    # Let's stick to Odd-Even Transposition for the seed, as it is guaranteed correct and simple.
    # The improvement will come from the *search* and *verification speed*.
    return _odd_even_transposition(n1 + n2) # Fallback to simple seed for safety

def _odd_even_transposition(n):
    """Generate the odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(n) for i in range(phase % 2, n - 1, 2)]

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Start with odd-even transposition network
    net = _odd_even_transposition(N)
    best_net = net[:]
    best_count = len(net)
    
    # Phase 1: Greedy removal
    # Use fast verify for speed, but only update best if full verify passes
    changed = True
    while changed:
        if time.time() > deadline - 5:
            break
        changed = False
        
        # Try removing from end to start
        for idx in range(len(net) - 1, -1, -1):
            if time.time() > deadline - 5:
                return best_net
            
            new_net = net[:idx] + net[idx+1:]
            
            # Quick check
            if _verify_fast(new_net):
                # Full check
                if _verify_full(new_net):
                    net = new_net
                    best_net = net[:]
                    best_count = len(net)
                    changed = True
                    break
        
        # If no change from end-to-start, try start-to-end
        if not changed:
            for idx in range(len(net)):
                if time.time() > deadline - 5:
                    return best_net
                
                new_net = net[:idx] + net[idx+1:]
                
                if _verify_fast(new_net):
                    if _verify_full(new_net):
                        net = new_net
                        best_net = net[:]
                        best_count = len(net)
                        changed = True
                        break
    
    # Phase 2: Simulated Annealing
    if time.time() < deadline - 3:
        T = 1.0
        min_T = 0.01
        cooling = 0.999
        current_count = len(net)
        
        while time.time() < deadline - 2:
            if T < min_T:
                # Restart from best
                net = best_net[:]
                current_count = best_count
                T = 1.0
                
            # Generate move
            r = random.random()
            if r < 0.4:
                # Remove
                if len(net) == 0:
                    break
                idx = random.randint(0, len(net) - 1)
                new_net = net[:idx] + net[idx+1:]
            elif r < 0.7:
                # Insert
                idx = random.randint(0, len(net))
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                new_net = net[:idx] + [(i, j)] + net[idx:]
            else:
                # Mutate
                if len(net) == 0:
                    continue
                idx = random.randint(0, len(net) - 1)
                i, j = net[idx]
                ni = random.randint(0, N - 2)
                nj = random.randint(ni + 1, N - 1)
                if (ni, nj) == (i, j):
                    continue
                new_net = net[:]
                new_net[idx] = (ni, nj)
            
            # Two-stage verification
            if not _verify_fast(new_net):
                continue
                
            if not _verify_full(new_net):
                continue
                
            new_count = len(new_net)
            delta = new_count - current_count
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                net = new_net
                current_count = new_count
                if current_count < best_count:
                    best_net = net[:]
                    best_count = current_count
                    
            T *= cooling
    
    # Final validation
    if not _verify_full(best_net):
        best_net = _odd_even_transposition(N)
        
    return best_net
