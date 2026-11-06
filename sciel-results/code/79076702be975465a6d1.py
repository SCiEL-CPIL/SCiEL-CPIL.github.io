import time
import random
import numpy as np

N = 18
MAX_COMPARATORS = 4096
DEADLINE = 55.0  # Safety margin before 60s kill

# Precompute full matrix of all 2^N boolean inputs
# Each row is a distinct input vector
all_masks = np.arange(1 << N, dtype=np.int32)
W0_full = np.zeros((1 << N, N), dtype=np.uint8)
for i in range(N):
    W0_full[:, i] = (all_masks >> i) & 1

# Precompute a subset for fast rejection
# Use a mix of low, high, and random inputs to catch errors early
subset_indices = np.unique(np.concatenate([
    np.arange(0, 200),       # Low weights
    np.arange((1 << N) - 200, 1 << N), # High weights
    np.random.choice(1 << N, 2000, replace=False)
]))
W0_subset = W0_full[subset_indices]

def _odd_even_transposition(n):
    """Generate the odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(n) for i in range(phase % 2, n - 1, 2)]

def _verify_subset(net):
    """
    Verify network on the subset of inputs.
    Returns True if sorted, False otherwise.
    """
    if not net:
        return True
        
    W = W0_subset.copy()
    for i, j in net:
        col_i = W[:, i]
        col_j = W[:, j]
        # min = AND, max = OR for bits
        tmp = col_i | col_j
        col_i[:] = col_i & col_j
        col_j[:] = tmp
        
    # Check if sorted: W[:, k] <= W[:, k+1] for all k
    # i.e., no 1 before 0
    # W[:, :-1] > W[:, 1:] should be all False
    if np.any(W[:, :-1] > W[:, 1:]):
        return False
    return True

def _verify_full(net):
    """
    Verify network on all inputs.
    Returns True if sorted, False otherwise.
    """
    if not net:
        return True
        
    W = W0_full.copy()
    for i, j in net:
        col_i = W[:, i]
        col_j = W[:, j]
        tmp = col_i | col_j
        col_i[:] = col_i & col_j
        col_j[:] = tmp
        
    if np.any(W[:, :-1] > W[:, 1:]):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Start with Odd-Even Transposition network
    net = _odd_even_transposition(N)
    best_net = net[:]
    best_count = len(net)
    
    # Phase 1: Greedy Removal
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
            
            # Quick subset check
            if _verify_subset(new_net):
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
                
                if _verify_subset(new_net):
                    if _verify_full(new_net):
                        net = new_net
                        best_net = net[:]
                        best_count = len(net)
                        changed = True
                        break
    
    # Phase 2: Simulated Annealing
    if time.time() < deadline - 3:
        T = 1.0
        min_T = 0.05
        cooling = 0.9995
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
            
            # Verify
            if not _verify_subset(new_net):
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
