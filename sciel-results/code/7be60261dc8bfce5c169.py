"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconvolution_max_abs(f):
    """
    Compute the maximum absolute value of the autoconvolution.
    We assume the metric is max |R(k)| for all k, or max |R(k)| for k != 0.
    Given the ambiguity, and that 'upper bound on C3' often relates to off-peak sidelobes,
    but the evaluator measures 'raw value', we compute max over all lags.
    Note: R(0) = sum(f^2). If we use +/-1 and 0, R(0) is fixed by sparsity.
    Usually, minimizing the peak sidelobe is the goal.
    """
    n = len(f)
    # Use FFT for efficient convolution
    # Zero-pad to avoid circular convolution effects
    size = 2 * n
    f_pad = np.zeros(size)
    f_pad[:n] = f
    F = np.fft.fft(f_pad, size)
    conv = np.fft.ifft(F * F).real
    # The convolution is symmetric. 
    # We take the max absolute value.
    return np.max(np.abs(conv))

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin before 150s limit
    
    # Strategy: Sparse Greedy + Local Search
    # We hypothesize that sparse signals (many zeros) have lower autoconvolution peaks
    # than dense +/-1 signals.
    
    best_f = np.zeros(N)
    best_val = np.inf
    
    # Try a range of sparsity levels
    # If the metric includes R(0), higher sparsity increases R(0).
    # If the metric is max_{k!=0} |R(k)|, higher sparsity might help or hurt.
    # Standard "Barker-like" sequences are dense. But "Bose-Chowla" are sparse.
    # Let's test both dense and sparse.
    
    sparsity_levels = [
        (N, "dense"), # All positions active
        (200, "semi_sparse"),
        (100, "sparse"),
        (50, "very_sparse"),
        (20, "ultra_sparse")
    ]
    
    for K, name in sparsity_levels:
        if time.time() > deadline - 10:
            break
            
        # Multiple random restarts for each sparsity
        for restart in range(5):
            if time.time() > deadline - 5:
                break
                
            # 1. Initialization: Greedy Construction
            # Start with empty set, add elements one by one at the position
            # that minimizes the current max autoconvolution.
            
            f = np.zeros(N)
            active_indices = []
            
            # Greedy placement
            for _ in range(K):
                best_pos = 0
                best_cost = np.inf
                # Sample a subset of positions to speed up greedy step if N is large
                # Here N=400 is small, so we can check all
                # But checking all 400 positions * K times * FFT cost might be slow
                # FFT is ~1ms. 400 * 100 * 1ms = 40s. Too slow for 5 restarts.
                
                # Optimization: Use a random subset of candidate positions for greedy
                candidates = np.random.choice(N, size=min(50, N), replace=False)
                
                for pos in candidates:
                    if f[pos] != 0:
                        continue
                    f[pos] = 1 # Try +1
                    cost = _autoconvolution_max_abs(f)
                    if cost < best_cost:
                        best_cost = cost
                        best_pos = pos
                    f[pos] = -1 # Try -1
                    cost = _autoconvolution_max_abs(f)
                    if cost < best_cost:
                        best_cost = cost
                        # Note: we need to track the sign too
                        # Let's store the best config
                        # For simplicity in this loop, we just find the pos, 
                        # then re-evaluate signs in the local search phase.
                        # Actually, greedy with sign is better.
                        pass 
                    f[pos] = 0 # Reset
                    
                # This greedy approach is a bit messy with signs. 
                # Let's simplify: 
                # 1. Randomly select K positions.
                # 2. Optimize signs for those positions using local search.
                # 3. Optimize positions using swaps.
                
                break # Exit candidate loop, we will use random init instead for simplicity and speed
                
            # Revised Init: Random Sparse
            np.random.seed(42 + restart + K)
            active_indices = np.random.choice(N, size=K, replace=False)
            f = np.zeros(N)
            f[active_indices] = 1 # Start all +1
            
            current_val = _autoconvolution_max_abs(f)
            
            # 2. Local Search Phase
            # A. Sign Flips
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for idx in active_indices:
                    if time.time() > deadline - 5:
                        break
                    old_val = f[idx]
                    f[idx] = -old_val
                    new_val = _autoconvolution_max_abs(f)
                    if new_val < current_val - 1e-12:
                        current_val = new_val
                        improved = True
                    else:
                        f[idx] = old_val
            
            # B. Position Swaps (Move active element to inactive position)
            # This changes the structure, not just the signs.
            # We try to move an element from a "crowded" area to an "empty" area.
            
            if time.time() < deadline - 5:
                # Determine inactive positions
                active_set = set(active_indices)
                inactive_indices = np.array([i for i in range(N) if i not in active_set])
                
                # Perform a limited number of swap attempts
                for _ in range(200):
                    if time.time() > deadline - 5:
                        break
                    
                    # Pick a random active index to move
                    i_move = active_indices[np.random.randint(K)]
                    # Pick a random inactive index to move to
                    if len(inactive_indices) == 0:
                        break
                    j_target = inactive_indices[np.random.randint(len(inactive_indices))]
                    
                    # Execute move
                    sign = f[i_move]
                    f[i_move] = 0
                    f[j_target] = sign
                    
                    new_val = _autoconvolution_max_abs(f)
                    
                    if new_val < current_val - 1e-12:
                        current_val = new_val
                        # Update active/inactive lists
                        active_indices[active_indices == i_move] = j_target
                        active_set.remove(i_move)
                        active_set.add(j_target)
                        inactive_indices[inactive_indices == j_target] = i_move
                        # Note: lists are not sorted, but we only use them for random selection
                    else:
                        # Revert
                        f[j_target] = 0
                        f[i_move] = sign
                        active_indices[active_indices == j_target] = i_move
                        active_set.remove(j_target)
                        active_set.add(i_move)
                        inactive_indices[inactive_indices == i_move] = j_target

            if current_val < best_val:
                best_val = current_val
                best_f = f.copy()
                
    # Final check: If best_val is still inf (should not happen), return zeros
    if best_val == np.inf:
        return np.zeros(N)
        
    return best_f
# EVOLVE-BLOCK-END
