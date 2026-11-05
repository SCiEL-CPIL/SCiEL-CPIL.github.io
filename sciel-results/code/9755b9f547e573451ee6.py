import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    N = 24
    
    # Precompute DFT matrix for fast evaluation
    # Use a fine grid to resolve peaks accurately
    M_grid = 2 ** 15  # 32768 points
    theta = 2 * np.pi * np.arange(M_grid) / M_grid
    # E[i, k] = exp(i * theta[i] * k)
    E = np.exp(1j * np.outer(theta, np.arange(N)))
    
    def get_supnorm_fast(coeffs):
        """Compute sup norm of polynomial with given coefficients (fast grid)."""
        P = E @ coeffs
        return float(np.max(np.abs(P)))
    
    def get_supnorm_precise(coeffs):
        """Compute sup norm with golden-section refinement of peaks."""
        P = E @ coeffs
        absP = np.abs(P)
        
        # Find local maxima on the grid
        # A point is a local max if it's >= neighbors
        local_max_mask = np.zeros(M_grid, dtype=bool)
        local_max_mask[1:-1] = (absP[1:-1] >= absP[:-2]) & (absP[1:-1] >= absP[2:])
        local_max_mask[0] = absP[0] >= absP[1]
        local_max_mask[-1] = absP[-1] >= absP[-2]
        
        # Get top peaks
        peak_indices = np.where(local_max_mask)[0]
        if len(peak_indices) == 0:
            return float(np.max(absP))
        
        peak_values = absP[peak_indices]
        # Sort by value descending, take top 10
        top_k = min(10, len(peak_indices))
        top_idx = peak_indices[np.argsort(peak_values)[-top_k:]]
        
        max_val = float(np.max(absP))
        
        # Refine each top peak with golden section
        for idx in top_idx:
            t0 = theta[idx]
            # Search in a small window around this peak
            half_width = 2 * 2 * np.pi / M_grid  # One grid spacing on each side
            a = t0 - half_width
            b = t0 + half_width
            
            # Golden section search
            gr = (np.sqrt(5) - 1) / 2  # golden ratio conjugate
            x1 = b - gr * (b - a)
            x2 = a + gr * (b - a)
            
            def eval_at(t):
                z = np.exp(1j * t * np.arange(N))
                return float(np.abs(np.sum(coeffs * z)))
            
            f1 = eval_at(x1)
            f2 = eval_at(x2)
            
            for _ in range(50):
                if f1 < f2:
                    b = x2
                    x2 = x1
                    f2 = f1
                    x1 = b - gr * (b - a)
                    f1 = eval_at(x1)
                else:
                    a = x1
                    x1 = x2
                    f1 = f2
                    x2 = a + gr * (b - a)
                    f2 = eval_at(x2)
            
            refined_val = max(f1, f2)
            if refined_val > max_val:
                max_val = refined_val
        
        return max_val
    
    # Exhaustive search with symmetry reduction
    # Fix c[0] = +1 (sign symmetry) and c[1] = +1 (reversal symmetry)
    # This reduces from 2^24 to 2^22 = 4194304 candidates
    
    best_val = float('inf')
    best_c = None
    
    # Precompute the DFT contributions for each coefficient position
    # E[:, k] is the contribution of coefficient k
    
    # We'll iterate over all 2^22 combinations of c[2..23]
    # c[0] = 1, c[1] = 1 are fixed
    
    # For speed, precompute the base polynomial (with c[0]=1, c[1]=1)
    # and the contribution vectors for each variable coefficient
    
    base_coeffs = np.zeros(N, dtype=float)
    base_coeffs[0] = 1.0
    base_coeffs[1] = 1.0
    
    # Precompute E @ base_coeffs
    base_P = E @ base_coeffs
    
    # For each variable position k (2 to 23), precompute E[:, k]
    # When we flip c[k] from +1 to -1, the change in P is -2 * E[:, k]
    # So P = base_P + sum_{k in flipped} (-2 * E[:, k])
    
    # We'll use a Gray code approach to minimize recomputation
    # But for simplicity and speed, let's just iterate and compute
    
    # Actually, let's use a smarter approach:
    # Iterate over all 2^22 bit patterns for positions 2-23
    # Use incremental updates
    
    num_vars = N - 2  # 22 variables
    total_candidates = 1 << num_vars  # 4194304
    
    # Precompute the contribution of each variable
    # var_contrib[k] = 2 * E[:, k+2]  (the change when flipping from +1 to -1)
    # P = base_P - sum_{k: bit_k=1} var_contrib[k]
    
    var_contrib = np.zeros((num_vars, M_grid), dtype=complex)
    for k in range(num_vars):
        var_contrib[k] = 2.0 * E[:, k + 2]
    
    # Use Gray code to iterate through all combinations with minimal changes
    # Gray code: g(n) = n ^ (n >> 1)
    # Consecutive Gray codes differ in exactly one bit
    
    best_val = float('inf')
    best_c = None
    
    current_P = base_P.copy()
    current_bits = np.zeros(num_vars, dtype=np.int32)
    
    for i in range(total_candidates):
        if i > 0:
            # Find which bit changed from previous Gray code
            prev_gray = (i - 1) ^ ((i - 1) >> 1)
            curr_gray = i ^ (i >> 1)
            diff = prev_gray ^ curr_gray
            bit_pos = int(np.log2(diff)) if diff > 0 else 0
            
            # Update current_P based on the bit change
            if (curr_gray >> bit_pos) & 1:
                # Bit turned on: flip from +1 to -1, subtract var_contrib
                current_P -= var_contrib[bit_pos]
                current_bits[bit_pos] = 1
            else:
                # Bit turned off: flip from -1 to +1, add var_contrib
                current_P += var_contrib[bit_pos]
                current_bits[bit_pos] = 0
        
        # Evaluate sup norm
        val = float(np.max(np.abs(current_P)))
        
        if val < best_val:
            best_val = val
            # Reconstruct coefficients
            best_c = np.ones(N, dtype=float)
            for k in range(num_vars):
                if current_bits[k] == 1:
                    best_c[k + 2] = -1.0
        
        # Check time periodically
        if i % 100000 == 0 and i > 0:
            if time.time() > deadline - 30.0:
                break
    
    # Now refine the best candidate with precise sup norm evaluation
    if best_c is not None:
        precise_val = get_supnorm_precise(best_c)
        
        # Try a few nearby candidates too
        # Get the top 20 candidates from the fast search and refine them
        # For simplicity, just refine the best one for now
        
        if precise_val < best_val:
            best_val = precise_val
    
    # If time permits, do a more thorough refinement
    if time.time() < deadline - 10.0 and best_c is not None:
        # Try flipping individual bits of the best candidate
        for i in range(N):
            if time.time() > deadline - 5.0:
                break
            trial_c = best_c.copy()
            trial_c[i] *= -1
            trial_val = get_supnorm_precise(trial_c)
            if trial_val < best_val:
                best_val = trial_val
                best_c = trial_c
        
        # Try pairs of flips
        for i in range(N):
            if time.time() > deadline - 5.0:
                break
            for j in range(i + 1, N):
                if time.time() > deadline - 5.0:
                    break
                trial_c = best_c.copy()
                trial_c[i] *= -1
                trial_c[j] *= -1
                trial_val = get_supnorm_precise(trial_c)
                if trial_val < best_val:
                    best_val = trial_val
                    best_c = trial_c
    
    return [int(x) for x in best_c]
