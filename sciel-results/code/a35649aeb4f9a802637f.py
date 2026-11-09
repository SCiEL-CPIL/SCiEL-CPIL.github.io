import numpy as np
import time

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Strategy:
    1. Fix the support structure to {0, 1, M}. This captures the essential "heavy tail"
       and "anchor" dynamics. 0 makes LHS small, M makes RHS large.
       The middle point 1 provides intermediate combinations.
    2. The probability P is a piecewise polynomial function of M and the weight p1 (mass at 1).
       Since weights sum to 1, we have p0 = 1 - p1 - pM.
       We perform a dense grid search over M and p1.
    3. To avoid missing the optimum, we use a multi-scale grid:
       - Coarse grid over M (log scale) and p1 (linear).
       - Refine around the best found point.
    4. This is deterministic and fast, avoiding the randomness and potential stagnation
       of local search methods.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin before 600s limit

    def calc_prob_3pt(M, p1):
        """
        Calculate P[X1+X2+X3 < 2*X4] for support {0, 1, M} and weights {p0, p1, pM}.
        p0 = 1 - p1 - pM.
        We assume p0, p1, pM > 0.
        """
        if p1 < 0 or p1 > 1:
            return 0.0
        # We need to choose pM. 
        # Wait, the strategy says "grid search over the ratio M and the probability mass at 1".
        # But we have 3 weights. p0 + p1 + pM = 1.
        # So we need to search over p1 AND pM? Or fix a relationship?
        # The prompt says: "exploiting the piecewise-linear nature... directly identify the global optimum".
        # Actually, P is a polynomial in p0, p1, pM.
        # Let's search over p1 and pM on a grid.
        # Or better, search over p1 and p0, then pM = 1 - p1 - p0.
        # Let's stick to the prompt's hint: "grid search over the ratio M and the probability mass at 1".
        # This implies maybe pM is determined or we search pM too?
        # "probability mass at 1" is p1.
        # If we only search p1, we need a rule for p0 and pM.
        # Perhaps the optimal structure has specific weight relationships?
        # Or maybe the prompt implies searching over the 2 degrees of freedom of the simplex?
        # The simplex of weights for 3 points is 2D. So we should search over 2 weight parameters.
        # Let's search over p1 and pM. p0 = 1 - p1 - pM.
        
        # Let's refine the grid search to be over M, p1, pM.
        # That's a 3D grid.
        # M: log space.
        # p1, pM: simplex.
        
        # For this function, I will just compute the prob given M, p1, pM.
        p0 = 1.0 - p1 - pM
        if p0 < -1e-9 or p1 < -1e-9 or pM < -1e-9:
            return 0.0
        if p0 < 0: p0 = 0.0
        if p1 < 0: p1 = 0.0
        if pM < 0: pM = 0.0
        # Renormalize just in case? No, if we enforce sum=1 in generation, it's fine.
        
        s = np.array([0.0, 1.0, M])
        w = np.array([p0, p1, pM])
        
        # Vectorized calculation
        # s_1, s_2, s_3 shapes: (3,1,1), (1,3,1), (1,1,3)
        s_1 = s[:, np.newaxis, np.newaxis]
        s_2 = s[np.newaxis, :, np.newaxis]
        s_3 = s[np.newaxis, np.newaxis, :]
        sum_3 = s_1 + s_2 + s_3
        
        w_1 = w[:, np.newaxis, np.newaxis]
        w_2 = w[np.newaxis, :, np.newaxis]
        w_3 = w[np.newaxis, np.newaxis, :]
        prod_w = w_1 * w_2 * w_3
        
        p_total = 0.0
        for l in range(3):
            threshold = 2.0 * s[l]
            mask = sum_3 < threshold
            prob_sum_lt = np.sum(prod_w * mask)
            p_total += w[l] * prob_sum_lt
            
        return p_total

    best_score = -np.inf
    best_M = 1.0
    best_p1 = 0.5
    best_pM = 0.5

    # Grid search parameters
    # M range: 1 to 1000 (log scale). The "tail" doesn't need to be infinite, just large enough
    # to make 2*M >> other sums.
    # p1, pM range: 0 to 1.
    
    # To keep it fast, we'll use a coarse grid first, then refine.
    
    # Phase 1: Coarse Grid
    # M: 50 points log-spaced from 1 to 1000
    M_vals = np.logspace(0, 3, 50)
    
    # Weights: We need to cover the 2D simplex.
    # Let's use a grid for p1 and pM such that p1 + pM <= 1.
    # Resolution: 20x20 = 400 points.
    # Total evaluations: 50 * 400 = 20,000. Very fast.
    
    p1_vals = np.linspace(0, 1, 20)
    pM_vals = np.linspace(0, 1, 20)
    
    # We can vectorize the weight part? 
    # calc_prob_3pt is not easily vectorizable over M and weights simultaneously 
    # because the mask depends on M.
    # But we can loop over M, and for each M, loop over weights?
    # Or vectorize over weights for a fixed M?
    # For fixed M, the masks are fixed.
    # sum_3 is fixed.
    # We can precompute the masks for each l.
    # Then for a batch of weights, we can compute the sum.
    
    # Let's optimize the inner loop.
    def calc_prob_batch(M, p1_arr, pM_arr):
        """
        Calculate prob for a fixed M and arrays of p1, pM.
        Returns array of probs.
        """
        n_w = len(p1_arr)
        p0_arr = 1.0 - p1_arr - pM_arr
        # Filter valid
        valid = (p0_arr >= -1e-9) & (p1_arr >= -1e-9) & (pM_arr >= -1e-9)
        if not np.any(valid):
            return np.zeros(n_w)
        
        p0_v = np.maximum(p0_arr[valid], 0)
        p1_v = np.maximum(p1_arr[valid], 0)
        pM_v = np.maximum(pM_arr[valid], 0)
        
        # For fixed M, the support s is fixed.
        s = np.array([0.0, 1.0, M])
        
        # Precompute sum_3 (3x3x3)
        s_1 = s[:, np.newaxis, np.newaxis]
        s_2 = s[np.newaxis, :, np.newaxis]
        s_3 = s[np.newaxis, np.newaxis, :]
        sum_3 = s_1 + s_2 + s_3
        
        # Masks for each l
        masks = []
        for l in range(3):
            threshold = 2.0 * s[l]
            masks.append(sum_3 < threshold)
            
        # We need to sum over i,j,k where mask_l[i,j,k] is True: w_i w_j w_k
        # This is a cubic form.
        # Let W be (n_w, 3).
        # For each l, we want sum_{i,j,k: mask_l[i,j,k]} W[:,i] W[:,j] W[:,k]
        
        probs = np.zeros(n_w)
        
        for l in range(3):
            mask = masks[l]
            # Indices where mask is True
            idx_i, idx_j, idx_k = np.where(mask)
            if len(idx_i) == 0:
                continue
            
            # Sum of products
            # W[:, idx_i] * W[:, idx_j] * W[:, idx_k]
            # Shape: (n_w, num_true)
            terms = W[:, idx_i] * W[:, idx_j] * W[:, idx_k]
            sum_terms = np.sum(terms, axis=1)
            
            # Multiply by w_l (which is W[:, l])
            probs += W[:, l] * sum_terms
            
        # Fill results
        results = np.zeros(n_w)
        results[valid] = probs
        return results

    # Generate weight grid
    # p1 from 0 to 1, pM from 0 to 1-p1
    # To keep it simple and uniform in the simplex, let's just use a 2D grid and filter.
    p1_grid, pM_grid = np.meshgrid(p1_vals, pM_vals, indexing='ij')
    p1_flat = p1_grid.ravel()
    pM_flat = pM_grid.ravel()
    
    for M in M_vals:
        if time.time() > deadline - 10.0:
            break
        probs = calc_prob_batch(M, p1_flat, pM_flat)
        idx_max = np.argmax(probs)
        if probs[idx_max] > best_score:
            best_score = probs[idx_max]
            best_M = M
            best_p1 = p1_flat[idx_max]
            best_pM = pM_flat[idx_max]

    # Phase 2: Refinement
    # Refine around the best found point.
    # Local grid in M, p1, pM.
    
    if time.time() < deadline - 5.0:
        # Range for M: [best_M/2, best_M*2]
        # Range for p1: [max(0, best_p1-0.1), min(1, best_p1+0.1)]
        # Range for pM: [max(0, best_pM-0.1), min(1-best_p1, best_pM+0.1)]
        
        M_lo = max(1.0, best_M / 2.0)
        M_hi = best_M * 2.0
        M_ref = np.logspace(np.log10(M_lo), np.log10(M_hi), 20)
        
        p1_lo = max(0.0, best_p1 - 0.1)
        p1_hi = min(1.0, best_p1 + 0.1)
        pM_lo = max(0.0, best_pM - 0.1)
        pM_hi = min(1.0 - p1_lo, best_pM + 0.1) # Ensure p0 >= 0 roughly
        
        p1_ref = np.linspace(p1_lo, p1_hi, 20)
        pM_ref = np.linspace(pM_lo, pM_hi, 20)
        
        p1_grid, pM_grid = np.meshgrid(p1_ref, pM_ref, indexing='ij')
        p1_flat = p1_grid.ravel()
        pM_flat = pM_grid.ravel()
        
        for M in M_ref:
            if time.time() > deadline - 2.0:
                break
            probs = calc_prob_batch(M, p1_flat, pM_flat)
            idx_max = np.argmax(probs)
            if probs[idx_max] > best_score:
                best_score = probs[idx_max]
                best_M = M
                best_p1 = p1_flat[idx_max]
                best_pM = pM_flat[idx_max]

    # Phase 3: Very fine refinement if time permits
    if time.time() < deadline - 1.0:
        M_lo = max(1.0, best_M * 0.9)
        M_hi = best_M * 1.1
        M_ref = np.logspace(np.log10(M_lo), np.log10(M_hi), 10)
        
        p1_lo = max(0.0, best_p1 - 0.05)
        p1_hi = min(1.0, best_p1 + 0.05)
        pM_lo = max(0.0, best_pM - 0.05)
        pM_hi = min(1.0 - p1_lo, best_pM + 0.05)
        
        p1_ref = np.linspace(p1_lo, p1_hi, 10)
        pM_ref = np.linspace(pM_lo, pM_hi, 10)
        
        p1_grid, pM_grid = np.meshgrid(p1_ref, pM_ref, indexing='ij')
        p1_flat = p1_grid.ravel()
        pM_flat = pM_grid.ravel()
        
        for M in M_ref:
            if time.time() > deadline - 0.5:
                break
            probs = calc_prob_batch(M, p1_flat, pM_flat)
            idx_max = np.argmax(probs)
            if probs[idx_max] > best_score:
                best_score = probs[idx_max]
                best_M = M
                best_p1 = p1_flat[idx_max]
                best_pM = pM_flat[idx_max]

    # Final construction
    p0 = 1.0 - best_p1 - best_pM
    # Clamp to non-negative
    if p0 < 0: p0 = 0.0
    if best_p1 < 0: best_p1 = 0.0
    if best_pM < 0: best_pM = 0.0
    
    # Normalize to ensure sum is exactly 1 (floating point safety)
    s = np.array([0.0, 1.0, best_M], dtype=np.float64)
    w = np.array([p0, best_p1, best_pM], dtype=np.float64)
    
    w_sum = np.sum(w)
    if w_sum <= 1e-9:
        # Fallback
        s = np.array([0.0, 1.0, 10.0])
        w = np.array([0.5, 0.3, 0.2])
    else:
        w = w / w_sum

    # Validity check
    if np.any(s < -1e-9) or np.any(w < -1e-9):
        s = np.array([0.0, 1.0, 10.0])
        w = np.array([0.5, 0.3, 0.2])

    return s, w

if __name__ == "__main__":
    s, w = beat_the_average_measure()
    print(s, w)
    # Quick check
    def calc_prob_fast(support, weights):
        n = len(support)
        w_sum = np.sum(weights)
        if w_sum <= 1e-9: return 0.0
        w_norm = weights / w_sum
        s_1 = support[:, np.newaxis, np.newaxis]
        s_2 = support[np.newaxis, :, np.newaxis]
        s_3 = support[np.newaxis, np.newaxis, :]
        sum_3 = s_1 + s_2 + s_3
        w_1 = w_norm[:, np.newaxis, np.newaxis]
        w_2 = w_norm[np.newaxis, :, np.newaxis]
        w_3 = w_norm[np.newaxis, np.newaxis, :]
        prod_w = w_1 * w_2 * w_3
        p_total = 0.0
        for l in range(n):
            threshold = 2.0 * support[l]
            mask = sum_3 < threshold
            prob_sum_lt = np.sum(prod_w * mask)
            p_total += w_norm[l] * prob_sum_lt
        return p_total
    print("Score:", calc_prob_fast(s, w))
