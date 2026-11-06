import numpy as np
import time

def construct():
    deadline = time.time() + 115.0  # Safety margin for 120s limit
    
    N_COEFF = 24
    sqrt_N = np.sqrt(N_COEFF)
    
    # Precompute z^k for k=0..23 on a dense grid for fast scoring
    # N=2048 is a good balance. The grid spacing is ~0.003 radians.
    N_GRID = 2048
    theta = 2 * np.pi * np.arange(N_GRID) / N_GRID
    z = np.exp(1j * theta)
    # Z[k, m] = z_m^k
    Z = z[None, :] ** np.arange(N_COEFF)[:, None]
    
    def score_fast(coeffs):
        """
        Fast evaluation using dense grid only.
        Returns max |P(z)| on the grid.
        """
        vals = coeffs @ Z
        return np.max(np.abs(vals))

    def score_precise(coeffs):
        """
        Precise evaluation with golden-section refinement of top peaks.
        Returns max |P(z)|.
        """
        vals = coeffs @ Z
        abs_vals = np.abs(vals)
        m = np.max(abs_vals)
        
        # Refine top peaks with golden section
        # We only need to refine if the grid max is close to the true max.
        # The grid max is a lower bound.
        # If the true max is significantly higher, it must be in a peak region.
        top_k = 10
        idx = np.argpartition(abs_vals, -top_k)[-top_k:]
        best = m
        
        dtheta = 2 * np.pi / N_GRID
        gr = (np.sqrt(5) - 1) / 2
        
        for i in idx:
            # Define interval around the peak
            # Handle circularity if needed, but for interior peaks it's fine.
            # If i is 0 or N-1, wrap around.
            lo_idx = (i - 1) % N_GRID
            hi_idx = (i + 1) % N_GRID
            
            # Unwrap angles if necessary for golden section
            # Simpler: just use the angle values
            a = theta[lo_idx]
            b = theta[hi_idx]
            
            # Ensure a < b for golden section
            if a > b:
                b += 2 * np.pi
            
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            
            # Evaluate P at c and d
            # We compute z^k for z = exp(i*c) and z = exp(i*d)
            # To avoid recomputing powers from scratch, we can use the fact that
            # we are just evaluating a dot product.
            # 24 multiplications is cheap.
            
            for _ in range(40):
                zc = np.exp(1j * c)
                zd = np.exp(1j * d)
                
                # Compute P(zc) and P(zd)
                # Using dot product with precomputed powers? No, powers depend on z.
                # So we compute zc^k and zd^k.
                # Vectorized power calculation for 24 elements is fast.
                kc = np.arange(N_COEFF)
                pc = np.sum(coeffs * (zc ** kc))
                pd = np.sum(coeffs * (zd ** kc))
                
                ac = abs(pc)
                ad = abs(pd)
                
                if ac < ad:
                    b = d
                    d = c
                    zd = zc
                    ad = ac
                    c = b - gr * (b - a)
                else:
                    a = c
                    c = d
                    zc = zd
                    ac = ad
                    d = a + gr * (b - a)
                
                if time.time() > deadline - 5:
                    break
            
            zm = np.exp(1j * (a + b) / 2)
            pm = np.sum(coeffs * (zm ** np.arange(N_COEFF)))
            am = abs(pm)
            if am > best:
                best = am
        
        return best

    best_coeffs = None
    best_score = float('inf')
    
    # Generate Seeds
    rng = np.random.default_rng(42)
    
    # Seed 1: Golay-Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < N_COEFF:
        p, q = p + q, p + [-c for c in q]
    gr_coeffs = np.array(p[:N_COEFF], dtype=np.float64)
    
    seeds = [gr_coeffs]
    
    # Add random seeds
    for _ in range(30):
        seeds.append(rng.choice([-1.0, 1.0], size=N_COEFF))
    
    # Local search from each seed
    for s_idx, seed in enumerate(seeds):
        if time.time() > deadline - 10:
            break
        
        coeffs = seed.copy()
        cur_score = score_precise(coeffs)
        cur_fast_score = score_fast(coeffs)
        
        if cur_score < best_score:
            best_score = cur_score
            best_coeffs = coeffs.copy()
            
        # Precompute current P(z) on grid for fast neighbor evaluation
        P_vals = coeffs @ Z
        
        improved = True
        # Use a loop that allows multiple flips
        # Alternate between single and double flips to escape local optima
        flip_type = 0 
        while improved and time.time() < deadline - 15:
            improved = False
            best_flip = None
            best_flip_score = cur_score
            best_flip_fast = cur_fast_score
            best_flip_indices = None
            
            if flip_type == 0:
                # Single flip
                for k in range(N_COEFF):
                    # P_new = P - 2 * c_k * z^k
                    P_new = P_vals - 2.0 * coeffs[k] * Z[k]
                    fast_score = np.max(np.abs(P_new))
                    
                    if fast_score < best_flip_fast:
                        new_coeffs = coeffs.copy()
                        new_coeffs[k] *= -1
                        precise_score = score_precise(new_coeffs)
                        
                        if precise_score < best_flip_score:
                            best_flip_score = precise_score
                            best_flip_fast = fast_score
                            best_flip = k
                            best_flip_indices = [k]
            else:
                # Double flip
                # Try all pairs (i, j) with i < j
                # There are 24*23/2 = 276 pairs.
                # This is more expensive, so we do it less frequently.
                for i in range(N_COEFF):
                    for j in range(i + 1, N_COEFF):
                        # P_new = P - 2*c_i*z^i - 2*c_j*z^j
                        P_new = P_vals - 2.0 * coeffs[i] * Z[i] - 2.0 * coeffs[j] * Z[j]
                        fast_score = np.max(np.abs(P_new))
                        
                        if fast_score < best_flip_fast:
                            new_coeffs = coeffs.copy()
                            new_coeffs[i] *= -1
                            new_coeffs[j] *= -1
                            precise_score = score_precise(new_coeffs)
                            
                            if precise_score < best_flip_score:
                                best_flip_score = precise_score
                                best_flip_fast = fast_score
                                best_flip_indices = [i, j]
            
            if best_flip_indices is not None:
                for k in best_flip_indices:
                    coeffs[k] *= -1
                cur_score = best_flip_score
                cur_fast_score = best_flip_fast
                P_vals = coeffs @ Z
                improved = True
                if cur_score < best_score:
                    best_score = cur_score
                    best_coeffs = coeffs.copy()
            
            # Switch flip type occasionally to ensure exploration
            if not improved:
                flip_type = 1 - flip_type
            else:
                # If we found an improvement, keep the same type for a while?
                # Or switch to explore more?
                # Let's switch after a few improvements to escape deeper local minima
                if flip_type == 0 and rng.random() < 0.1:
                    flip_type = 1
                elif flip_type == 1 and rng.random() < 0.5:
                    flip_type = 0

    # Final polish: try to improve best_coeffs with more precise search if time remains
    if best_coeffs is not None and time.time() < deadline - 5:
        # Try random perturbations of the best solution
        for _ in range(50):
            if time.time() > deadline - 2:
                break
            cand = best_coeffs.copy()
            # Flip 1 or 2 random coefficients
            n_flips = rng.integers(1, 3)
            for _ in range(n_flips):
                k = rng.integers(0, N_COEFF)
                cand[k] *= -1
            s = score_precise(cand)
            if s < best_score:
                best_score = s
                best_coeffs = cand.copy()

    if best_coeffs is None:
        return [1] * N_COEFF
    return [int(c) for c in best_coeffs]
