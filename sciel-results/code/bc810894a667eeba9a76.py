# EVOLVE-BLOCK-START
import time
import numpy as np

def kakeya_needle_3d8() -> np.ndarray:
    """
    Discrete 3-D Kakeya set: union of 64 sheared tubes (N = 8).
    
    Strategy: Minimize integral_0^1 U(c)^2 dc where U(c) is 1D union length.
    Uses vectorized union computation with many c-samples and Nelder-Mead refinement.
    """
    N = 8
    deadline = time.time() + 620.0  # Safety margin
    
    def compute_1d_union_sq(x, c_samples):
        """
        Computes the mean of U(c)^2 where U(c) is the 1D union length.
        x: shape (N,) - base positions
        c_samples: shape (M,) - samples of c in [0,1]
        Returns: scalar mean of U(c)^2
        """
        i_vals = np.arange(N, dtype=np.float64)
        shear_coeffs = i_vals / N  # shape (N,)
        
        # L: (N, M), R: (N, M)
        L = x[:, None] + c_samples[None, :] * shear_coeffs[:, None]
        R = L + 1.0 / N
        
        # Transpose to (M, N) for per-sample sorting
        L_t = L.T
        R_t = R.T
        
        # Sort intervals by start point
        order = np.argsort(L_t, axis=1, kind='mergesort')
        L_sorted = np.take_along_axis(L_t, order, axis=1)
        R_sorted = np.take_along_axis(R_t, order, axis=1)
        
        # Compute union length via sweep
        max_r_up_to = np.maximum.accumulate(R_sorted, axis=1)
        max_r_before = np.empty_like(R_sorted)
        max_r_before[:, 0] = -np.inf
        max_r_before[:, 1:] = max_r_up_to[:, :-1]
        
        effective_start = np.maximum(L_sorted, max_r_before)
        contributions = np.maximum(0, R_sorted - effective_start)
        union_lens = np.sum(contributions, axis=1)
        
        return np.mean(union_lens ** 2)

    # Use more c-samples for better accuracy
    rng = np.random.default_rng(42)
    M_obj = 200_000
    c_samples = rng.uniform(0, 1, M_obj)
    
    def objective(x):
        """Evaluate objective for candidate x (shape N,)."""
        if time.time() > deadline:
            return 1e9
        val = compute_1d_union_sq(x, c_samples)
        if not np.isfinite(val):
            return 1e9
        return val

    def make_positions(x):
        """Convert 1D positions x (shape N,) to 64x2 array."""
        positions = np.zeros((64, 2), dtype=np.float64)
        for k in range(64):
            i = k // 8
            j = k % 8
            positions[k, 0] = x[i]
            positions[k, 1] = x[j]
        return positions

    def validate_positions(pos):
        """Check if positions are valid (finite, V > 0)."""
        if not np.all(np.isfinite(pos)):
            return False
        x_k = pos[:, 0]
        y_k = pos[:, 1]
        i_vals = np.arange(64) // 8
        j_vals = np.arange(64) % 8
        
        X0 = np.min(x_k)
        X1 = np.max(x_k + (i_vals + 1) / 8.0)
        Y0 = np.min(y_k)
        Y1 = np.max(y_k + (j_vals + 1) / 8.0)
        
        if (X1 - X0) * (Y1 - Y0) <= 1e-9:
            return False
        return True

    # Generate diverse candidates
    candidates = []
    
    # 1. Centered shear alignment (known good starting point)
    k = np.arange(N, dtype=np.float64)
    candidates.append(-k / (2.0 * N))
    
    # 2. Various structured families
    # Linear with different scales
    for scale in [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0]:
        candidates.append(-k / (2.0 * N) * scale)
    
    # Quadratic family
    for a in [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5]:
        candidates.append(a * (k - (N-1)/2.0)**2 / (N-1)**2)
    
    # Sine family
    for amp in [0.2, 0.4, 0.6, 0.8, 1.0]:
        candidates.append(amp * np.sin(k * np.pi / N) / N)
    
    # Alternating patterns
    for amp in [0.1, 0.2, 0.3, 0.5]:
        candidates.append(amp * np.sin(k * np.pi) / N)
    
    # 3. Random candidates
    for _ in range(50):
        # Random positions in a reasonable range
        x_rand = np.random.uniform(-1.0, 1.0, N)
        candidates.append(x_rand)
    
    # 4. Perturbed versions of structured candidates
    good_structured = candidates[:20]
    for base in good_structured[:10]:
        for _ in range(5):
            perturbed = base + np.random.normal(0, 0.05, N)
            candidates.append(perturbed)
    
    # 5. Sort by a quick proxy: sum of squared differences (lower = more compact)
    # This gives us a rough ordering before expensive evaluation
    proxy_scores = []
    for x in candidates:
        # Proxy: variance of positions + interaction with shear
        var = np.var(x)
        # Also consider the spread after shearing at c=0.5
        c_mid = 0.5
        x_sheared = x + c_mid * np.arange(N) / N
        var_sheared = np.var(x_sheared)
        proxy = var + var_sheared
        proxy_scores.append(proxy)
    
    proxy_scores = np.array(proxy_scores)
    sort_idx = np.argsort(proxy_scores)
    
    # Evaluate candidates in order of proxy score
    best_x = candidates[0].copy()
    best_val = 1e9
    evaluated_count = 0
    max_candidates = len(candidates)
    
    # Evaluate top candidates by proxy
    eval_limit = min(max_candidates, 150)
    for idx in sort_idx[:eval_limit]:
        if time.time() > deadline - 30:
            break
        
        x = candidates[idx]
        if not np.all(np.isfinite(x)):
            continue
        
        val = objective(x)
        evaluated_count += 1
        
        if val < best_val:
            best_val = val
            best_x = x.copy()
    
    # 6. Nelder-Mead style local refinement
    def nelder_mead(x0, max_iters=500):
        """Nelder-Mead optimization."""
        x = x0.copy()
        
        # Initialize simplex
        alpha = 1.0
        beta = 0.5
        gamma = 2.0
        eps = 1e-4
        
        simplex = [x.copy()]
        for i in range(N):
            xi = x.copy()
            xi[i] += eps if x[i] == 0 else eps * x[i]
            simplex.append(xi)
        
        vals = [objective(s) for s in simplex]
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Sort simplex by objective value
            order = np.argsort(vals)
            simplex = [simplex[i] for i in order]
            vals = [vals[i] for i in order]
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            val_r = objective(x_r)
            
            if val_r < vals[0]:
                # Expansion
                x_e = centroid + gamma * (centroid - simplex[-1])
                val_e = objective(x_e)
                if val_e < val_r:
                    simplex[-1] = x_e
                    vals[-1] = val_e
                else:
                    simplex[-1] = x_r
                    vals[-1] = val_r
            elif val_r < vals[-2]:
                simplex[-1] = x_r
                vals[-1] = val_r
            else:
                # Contraction
                x_c = centroid + beta * (simplex[-1] - centroid)
                val_c = objective(x_c)
                if val_c < vals[-1]:
                    simplex[-1] = x_c
                    vals[-1] = val_c
                else:
                    # Shrink
                    for i in range(1, N + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        vals[i] = objective(simplex[i])
        
        # Return best point
        best_idx = np.argmin(vals)
        return simplex[best_idx], vals[best_idx]
    
    if time.time() < deadline - 20:
        best_x, best_val = nelder_mead(best_x)
    
    # 7. Random restarts around best
    if time.time() < deadline - 30:
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            x_trial = best_x + np.random.normal(0, 0.01, N)
            val = objective(x_trial)
            if val < best_val:
                best_val = val
                best_x = x_trial.copy()
        
        # Another Nelder-Mead
        if time.time() < deadline - 10:
            best_x, best_val = nelder_mead(best_x, max_iters=300)
    
    # 8. Final validation
    positions = make_positions(best_x)
    if not validate_positions(positions):
        # Fallback to known good centered configuration
        fallback = -k / (2.0 * N)
        positions = make_positions(fallback)
    
    return positions

# EVOLVE-BLOCK-END
