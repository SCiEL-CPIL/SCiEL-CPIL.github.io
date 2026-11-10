import numpy as np
import time

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    delta = 1.0 / N
    
    # Deadline guard
    deadline = time.time() + 620.0
    
    # Helper: compute exact union volume (optimized)
    def compute_volume(X, Y):
        """
        X, Y: arrays of length N (base positions for each i or j)
        Computes V = integral_0^1 Lx(c) * Ly(c) dc
        """
        def get_endpoint_arrays(pos):
            """Return arrays A, B, is_right for 2N endpoints"""
            idxs = np.arange(N)
            B = idxs / N
            A_left = pos
            A_right = pos + delta
            A = np.concatenate([A_left, A_right])
            B_full = np.concatenate([B, B])
            is_right = np.concatenate([np.zeros(N, dtype=int), np.ones(N, dtype=int)])
            return A, B_full, is_right
        
        Ax, Bx, rx = get_endpoint_arrays(X)
        Ay, By, ry = get_endpoint_arrays(Y)
        
        # Find critical c values
        def find_critical_vec(A, B):
            """Vectorized critical point finding"""
            n = len(A)
            crits = []
            for i in range(n):
                for j in range(i+1, n):
                    if abs(B[i] - B[j]) > 1e-15:
                        c = (A[j] - A[i]) / (B[i] - B[j])
                        if 0.0 < c < 1.0:
                            crits.append(c)
            return set(crits)
        
        crit_x = find_critical_vec(Ax, Bx)
        crit_y = find_critical_vec(Ay, By)
        all_crit = sorted(crit_x | crit_y)
        all_crit = [0.0] + [c for c in all_crit if 0.0 < c < 1.0] + [1.0]
        all_crit = list(dict.fromkeys(all_crit))
        all_crit.sort()
        
        # Precompute endpoint values at all critical points
        n_crit = len(all_crit)
        # Shape: (n_crit, 2N)
        vals_x = Ax[None, :] + Bx[None, :] * np.array(all_crit)[:, None]
        vals_y = Ay[None, :] + By[None, :] * np.array(all_crit)[:, None]
        
        # Sort endpoints at each critical point and compute union lengths
        def compute_union_lengths(vals, is_right):
            """Compute union lengths at all critical points. vals shape (n_crit, 2N)"""
            n_c, n_e = vals.shape
            lengths = np.zeros(n_c)
            for k in range(n_c):
                v = vals[k]
                order = np.argsort(v)
                v_sorted = v[order]
                r_sorted = is_right[order]
                # Sweep to compute union length
                count = 0
                total = 0.0
                start = 0.0
                for idx in range(n_e):
                    if r_sorted[idx] == 0:
                        if count == 0:
                            start = v_sorted[idx]
                        count += 1
                    else:
                        count -= 1
                        if count == 0:
                            total += v_sorted[idx] - start
                lengths[k] = total
            return lengths
        
        Lx = compute_union_lengths(vals_x, rx)
        Ly = compute_union_lengths(vals_y, ry)
        
        # Integrate piecewise
        vol = 0.0
        for k in range(n_crit - 1):
            h = all_crit[k+1] - all_crit[k]
            if h < 1e-15:
                continue
            Lx0, Lx1 = Lx[k], Lx[k+1]
            Ly0, Ly1 = Ly[k], Ly[k+1]
            dx = Lx1 - Lx0
            dy = Ly1 - Ly0
            term = Lx0 * Ly0 + 0.5 * (Lx0 * dy + Ly0 * dx) + (1.0 / 3.0) * dx * dy
            vol += h * term
        
        return vol
    
    # Helper: simulate Monte Carlo volume (for validation only, not used in optimization)
    def mc_volume(positions, n_samples=100000):
        rng = np.random.default_rng(12345)
        X0 = min(positions[:, 0])
        X1 = max(positions[:, 0] + 1.0/8)
        Y0 = min(positions[:, 1])
        Y1 = max(positions[:, 1] + 1.0/8)
        a = rng.uniform(X0, X1, n_samples)
        b = rng.uniform(Y0, Y1, n_samples)
        c = rng.uniform(0, 1, n_samples)
        
        inside = np.zeros(n_samples, dtype=bool)
        for k in range(64):
            i = k // 8
            j = k % 8
            xk = positions[k, 0]
            yk = positions[k, 1]
            # Tube k at height c: [xk + c*i/8, xk + c*i/8 + 1/8] x [yk + c*j/8, yk + c*j/8 + 1/8]
            lo_x = xk + c * i / 8
            hi_x = lo_x + 1.0 / 8
            lo_y = yk + c * j / 8
            hi_y = lo_y + 1.0 / 8
            mask = (a >= lo_x) & (a <= hi_x) & (b >= lo_y) & (b <= hi_y)
            inside |= mask
        
        frac = np.mean(inside)
        return frac * (X1 - X0) * (Y1 - Y0)
    
    def build_positions(X, Y):
        """Build 64x2 positions array from X (length 8) and Y (length 8)"""
        i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
        i_flat = i_grid.flatten()
        j_flat = j_grid.flatten()
        base_x = X[i_flat]
        base_y = Y[j_flat]
        return np.stack([base_x, base_y], axis=1)
    
    # Objective for optimization
    def objective(params):
        X = params[:N]
        Y = params[N:]
        return compute_volume(X, Y)
    
    # Initial guess: try several heuristics
    # Heuristic 1: Keich-like construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps_bits = np.array(list(bin(idx)[2:]), dtype=int)
        eps_bits = np.pad(eps_bits, (pos - len(eps_bits), 0), 'constant')
        keich[idx] = np.sum((1 - j_terms) * eps_bits * (2.0 ** -j_terms) / pos)
    
    # Heuristic 2: Uniform spread
    uniform = np.linspace(0, 0.5, N)
    
    # Heuristic 3: Symmetric spread (centered)
    symmetric = np.linspace(-0.25, 0.25, N)
    
    # Heuristic 4: Geometric spread
    geo = np.geomspace(0.01, 0.49, N)
    
    # Heuristic 5: Random restarts
    rng_init = np.random.default_rng(42)
    
    # Start with best of initial guesses
    best_v = float('inf')
    best_p = None
    
    candidates = [
        np.concatenate([keich, keich]),
        np.concatenate([uniform, uniform]),
        np.concatenate([symmetric, symmetric]),
        np.concatenate([geo, geo]),
    ]
    
    for _ in range(10):
        rand_x = rng_init.uniform(-0.3, 0.8, N)
        rand_y = rng_init.uniform(-0.3, 0.8, N)
        candidates.append(np.concatenate([rand_x, rand_y]))
    
    for p0 in candidates:
        v = objective(p0)
        if v < best_v:
            best_v = v
            best_p = p0.copy()
    
    # Local optimization using gradient-free methods
    import scipy.optimize as so
    
    t_start = time.time()
    
    # Multiple restarts of Nelder-Mead
    for restart in range(15):
        if time.time() > deadline - 120:
            break
        
        if restart == 0:
            p_current = best_p.copy()
        else:
            # Perturb the best solution with varying scales
            scale = 0.03 + 0.02 * restart
            perturbation = rng_init.normal(0, scale, 2*N)
            p_current = best_p + perturbation
        
        try:
            res = so.minimize(objective, p_current, method='Nelder-Mead',
                              options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-9,
                                       'adaptive': True})
            if res.fun < best_v:
                best_v = res.fun
                best_p = res.x.copy()
        except Exception:
            pass
        
        # Try Powell from current best
        if time.time() > deadline - 120:
            break
        try:
            res2 = so.minimize(objective, best_p, method='Powell',
                               options={'maxiter': 800, 'xtol': 1e-9, 'ftol': 1e-9})
            if res2.fun < best_v:
                best_v = res2.fun
                best_p = res2.x.copy()
        except Exception:
            pass
    
    # Final high-precision polish pass
    if time.time() < deadline - 60:
        try:
            res3 = so.minimize(objective, best_p, method='Nelder-Mead',
                               options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12,
                                        'adaptive': True})
            if res3.fun < best_v:
                best_v = res3.fun
                best_p = res3.x.copy()
        except Exception:
            pass
        
        # Additional Powell polish
        if time.time() < deadline - 30:
            try:
                res4 = so.minimize(objective, best_p, method='Powell',
                                   options={'maxiter': 2000, 'xtol': 1e-12, 'ftol': 1e-12})
                if res4.fun < best_v:
                    best_v = res4.fun
                    best_p = res4.x.copy()
            except Exception:
                pass
    
    X_opt = best_p[:N]
    Y_opt = best_p[N:]
    
    positions = build_positions(X_opt, Y_opt)
    
    # Validate: check finite and non-degenerate
    if not np.all(np.isfinite(positions)):
        # Fallback to initial guess
        positions = build_positions(keich, keich)
    
    return positions
