import numpy as np
from scipy.optimize import minimize
import time

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    w = 1.0 / N
    deadline = time.time() + 600.0

    # Keich binary construction for initial guess
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)

    U0 = keich.copy()

    # Precompute indices for final position construction
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()

    def union_length(U, c):
        """
        Computes the length of the union of N intervals [U_i + c*i/N, U_i + c*i/N + w].
        Vectorized merge.
        """
        starts = U + c * np.arange(N) / N
        ends = starts + w
        
        # Sort intervals by start point
        order = np.argsort(starts)
        s = starts[order]
        e = ends[order]
        
        # Merge overlapping intervals
        # We can do this with a simple loop since N=8 is tiny, or vectorized.
        # Loop is fast enough and robust.
        length = 0.0
        cur_s = s[0]
        cur_e = e[0]
        for i in range(1, N):
            if s[i] > cur_e:
                length += cur_e - cur_s
                cur_s = s[i]
                cur_e = e[i]
            else:
                if e[i] > cur_e:
                    cur_e = e[i]
        length += cur_e - cur_s
        return length

    def objective(params):
        """
        Minimizes the integral of L(c)^2 over c in [0, 1].
        This corresponds to minimizing the volume when U_x = U_y = U.
        Uses a fixed grid for determinism and speed.
        """
        if time.time() > deadline - 5:
            return 1e9
            
        U = params
        # Use a fine grid for integration. 200 points is a good balance.
        c_vals = np.linspace(0, 1, 200)
        total = 0.0
        for c in c_vals:
            l = union_length(U, c)
            total += l * l
        return total / len(c_vals)

    # Optimize U
    # Start from Keich
    best_U = U0.copy()
    best_score = objective(best_U)

    # Strategy:
    # 1. L-BFGS-B for fast gradient-based local refinement (good for smooth parts, but union length has kinks).
    #    However, the objective is continuous and differentiable almost everywhere.
    #    L-BFGS-B can handle this well if we use a fine grid.
    # 2. Nelder-Mead for robustness against kinks.
    # 3. Multiple restarts with noise.

    # Phase 1: L-BFGS-B from Keich
    try:
        res = minimize(objective, best_U, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        if res.fun < best_score:
            best_score = res.fun
            best_U = res.x.copy()
    except Exception:
        pass

    # Phase 2: Nelder-Mead polish from best_U
    if time.time() < deadline - 30:
        try:
            res = minimize(objective, best_U, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_score:
                best_score = res.fun
                best_U = res.x.copy()
        except Exception:
            pass

    # Phase 3: Multi-start Nelder-Mead with noise
    rng = np.random.default_rng(42)
    for restart in range(20):
        if time.time() > deadline - 10:
            break
        
        # Generate a noisy starting point around the current best
        # Noise scale decreases as we go? Or fixed?
        # Let's try a mix: some small noise, some larger.
        scale = 0.1 / (restart + 1)
        if restart == 0:
            start = best_U.copy()
        else:
            start = best_U + rng.normal(0, scale, size=N)
        
        try:
            res = minimize(objective, start, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_score:
                best_score = res.fun
                best_U = res.x.copy()
        except Exception:
            pass

    # Final high-precision polish
    if time.time() < deadline - 5:
        try:
            res = minimize(objective, best_U, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14})
            if res.fun < best_score:
                best_score = res.fun
                best_U = res.x.copy()
        except Exception:
            pass

    # Construct the 64 positions
    base_x = best_U[i_flat]
    base_y = best_U[j_flat]
    
    positions = np.stack([base_x, base_y], axis=1)
    
    # Sanity check: ensure finite
    if not np.all(np.isfinite(positions)):
        # Fallback to Keich
        base_x = U0[i_flat]
        base_y = U0[j_flat]
        positions = np.stack([base_x, base_y], axis=1)

    return positions
