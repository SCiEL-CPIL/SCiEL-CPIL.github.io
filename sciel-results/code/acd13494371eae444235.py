# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    deadline = time.time() + 590.0  # Safety margin
    
    # Precompute shear rates
    i_vals = np.arange(N, dtype=np.float64)
    shear = i_vals / N  # shape (N,)
    width = 1.0 / N
    
    def compute_1d_union_len(x, c):
        """Compute 1D union length of intervals [x_i + c*shear_i, x_i + c*shear_i + width] at height c."""
        L = x + c * shear
        R = L + width
        order = np.argsort(L)
        Ls = L[order]
        Rs = R[order]
        # Standard interval union
        total = 0.0
        cur_start = Ls[0]
        cur_end = Rs[0]
        for k in range(1, N):
            if Ls[k] > cur_end:
                total += cur_end - cur_start
                cur_start = Ls[k]
                cur_end = Rs[k]
            else:
                cur_end = max(cur_end, Rs[k])
        total += cur_end - cur_start
        return total
    
    def exact_volume(x):
        """Compute exact volume V = integral_0^1 Ux(c)^2 dc (using symmetry x=y)."""
        # Find all breakpoints in [0,1]
        breakpoints = {0.0, 1.0}
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                di = shear[i] - shear[j]
                if abs(di) < 1e-15:
                    continue
                # Right endpoints meet: x_i + c*s_i + w = x_j + c*s_j
                c1 = (x[j] - x[i] - width) / di
                if 0.0 <= c1 <= 1.0:
                    breakpoints.add(c1)
                # Left endpoints meet: x_i + c*s_i = x_j + c*s_j
                c2 = (x[j] - x[i]) / di
                if 0.0 <= c2 <= 1.0:
                    breakpoints.add(c2)
        
        pts = sorted(breakpoints)
        
        # Integrate Ux(c)^2 over each subinterval
        V = 0.0
        for m in range(len(pts) - 1):
            c0 = pts[m]
            c1 = pts[m+1]
            if c1 - c0 < 1e-12:
                continue
            u0 = compute_1d_union_len(x, c0)
            u1 = compute_1d_union_len(x, c1)
            diff = u1 - u0
            integral_t = u0*u0 + u0*diff + diff*diff/3.0
            V += integral_t * (c1 - c0)
        
        return V
    
    # Initial guesses
    k = np.arange(N, dtype=np.float64)
    init_candidates = [
        -k / (2.0 * N),           # Centered
        -k / (2.0 * N) * 0.5,     # Compressed
        -k / (2.0 * N) * 1.5,     # Expanded
        np.linspace(-0.5, 0.5, N),
    ]
    
    best_x = init_candidates[0].copy()
    best_val = exact_volume(best_x)
    
    # Global optimization using differential evolution
    bounds = [(-2.0, 2.0)] * N
    
    if time.time() < deadline - 60:
        try:
            result = differential_evolution(
                exact_volume, bounds, seed=42, maxiter=200, 
                tol=1e-8, popsize=15, polish=False,
                updating='immediate'
            )
            if np.all(np.isfinite(result.x)) and result.fun < best_val:
                best_x = result.x.copy()
                best_val = result.fun
        except Exception:
            pass
    
    # Local refinement with Nelder-Mead
    if time.time() < deadline - 30:
        try:
            result2 = minimize(exact_volume, best_x, method='Nelder-Mead',
                              options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            if np.all(np.isfinite(result2.x)) and result2.fun < best_val:
                best_x = result2.x.copy()
                best_val = result2.fun
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 10:
        try:
            result3 = minimize(exact_volume, best_x, method='Nelder-Mead',
                              options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-15})
            if np.all(np.isfinite(result3.x)) and result3.fun < best_val:
                best_x = result3.x.copy()
                best_val = result3.fun
        except Exception:
            pass
    
    # Construct the 64x2 array
    positions = np.zeros((64, 2), dtype=np.float64)
    for k_idx in range(64):
        i = k_idx // N
        j = k_idx % N
        positions[k_idx, 0] = best_x[i]
        positions[k_idx, 1] = best_x[j]
    
    return positions

# EVOLVE-BLOCK-END
