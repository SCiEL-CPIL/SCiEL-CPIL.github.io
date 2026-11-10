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
    
    def compute_1d_union_len(pos, c):
        """
        Compute 1D union length of intervals [pos_i + c*shear_i, pos_i + c*shear_i + width]
        at height c.
        pos: array of shape (N,)
        c: scalar height
        """
        L = pos + c * shear
        R = L + width
        order = np.argsort(L)
        Ls = L[order]
        Rs = R[order]
        
        # Standard interval union algorithm
        if N == 0:
            return 0.0
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

    def exact_volume(params):
        """
        Compute exact volume V = integral_0^1 Ux(c) * Uy(c) dc.
        params: array of shape (2*N,) where first N are x positions, next N are y positions.
        """
        x = params[:N]
        y = params[N:]
        
        # Find all breakpoints in [0,1] where the union structure changes.
        # The union structure changes when any left endpoint meets any other left endpoint,
        # any left meets any right, or any right meets any other right.
        # Since intervals are [L_i, R_i] and R_i = L_i + w, the relative order of endpoints
        # only changes when L_i = L_j, L_i = R_j, or R_i = R_j.
        # L_i(c) = x_i + c*s_i
        # R_i(c) = x_i + c*s_i + w
        
        breakpoints = {0.0, 1.0}
        
        # Generate breakpoints for x-dimension
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                di = shear[i] - shear[j]
                if abs(di) < 1e-15:
                    continue
                # L_i = L_j => x_i + c*s_i = x_j + c*s_j => c = (x_j - x_i) / (s_i - s_j)
                c1 = (x[j] - x[i]) / di
                if 0.0 <= c1 <= 1.0:
                    breakpoints.add(c1)
                
                # L_i = R_j => x_i + c*s_i = x_j + c*s_j + w => c = (x_j - x_i + w) / (s_i - s_j)
                c2 = (x[j] - x[i] + width) / di
                if 0.0 <= c2 <= 1.0:
                    breakpoints.add(c2)
                
                # R_i = R_j is same condition as L_i = L_j shifted by w, but since w is constant,
                # R_i - R_j = L_i - L_j, so the meeting times are identical to L_i = L_j.
                # However, we must also consider L_i meeting R_k for k != j?
                # Actually, the set of "critical" c values for the union length function U(c)
                # are the values where any pair of endpoints coincide.
                # Endpoints are L_i and R_i.
                # Pairs: (L_i, L_j), (L_i, R_j), (R_i, L_j), (R_i, R_j).
                # (L_i, L_j) -> c = (x_j - x_i) / di
                # (L_i, R_j) -> c = (x_j - x_i + w) / di
                # (R_i, L_j) -> c = (x_j - x_i - w) / di
                # (R_i, R_j) -> c = (x_j - x_i) / di (same as L_i=L_j)
                
                c3 = (x[j] - x[i] - width) / di
                if 0.0 <= c3 <= 1.0:
                    breakpoints.add(c3)

        # Generate breakpoints for y-dimension
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                di = shear[i] - shear[j]
                if abs(di) < 1e-15:
                    continue
                c1 = (y[j] - y[i]) / di
                if 0.0 <= c1 <= 1.0:
                    breakpoints.add(c1)
                
                c2 = (y[j] - y[i] + width) / di
                if 0.0 <= c2 <= 1.0:
                    breakpoints.add(c2)
                
                c3 = (y[j] - y[i] - width) / di
                if 0.0 <= c3 <= 1.0:
                    breakpoints.add(c3)
        
        pts = sorted(breakpoints)
        
        # Integrate Ux(c) * Uy(c) over each subinterval.
        # Ux(c) and Uy(c) are piecewise linear. Their product is piecewise quadratic.
        V = 0.0
        for m in range(len(pts) - 1):
            c0 = pts[m]
            c1 = pts[m+1]
            if c1 - c0 < 1e-12:
                continue
            
            u0_x = compute_1d_union_len(x, c0)
            u1_x = compute_1d_union_len(x, c1)
            u0_y = compute_1d_union_len(y, c0)
            u1_y = compute_1d_union_len(y, c1)
            
            # u_x(c) = u0_x + (u1_x - u0_x) * (c - c0) / (c1 - c0)
            # Let t = (c - c0) / (c1 - c0), c = c0 + t * (c1 - c0)
            # u_x(t) = u0_x + dx * t, where dx = u1_x - u0_x
            # u_y(t) = u0_y + dy * t, where dy = u1_y - u0_y
            # f(t) = (u0_x + dx*t) * (u0_y + dy*t)
            #      = u0_x*u0_y + (u0_x*dy + u0_y*dx)*t + dx*dy*t^2
            # Integral from 0 to 1 of f(t) dt:
            # = u0_x*u0_y + 0.5*(u0_x*dy + u0_y*dx) + (1/3)*dx*dy
            
            dx = u1_x - u0_x
            dy = u1_y - u0_y
            integral_t = u0_x * u0_y + 0.5 * (u0_x * dy + u0_y * dx) + (dx * dy) / 3.0
            
            V += integral_t * (c1 - c0)
        
        return V

    # Initial guesses
    # Start with symmetric centered configuration
    k = np.arange(N, dtype=np.float64)
    init_x = -k / (2.0 * N)
    init_y = -k / (2.0 * N)
    
    # Try a few different initial configurations to see if we can find better basins
    init_candidates = [
        np.concatenate([init_x, init_y]),
        np.concatenate([init_x * 0.5, init_y * 0.5]),
        np.concatenate([init_x * 1.5, init_y * 1.5]),
        np.concatenate([np.linspace(-0.5, 0.5, N), np.linspace(-0.5, 0.5, N)]),
    ]
    
    best_params = init_candidates[0].copy()
    best_val = exact_volume(best_params)
    
    # Global optimization using differential evolution
    # Bounds: positions can range reasonably. The tubes are width 1/8.
    # If we spread them too much, volume increases.
    # A reasonable bound is [-2, 2]
    bounds = [(-2.0, 2.0)] * (2 * N)
    
    if time.time() < deadline - 120:
        try:
            result = differential_evolution(
                exact_volume, bounds, seed=42, maxiter=150, 
                tol=1e-8, popsize=15, polish=False,
                updating='immediate'
            )
            if np.all(np.isfinite(result.x)) and result.fun < best_val:
                best_params = result.x.copy()
                best_val = result.fun
        except Exception:
            pass
    
    # Local refinement with Nelder-Mead
    if time.time() < deadline - 60:
        try:
            result2 = minimize(exact_volume, best_params, method='Nelder-Mead',
                              options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            if np.all(np.isfinite(result2.x)) and result2.fun < best_val:
                best_params = result2.x.copy()
                best_val = result2.fun
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 20:
        try:
            result3 = minimize(exact_volume, best_params, method='Nelder-Mead',
                              options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-15})
            if np.all(np.isfinite(result3.x)) and result3.fun < best_val:
                best_params = result3.x.copy()
                best_val = result3.fun
        except Exception:
            pass
            
    # Construct the 64x2 array
    x_best = best_params[:N]
    y_best = best_params[N:]
    
    positions = np.zeros((64, 2), dtype=np.float64)
    for k_idx in range(64):
        i = k_idx // N
        j = k_idx % N
        positions[k_idx, 0] = x_best[i]
        positions[k_idx, 1] = y_best[j]
    
    return positions

# EVOLVE-BLOCK-END
