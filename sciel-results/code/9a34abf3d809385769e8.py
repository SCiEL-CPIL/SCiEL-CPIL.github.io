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
    
    def compute_1d_union_len_vec(x, c):
        """Compute 1D union length of intervals [x_i + c*shear_i, x_i + c*shear_i + width] at height c."""
        L = x + c * shear
        R = L + width
        order = np.argsort(L)
        Ls = L[order]
        Rs = R[order]
        
        # Vectorized interval union length
        # We need to sum the lengths of disjoint segments.
        # A point is covered if it is in any interval.
        # The union length is the measure of the set.
        # We can compute this by finding the gaps.
        # Or simpler: 
        # max_r = -inf
        # total = 0
        # for each interval in sorted order:
        #   if L > max_r: total += max_r - L_prev_end? No.
        #   Standard algorithm:
        #   cur_end = Rs[0]
        #   total = 0
        #   for k in 1..N-1:
        #      if Ls[k] > cur_end:
        #         total += cur_end - (Ls[k-1] if k==1 else Ls[k-1]?) 
        #         # Actually, we just add the length of the previous merged component.
        #         # But we don't track start of current component easily in pure vector ops without loops.
        #   Let's stick to a small loop for N=8, it's fast enough and less error prone.
        total = 0.0
        cur_start = Ls[0]
        cur_end = Rs[0]
        for k in range(1, N):
            if Ls[k] > cur_end:
                total += cur_end - cur_start
                cur_start = Ls[k]
                cur_end = Rs[k]
            else:
                if Rs[k] > cur_end:
                    cur_end = Rs[k]
        total += cur_end - cur_start
        return total

    def exact_volume_3d(params):
        """
        Compute exact volume V = integral_0^1 Ux(c) * Uy(c) dc.
        params: 128-dim vector. First 8 are x_0..x_7, last 8 are y_0..y_7.
        """
        x = params[:N]
        y = params[N:]
        
        # Find breakpoints for x
        bps_x = {0.0, 1.0}
        for i in range(N):
            for j in range(N):
                if i == j: continue
                di = shear[i] - shear[j]
                if abs(di) < 1e-15: continue
                c1 = (x[j] - x[i] - width) / di
                if 0.0 <= c1 <= 1.0: bps_x.add(c1)
                c2 = (x[j] - x[i]) / di
                if 0.0 <= c2 <= 1.0: bps_x.add(c2)
                
        # Find breakpoints for y
        bps_y = {0.0, 1.0}
        for i in range(N):
            for j in range(N):
                if i == j: continue
                di = shear[i] - shear[j]
                if abs(di) < 1e-15: continue
                c1 = (y[j] - y[i] - width) / di
                if 0.0 <= c1 <= 1.0: bps_y.add(c1)
                c2 = (y[j] - y[i]) / di
                if 0.0 <= c2 <= 1.0: bps_y.add(c2)
        
        # Combine breakpoints
        all_bps = sorted(set(bps_x).union(set(bps_y)))
        
        V = 0.0
        for m in range(len(all_bps) - 1):
            c0 = all_bps[m]
            c1 = all_bps[m+1]
            if c1 - c0 < 1e-12: continue
            
            # Evaluate Ux and Uy at endpoints
            ux0 = compute_1d_union_len_vec(x, c0)
            ux1 = compute_1d_union_len_vec(x, c1)
            uy0 = compute_1d_union_len_vec(y, c0)
            uy1 = compute_1d_union_len_vec(y, c1)
            
            # Integrate ux(c)*uy(c) over [c0, c1]
            # ux(c) = ux0 + (ux1-ux0)*(c-c0)/(c1-c0)
            # uy(c) = uy0 + (uy1-uy0)*(c-c0)/(c1-c0)
            # Let t = (c-c0)/(c1-c0), dc = (c1-c0)dt
            # f(t) = (ux0 + dx*t) * (uy0 + dy*t)
            #      = ux0*uy0 + (ux0*dy + uy0*dx)*t + dx*dy*t^2
            # integral_0^1 f(t) dt = ux0*uy0 + (ux0*dy + uy0*dx)/2 + dx*dy/3
            dx = ux1 - ux0
            dy = uy1 - uy0
            integral_t = ux0*uy0 + 0.5*(ux0*dy + uy0*dx) + (dx*dy)/3.0
            V += integral_t * (c1 - c0)
            
        return V

    # Initial guess: Symmetric centered
    init_x = -np.arange(N, dtype=np.float64) / (2.0 * N)
    init_y = -np.arange(N, dtype=np.float64) / (2.0 * N)
    init_params = np.concatenate([init_x, init_y])
    
    best_params = init_params.copy()
    best_val = exact_volume_3d(best_params)
    
    # Global optimization
    # 128 dimensions is high for DE, but we have time.
    # Reduce population size to fit in time? 
    # 128 * 15 = 1920 evals per generation.
    # Exact volume is fast now (vectorized union + few breakpoints).
    
    bounds = [(-2.0, 2.0)] * (2 * N)
    
    if time.time() < deadline - 120:
        try:
            # Use smaller popsize for higher dim
            result = differential_evolution(
                exact_volume_3d, bounds, seed=42, maxiter=100, 
                tol=1e-6, popsize=10, polish=False,
                updating='immediate'
            )
            if np.all(np.isfinite(result.x)) and result.fun < best_val:
                best_params = result.x.copy()
                best_val = result.fun
        except Exception:
            pass
    
    # Local refinement
    # Nelder-Mead in 128D is slow. 
    # Maybe use L-BFGS-B with numerical gradient? 
    # The function is piecewise smooth, so gradients are undefined at breakpoints.
    # NM is robust to this.
    
    if time.time() < deadline - 60:
        try:
            # Coarse NM
            result2 = minimize(exact_volume_3d, best_params, method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if np.all(np.isfinite(result2.x)) and result2.fun < best_val:
                best_params = result2.x.copy()
                best_val = result2.fun
        except Exception:
            pass
            
    if time.time() < deadline - 30:
        try:
            # Fine NM
            result3 = minimize(exact_volume_3d, best_params, method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            if np.all(np.isfinite(result3.x)) and result3.fun < best_val:
                best_params = result3.x.copy()
                best_val = result3.fun
        except Exception:
            pass
            
    # Construct the 64x2 array
    positions = np.zeros((64, 2), dtype=np.float64)
    for k_idx in range(64):
        i = k_idx // N
        j = k_idx % N
        positions[k_idx, 0] = best_params[i]
        positions[k_idx, 1] = best_params[N + j]
    
    return positions

# EVOLVE-BLOCK-END
