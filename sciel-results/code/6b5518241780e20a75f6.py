# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    Optimizes the 1D base positions to minimize the union volume.
    Uses a fast exact volume calculation based on piecewise linear integration.
    """
    N = 8
    start_time = time.time()
    deadline = start_time + 600.0
    safety_margin = 30.0

    # Precompute indices
    m = np.arange(N)
    
    def _exact_volume_fast(z):
        """
        Compute the exact volume of the union of tubes.
        The volume is integral_0^1 L(c)^2 dc.
        L(c) is the length of the union of intervals [z_i + c*i/N, z_i + c*i/N + 1/N].
        L(c) is piecewise linear in c. We find all critical points and integrate exactly.
        """
        # Critical points where order of starts changes: z_i + c*i/N = z_j + c*j/N
        # c = (z_j - z_i) / ((i-j)/N)
        # Critical points where gap closes/opens: z_i + c*i/N = z_j + c*j/N + 1/N
        # c = (z_j - z_i + 1/N) / ((i-j)/N)
        
        critical_cs = [0.0, 1.0]
        
        # Generate critical points
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                denom = (i - j) / N
                if abs(denom) < 1e-12:
                    continue
                # Order change
                c1 = (z[j] - z[i]) / denom
                # Gap change
                c2 = (z[j] - z[i] + 1.0/N) / denom
                
                if 0.0 <= c1 <= 1.0:
                    critical_cs.append(c1)
                if 0.0 <= c2 <= 1.0:
                    critical_cs.append(c2)
        
        critical_cs = np.array(critical_cs)
        critical_cs = np.sort(np.unique(critical_cs))
        
        volume = 0.0
        
        for k in range(len(critical_cs) - 1):
            c_start = critical_cs[k]
            c_end = critical_cs[k+1]
            
            if c_end - c_start < 1e-12:
                continue
                
            # Pick a point in the interval to determine the structure
            c_mid = 0.5 * (c_start + c_end)
            
            starts = z + c_mid * m / N
            ends = starts + 1.0 / N
            
            order = np.argsort(starts)
            s_sorted = starts[order]
            e_sorted = ends[order]
            
            # Determine which gaps are active (gap > 0)
            gaps = s_sorted[1:] - e_sorted[:-1]
            active_gaps = gaps > 0
            
            # L(c) = max(ends) - min(starts) - sum(active_gaps)
            # In this interval, the order is constant, so max is e_sorted[-1] and min is s_sorted[0].
            # The set of active gaps is constant.
            # L(c) = (e_sorted[-1](c)) - (s_sorted[0](c)) - sum_{k: active} (s_sorted[k+1](c) - e_sorted[k](c))
            
            # Let's compute the slope and intercept of L(c)
            # s_i(c) = z_i + (i/N) c
            # e_i(c) = z_i + (i/N) c + 1/N
            
            # Slope of s_sorted[0] is order[0]/N
            # Slope of e_sorted[-1] is order[-1]/N
            
            slope_min_s = order[0] / N
            slope_max_e = order[-1] / N
            
            # Slope of sum of active gaps
            slope_gaps = 0.0
            for idx in range(N - 1):
                if active_gaps[idx]:
                    # gap = s_sorted[idx+1] - e_sorted[idx]
                    # slope = (order[idx+1]/N) - (order[idx]/N)
                    slope_gaps += (order[idx+1] - order[idx]) / N
            
            A = slope_max_e - slope_min_s - slope_gaps
            
            # Intercept B = L(c_mid) - A * c_mid
            L_mid = (e_sorted[-1] - s_sorted[0]) - np.sum(gaps[active_gaps])
            B = L_mid - A * c_mid
            
            # Integrate (A c + B)^2 from c_start to c_end
            # = [ (A c + B)^3 / (3 A) ] from c_start to c_end
            # = ( (A c_end + B)^3 - (A c_start + B)^3 ) / (3 A)
            # If A is close to 0, use trapezoidal or analytic for constant
            if abs(A) < 1e-12:
                # L is constant B
                volume += B * B * (c_end - c_start)
            else:
                val_end = A * c_end + B
                val_start = A * c_start + B
                volume += (val_end**3 - val_start**3) / (3.0 * A)
        
        return volume

    def cost_smooth(z, eps=0.01):
        """
        Smooth approximation for gradient-based optimization.
        """
        c_quad, w_quad = np.polynomial.legendre.leggauss(50)
        c_quad = 0.5 * (c_quad + 1)
        w_quad = 0.5 * w_quad
        
        cost = 0.0
        for c, w in zip(c_quad, w_quad):
            starts = z + c * m / N
            ends = starts + 1.0 / N
            order = np.argsort(starts)
            s_sorted = starts[order]
            e_sorted = ends[order]
            gaps = s_sorted[1:] - e_sorted[:-1]
            # Softplus approximation for max(0, gap)
            soft_gaps = eps * np.logaddexp(0, gaps / eps)
            L = e_sorted[-1] - s_sorted[0] - np.sum(soft_gaps)
            cost += w * L * L
        return cost

    # Initial guess: Keich binary construction
    pos_k = int(np.log2(N))
    j_terms = np.arange(1, pos_k + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos_k - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos_k)
    
    # Shift so z_0 = 0
    keich = keich - keich[0]
    x0 = keich[1:].copy()
    
    best_z = x0.copy()
    z_full_init = np.zeros(N)
    z_full_init[1:] = best_z
    best_cost = _exact_volume_fast(z_full_init)
    
    rng = np.random.default_rng(12345)
    
    # Optimization loop
    for run in range(50):
        if time.time() > deadline - safety_margin:
            break
        
        if run == 0:
            start = best_z.copy()
        else:
            # Random restart with adaptive noise
            scale = 0.05 + 0.01 * (run % 10)
            start = best_z + rng.normal(0, scale, N - 1)
        
        current_z = start.copy()
        
        try:
            # Stage 1: Smooth optimization
            def cost_smooth_params(params, eps=0.01):
                z = np.zeros(N)
                z[1:] = params
                return cost_smooth(z, eps)
            
            res1 = scipy.optimize.minimize(
                lambda p: cost_smooth_params(p, 0.05),
                current_z, 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if res1.fun < best_cost:
                current_z = res1.x
            
            # Stage 2: Exact Nelder-Mead polish
            def exact_cost_params(params):
                z = np.zeros(N)
                z[1:] = params
                return _exact_volume_fast(z)
            
            res2 = scipy.optimize.minimize(
                exact_cost_params,
                current_z,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if res2.fun < best_cost:
                best_cost = res2.fun
                best_z = res2.x.copy()
                
        except Exception:
            pass

    # Final polish on the best found
    if time.time() < deadline - safety_margin:
        try:
            def exact_cost_params(params):
                z = np.zeros(N)
                z[1:] = params
                return _exact_volume_fast(z)
            
            res_fin = scipy.optimize.minimize(
                exact_cost_params,
                best_z,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res_fin.fun < best_cost:
                best_cost = res_fin.fun
                best_z = res_fin.x.copy()
        except Exception:
            pass

    # Construct full z array
    z_full = np.zeros(N)
    z_full[1:] = best_z
    
    # Construct 64 positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = z_full[i_flat]
    base_y = z_full[j_flat]
    
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions

# EVOLVE-BLOCK-END
