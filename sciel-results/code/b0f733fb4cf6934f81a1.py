# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    Optimizes the 1D base positions to minimize the union volume.
    Uses a hybrid strategy:
    1. Smooth L-BFGS-B optimization on a softplus-approximated volume.
    2. Exact Nelder-Mead polish on the true piecewise-linear volume.
    """
    N = 8
    start_time = time.time()
    deadline = start_time + 600.0
    safety_margin = 30.0

    # Quadrature points for integrating over c in [0, 1]
    c_quad, w_quad = np.polynomial.legendre.leggauss(200)
    c_quad = 0.5 * (c_quad + 1)
    w_quad = 0.5 * w_quad

    def _union_length_exact(z, c):
        """
        Compute the exact union length of intervals [z[i] + c*i/N, z[i] + c*i/N + 1/N].
        Returns (length, gradient w.r.t z).
        """
        m = np.arange(N)
        starts = z + c * m / N
        ends = starts + 1.0 / N
        
        order = np.argsort(starts)
        s_sorted = starts[order]
        e_sorted = ends[order]
        
        L = e_sorted[-1] - s_sorted[0]
        
        grad_z = np.zeros(N)
        grad_z[order[-1]] += 1.0
        grad_z[order[0]] -= 1.0
        
        for k in range(N - 1):
            gap = s_sorted[k+1] - e_sorted[k]
            if gap > 0:
                grad_z[order[k+1]] -= 1.0
                grad_z[order[k]] += 1.0
                L -= gap
        
        return L, grad_z

    def _union_length_smooth(z, c, eps=0.01):
        """
        Compute a smooth approximation of the union length.
        Uses softplus for the max(0, gap) term. Vectorized.
        """
        m = np.arange(N)
        starts = z + c * m / N
        ends = starts + 1.0 / N
        
        order = np.argsort(starts)
        s_sorted = starts[order]
        e_sorted = ends[order]
        
        gaps = s_sorted[1:] - e_sorted[:-1]
        # softplus(gaps/eps) * eps
        L = e_sorted[-1] - s_sorted[0] - np.sum(eps * np.logaddexp(0, gaps / eps))
        
        grad_z = np.zeros(N)
        grad_z[order[-1]] += 1.0
        grad_z[order[0]] -= 1.0
        
        sig = 1.0 / (1.0 + np.exp(-gaps / eps))
        
        # Vectorized gradient accumulation
        np.add.at(grad_z, order[1:], -sig)
        np.add.at(grad_z, order[:-1], sig)
        
        return L, grad_z

    def volume_exact(z):
        """
        Compute the exact volume.
        """
        cost = 0.0
        for c, w in zip(c_quad, w_quad):
            L, _ = _union_length_exact(z, c)
            cost += w * L * L
        return cost

    def volume_smooth(z, eps=0.01):
        """
        Compute the smooth volume and gradient.
        """
        cost = 0.0
        grad = np.zeros_like(z)
        for c, w in zip(c_quad, w_quad):
            L, g = _union_length_smooth(z, c, eps)
            cost += w * L * L
            grad += w * 2.0 * L * g
        return cost, grad

    def volume_smooth_batch(z, eps_list):
        """
        Compute smooth volumes for multiple eps values.
        """
        results = []
        for eps in eps_list:
            cost, grad = volume_smooth(z, eps)
            results.append((cost, grad))
        return results

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
    best_cost = volume_exact(z_full_init)
    
    rng = np.random.default_rng(12345)
    
    # Optimization loop
    # 1. Smooth L-BFGS-B
    # 2. Exact Nelder-Mead polish
    
    def cost_smooth_params(params, eps):
        z = np.zeros(N)
        z[1:] = params
        cost, grad_full = volume_smooth(z, eps)
        return cost, grad_full[1:]

    # Try multiple restarts
    for run in range(30):
        if time.time() > deadline - safety_margin:
            break
        
        if run == 0:
            start = best_z.copy()
        else:
            # Random restart with adaptive noise
            scale = 0.03 + 0.02 * (run % 5)
            start = best_z + rng.normal(0, scale, N - 1)
        
        current_z = start.copy()
        
        try:
            # Stage 1: Smooth optimization with coarse eps
            res1 = scipy.optimize.minimize(
                lambda p: cost_smooth_params(p, 0.1)[0],
                current_z, 
                jac=lambda p: cost_smooth_params(p, 0.1)[1],
                method='L-BFGS-B',
                options={'maxiter': 300, 'gtol': 1e-4, 'ftol': 1e-6}
            )
            if res1.fun < best_cost:
                current_z = res1.x
            
            # Stage 2: Smooth optimization with fine eps
            res2 = scipy.optimize.minimize(
                lambda p: cost_smooth_params(p, 0.01)[0],
                current_z, 
                jac=lambda p: cost_smooth_params(p, 0.01)[1],
                method='L-BFGS-B',
                options={'maxiter': 300, 'gtol': 1e-5, 'ftol': 1e-8}
            )
            if res2.fun < best_cost:
                current_z = res2.x
            
            # Stage 3: Very fine eps polish
            res3_smooth = scipy.optimize.minimize(
                lambda p: cost_smooth_params(p, 0.001)[0],
                current_z, 
                jac=lambda p: cost_smooth_params(p, 0.001)[1],
                method='L-BFGS-B',
                options={'maxiter': 200, 'gtol': 1e-6, 'ftol': 1e-10}
            )
            if res3_smooth.fun < best_cost:
                current_z = res3_smooth.x
            
            # Stage 4: Exact Nelder-Mead polish
            def exact_cost_params(params):
                z = np.zeros(N)
                z[1:] = params
                return volume_exact(z)
            
            res4 = scipy.optimize.minimize(
                exact_cost_params,
                current_z,
                method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-7, 'fatol': 1e-10}
            )
            
            if res4.fun < best_cost:
                best_cost = res4.fun
                best_z = res4.x.copy()
                
        except Exception:
            pass

    # Final polish on the best found
    if time.time() < deadline - safety_margin:
        try:
            def exact_cost_params(params):
                z = np.zeros(N)
                z[1:] = params
                return volume_exact(z)
            
            # Multi-stage final polish
            # 1. Very fine smooth
            res_fin1 = scipy.optimize.minimize(
                lambda p: cost_smooth_params(p, 0.0005)[0],
                best_z, 
                jac=lambda p: cost_smooth_params(p, 0.0005)[1],
                method='L-BFGS-B',
                options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-12}
            )
            if res_fin1.fun < best_cost:
                best_z = res_fin1.x
                best_cost = res_fin1.fun
            
            # 2. Exact Nelder-Mead
            res_fin2 = scipy.optimize.minimize(
                exact_cost_params,
                best_z,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-12}
            )
            if res_fin2.fun < best_cost:
                best_cost = res_fin2.fun
                best_z = res_fin2.x.copy()
                
            # 3. One more very fine smooth pass
            if time.time() < deadline - safety_margin:
                res_fin3 = scipy.optimize.minimize(
                    lambda p: cost_smooth_params(p, 0.0001)[0],
                    best_z, 
                    jac=lambda p: cost_smooth_params(p, 0.0001)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'gtol': 1e-10, 'ftol': 1e-14}
                )
                if res_fin3.fun < best_cost:
                    best_z = res_fin3.x
                    best_cost = res_fin3.fun
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
