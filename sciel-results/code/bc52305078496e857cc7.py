# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    Optimizes the 1D base positions to minimize the union volume.
    """
    N = 8
    start_time = time.time()
    deadline = start_time + 630.0
    safety_margin = 30.0

    # Quadrature points for integrating over c in [0, 1]
    # Using more points for better accuracy
    c_quad, w_quad = np.polynomial.legendre.leggauss(300)
    c_quad = 0.5 * (c_quad + 1)
    w_quad = 0.5 * w_quad

    def _union_length_exact(z, c):
        """
        Compute the exact union length of intervals [z[i] + c*i/N, z[i] + c*i/N + 1/N].
        """
        m = np.arange(N)
        starts = z + c * m / N
        ends = starts + 1.0 / N
        
        order = np.argsort(starts)
        s_sorted = starts[order]
        e_sorted = ends[order]
        
        L = e_sorted[-1] - s_sorted[0]
        
        for k in range(N - 1):
            gap = s_sorted[k+1] - e_sorted[k]
            if gap > 0:
                L -= gap
        
        return L

    def _union_length_grad_exact(z, c):
        """
        Compute the gradient of union length w.r.t z at a given c.
        """
        m = np.arange(N)
        starts = z + c * m / N
        ends = starts + 1.0 / N
        
        order = np.argsort(starts)
        s_sorted = starts[order]
        e_sorted = ends[order]
        
        grad_z = np.zeros(N)
        grad_z[order[-1]] += 1.0
        grad_z[order[0]] -= 1.0
        
        for k in range(N - 1):
            gap = s_sorted[k+1] - e_sorted[k]
            if gap > 0:
                grad_z[order[k+1]] -= 1.0
                grad_z[order[k]] += 1.0
        
        return grad_z

    def _union_length_smooth(z, c, eps=0.01):
        """
        Compute a smooth approximation of the union length.
        Uses softplus for the max(0, gap) term.
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
        
        np.add.at(grad_z, order[1:], -sig)
        np.add.at(grad_z, order[:-1], sig)
        
        return L, grad_z

    def volume_exact(z):
        """
        Compute the exact volume.
        """
        cost = 0.0
        for c, w in zip(c_quad, w_quad):
            L = _union_length_exact(z, c)
            cost += w * L * L
        return cost

    def volume_grad_exact(z):
        """
        Compute the exact volume and its gradient.
        """
        cost = 0.0
        grad = np.zeros_like(z)
        for c, w in zip(c_quad, w_quad):
            L = _union_length_exact(z, c)
            g = _union_length_grad_exact(z, c)
            cost += w * L * L
            grad += w * 2.0 * L * g
        return cost, grad

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
    # Use multiple restarts with different strategies
    
    def cost_smooth_params(params, eps):
        z = np.zeros(N)
        z[1:] = params
        cost, grad_full = volume_smooth(z, eps)
        return cost, grad_full[1:]

    def cost_exact_params(params):
        z = np.zeros(N)
        z[1:] = params
        return volume_exact(z)

    def cost_grad_exact_params(params):
        z = np.zeros(N)
        z[1:] = params
        cost, grad_full = volume_grad_exact(z)
        return cost, grad_full[1:]

    # Try multiple restarts with different strategies
    for run in range(40):
        if time.time() > deadline - safety_margin:
            break
        
        if run == 0:
            start = best_z.copy()
        elif run % 4 == 1:
            # Random restart with moderate noise
            scale = 0.05
            start = best_z + rng.normal(0, scale, N - 1)
        elif run % 4 == 2:
            # Random restart with small noise
            scale = 0.02
            start = best_z + rng.normal(0, scale, N - 1)
        elif run % 4 == 3:
            # Random restart with large noise
            scale = 0.1
            start = best_z + rng.normal(0, scale, N - 1)
        else:
            # Random restart with adaptive noise
            scale = 0.03 + 0.02 * (run % 5)
            start = best_z + rng.normal(0, scale, N - 1)
        
        current_z = start.copy()
        
        try:
            # Stage 1: Smooth optimization with coarse eps
            if time.time() < deadline - safety_margin:
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
            if time.time() < deadline - safety_margin:
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
            if time.time() < deadline - safety_margin:
                res3_smooth = scipy.optimize.minimize(
                    lambda p: cost_smooth_params(p, 0.001)[0],
                    current_z, 
                    jac=lambda p: cost_smooth_params(p, 0.001)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'gtol': 1e-6, 'ftol': 1e-10}
                )
                if res3_smooth.fun < best_cost:
                    current_z = res3_smooth.x
            
            # Stage 4: Exact gradient-based polish
            if time.time() < deadline - safety_margin:
                res4 = scipy.optimize.minimize(
                    lambda p: cost_grad_exact_params(p)[0],
                    current_z,
                    jac=lambda p: cost_grad_exact_params(p)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-12}
                )
                if res4.fun < best_cost:
                    current_z = res4.x
            
            # Stage 5: Exact Nelder-Mead polish
            if time.time() < deadline - safety_margin:
                res5 = scipy.optimize.minimize(
                    cost_exact_params,
                    current_z,
                    method='Nelder-Mead',
                    options={'maxiter': 1500, 'xatol': 1e-7, 'fatol': 1e-10}
                )
                
                if res5.fun < best_cost:
                    best_cost = res5.fun
                    best_z = res5.x.copy()
                    
        except Exception:
            pass

    # Final polish on the best found
    if time.time() < deadline - safety_margin:
        try:
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
            
            # 2. Exact gradient-based
            if time.time() < deadline - safety_margin:
                res_fin2 = scipy.optimize.minimize(
                    lambda p: cost_grad_exact_params(p)[0],
                    best_z,
                    jac=lambda p: cost_grad_exact_params(p)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-14}
                )
                if res_fin2.fun < best_cost:
                    best_cost = res_fin2.fun
                    best_z = res_fin2.x.copy()
            
            # 3. Exact Nelder-Mead
            if time.time() < deadline - safety_margin:
                res_fin3 = scipy.optimize.minimize(
                    cost_exact_params,
                    best_z,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-12}
                )
                if res_fin3.fun < best_cost:
                    best_cost = res_fin3.fun
                    best_z = res_fin3.x.copy()
                    
            # 4. One more very fine smooth pass
            if time.time() < deadline - safety_margin:
                res_fin4 = scipy.optimize.minimize(
                    lambda p: cost_smooth_params(p, 0.0001)[0],
                    best_z, 
                    jac=lambda p: cost_smooth_params(p, 0.0001)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'gtol': 1e-10, 'ftol': 1e-14}
                )
                if res_fin4.fun < best_cost:
                    best_z = res_fin4.x
                    best_cost = res_fin4.fun
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
