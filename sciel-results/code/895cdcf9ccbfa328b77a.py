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
    deadline = start_time + 600.0
    safety_margin = 30.0

    # Quadrature points for integrating over c in [0, 1]
    # Use Gauss-Legendre for better accuracy with smooth functions
    c_quad, w_quad = np.polynomial.legendre.leggauss(100)
    c_quad = 0.5 * (c_quad + 1)  # Map from [-1, 1] to [0, 1]
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
        Uses softplus for the max(0, gap) term.
        """
        m = np.arange(N)
        starts = z + c * m / N
        ends = starts + 1.0 / N
        
        order = np.argsort(starts)
        s_sorted = starts[order]
        e_sorted = ends[order]
        
        # L = e_sorted[-1] - s_sorted[0] - sum softplus(gap_k / eps) * eps
        gaps = s_sorted[1:] - e_sorted[:-1]
        # softplus(x) = log(1 + exp(x))
        # To avoid overflow, use np.logaddexp(0, x)
        # softplus(gaps/eps) * eps = eps * logaddexp(0, gaps/eps)
        L = e_sorted[-1] - s_sorted[0] - np.sum(eps * np.logaddexp(0, gaps / eps))
        
        grad_z = np.zeros(N)
        grad_z[order[-1]] += 1.0
        grad_z[order[0]] -= 1.0
        
        # Gradient of softplus(gap/eps) * eps w.r.t gap is sigmoid(gap/eps)
        # d/dgap (eps * log(1+exp(gap/eps))) = eps * (exp(gap/eps)/(1+exp(gap/eps))) = eps * sigmoid(gap/eps)
        # sigmoid(x) = 1 / (1 + exp(-x))
        sig = 1.0 / (1.0 + np.exp(-gaps / eps))
        
        # dL/dgap = -sig
        # gap_k = s_sorted[k+1] - e_sorted[k]
        # d(gap_k)/d(s_sorted[k+1]) = 1
        # d(gap_k)/d(e_sorted[k]) = -1
        # So dL/d(s_sorted[k+1]) -= sig[k]
        # dL/d(e_sorted[k]) -= (-sig[k]) = +sig[k]
        
        # Map back to z
        # s_sorted[j] depends on z[order[j]] with derivative 1
        # e_sorted[j] depends on z[order[j]] with derivative 1
        
        for k in range(N - 1):
            g = sig[k]
            grad_z[order[k+1]] -= g
            grad_z[order[k]] += g
        
        return L, grad_z

    def volume_exact(z):
        """
        Compute the exact volume and gradient.
        """
        cost = 0.0
        grad = np.zeros_like(z)
        for c, w in zip(c_quad, w_quad):
            L, g = _union_length_exact(z, c)
            # V = int A^2 dc. dV/dz = int 2 A dA/dz dc
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
    best_cost = volume_exact(z_full_init)[0]
    
    rng = np.random.default_rng(12345)
    
    # Optimization loop
    # 1. Smooth optimization with L-BFGS-B
    # 2. Exact polish with L-BFGS-B
    
    def cost_smooth_params(params):
        z = np.zeros(N)
        z[1:] = params
        cost, grad_full = volume_smooth(z, eps=0.05)
        return cost, grad_full[1:]

    def cost_exact_params(params):
        z = np.zeros(N)
        z[1:] = params
        cost, grad_full = volume_exact(z)
        return cost, grad_full[1:]

    # Try multiple restarts
    for run in range(20):
        if time.time() > deadline - safety_margin:
            break
            
        if run == 0:
            start = best_z.copy()
        else:
            # Random restart with increasing noise
            scale = 0.05 * (run + 1)
            start = best_z + rng.normal(0, scale, N - 1)
        
        try:
            # Smooth phase
            res_smooth = scipy.optimize.minimize(
                cost_smooth_params, 
                start, 
                jac=True, 
                method='L-BFGS-B',
                options={'maxiter': 200, 'gtol': 1e-5, 'ftol': 1e-8}
            )
            
            if res_smooth.fun < best_cost:
                # Polish with exact gradient
                res_exact = scipy.optimize.minimize(
                    cost_exact_params, 
                    res_smooth.x, 
                    jac=True, 
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'gtol': 1e-6, 'ftol': 1e-10}
                )
                
                if res_exact.fun < best_cost:
                    best_cost = res_exact.fun
                    best_z = res_exact.x.copy()
            else:
                # Even if smooth didn't improve, maybe exact polish on smooth result helps?
                # Or maybe exact polish on current best?
                pass
                
        except Exception:
            pass

    # Final polish on the best found
    if time.time() < deadline - safety_margin:
        try:
            res_final = scipy.optimize.minimize(
                cost_exact_params, 
                best_z, 
                jac=True, 
                method='L-BFGS-B',
                options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-12}
            )
            if res_final.fun < best_cost:
                best_cost = res_final.fun
                best_z = res_final.x.copy()
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
