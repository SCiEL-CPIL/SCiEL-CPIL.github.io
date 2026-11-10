import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_3d8() -> np.ndarray:
    """
    Optimizes the base positions of 64 sheared tubes in 3D to minimize the union volume.
    Exploits the product structure of the problem to decouple it into two identical 1D problems.
    Uses L-BFGS-B with analytic gradients for the 1D objective.
    """
    start_time = time.time()
    deadline = 600.0  # Safety margin from 630s limit
    
    N = 8
    h = 1.0 / N
    
    # Keich construction as initial guess
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    # 1D Optimization Problem:
    # Minimize J(x) = integral_0^1 L(z)^2 dz
    # where L(z) = len(union_{k=0}^{N-1} [x_k + k/N*z, x_k + k/N*z + h])
    
    # Quadrature points for the integral over z
    M = 200
    z_grid = np.linspace(0, 1, M)
    dz = 1.0 / M
    
    def compute_union_length(z, x):
        """Computes the length of the union of intervals [x_k + k/N*z, x_k + k/N*z + h] for k=0..N-1."""
        starts = x + (np.arange(N) * z / N)
        ends = starts + h
        # Sort by start
        order = np.argsort(starts)
        starts = starts[order]
        ends = ends[order]
        
        total_len = 0.0
        current_end = -np.inf
        for i in range(N):
            if starts[i] > current_end:
                total_len += ends[i] - starts[i]
                current_end = ends[i]
            else:
                if ends[i] > current_end:
                    total_len += ends[i] - current_end
                    current_end = ends[i]
        return total_len

    def objective(x):
        """
        Computes J(x) = sum_{j=0}^{M-1} L(z_j)^2 * dz
        """
        total = 0.0
        for z in z_grid:
            L = compute_union_length(z, x)
            total += L * L * dz
        return total

    def gradient(x):
        """
        Computes dJ/dx_k = integral_0^1 2 L(z) * g_k(z) dz
        where g_k(z) = 1_{u_k+h in U_{-k}} - 1_{u_k in U_{-k}}
        """
        grad = np.zeros(N)
        for z in z_grid:
            # Compute starts for all k
            starts = x + (np.arange(N) * z / N)
            ends = starts + h
            
            # Compute L(z)
            L = compute_union_length(z, x)
            
            if L == 0:
                continue
                
            # For each k, compute g_k(z)
            for k in range(N):
                u_k = starts[k]
                u_k_end = ends[k]
                
                # Check if u_k is in union of other intervals
                in_union_start = False
                in_union_end = False
                
                for i in range(N):
                    if i == k:
                        continue
                    s_i = starts[i]
                    e_i = ends[i]
                    if s_i <= u_k <= e_i:
                        in_union_start = True
                    if s_i <= u_k_end <= e_i:
                        in_union_end = True
                        
                g_k = (1.0 if in_union_end else 0.0) - (1.0 if in_union_start else 0.0)
                grad[k] += 2.0 * L * g_k * dz
                
        return grad

    # Run optimization
    x0 = keich.copy()
    
    # We can try multiple restarts if time permits, but L-BFGS-B is usually robust.
    # Let's start with the Keich guess.
    
    res = minimize(
        objective, 
        x0, 
        jac=gradient, 
        method='L-BFGS-B', 
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False}
    )
    
    if res.success:
        x_opt = res.x
    else:
        # Fallback to initial if optimization fails, or use best found
        x_opt = res.x if hasattr(res, 'x') else x0

    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        # If we are close to deadline, just return what we have
        pass

    # Construct the 64x2 array
    # Row k is tube (i, j) with i = k // 8 and j = k % 8
    # base_x = x_opt[i], base_y = x_opt[j]
    
    positions = np.zeros((64, 2))
    for k in range(64):
        i = k // N
        j = k % N
        positions[k, 0] = x_opt[i]
        positions[k, 1] = x_opt[j]
        
    return positions

# EVOLVE-BLOCK-END
