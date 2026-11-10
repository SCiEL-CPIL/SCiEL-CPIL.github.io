# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

def kakeya_needle_3d8() -> np.ndarray:
    """
    Discrete 3-D Kakeya set: union of 64 sheared tubes (N = 8).
    
    Strategy:
    1. Exploit the separable structure: assume x_k depends only on i and y_k only on j.
    2. Due to symmetry, assume x_i = y_i = X_i.
    3. The volume is V = integral_0^1 U(c)^2 dc, where U(c) is the 1D union length.
    4. Compute V exactly using piecewise quadratic integration.
    5. Use Differential Evolution for global search, then Nelder-Mead for polishing.
    """
    N = 8
    deadline = time.time() + 580.0
    
    i_vals = np.arange(N, dtype=np.float64)
    shear = i_vals / N  # shape (N,)
    width = 1.0 / N

    def exact_objective(X):
        """
        Compute integral_0^1 U(c)^2 dc exactly.
        U(c) is the union length of intervals [X_i + c * shear_i, X_i + c * shear_i + width].
        """
        # Breakpoints where ordering of endpoints changes
        # Endpoints are L_i(c) = X_i + c * shear_i and R_i(c) = X_i + c * shear_i + width
        # Pairs of endpoints are equal when:
        # L_i(c) = L_j(c) => X_i - X_j + c(shear_i - shear_j) = 0
        # L_i(c) = R_j(c) => X_i - X_j - width + c(shear_i - shear_j) = 0
        # R_i(c) = L_j(c) => X_i + width - X_j + c(shear_i - shear_j) = 0
        # R_i(c) = R_j(c) => X_i - X_j + c(shear_i - shear_j) = 0 (same as L_i=L_j)
        
        breakpoints = {0.0, 1.0}
        
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                da = shear[i] - shear[j]
                if abs(da) < 1e-15:
                    continue
                # L_i = L_j => X_i + c da = X_j => c = (X_j - X_i) / da
                c1 = (X[j] - X[i]) / da
                if 0.0 <= c1 <= 1.0:
                    breakpoints.add(c1)
                
                # L_i = R_j => X_i + c da = X_j + width => c = (X_j + width - X[i]) / da
                c2 = (X[j] + width - X[i]) / da
                if 0.0 <= c2 <= 1.0:
                    breakpoints.add(c2)
                    
                # R_i = L_j => X_i + width + c da = X[j] => c = (X[j] - X[i] - width) / da
                c3 = (X[j] - X[i] - width) / da
                if 0.0 <= c3 <= 1.0:
                    breakpoints.add(c3)
                    
                # R_i = R_j => same as L_i = L_j

        breakpoints = np.sort(np.array(list(breakpoints), dtype=np.float64))
        
        if len(breakpoints) < 2:
            return 0.0
            
        total_integral = 0.0
        for idx in range(len(breakpoints) - 1):
            c0 = breakpoints[idx]
            c1 = breakpoints[idx + 1]
            
            if c1 - c0 < 1e-15:
                continue
            
            # Compute U(c) at c0 and c1
            def union_len(c):
                L = X + c * shear
                R = L + width
                order = np.argsort(L)
                L_s = L[order]
                R_s = R[order]
                max_r = -np.inf
                u_len = 0.0
                for ii in range(N):
                    if L_s[ii] > max_r:
                        u_len += R_s[ii] - L_s[ii]
                        max_r = R_s[ii]
                    else:
                        if R_s[ii] > max_r:
                            max_r = R_s[ii]
                return u_len
            
            u0 = union_len(c0)
            u1 = union_len(c1)
            
            m = (u1 - u0) / (c1 - c0)
            b = u0 - m * c0
            
            # Integral of (m*c + b)^2 from c0 to c1
            # = m^2 * (c1^3 - c0^3)/3 + 2mb * (c1^2 - c0^2)/2 + b^2 * (c1 - c0)
            integral = (m ** 2) * (c1 ** 3 - c0 ** 3) / 3.0 + \
                       (2 * m * b) * (c1 ** 2 - c0 ** 2) / 2.0 + \
                       (b ** 2) * (c1 - c0)
            
            total_integral += integral
        
        return total_integral

    # Initial guess
    k = np.arange(N, dtype=np.float64)
    x0 = -k / (2.0 * N)

    # Global Search with Differential Evolution
    best_x = x0.copy()
    best_val = exact_objective(x0)
    
    # Bounds for X: The relative positions matter. 
    # We can shift all X by a constant without changing the volume.
    # Let's fix X[0] = 0 to reduce dimensionality? 
    # DE handles unconstrained, but fixing a variable helps convergence.
    # However, the objective is translation invariant.
    # Let's just use bounds [-1, 1] for all.
    bounds = [(-1.0, 1.0) for _ in range(N)]
    
    if time.time() < deadline - 60:
        try:
            de_res = differential_evolution(
                exact_objective, 
                bounds, 
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-8,
                polish=False, # We will polish later
                init='sobol'
            )
            if np.all(np.isfinite(de_res.x)) and de_res.fun < best_val:
                best_val = de_res.fun
                best_x = de_res.x
        except Exception:
            pass

    # Polish with Nelder-Mead
    if time.time() < deadline - 10:
        try:
            nm_res = minimize(exact_objective, best_x, method='Nelder-Mead',
                              options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14,
                                       'adaptive': True})
            if np.all(np.isfinite(nm_res.x)) and nm_res.fun < best_val:
                best_val = nm_res.fun
                best_x = nm_res.x
        except Exception:
            pass

    # Construct the 64x2 array
    positions = np.zeros((64, 2), dtype=np.float64)
    for k_idx in range(64):
        i = k_idx // 8
        j = k_idx % 8
        positions[k_idx, 0] = best_x[i]
        positions[k_idx, 1] = best_x[j]

    return positions

# EVOLVE-BLOCK-END
