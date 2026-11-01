import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 195.0

def construct():
    """
    Finds coefficients c0, c4, c8 for the Hermite combination
    g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    such that g(0)=0 and the upper bound on A(g)*A(hat{g}) is minimized.
    Returns np.array([c0, c4, c8]).
    """
    
    # Degrees involved
    degrees = [0, 4, 8, 12]
    n_degrees = len(degrees)
    
    # Precompute Q0 (Norm matrix) and Q4 (4th Moment matrix)
    # Q0[j, k] = int H_j(x) H_k(x) exp(-x^2) dx = sqrt(pi) * 2^j * j! * delta_jk
    # Q4[j, k] = int x^4 H_j(x) H_k(x) exp(-x^2) dx
    
    def factorial(n):
        if n == 0: return 1
        res = 1
        for i in range(2, n+1):
            res *= i
        return res

    Q0 = np.zeros((n_degrees, n_degrees))
    for i, d in enumerate(degrees):
        Q0[i, i] = np.sqrt(np.pi) * (2 ** d) * factorial(d)
        
    # Compute Q4 using the recurrence relation for x^4 H_j
    # x^2 H_n = 0.25 H_{n+2} + (n + 0.5) H_n + 0.5 n(n-1) H_{n-2}
    # x^4 H_n = x^2 (x^2 H_n)
    # We need the coefficients of H_m in the expansion of x^4 H_n.
    # Let x^4 H_n = sum_k a_{n,k} H_k
    # Then Q4[j, k] = a_{j, k} * Q0[j, j] if j==k? No.
    # Q4[j, k] = int (sum_m a_{j,m} H_m) H_k e^{-x^2} = a_{j,k} Q0[j,j] (since Q0 is diagonal)
    
    def get_x4_coeffs(n):
        """Returns dict {m: coeff} for x^4 H_n = sum coeff * H_m"""
        # First x^2 H_n
        # x^2 H_n = 0.25 H_{n+2} + (n+0.5) H_n + 0.5*n*(n-1) H_{n-2}
        c_x2 = {}
        c_x2[n+2] = 0.25
        c_x2[n] = n + 0.5
        if n >= 2:
            c_x2[n-2] = 0.5 * n * (n-1)
            
        # Now x^4 H_n = x^2 (sum_m c_x2[m] H_m) = sum_m c_x2[m] (x^2 H_m)
        c_x4 = {}
        for m, coeff_m in c_x2.items():
            # Expand x^2 H_m
            # 0.25 H_{m+2} + (m+0.5) H_m + 0.5*m*(m-1) H_{m-2}
            c_x4[m+2] = c_x4.get(m+2, 0) + coeff_m * 0.25
            c_x4[m] = c_x4.get(m, 0) + coeff_m * (m + 0.5)
            if m >= 2:
                c_x4[m-2] = c_x4.get(m-2, 0) + coeff_m * 0.5 * m * (m-1)
                
        return c_x4

    Q4 = np.zeros((n_degrees, n_degrees))
    for i, d in enumerate(degrees):
        coeffs = get_x4_coeffs(d)
        for m, val in coeffs.items():
            if m in degrees:
                j = degrees.index(m)
                # Q4[i, j] = val * Q0[i, i]  (because int H_j H_i e^{-x^2} is 0 if i!=j? No, Q0 is diagonal)
                # Wait, Q4[i, j] = int x^4 H_i H_j e^{-x^2}.
                # x^4 H_i = sum_k a_{i,k} H_k.
                # int (sum_k a_{i,k} H_k) H_j e^{-x^2} = a_{i,j} int H_j H_j e^{-x^2} = a_{i,j} Q0[j,j].
                # So Q4[i, j] = a_{i,j} * Q0[j,j].
                Q4[i, j] = val * Q0[j, j]
    
    # Symmetry check: Q4 should be symmetric?
    # int x^4 H_i H_j = int x^4 H_j H_i. Yes.
    # Let's enforce symmetry to be safe against floating point errors in coefficient extraction
    Q4 = (Q4 + Q4.T) / 2.0

    def get_c12(c0, c4, c8):
        return -(c0 + 3.0*c4 + 105.0*c8) / 945.0

    def compute_objective_and_grad(coeffs):
        c0, c4, c8 = coeffs
        c12 = get_c12(c0, c4, c8)
        c = np.array([c0, c4, c8, c12])
        
        # J = c^T Q4 c
        J = c.T @ Q4 @ c
        # N = c^T Q0 c
        N = c.T @ Q0 @ c
        
        if N <= 0:
            return 1e10, np.zeros(3)
            
        S = J / (N * N)
        
        # Gradient with respect to c0, c4, c8
        # S = J N^{-2}
        # dS/dc = (dJ/dc N^{-2}) - J * 2 N^{-3} * (dN/dc)
        # dJ/dc = 2 Q4 c
        # dN/dc = 2 Q0 c
        
        dJ_dc = 2.0 * (Q4 @ c)
        dN_dc = 2.0 * (Q0 @ c)
        
        dS_dc = (dJ_dc / (N*N)) - (J * 2.0 * dN_dc / (N**3))
        
        # dS_dc is gradient wrt c = [c0, c4, c8, c12]
        # We need gradient wrt [c0, c4, c8]
        # c12 = -(c0 + 3c4 + 105c8)/945
        # dc12/dc0 = -1/945
        # dc12/dc4 = -3/945
        # dc12/dc8 = -105/945
        
        grad_3 = np.zeros(3)
        grad_3[0] = dS_dc[0] + dS_dc[3] * (-1.0/945.0)
        grad_3[1] = dS_dc[1] + dS_dc[3] * (-3.0/945.0)
        grad_3[2] = dS_dc[2] + dS_dc[3] * (-105.0/945.0)
        
        return S, grad_3

    # Initial guess from seed
    best_coeffs = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Multi-start optimization
    best_score = np.inf
    np.random.seed(42)
    
    # Widen bounds to allow more exploration
    bounds = [(-50.0, 50.0), (-10.0, 10.0), (-1.0, 1.0)]
    
    # Start with the seed point
    score, _ = compute_objective_and_grad(best_coeffs)
    if score < best_score:
        best_score = score
        best_result = best_coeffs.copy()

    # Generate diverse initial points using structured exploration
    initial_points = [best_coeffs.copy()]
    
    # Analytical insight: try points where the Hermite combination has
    # specific properties. The constraint g(0)=0 means:
    # c0 + 3*c4 + 105*c8 + 945*c12 = 0
    # Let's explore the 2D manifold of (c4, c8) with c0 determined
    
    # Add points along specific directions in coefficient space
    directions = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([1.0, -1.0, 1.0]),
        np.array([-1.0, 1.0, 1.0]),
        np.array([1.0, 1.0, -1.0]),
        np.array([-1.0, -1.0, 1.0]),
    ]
    
    for d in directions:
        for scale in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
            x0 = best_coeffs + scale * d
            x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
            initial_points.append(x0)
    
    # Add points near the seed with small perturbations
    for _ in range(10):
        perturbation = np.random.normal(0, 0.05, size=3)
        x0 = best_coeffs + perturbation
        x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
        initial_points.append(x0)
    
    # Add random points in the wider space
    for i in range(20):
        x0 = np.array([
            np.random.uniform(bounds[0][0], bounds[0][1]),
            np.random.uniform(bounds[1][0], bounds[1][1]),
            np.random.uniform(bounds[2][0], bounds[2][1])
        ])
        initial_points.append(x0)
    
    # Try all initial points with adaptive time budget
    time_per_opt = max(1.0, (TIME_LIMIT - 20.0) / len(initial_points))
    for x0 in initial_points:
        if time.time() > TIME_LIMIT - 20.0:
            break
        try:
            res = minimize(
                lambda x: compute_objective_and_grad(x)[0],
                x0,
                jac=lambda x: compute_objective_and_grad(x)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-11, 'maxls': 50}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
            
    # Final high-precision polish using Newton's method for better convergence
    if time.time() < TIME_LIMIT - 15:
        try:
            # Compute Hessian numerically
            def hessian(x):
                eps = 1e-7
                H = np.zeros((3, 3))
                for i in range(3):
                    for j in range(3):
                        if i == j:
                            fpp = (compute_objective_and_grad(x + eps * np.eye(3)[i])[0] 
                                 - 2 * compute_objective_and_grad(x)[0] 
                                 + compute_objective_and_grad(x - eps * np.eye(3)[i])[0]) / eps**2
                            H[i, j] = fpp
                        else:
                            fpp = (compute_objective_and_grad(x + eps * np.eye(3)[i] + eps * np.eye(3)[j])[0]
                                 - compute_objective_and_grad(x + eps * np.eye(3)[i] - eps * np.eye(3)[j])[0]
                                 - compute_objective_and_grad(x - eps * np.eye(3)[i] + eps * np.eye(3)[j])[0]
                                 + compute_objective_and_grad(x - eps * np.eye(3)[i] - eps * np.eye(3)[j])[0]) / (4 * eps**2)
                            H[i, j] = fpp
                return H
            
            # Newton's method iterations
            x_newton = best_result.copy()
            for _ in range(50):
                if time.time() > TIME_LIMIT - 10:
                    break
                try:
                    f, g = compute_objective_and_grad(x_newton)
                    H = hessian(x_newton)
                    # Regularize if needed
                    eigvals = np.linalg.eigvalsh(H)
                    if np.min(eigvals) < 1e-8:
                        H = H + 1e-8 * np.eye(3)
                    try:
                        step = np.linalg.solve(H, -g)
                    except np.linalg.LinAlgError:
                        step = -g / np.max(np.abs(g)) if np.max(np.abs(g)) > 0 else np.zeros(3)
                    # Line search
                    alpha = 1.0
                    for _ in range(30):
                        x_trial = x_newton + alpha * step
                        x_trial = np.clip(x_trial, [b[0] for b in bounds], [b[1] for b in bounds])
                        f_trial, _ = compute_objective_and_grad(x_trial)
                        if f_trial < f - 1e-4 * alpha * g @ step:
                            break
                        alpha *= 0.5
                    x_newton = x_trial
                    if np.linalg.norm(step * alpha) < 1e-12:
                        break
                except Exception:
                    break
            
            f_final, _ = compute_objective_and_grad(x_newton)
            if f_final < best_score:
                best_score = f_final
                best_result = x_newton.copy()
        except Exception:
            pass
    
    # Final L-BFGS-B polish with very tight tolerances
    if time.time() < TIME_LIMIT - 5:
        try:
            res = minimize(
                lambda x: compute_objective_and_grad(x)[0],
                best_result,
                jac=lambda x: compute_objective_and_grad(x)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 20000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x.copy()
        except Exception:
            pass

    return best_result
