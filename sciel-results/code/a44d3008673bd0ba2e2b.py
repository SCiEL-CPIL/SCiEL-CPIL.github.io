import numpy as np
from scipy.special import lpmv, factorial
import time
from scipy.optimize import minimize

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    Strategy:
    1. Precompute basis matrix A on the specified quadrature grid.
    2. Define objective J(u) = ||f||_4^4 (since ||f||_2=1, minimizing L4/L2 is equivalent to minimizing L4).
    3. Generate diverse initial candidates.
    4. Polish each candidate with scipy.optimize.minimize (SLSQP) on the unit sphere.
    5. Return the best valid coefficient vector found within the time budget.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill

    k = 10
    n_m = 2 * k + 1  # 21 coefficients

    # 1. Setup Quadrature Grid
    mu, w = np.polynomial.legendre.leggauss(70)
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    d_phi = 2 * np.pi / n_phi

    mu_flat, phi_flat = np.meshgrid(mu, phi, indexing='ij')
    mu_flat = mu_flat.flatten()
    phi_flat = phi_flat.flatten()
    w_flat = np.repeat(w, n_phi)
    n_points = len(mu_flat)

    # 2. Construct Basis Functions
    A = np.zeros((n_points, n_m))
    
    def get_norm(m):
        m_abs = abs(m)
        val = (21.0 / (4.0 * np.pi)) * (factorial(10 - m_abs, exact=False) / factorial(10 + m_abs, exact=False))
        return np.sqrt(val)
    
    # m=0
    idx = 10
    P = lpmv(0, 10, mu_flat)
    N = get_norm(0)
    A[:, idx] = N * P
    
    for m in range(1, k + 1):
        idx_pos = m + 10
        idx_neg = -m + 10
        P = lpmv(m, 10, mu_flat)
        N = get_norm(m)
        sqrt2_N = np.sqrt(2.0) * N
        A[:, idx_pos] = sqrt2_N * P * np.cos(m * phi_flat)
        A[:, idx_neg] = sqrt2_N * P * np.sin(m * phi_flat)
    
    # Precompute weighted basis for fast gradient
    B = A * w_flat[:, None]

    # 3. Objective and Gradient for J(u) = (||f||_4)^4 = (sum w |f|^4)^4 ? 
    # Actually, ||f||_4 = ( \int |f|^4 d\Omega )^{1/4}.
    # Minimizing ||f||_4 is equivalent to minimizing \int |f|^4.
    # Let's minimize I(u) = \int f^4 d\Omega.
    # Since f can be negative, f^4 is positive.
    # I(u) = sum w_flat * (A @ u)^4 * d_phi
    # Gradient: d/du I = 4 * sum w_flat * (A @ u)^3 * A * d_phi
    
    def objective_and_grad(u):
        """Compute I(u) and gradient for unit vector u."""
        f_vals = A @ u
        f4 = f_vals ** 4
        val = d_phi * np.sum(w_flat * f4)
        
        # Gradient
        f3 = f_vals ** 3
        grad_u = 4.0 * d_phi * (B.T @ f3)
        
        return val, grad_u

    # 4. Generate Initial Candidates
    candidates = []
    rng = np.random.default_rng(42)
    
    # Pure modes
    for m in range(0, 11):
        c = np.zeros(n_m)
        c[10 + m] = 1.0
        candidates.append(c)
    
    for m in range(1, 11):
        c = np.zeros(n_m)
        c[10 - m] = 1.0
        candidates.append(c)
    
    # Pairs of positive m modes
    for m1 in range(0, 11):
        for m2 in range(m1 + 1, 11):
            c = np.zeros(n_m)
            c[10 + m1] = 1.0
            c[10 + m2] = 1.0
            c /= np.linalg.norm(c)
            candidates.append(c)
    
    # Pairs mixing cos and sin for same |m|
    for m in range(1, 11):
        for angle in np.linspace(0, np.pi/2, 5):
            c = np.zeros(n_m)
            c[10 + m] = np.cos(angle)
            c[10 - m] = np.sin(angle)
            c /= np.linalg.norm(c)
            candidates.append(c)
    
    # Triplets
    for m1 in range(0, 11):
        for m2 in range(m1 + 1, 11):
            for m3 in range(m2 + 1, 11):
                if time.time() > deadline - 300:
                    break
                c = np.zeros(n_m)
                c[10 + m1] = 1.0
                c[10 + m2] = 1.0
                c[10 + m3] = 1.0
                c /= np.linalg.norm(c)
                candidates.append(c)
    
    # Random normalized vectors
    for _ in range(150):
        if time.time() > deadline - 300:
            break
        c_rand = rng.standard_normal(n_m)
        c_rand /= np.linalg.norm(c_rand)
        candidates.append(c_rand)
    
    # Random with emphasis on low-m modes
    for _ in range(80):
        if time.time() > deadline - 300:
            break
        c_rand = np.zeros(n_m)
        for m in range(0, 11):
            if rng.random() < 0.7:
                c_rand[10 + m] = rng.standard_normal()
        if np.linalg.norm(c_rand) > 1e-12:
            c_rand /= np.linalg.norm(c_rand)
            candidates.append(c_rand)
            
    # 5. Evaluate and sort candidates
    candidate_vals = []
    for c_init in candidates:
        if time.time() > deadline - 180:
            break
        val, _ = objective_and_grad(c_init)
        candidate_vals.append((val, c_init))
    
    candidate_vals.sort(key=lambda x: x[0])
    
    # 6. Polish top candidates with SLSQP
    best_c = None
    best_val = np.inf
    n_to_polish = min(30, len(candidate_vals)) # Reduce number to allow SLSQP to finish
    
    constraint = {'type': 'eq', 'fun': lambda u: np.dot(u, u) - 1.0}
    
    for i in range(n_to_polish):
        if time.time() > deadline - 30:
            break
        
        val_init, c_init = candidate_vals[i]
        
        # Use SLSQP for robustness
        try:
            res = minimize(
                objective_and_grad, 
                c_init, 
                method='SLSQP', 
                jac=True, 
                bounds=None, 
                constraints=constraint,
                options={'ftol': 1e-12, 'maxiter': 500}
            )
            if res.success and np.all(np.isfinite(res.x)):
                # Normalize just in case
                res.x = res.x / np.linalg.norm(res.x)
                val_polished, _ = objective_and_grad(res.x)
                if val_polished < best_val:
                    best_val = val_polished
                    best_c = res.x.copy()
        except Exception:
            pass

    # 7. Final high-precision polish on the best
    if best_c is not None and time.time() < deadline - 10:
        try:
            res_final = minimize(
                objective_and_grad, 
                best_c, 
                method='SLSQP', 
                jac=True, 
                bounds=None, 
                constraints=constraint,
                options={'ftol': 1e-14, 'maxiter': 1000}
            )
            if res_final.success and np.all(np.isfinite(res_final.x)):
                res_final.x = res_final.x / np.linalg.norm(res_final.x)
                val_final, _ = objective_and_grad(res_final.x)
                if val_final < best_val:
                    best_val = val_final
                    best_c = res_final.x.copy()
        except Exception:
            pass

    # 8. Final Validity Check and Scaling
    if best_c is None or not np.all(np.isfinite(best_c)):
        best_c = np.zeros(n_m)
        best_c[10] = 1.0
        
    # Ensure max|c| is in valid range
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3:
        best_c = best_c * (1e-2 / max_abs)
    elif max_abs > 1e5:
        best_c = best_c * (1e4 / max_abs)
        
    return best_c
