import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize
import time

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    Uses a spectral tensor formulation for fast evaluation of the L4 norm,
    enabling a large multi-start optimization search.
    """
    start_time = time.time()
    deadline = start_time + 570  # Safety margin

    k = 10
    n_m = 2 * k + 1  # 21 coefficients

    # 1. Setup Quadrature Grid
    # 70-point Gauss-Legendre in mu = cos(theta)
    mu, w = np.polynomial.legendre.leggauss(70)
    # 130 equally spaced phi
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    d_phi = 2 * np.pi / n_phi

    # Flatten grid
    mu_flat, phi_flat = np.meshgrid(mu, phi, indexing='ij')
    mu_flat = mu_flat.flatten()
    phi_flat = phi_flat.flatten()
    w_flat = np.repeat(w, n_phi) * d_phi  # Include d_phi in weights for the integral
    
    n_points = len(mu_flat)

    # 2. Construct Basis Functions A
    A = np.zeros((n_points, n_m))
    
    def get_norm(m):
        m_abs = abs(m)
        val = (21.0 / (4.0 * np.pi)) * (factorial(10 - m_abs) / factorial(10 + m_abs))
        return np.sqrt(val)
    
    # m = 0
    idx = 10 
    P = lpmv(0, 10, mu_flat)
    N = get_norm(0)
    A[:, idx] = N * P
    
    # m > 0
    for m in range(1, k + 1):
        idx_pos = m + 10
        idx_neg = -m + 10
        P = lpmv(m, 10, mu_flat)
        N = get_norm(m)
        sqrt2_N = np.sqrt(2.0) * N
        A[:, idx_pos] = sqrt2_N * P * np.cos(m * phi_flat)
        A[:, idx_neg] = sqrt2_N * P * np.sin(m * phi_flat)
        
    # 3. Compute the 4th-order tensor Q for fast L4 norm evaluation
    # Q_ijkl = sum_p w_p A_pi A_pj A_pk A_pl
    # We use B = A * sqrt(w) to simplify
    B = A * np.sqrt(w_flat)[:, None]
    # Compute Q using einsum
    Q = np.einsum('pi,pj,pk,pl->ijkl', B, B, B, B)
    
    # Optional: Symmetrize Q to reduce numerical asymmetry
    # Q = (Q + Q.transpose(1,0,2,3) + ... ) / 24 ... 
    # But einsum on symmetric B should be symmetric enough.

    def objective_and_grad(c):
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-12:
            return 1e10, np.zeros(n_m)
        
        # F(c) = ||f||_4^4 = sum Q_ijkl c_i c_j c_k c_l
        F = np.einsum('ijkl,i,j,k,l->', Q, c, c, c, c)
        
        if F < 1e-15:
            return 1e10, np.zeros(n_m)
            
        # Gradient of F: G_i = 4 * sum_jkl Q_ijkl c_j c_k c_l
        G = 4.0 * np.einsum('ijkl,j,k,l->i', Q, c, c, c)
        
        # R = F^0.25 / norm_c
        R = F**0.25 / norm_c
        
        # Gradient of R
        # dR/dc_i = (0.25 * F^-0.75 * G_i * norm_c - F^0.25 * c_i / norm_c^2) / norm_c
        # = 0.25 * F^-0.75 * G_i / norm_c - F^0.25 * c_i / norm_c^3
        grad_R = 0.25 * (F ** -0.75) * (G / norm_c) - (F ** 0.25) * (c / (norm_c ** 3))
        
        return R, grad_R

    # 4. Generate Initial Candidates
    candidates = []
    
    # Pure modes
    for i in range(n_m):
        c = np.zeros(n_m)
        c[i] = 1.0
        candidates.append(c)
        
    # Pairs of modes
    for i in range(n_m):
        for j in range(i + 1, n_m):
            if time.time() > deadline - 60:
                break
            c = np.zeros(n_m)
            c[i] = 1.0
            c[j] = 1.0
            c /= np.linalg.norm(c)
            candidates.append(c)
            
    # Random normalized vectors
    rng = np.random.default_rng(42)
    n_random = 500
    for _ in range(n_random):
        if time.time() > deadline - 30:
            break
        c_rand = rng.standard_normal(n_m)
        c_rand /= np.linalg.norm(c_rand)
        candidates.append(c_rand)

    # 5. Optimize Each Candidate
    best_c = None
    best_val = np.inf
    
    # Sort by initial objective to prioritize? 
    # With fast eval, we can just run all.
    for c_init in candidates:
        if time.time() > deadline - 5:
            break
            
        try:
            res = minimize(
                objective_and_grad, 
                c_init, 
                jac=True, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            val = res.fun
            if val < best_val and np.all(np.isfinite(res.x)):
                best_val = val
                best_c = res.x.copy()
        except Exception:
            # If optimization fails, check the initial point itself
            try:
                val_init, _ = objective_and_grad(c_init)
                if val_init < best_val:
                    best_val = val_init
                    best_c = c_init.copy()
            except:
                pass

    # 6. Final Validity Check and Scaling
    if best_c is None or not np.all(np.isfinite(best_c)):
        # Fallback to m=0
        best_c = np.zeros(n_m)
        best_c[10] = 1.0
        
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3 or max_abs > 1e5:
        if max_abs > 1e5:
            best_c = best_c * (1e4 / max_abs)
        elif max_abs < 1e-3:
            best_c = best_c * (1e-2 / max_abs)
            
    return best_c
