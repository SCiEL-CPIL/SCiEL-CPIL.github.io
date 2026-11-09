import numpy as np
from scipy.special import lpmv
from scipy.linalg import eigh
from scipy.optimize import minimize

def rudin_problem10() -> np.ndarray:
    k = 10
    n_theta = 70
    n_phi = 130
    
    # Grid
    # Gauss-Legendre nodes in cos(theta)
    cos_theta, w_theta = np.polynomial.legendre.leggauss(n_theta)
    # Map from [-1, 1] to [0, pi] is not needed for cos_theta directly, 
    # leggauss gives nodes in [-1, 1] which correspond to cos(theta) in [-1, 1].
    # theta = arccos(cos_theta)
    
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    w_phi = np.ones(n_phi) * (2 * np.pi / n_phi)
    
    # Create meshgrid
    cos_theta_grid, phi_grid = np.meshgrid(cos_theta, phi, indexing='ij')
    cos_theta_flat = cos_theta_grid.flatten()
    phi_flat = phi_grid.flatten()
    
    # Weights
    w = np.outer(w_theta, w_phi).flatten()
    
    # Normalization constants N_m
    # N_m = sqrt(21/(4*pi) * (10-m)! / (10+m)!)
    # Note: The problem statement defines N_m for the basis.
    # Y_0 = N_0 P^0
    # Y_m (m>0) = sqrt(2) N_m P^m cos(m phi)
    # Y_m (m<0) = sqrt(2) N_|m| P^|m| sin(|m| phi)
    
    # We need to construct the basis matrix Y of shape (N_points, 21)
    N_points = n_theta * n_phi
    Y = np.zeros((N_points, 2 * k + 1))
    
    for m in range(-k, k + 1):
        abs_m = abs(m)
        # Factorial ratio
        # (10-m)! / (10+m)!
        # Use log gamma for stability? 10 is small, direct factorial is fine.
        num = 1
        for i in range(1, 10 - abs_m + 1):
            num *= i
        den = 1
        for i in range(1, 10 + abs_m + 1):
            den *= i
        ratio = num / den
        N_m = np.sqrt(21.0 / (4.0 * np.pi) * ratio)
        
        # P_10^m(cos_theta)
        # scipy.special.lpmv(m, n, x) includes Condon-Shortley phase
        P = lpmv(abs_m, k, cos_theta_flat)
        
        idx = m + k
        if m == 0:
            Y[:, idx] = N_m * P
        elif m > 0:
            Y[:, idx] = np.sqrt(2.0) * N_m * P * np.cos(m * phi_flat)
        else:
            Y[:, idx] = np.sqrt(2.0) * N_m * P * np.sin(abs_m * phi_flat)
            
    # Verify orthonormality (approximate)
    # Gram = Y.T @ (w[:, None] * Y)
    # Should be identity.
    
    def compute_R(c):
        # c is normalized
        f = Y @ c
        f2 = f * f
        f4 = f2 * f2
        norm2_sq = np.sum(w * f2)
        norm4_sq = np.sum(w * f4)
        # R = ||f||_4 / ||f||_2 = sqrt(norm4_sq) / sqrt(norm2_sq)
        # Since norm2_sq should be 1 if c is normalized and basis orthonormal
        # R = sqrt(norm4_sq) / sqrt(norm2_sq)
        return np.sqrt(norm4_sq / norm2_sq)
        
    def find_best_coeff():
        best_c = None
        best_R = np.inf
        
        # Try a few random starts
        rng = np.random.default_rng(42)
        
        # Generate initial guesses
        initial_guesses = []
        for i in range(30):
            c = rng.normal(size=2 * k + 1)
            c /= np.linalg.norm(c)
            initial_guesses.append(c)
            
        # Add some structured guesses
        c0 = np.zeros(2 * k + 1)
        c0[k] = 1.0
        initial_guesses.append(c0)
        
        c10 = np.zeros(2 * k + 1)
        c10[0] = 1.0
        initial_guesses.append(c10)
        
        c10b = np.zeros(2 * k + 1)
        c10b[20] = 1.0
        initial_guesses.append(c10b)
        
        c010 = np.zeros(2 * k + 1)
        c010[k] = 1.0
        c010[20] = 1.0
        c010 /= np.linalg.norm(c010)
        initial_guesses.append(c010)
        
        for c_init in initial_guesses:
            c = c_init.copy()
            
            # Phase 1: Fast eigenvalue iteration
            for _ in range(10):
                f = Y @ c
                f2 = f * f
                M = (Y.T * (w * f2)) @ Y
                evals, evecs = eigh(M)
                new_c = evecs[:, 0]
                if np.dot(new_c, c) < 0:
                    new_c = -new_c
                c = new_c
                c /= np.linalg.norm(c)
                
            # Phase 2: Polish with SLSQP
            def obj(c_vec):
                return compute_R(c_vec)
            
            def constraint(c_vec):
                return np.sum(c_vec**2) - 1.0
            
            try:
                res = minimize(obj, c, method='SLSQP', 
                               constraints={'type': 'eq', 'fun': constraint}, 
                               options={'maxiter': 100, 'ftol': 1e-12})
                if res.success:
                    c_final = res.x
                    c_final /= np.linalg.norm(c_final)
                    R_final = compute_R(c_final)
                    if R_final < best_R:
                        best_R = R_final
                        best_c = c_final.copy()
                else:
                    R_final = compute_R(c)
                    if R_final < best_R:
                        best_R = R_final
                        best_c = c.copy()
            except Exception:
                R_final = compute_R(c)
                if R_final < best_R:
                    best_R = R_final
                    best_c = c.copy()
                
        return best_c, best_R

    coeffs, _ = find_best_coeff()
    return coeffs
