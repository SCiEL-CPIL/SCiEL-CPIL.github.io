import numpy as np
import scipy.special
import scipy.optimize

def rudin_problem10() -> np.ndarray:
    k = 10
    n_theta = 70
    n_phi = 130
    
    # Gauss-Legendre nodes and weights in cos(theta)
    t, w = np.polynomial.legendre.leggauss(n_theta)
    theta = np.arccos(t)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    
    Theta, Phi = np.meshgrid(theta, phi, indexing='ij')
    T = np.cos(Theta)
    
    W = np.tile(w[:, np.newaxis], (1, n_phi)) * (2 * np.pi / n_phi)
    W_flat = W.flatten()
    
    m_vals = np.arange(-k, k + 1)
    A = np.zeros((n_theta * n_phi, 2 * k + 1), dtype=np.float64)
    
    for idx, m in enumerate(m_vals):
        abs_m = abs(m)
        log_N = 0.5 * (np.log(21) - np.log(4 * np.pi) + scipy.special.gammaln(11 - abs_m) - scipy.special.gammaln(11 + abs_m))
        N_m = np.exp(log_N)
        
        P_lm = scipy.special.lpmv(abs_m, k, T)
        
        if m == 0:
            Y_m = N_m * P_lm
        elif m > 0:
            Y_m = np.sqrt(2) * N_m * P_lm * np.cos(m * Phi)
        else:
            Y_m = np.sqrt(2) * N_m * P_lm * np.sin(abs_m * Phi)
            
        A[:, idx] = Y_m.flatten()
        
    dim = 2 * k + 1
    
    # Objective: Minimize R = ||f||_4 / ||f||_2
    # Equivalent to minimizing R^4 = ||f||_4^4 / ||f||_2^4
    # This avoids square roots and handles scaling.
    def objective(c):
        y = A.dot(c)
        norm2_sq = np.dot(W_flat, y**2)
        if norm2_sq == 0:
            return 1e10
        norm4 = np.dot(W_flat, y**4)
        return norm4 / (norm2_sq ** 2)
        
    # Bounds for differential evolution
    # We can restrict to [-1, 1] since the function is homogeneous.
    bounds = [(-1.0, 1.0) for _ in range(dim)]
    
    # Use Differential Evolution for global search
    # popsize=10, maxiter=1000, tol=1e-8, mutation=(0.5, 1.0), recombination=0.9
    try:
        res = scipy.optimize.differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=2000, 
            tol=1e-10, 
            polish=True,  # Uses L-BFGS-B at the end for fine-tuning
            init='sobol' # Quasi-random initialization for better coverage
        )
        best_c = res.x
    except Exception:
        # Fallback if DE fails
        best_c = np.zeros(dim)
        best_c[0] = 1.0

    # Normalize and validate
    norm_c = np.linalg.norm(best_c)
    if norm_c > 1e-12:
        best_c = best_c / norm_c
    else:
        best_c = np.zeros(dim)
        best_c[0] = 1.0
        
    # Check validity constraints from task
    if np.any(np.isnan(best_c)) or np.any(np.isinf(best_c)):
        best_c = np.zeros(dim)
        best_c[0] = 1.0
        
    max_c = np.max(np.abs(best_c))
    if max_c < 1e-3:
        best_c = np.zeros(dim)
        best_c[0] = 1.0
    elif max_c > 1e5:
        best_c = best_c / max_c * 1.0 # Scale down, though normalization should prevent this if norm is 1
        
    return best_c
