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
    
    def objective(c):
        y = A.dot(c)
        return np.sum(W_flat * y**4)
        
    def gradient(c):
        y = A.dot(c)
        return 4.0 * A.T.dot(W_flat * y**3)
        
    def normalize(c):
        n = np.linalg.norm(c)
        if n == 0:
            return c
        return c / n
        
    def constraint_func(c):
        return np.dot(c, c) - 1.0

    def constraint_grad(c):
        return 2.0 * c
        
    # Generate candidates
    candidates = []
    # Basis vectors
    for i in range(dim):
        e = np.zeros(dim)
        e[i] = 1.0
        candidates.append(e)
        candidates.append(-e)
        
    # Random vectors
    rng = np.random.default_rng(42)
    for _ in range(100):
        c = rng.normal(size=dim)
        candidates.append(normalize(c))
        
    best_c = None
    best_val = np.inf
    
    # Optimize from each candidate using trust-constr
    for c0 in candidates:
        try:
            res = scipy.optimize.minimize(
                objective, 
                c0, 
                method='trust-constr', 
                jac=gradient,
                constraints=[{'type': 'eq', 'fun': constraint_func, 'jac': constraint_grad}],
                options={'maxiter': 500, 'gtol': 1e-8}
            )
            if res.success:
                c_opt = res.x
                c_opt = c_opt / np.linalg.norm(c_opt)
                val = objective(c_opt)
                if val < best_val:
                    best_val = val
                    best_c = c_opt
        except Exception:
            pass

    if best_c is None:
        # Fallback to best candidate without optimization
        for c0 in candidates:
            val = objective(c0)
            if val < best_val:
                best_val = val
                best_c = c0.copy()
                
    return best_c
