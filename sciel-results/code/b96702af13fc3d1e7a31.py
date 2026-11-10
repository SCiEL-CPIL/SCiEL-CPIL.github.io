import numpy as np
import scipy.optimize
import scipy.special

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
    
    # Weights
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
        
    def objective(c):
        y = A.dot(c)
        return np.sum(W_flat * y**4)
        
    def constraint_eq(c):
        return np.sum(c**2) - 1.0
        
    # Use Basin Hopping for global search
    # Initial guess: random unit vector
    rng = np.random.default_rng(42)
    c0 = rng.normal(size=2 * k + 1)
    c0 /= np.linalg.norm(c0)
    
    # Basin Hopping parameters
    # stepsize: magnitude of perturbations
    # niter: number of basin hopping steps
    # Each step involves a local minimization (SLSQP)
    
    try:
        res = scipy.optimize.basinhopping(
            objective,
            c0,
            minimizer_kwargs={
                'constraints': [{'type': 'eq', 'fun': constraint_eq}],
                'method': 'SLSQP',
                'maxiter': 200,
                'ftol': 1e-12
            },
            stepsize=0.5,
            niter=100,
            seed=42,
            disp=False
        )
        best_c = res.x
        best_val = res.fun
    except Exception:
        # Fallback to simple random restarts if BH fails
        best_c = None
        best_val = np.inf
        for _ in range(10):
            c_rand = rng.normal(size=2 * k + 1)
            c_rand /= np.linalg.norm(c_rand)
            try:
                res_local = scipy.optimize.minimize(
                    objective, 
                    c_rand, 
                    method='SLSQP', 
                    constraints=[{'type': 'eq', 'fun': constraint_eq}],
                    options={'maxiter': 500, 'ftol': 1e-12}
                )
                if res_local.success and res_local.fun < best_val:
                    best_val = res_local.fun
                    best_c = res_local.x
            except:
                continue
        if best_c is None:
            best_c = np.ones(2 * k + 1)
            best_c /= np.linalg.norm(best_c)

    return best_c
