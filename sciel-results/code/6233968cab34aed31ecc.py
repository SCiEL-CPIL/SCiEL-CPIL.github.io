import numpy as np
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N

def _score_raw(h):
    """Compute the raw score: L2^2 / (L1 * AutoConvMax)."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def construct():
    # Formulate as: Maximize ||h||_2^2 subject to sum(h)=1, h>=0, max(h*h)<=1
    # This is a Quadratic Programming problem.
    
    def objective(h):
        return -np.sum(h**2)
    
    cons = []
    
    # Equality: sum(h) = 1
    cons.append({
        'type': 'eq',
        'fun': lambda h: np.sum(h) - 1.0,
        'jac': lambda h: np.ones(N)
    })
    
    # Inequality: (h*h)_k <= 1 for all k
    for k in range(2*N - 1):
        i_start = max(0, k - (N - 1))
        i_end = min(N - 1, k)
        
        def conv_func(h, k=k, i_start=i_start, i_end=i_end):
            s = 0.0
            for i in range(i_start, i_end + 1):
                j = k - i
                s += h[i] * h[j]
            return s - 1.0
        
        def conv_jac(h, k=k, i_start=i_start, i_end=i_end):
            jac = np.zeros(N)
            for i in range(i_start, i_end + 1):
                j = k - i
                if i == j:
                    jac[i] += 2.0 * h[i]
                else:
                    jac[i] += h[j]
                    jac[j] += h[i]
            return jac

        cons.append({
            'type': 'ineq',
            'fun': conv_func,
            'jac': conv_jac
        })
        
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    # Initial guesses
    x0_uniform = np.ones(N) / N
    np.random.seed(42)
    x0_sparse = np.zeros(N)
    support = np.random.choice(N, size=15, replace=False)
    x0_sparse[support] = 1.0 / 15
    
    best_h = None
    best_score = -np.inf
    
    for x0 in [x0_uniform, x0_sparse]:
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 2000, 'ftol': 1e-12})
            if res.success:
                h_sol = res.x
                h_sol = np.clip(h_sol, 0, None)
                s = np.sum(h_sol)
                if s > 0:
                    h_sol /= s
                score = _score_raw(h_sol)
                if score > best_score:
                    best_score = score
                    best_h = h_sol
        except Exception:
            pass
            
    if best_h is None:
        return list(np.ones(N) / N)
        
    return list(best_h)
