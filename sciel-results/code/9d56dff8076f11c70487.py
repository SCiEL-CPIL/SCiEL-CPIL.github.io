import numpy as np
from scipy.fft import dct, idct
from scipy.optimize import minimize

N = 50

def construct():
    K = 40  # Increased from 20

    def compute_value_from_h(h):
        s = np.sum(h)
        if s == 0:
            return 0.0
        h = h / s
        # Enforce symmetry to reduce search space and potentially find better symmetric solutions
        h = (h + h[::-1]) / 2.0
        s = np.sum(h)
        if s == 0:
            return 0.0
        h = h / s
        
        g = np.convolve(h, h)
        m = len(g)
        y = np.zeros(m + 2)
        y[1:-1] = g
        w = 1.0 / (m + 1)
        y_sq = y**2
        y_prod = y[:-1] * y[1:]
        l2 = (w / 3.0) * np.sum(y_sq[:-1] + y_prod + y_sq[1:])
        norm_1 = np.sum(np.abs(g)) / (m + 1)
        norm_inf = np.max(np.abs(g))
        if norm_1 == 0 or norm_inf == 0:
            return 0.0
        return l2 / (norm_1 * norm_inf)

    def objective(c):
        c_full = np.zeros(N)
        c_full[:K] = c
        h_raw = idct(c_full, type=2, norm='ortho')
        h = np.abs(h_raw)
        return -compute_value_from_h(h)

    best_val = -np.inf
    best_c = None

    np.random.seed(42)

    # Define restart strategies
    restarts = []
    
    # 1. DC component only
    c0 = np.zeros(K)
    c0[0] = 1.0
    restarts.append(c0)
    
    # 2. Low frequency ramp
    c0 = np.zeros(K)
    c0[0] = 1.0
    for i in range(1, 10):
        c0[i] = 0.5 * (1 - i/10)
    restarts.append(c0)
    
    # 3. Random small perturbations
    for _ in range(5):
        c0 = np.random.randn(K) * 0.1
        c0[0] = 1.0
        restarts.append(c0)
        
    # 4. Peaky shape (high DC, low others)
    c0 = np.zeros(K)
    c0[0] = 2.0
    c0[1:5] = -0.2
    restarts.append(c0)
    
    # 5. Broad shape
    c0 = np.zeros(K)
    c0[0] = 0.5
    c0[1:10] = 0.3
    restarts.append(c0)

    for i, c0 in enumerate(restarts):
        # Use L-BFGS-B with numerical gradient
        # Note: abs() makes it non-smooth, but L-BFGS-B often works well enough
        # We can also try Powell for robustness
        method = 'L-BFGS-B'
        jac = '2-point' # Numerical gradient
        
        try:
            res = minimize(objective, c0, method=method, jac=jac,
                          options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-8})
            val = -res.fun
            if val > best_val:
                best_val = val
                best_c = res.x.copy()
        except Exception:
            pass
            
        # Also try Powell for some restarts to handle non-smoothness better
        if i % 2 == 0:
            try:
                res_p = minimize(objective, c0, method='Powell',
                               options={'maxiter': 500, 'ftol': 1e-12, 'xtol': 1e-8})
                val_p = -res_p.fun
                if val_p > best_val:
                    best_val = val_p
                    best_c = res_p.x.copy()
            except Exception:
                pass

    if best_c is None:
        best_c = np.zeros(K)
        best_c[0] = 1.0

    c_full = np.zeros(N)
    c_full[:K] = best_c
    h_raw = idct(c_full, type=2, norm='ortho')
    h = np.abs(h_raw)
    h = (h + h[::-1]) / 2.0  # Symmetrize final result
    
    s = np.sum(h)
    if s > 0:
        h = h / s
    else:
        h = np.ones(N) / N

    return h
