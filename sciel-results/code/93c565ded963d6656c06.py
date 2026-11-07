"""Optimized program for second autocorrelation inequality."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 50

def _evaluate(h):
    """Compute the objective value for given heights h."""
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    
    g = np.convolve(h, h)
    m = len(g)
    
    y = np.zeros(m + 2)
    y[1:-1] = g
    
    w = 1.0 / (m + 1)
    
    terms = (y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2)
    l2 = (w / 3.0) * np.sum(terms)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 * norm_inf == 0:
        return 0.0
    
    value = l2 / (norm_1 * norm_inf)
    return value

def _negative_value(h):
    """Negative of the objective for minimization."""
    return -_evaluate(h)

# EVOLVE-BLOCK-START
def construct():
    """Construct 50 nonnegative step heights to maximize the objective."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin under 150s limit
    
    # Start with the seed
    best_h = np.ones(N)
    best_val = _evaluate(best_h)
    
    def _time_left():
        return deadline - time.time()
    
    # Strategy 1: DCT-based parameterization
    # Represent h as |IDCT(c)| where c is a low-dimensional coefficient vector
    # This enforces non-negativity and allows exploration of smooth shapes
    
    def _dct_to_h(coeffs):
        """Convert DCT coefficients to nonnegative heights."""
        # coeffs has length N
        # Use inverse DCT (type II) to get h
        # scipy.fft has idct, but we can use numpy
        # Actually, let's use scipy.fft.idct
        try:
            from scipy.fft import idct
            h = idct(coeffs, type=2, norm='ortho')
        except:
            # Fallback: manual DCT inverse
            h = np.zeros(N)
            for i in range(N):
                s = 0.0
                for k in range(N):
                    s += coeffs[k] * np.cos(np.pi * k * (2*i + 1) / (2*N))
                h[i] = s
        return np.abs(h)
    
    def _neg_value_dct(coeffs):
        """Negative value for DCT parameterization."""
        h = _dct_to_h(coeffs)
        # Normalize to avoid scale issues (scale invariance)
        h_max = np.max(h)
        if h_max > 0:
            h = h / h_max
        return -_evaluate(h)
    
    # Initial DCT coefficients: all ones in h corresponds to specific DCT
    # For h = ones(N), the DCT is concentrated at k=0
    # Let's start with a few initializations in DCT space
    
    inits_dct = []
    
    # Uniform: DCT of ones
    h_init = np.ones(N)
    try:
        from scipy.fft import dct
        c_init = dct(h_init, type=2, norm='ortho')
        inits_dct.append(c_init.copy())
    except:
        pass
    
    # Gaussian-like in DCT space
    np.random.seed(42)
    for _ in range(5):
        c = np.zeros(N)
        c[0] = np.random.uniform(1, 3)
        c[1] = np.random.uniform(-0.5, 0.5)
        c[2] = np.random.uniform(-0.3, 0.3)
        inits_dct.append(c)
    
    # Try optimizing in DCT space
    for init_c in inits_dct:
        if _time_left() < 20:
            break
        try:
            res = minimize(_neg_value_dct, init_c.copy(), method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
            h_candidate = _dct_to_h(res.x)
            h_max = np.max(h_candidate)
            if h_max > 0:
                h_candidate = h_candidate / h_max
            val_candidate = _evaluate(h_candidate)
            if val_candidate > best_val:
                best_val = val_candidate
                best_h = h_candidate
        except Exception:
            pass
    
    if _time_left() < 30:
        return list(np.maximum(best_h, 0.0))
    
    # Strategy 2: Direct L-BFGS-B from multiple starts
    np.random.seed(123)
    inits = [
        np.ones(N),
        np.ones(N) * 0.5,
        np.ones(N) * 2.0,
        np.linspace(0.1, 1.0, N),
        np.linspace(1.0, 0.1, N),
        np.exp(-np.linspace(0, 2, N)),
        1.0 / (1.0 + np.linspace(0, 5, N)),
        np.sin(np.linspace(0, np.pi, N)) + 0.1,
        np.cos(np.linspace(0, 2*np.pi, N)) + 1.5,
        np.random.rand(N) * 2,
        # Bimodal
        np.concatenate([np.linspace(1, 0.2, N//2), np.linspace(0.2, 1, N - N//2)]),
        # Plateau
        np.concatenate([np.zeros(5), np.ones(N-10), np.zeros(5)]),
    ]
    
    for init in inits:
        if _time_left() < 25:
            break
        init = np.maximum(init, 0.0)
        try:
            res = minimize(_negative_value, init.copy(), method='L-BFGS-B',
                           bounds=[(0, None)] * N,
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
            h_candidate = np.maximum(res.x, 0.0)
            val_candidate = _evaluate(h_candidate)
            if val_candidate > best_val:
                best_val = val_candidate
                best_h = h_candidate
        except Exception:
            pass
    
    if _time_left() < 20:
        return list(np.maximum(best_h, 0.0))
    
    # Strategy 3: Differential evolution for global search
    try:
        res_de = differential_evolution(_negative_value, 
                                         bounds=[(0, 5.0)] * N,
                                         seed=42, maxiter=80, 
                                         popsize=20, tol=1e-8,
                                         polish=True,
                                         init='latinhypercube',
                                         mutation=(0.3, 1.0),
                                         recombination=0.8)
        if res_de.success or res_de.fun < -best_val:
            h_candidate = np.maximum(res_de.x, 0.0)
            val_candidate = _evaluate(h_candidate)
            if val_candidate > best_val:
                best_val = val_candidate
                best_h = h_candidate
    except Exception:
        pass
    
    if _time_left() < 10:
        return list(np.maximum(best_h, 0.0))
    
    # Strategy 4: High-precision polish
    try:
        res = minimize(_negative_value, best_h.copy(), method='L-BFGS-B',
                       bounds=[(0, None)] * N,
                       options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-15, 'maxfun': 100000})
        h_candidate = np.maximum(res.x, 0.0)
        val_candidate = _evaluate(h_candidate)
        if val_candidate > best_val:
            best_val = val_candidate
            best_h = h_candidate
    except Exception:
        pass
    
    # Final: try a few random perturbations around best
    if _time_left() > 5:
        np.random.seed(999)
        for _ in range(10):
            if _time_left() < 3:
                break
            perturbation = np.random.randn(N) * 0.05
            h_candidate = np.maximum(best_h + perturbation, 0.0)
            try:
                res = minimize(_negative_value, h_candidate.copy(), method='L-BFGS-B',
                               bounds=[(0, None)] * N,
                               options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-14})
                h_candidate = np.maximum(res.x, 0.0)
                val_candidate = _evaluate(h_candidate)
                if val_candidate > best_val:
                    best_val = val_candidate
                    best_h = h_candidate
            except Exception:
                pass
    
    return list(np.maximum(best_h, 0.0))
# EVOLVE-BLOCK-END
