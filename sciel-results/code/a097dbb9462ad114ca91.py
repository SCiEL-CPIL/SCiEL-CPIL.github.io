import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 140.0  # Safety margin before 150s limit

def _score(h):
    """Compute the score: ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    h_norm = h / l1
    l2_sq = np.sum(h_norm ** 2)
    conv = np.convolve(h_norm, h_norm, mode='full')
    max_val = np.max(conv)
    if max_val <= 0:
        return 0.0
    return l2_sq / max_val

def _score_neg(h):
    return -_score(h)

def _gaussian_profile(alpha, mu=0.0):
    """
    Generate heights h_k = exp(-alpha * (k - mu)^2) for k in [0, N).
    """
    k = np.arange(N)
    h = np.exp(-alpha * (k - mu) ** 2)
    return h

def _golden_section_search(f, a, b, tol=1e-6, max_iter=50):
    """
    Golden-section search to find the minimum of f in [a, b].
    Returns the x value that minimizes f.
    """
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = f(c)
    fd = f(d)
    
    for _ in range(max_iter):
        if b - a < tol:
            break
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = f(d)
            
    return (a + b) / 2.0

def _optimize_gaussian_family(start_time):
    """
    Optimize within the family h_k = exp(-alpha * (k - mu)^2).
    Uses golden-section search for alpha and grid search for mu.
    """
    best_h = np.zeros(N)
    best_score = 0.0
    
    # Define a wrapper for the objective function for a fixed mu
    # We want to maximize score, so we minimize -score
    def obj_alpha_neg(mu):
        def f(alpha):
            h = _gaussian_profile(alpha, mu)
            return -_score(h)
        return f
    
    # Search over a range of mus and alphas
    # mu can be anywhere in [0, N)
    # alpha > 0
    
    # Coarse grid for mu
    mus = np.linspace(0, N-1, 25)
    
    # For each mu, find optimal alpha
    for mu in mus:
        if time.time() - start_time > DEADLINE:
            break
            
        f = obj_alpha_neg(mu)
        
        # Golden section search for alpha in [1e-4, 10.0]
        # The optimal alpha is likely in this range for N=50
        alpha_opt = _golden_section_search(f, 1e-4, 10.0, tol=1e-5, max_iter=30)
        
        h_opt = _gaussian_profile(alpha_opt, mu)
        s_opt = _score(h_opt)
        
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()
            
    return best_h, best_score

def construct():
    start_time = time.time()
    
    # 1. Initialize with best known heuristics
    # Uniform
    best_h = np.ones(N)
    best_score = _score(best_h)
    
    # 2. Sidon-like sets (from current program)
    sidon_sets = [
        [0, 1, 4, 10, 18, 23, 31, 37, 46],
        [0, 2, 5, 11, 17, 24, 30, 38, 45],
        [0, 3, 6, 12, 19, 25, 33, 39, 48],
        [0, 1, 3, 7, 12, 18, 25, 31, 40],
    ]
    
    for idx_set in sidon_sets:
        h = np.zeros(N)
        h[idx_set] = 1.0
        s = _score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
            
    # 3. Structural Change: Optimize Gaussian Family
    # This is the "new direction" suggested.
    # It's fast and might find a better local optimum than random sparse sets.
    if time.time() - start_time < DEADLINE - 10:
        h_gauss, s_gauss = _optimize_gaussian_family(start_time)
        if s_gauss > best_score:
            best_score = s_gauss
            best_h = h_gauss.copy()
            
    # 4. Random sparse initializations
    rng = np.random.RandomState(42)
    candidates = [best_h.copy()]
    
    for i in range(50):
        k = rng.randint(5, 15)
        idx = rng.choice(N, k, replace=False)
        h = np.zeros(N)
        h[idx] = 1.0
        candidates.append(h)
        
    # Sort candidates by score
    candidates.sort(key=_score, reverse=True)
    
    # Optimization loop
    # We will use L-BFGS-B to maximize the score directly.
    
    for i in range(min(10, len(candidates))):
        if time.time() - start_time > DEADLINE - 5:
            break
            
        h0 = candidates[i]
        
        # Direct optimization
        try:
            res = minimize(_score_neg, h0, method='L-BFGS-B', 
                           bounds=[(0, None)] * N,
                           options={'maxiter': 50, 'gtol': 1e-8, 'maxls': 20})
            h_opt = np.maximum(res.x, 0.0)
            s_opt = _score(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()
        except Exception:
            pass
            
        # If we still have time, do a second pass with more iterations on the best found so far
        if i == 0 and time.time() - start_time < DEADLINE - 20:
            try:
                res2 = minimize(_score_neg, best_h, method='L-BFGS-B', 
                                bounds=[(0, None)] * N,
                                options={'maxiter': 200, 'gtol': 1e-10, 'maxls': 20})
                h_opt2 = np.maximum(res2.x, 0.0)
                s_opt2 = _score(h_opt2)
                if s_opt2 > best_score:
                    best_score = s_opt2
                    best_h = h_opt2.copy()
            except Exception:
                pass

    return best_h.tolist()
