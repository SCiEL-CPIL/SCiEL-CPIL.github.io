import numpy as np
import time
from scipy.optimize import minimize

N = 50

def evaluate(h):
    """Compute the value for a given set of step heights."""
    h = np.array(h, dtype=np.float64)
    if np.any(h < -1e-12) or not np.all(np.isfinite(h)):
        return -1.0
    h = np.clip(h, 0, None)
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:m+1] = g
    w = 1.0 / (m + 1)
    terms = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(terms)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 * norm_inf == 0:
        return -1.0
    return l2 / (norm_1 * norm_inf)

def construct():
    deadline = time.time() + 115.0  # Safety margin
    best_h = None
    best_val = -1.0
    
    # Strategy: Structured Spectral Parameterization
    # Instead of optimizing 50 free variables, we represent h as a sum of cosine modes.
    # h_k = sum_{j=1}^{K} a_j * cos(2 * pi * j * k / (N+1)) + a_0
    # This restricts the search to smooth, band-limited functions, which are likely candidates
    # for optimal autocorrelation properties. We optimize the coefficients a_j.
    # We use Nelder-Mead on the coefficients, with a penalty for negative h.
    
    K = 10  # Number of cosine modes
    
    def params_to_h(params):
        a0 = params[0]
        a = params[1:K+1]
        k = np.arange(N)
        h = np.zeros(N)
        h += a0
        for j in range(1, K+1):
            h += a[j-1] * np.cos(2 * np.pi * j * k / (N+1))
        return h
    
    def objective(params):
        h = params_to_h(params)
        # Penalize negative heights heavily to guide optimization towards valid region
        # or just clip and evaluate. Clipping is better for Nelder-Mead continuity if penalty is soft.
        # Let's use a smooth penalty: -M * sum(h[h<0]^2)
        neg_mask = h < 0
        if np.any(neg_mask):
            penalty = 1000.0 * np.sum(h[neg_mask]**2)
        else:
            penalty = 0.0
        val = evaluate(h)
        return -val + penalty
    
    # Generate diverse initializations in parameter space
    initializations = []
    
    # 1. Constant
    initializations.append(np.zeros(K+1)) # a0=0, others 0 -> h=0 (bad, but NM will move)
    initializations[0][0] = 1.0
    
    # 2. Low frequency dominant
    p = np.zeros(K+1)
    p[0] = 1.0
    p[1] = 0.5
    initializations.append(p)
    
    # 3. Medium frequency
    p = np.zeros(K+1)
    p[0] = 1.0
    p[2] = 0.5
    initializations.append(p)
    
    # 4. High frequency
    p = np.zeros(K+1)
    p[0] = 1.0
    p[5] = 0.5
    initializations.append(p)
    
    # 5. Random coefficients
    for _ in range(10):
        p = np.random.uniform(-0.5, 1.5, K+1)
        initializations.append(p)
        
    # 6. Specific shapes mapped to params (rough inverse)
    # Triangle
    h_tri = 1 - np.abs(2*np.linspace(0, 1, N) - 1)
    # Fit coeffs? Too complex. Just use random starts that cover the space.
    
    for p0 in initializations:
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                fun=objective,
                x0=p0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            h_opt = params_to_h(result.x)
            # Ensure non-negative
            h_opt = np.clip(h_opt, 0, None)
            
            val = evaluate(h_opt)
            
            if val > best_val:
                best_val = val
                best_h = h_opt
                
        except Exception:
            pass

    # Fallback: If spectral search failed or found nothing, try a few direct random searches
    # to ensure we beat the seed if possible
    if best_val < 1.0:
        for _ in range(20):
            if time.time() > deadline:
                break
            h_rand = np.random.uniform(0, 1, N)
            val = evaluate(h_rand)
            if val > best_val:
                best_val = val
                best_h = h_rand
                
    if best_h is None:
        best_h = np.ones(N)
        
    return list(best_h)

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
