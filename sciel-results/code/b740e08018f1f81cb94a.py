import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _ratio(h):
    """Compute R = ||h||_2^2 / (||h||_1 * ||h*h||_inf) for non-negative h."""
    if np.any(h < -1e-12):
        return -np.inf
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    l2_sq = np.sum(h * h)
    if l1 < 1e-12:
        return -np.inf
    # Autoconvolution (full convolution of h with itself)
    h_conv = np.convolve(h, h, mode='full')
    sup_norm = np.max(h_conv)
    if sup_norm < 1e-12:
        return -np.inf
    return l2_sq / (l1 * sup_norm)

def _negative_ratio(h):
    """Negative of ratio for minimization."""
    return -_ratio(h)

def construct():
    """
    Strategy: Model the nonnegative heights as the absolute values of the inverse 
    Fourier transform of a complex vector with fixed L2 norm. This decouples the 
    magnitude profile from the phase. We then optimize over the phase variables 
    to minimize the autoconvolution peak, thereby maximizing the ratio.
    
    We use L-BFGS-B on the phase variables. The magnitude is kept fixed at 
    uniform (1/sqrt(N)) to satisfy the L2 constraint ||h||_2 = 1. 
    Actually, the problem is scale invariant, so we can fix ||h||_2 = 1.
    
    The key insight is that for a given magnitude spectrum |H|, the time-domain 
    signal h = IFFT(H) can have its phase manipulated to change its shape. 
    However, we need h to be non-negative. This is a constraint.
    
    Instead, we can directly optimize the non-negative heights h with a simple 
    parameterization that ensures non-negativity and explores a rich space. 
    We'll use a softmax-like parameterization or simply optimize with bounds.
    
    Let's try a different approach: use the fact that the ratio is scale-invariant. 
    We can fix ||h||_1 = 1. Then we need to maximize ||h||_2^2 / ||h*h||_inf. 
    This is equivalent to maximizing the "concentration" of h in L2 while 
    minimizing the peak of its autoconvolution.
    
    We'll use scipy.optimize.minimize with L-BFGS-B to minimize the negative 
    of the ratio, with bounds [0, inf) for each h_i.
    """
    start_time = time.time()
    deadline = start_time + 115  # 5 seconds margin
    
    # Initial guess: uniform
    h0 = np.ones(N) / N
    
    # We'll fix L1 norm to 1 by using a parameterization that sums to 1.
    # Actually, L-BFGS-B with bounds is simpler. Let's just optimize h directly
    # with the constraint that we evaluate the ratio which is scale-invariant.
    
    # Let's use a logarithmic parameterization to ensure non-negativity.
    # h_i = exp(x_i) / sum(exp(x_j)) to enforce L1=1 and h_i > 0.
    # Then we optimize x.
    
    def _ratio_from_x(x):
        # Softmax to get h with L1=1
        x_shifted = x - np.max(x)
        h = np.exp(x_shifted)
        h /= np.sum(h)
        return _ratio(h)
    
    def _neg_ratio_from_x(x):
        return -_ratio_from_x(x)
    
    # Initial x: all zeros (uniform)
    x0 = np.zeros(N)
    
    best_h = np.ones(N) / N
    best_score = _ratio(best_h)
    
    # Try several random restarts
    rng = np.random.default_rng(42)
    n_restarts = 20
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        if r == 0:
            x_start = x0
        else:
            # Random initial phase
            x_start = rng.normal(0, 1, size=N)
        
        try:
            res = minimize(_neg_ratio_from_x, x_start, method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-12})
            
            if res.success:
                x_shifted = res.x - np.max(res.x)
                h_new = np.exp(x_shifted)
                h_new /= np.sum(h_new)
                score_new = _ratio(h_new)
                if score_new > best_score:
                    best_score = score_new
                    best_h = h_new
        except Exception:
            pass
    
    # Also try coordinate descent on the phase to see if we can improve
    # This is a simpler approach that might be more robust
    if time.time() < deadline:
        h_cd = np.ones(N) / N
        best_cd_score = _ratio(h_cd)
        
        for iteration in range(500):
            if time.time() > deadline:
                break
            improved = False
            for i in range(N):
                # Try to adjust h_i while keeping L1=1
                # This is tricky, so let's skip this for now
                pass
            if not improved:
                break
    
    # Final check
    final_h = np.maximum(best_h, 0)
    final_h /= np.sum(final_h)
    
    return [float(x) for x in final_h]
