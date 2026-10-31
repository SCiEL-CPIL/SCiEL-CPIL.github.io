import numpy as np
from scipy import optimize

N = 600

def compute_autoconv_max(h):
    """Compute the normalized maximum of the autoconvolution of step function h."""
    # h is array of N heights
    # The autoconvolution in discrete form: for step functions on equal intervals
    # (f*f)[k] = sum_i h[i] * h[k-i] (with appropriate bounds)
    # But we need to be careful about the discrete convolution
    
    # For a step function on [-1/4, 1/4] with N equal intervals of width dx:
    # The autoconvolution at point t is ∫ f(s)f(t-s)ds
    # In discrete terms, if we think of the convolution at the grid points:
    
    # Actually, let's think of it as: the autoconvolution of the step function
    # can be computed by convolving the height array with itself.
    
    # The key normalization: the "normalized maximum" is likely
    # max(autoconv) / (sum(h) * dx)^2 * (dx) ... let me think about units.
    
    # If f(x) = h_i for x in interval i, then ∫f = dx * sum(h_i)
    # The autoconvolution (f*f)(t) has units of [f]^2 * [x]
    # The maximum of (f*f) divided by (∫f)^2 gives a dimensionless quantity
    # that depends only on the shape.
    
    # For the discrete convolution: if we compute np.convolve(h, h, 'full'),
    # this gives the sum of products, which approximates (1/dx) * (f*f) at grid points
    # because the integral is approximated by sum * dx.
    
    # So (f*f)(t_k) ≈ dx * sum_i h[i] * h[k-i]
    # And (∫f)^2 = (dx * sum(h))^2
    # So normalized max = max_k [dx * (h*h)_k] / (dx * sum(h))^2
    #                   = max_k (h*h)_k / (dx * (sum(h))^2)
    
    # But wait, this depends on dx. Let me reconsider.
    
    # Actually, I think the normalization might just be:
    # max(autoconv) / (∫f)^2 where autoconv is the actual integral
    
    # Let me just compute: 
    # autoconv = np.convolve(h, h, 'full')  # this is sum of h[i]*h[j] where i+j=k
    # This approximates (1/dx) * (f*f) evaluated at grid points
    # So (f*f) ≈ dx * autoconv
    # ∫f = dx * sum(h)
    # normalized_max = max(dx * autoconv) / (dx * sum(h))^2
    #                = max(autoconv) / (dx * (sum(h))^2)
    
    # Hmm, this still has dx in it. Let me think again.
    
    # Actually, I think for the purpose of this problem, the "normalized maximum" 
    # might simply be max(convolve(h,h)) / (sum(h))^2, which is scale-invariant
    # and depends only on the shape. The dx factor might cancel or be absorbed.
    
    # Let me just use: score = max(np.convolve(h, h, 'full')) / (sum(h))**2
    
    h = np.asarray(h, dtype=float)
    conv = np.convolve(h, h, 'full')
    total = sum(h)
    if total == 0:
        return 0.0
    return max(conv) / (total * total)

def construct():
    import time
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, leaving 10s margin
    
    # Initial guess: all ones
    h = np.ones(N)
    
    # Compute initial score
    current_score = compute_autoconv_max(h)
    best_h = h.copy()
    best_score = current_score
    
    # Use optimization to minimize the maximum of the autoconvolution
    # We'll use a smooth approximation: soft-max via log-sum-exp
    # or we can directly use scipy.optimize.minimize with a penalty
    
    def objective(x):
        """Minimize the max of autoconvolution, normalized."""
        h = np.abs(x)  # enforce nonnegativity
        conv = np.convolve(h, h, 'full')
        total = sum(h)
        if total < 1e-10:
            return 1e10
        # Use soft-max for smoothness
        # Actually, let's just use the actual max for simplicity, 
        # or a soft approximation
        m = max(conv)
        # Soft max: log(sum(exp(conv/T))) * T for small T
        # But this might be slow. Let's try direct max first.
        return m / (total * total)
    
    # Try L-BFGS-B with bounds
    # But the objective is non-smooth due to max. Let's use a soft approximation.
    
    def soft_objective(x, T=0.1):
        h = np.abs(x)
        conv = np.convolve(h, h, 'full')
        total = sum(h)
        if total < 1e-10:
            return 1e10
        # Soft max
        conv_max = max(conv)
        soft_max = conv_max + T * np.log(np.sum(np.exp((conv - conv_max) / T)))
        return soft_max / (total * total)
    
    # Try optimization
    try:
        # Use multiple restarts with different initial conditions
        np.random.seed(42)
        
        for restart in range(5):
            if time.time() > deadline - 30:
                break
            
            if restart == 0:
                h0 = np.ones(N)
            elif restart == 1:
                # Triangle: high in center
                h0 = 1.0 - np.abs(np.linspace(-1, 1, N))
                h0 = np.maximum(h0, 0)
            elif restart == 2:
                # Inverse triangle: high at edges
                h0 = np.abs(np.linspace(-1, 1, N))
                h0 = np.maximum(h0, 0.1)
            elif restart == 3:
                # Random
                h0 = np.random.rand(N) + 0.1
            else:
                # Cosine bump
                h0 = 0.5 * (1 + np.cos(np.pi * np.linspace(-1, 1, N)))
                h0 = np.maximum(h0, 0)
            
            h0 = np.abs(h0)
            
            result = optimize.minimize(
                soft_objective, h0,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if result.fun < best_score:
                h_new = np.abs(result.x)
                actual_score = compute_autoconv_max(h_new)
                if actual_score < best_score:
                    best_score = actual_score
                    best_h = h_new.copy()
            
    except Exception:
        pass
    
    return list(best_h)
