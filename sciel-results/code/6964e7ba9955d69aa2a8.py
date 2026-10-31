"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt
import scipy.signal as sig

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-6:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # The problem is scale invariant.
    # We can fix the sum to 1 to simplify optimization.
    # Minimize max|autoconv| subject to sum(h) = 1.
    # Then the score is just max|autoconv|.
    
    # Initial guess: All ones scaled to sum 1.
    # h_i = 1/N.
    # max|autoconv| = 1/N.
    # Score = (1/N) / 1^2 = 1/N.
    
    # Let's try to find a shape that minimizes max|autoconv| given sum=1.
    
    def objective(x):
        # x is the vector h
        # Calculate autoconv
        n = len(x)
        size = 2 * n - 1
        fsize = 1
        while fsize < size:
            fsize *= 2
        H = np.fft.rfft(x, fsize)
        ac = np.fft.irfft(H * H, fsize)
        # Soft max to make it differentiable
        # Use a small epsilon
        eps = 0.01
        # We want max |ac|
        # log-sum-exp of |ac|/eps * eps
        # But we need to handle the constraint sum(x)=1.
        # We will use penalty method or constrained optimization.
        
        # Actually, let's just use the raw score with a constraint penalty.
        s = np.sum(x)
        if abs(s) < 1e-6:
            return 1e10
        val = _autoconv_max_abs(x) / (s * s)
        return val

    # We can use scipy.optimize.minimize with bounds?
    # Signs can be negative.
    # Let's try a few initializations and polish with local search on values.
    
    best_h = np.ones(N) / N
    best_val = _score(best_h)
    
    # Initialization strategies
    inits = []
    
    # 1. All ones
    inits.append(np.ones(N))
    
    # 2. Cosine window
    x = np.arange(N)
    inits.append(np.cos(np.pi * x / N))
    inits.append(np.cos(np.pi * x / (2*N))**2)
    
    # 3. Gaussian
    mu = N/2
    sigma = N/4
    inits.append(np.exp(-(x - mu)**2 / (2*sigma**2)))
    
    # 4. Triangle
    inits.append(np.maximum(0, 1 - np.abs(x - mu)/mu))
    
    # 5. Random smooth
    for i in range(5):
        np.random.seed(i)
        # Low pass filter random noise
        noise = np.random.randn(N)
        # Simple moving average
        kernel = np.ones(10)/10
        smooth = np.convolve(noise, kernel, mode='same')
        inits.append(smooth)
        
    for h_init in inits:
        if time.time() > deadline - 10:
            break
            
        # Normalize to sum 1 for optimization stability?
        # Or just optimize the ratio directly.
        # The ratio is homogeneous of degree 0.
        # Gradient descent on the ratio might be unstable due to scale.
        # Let's fix sum(h) = 1.
        
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        h_norm = h_init / s0
        
        # Define objective for constrained problem: sum(h)=1
        # We use a penalty for sum(h) != 1
        def obj_constrained(x):
            s = np.sum(x)
            penalty = 100.0 * (s - 1.0)**2
            return _autoconv_max_abs(x) + penalty
        
        # Use BFGS
        try:
            res = opt.minimize(obj_constrained, h_norm, method='BFGS', 
                               options={'maxiter': 100, 'disp': False, 'gtol': 1e-6})
            h_opt = res.x
            # Check if sum is close to 1
            if abs(np.sum(h_opt) - 1.0) < 0.1:
                val = _autoconv_max_abs(h_opt)
                # The score for this h_opt (with sum 1) is just val.
                # But we need to compare with best_val which is calculated on raw h.
                # Since score is scale invariant, score(h_opt) = score(h_opt * c).
                # If sum(h_opt) = 1, score = max|A| / 1^2 = max|A|.
                if val < best_val:
                    best_val = val
                    best_h = h_opt
        except Exception:
            pass

    # Final polish: Local sign/magnitude flips?
    # Since we allowed real values, we can do coordinate descent.
    # But BFGS should have done a decent job.
    
    # One more check: All ones is a strong baseline.
    # Ensure we don't return something worse.
    base_val = _score(np.ones(N))
    if best_val > base_val:
        best_val = base_val
        best_h = np.ones(N)
        
    return best_h

# EVOLVE-BLOCK-END
