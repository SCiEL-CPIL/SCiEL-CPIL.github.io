"""Program for signed step functions minimizing max|autoconv|/(integral)^2."""
import time
import numpy as np
from scipy.fft import fft, ifft

N = 400
DEADLINE_MARGIN = 10.0  # seconds to spare before the 150s kill

def _autoconv(x):
    """
    Compute the autoconvolution of a length-N vector via FFT.
    The full linear convolution has length 2N-1.
    We compute it and return the max absolute value.
    """
    # Use a size that is at least 2N-1 and is a power of 2 for speed, or just 2*N
    # scipy.fft is fast. Let's use n=2*N for simplicity, it's enough.
    n = 2 * N
    f = fft(x, n=n)
    conv = ifft(f * f, n=n)
    # The first 2N-1 elements are the valid linear convolution
    # The last element is wrap-around (which is 0 for linear conv of finite length if n >= 2N-1)
    # Actually, ifft returns complex. We take real part.
    conv_real = np.real(conv[:2*N-1])
    return np.max(np.abs(conv_real))

def _score(x):
    """
    Compute the ratio max|autoconv| / (integral)^2.
    Integral of step function is sum(x) * (width/N) ? 
    The problem says "squared integral".
    Usually for discrete signals, "integral" is approximated by the sum.
    Let's assume integral = sum(x).
    Ratio = max|autoconv| / (sum(x))^2
    """
    s = np.sum(x)
    if s == 0:
        return float('inf')
    return _autoconv(x) / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN
    
    # 1. Initial candidate: constant vector
    x_best = np.ones(N)
    score_best = _score(x_best)
    
    # Check deadline
    if time.time() > deadline:
        return x_best
    
    # 2. Local Search Strategy
    # We want to minimize the ratio.
    # The constant vector is a strong baseline.
    # Let's try a random restart approach with local search.
    # However, the search space is huge.
    # A structural change: Use a different representation or optimization.
    
    # Idea: Minimize the variance of the autoconvolution relative to its mean?
    # Or just do a simple hill climbing on signs?
    # But signs are binary {-1, 1} or real?
    # "Signed step heights" implies real values are allowed, or just signs?
    # "Signed" usually implies signs, but "heights" implies magnitude.
    # The seed uses np.ones(N), which are all +1.
    # If we allow real values, we can scale. Scaling x by c scales autoconv by c^2 and sum by c.
    # Ratio: (c^2 * A) / (c * S)^2 = A / S^2.
    # So scaling doesn't change the ratio. We can fix the sum to be 1, or fix the l2 norm, etc.
    # Let's fix the sum to be N (so average is 1).
    
    # Let's try a local search that flips individual values between +1 and -1?
    # Or perturb values?
    
    # Given the time limit, a heavy search might not finish.
    # The constant vector is likely optimal or near-optimal.
    # The "structural change" directive suggests trying something else.
    # Let's try a few random perturbations and see if they beat the constant.
    
    rng = np.random.default_rng(42)
    
    # Try a few different initial structures
    candidates = [
        np.ones(N),
        np.ones(N) * 1.0, # Same
    ]
    
    # Maybe a pattern?
    # Alternating signs?
    x_alt = np.ones(N)
    x_alt[::2] = -1
    candidates.append(x_alt)
    
    # Random signs
    x_rand = rng.choice([-1, 1], size=N)
    candidates.append(x_rand)
    
    # Random real values with mean 1
    x_real = 1.0 + 0.1 * rng.standard_normal(N)
    # Renormalize sum to N?
    x_real = x_real * (N / np.sum(x_real))
    candidates.append(x_real)
    
    best_x = None
    best_score = float('inf')
    
    for x in candidates:
        s = _score(x)
        if s < best_score:
            best_score = s
            best_x = x
            
        if time.time() > deadline:
            return x_best
    
    # If any candidate beat the constant vector, perform a local search on it
    # Otherwise, return the constant vector (which is best_x if it was best)
    
    if best_score < score_best:
        # Local search on the best candidate
        x_current = best_x.copy()
        s_current = best_score
        no_improvement = 0
        
        # Simple coordinate descent or random perturbations
        # Since we want to be fast, let's do a limited number of steps
        max_steps = 1000
        
        for i in range(max_steps):
            if time.time() > deadline:
                return x_current
            
            # Pick a random index to perturb
            idx = rng.integers(0, N)
            old_val = x_current[idx]
            
            # Try a few new values for this index
            # If it was binary, try flipping. If real, try small step.
            # Let's try a small random perturbation
            x_trial = x_current.copy()
            x_trial[idx] = x_trial[idx] + 0.5 * rng.standard_normal()
            
            # Renormalize to keep sum constant?
            # Actually, the ratio is scale invariant. So we can fix sum=1 or sum=N.
            # Let's fix sum = N.
            if np.sum(x_trial) != 0:
                x_trial = x_trial * (N / np.sum(x_trial))
            
            s_trial = _score(x_trial)
            if s_trial < s_current:
                x_current = x_trial
                s_current = s_trial
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement > 100:
                    break
        
        if s_current < score_best:
            x_best = x_current
            score_best = s_current
            
    return x_best

# EVOLVE-BLOCK-START
# The construct function is defined above.
# The evaluator calls construct().
# EVOLVE-BLOCK-END
