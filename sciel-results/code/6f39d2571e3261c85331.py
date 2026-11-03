"""Third autocorrelation inequality: signed step function on 400 intervals in [-1/4,1/4]."""
N = 400
import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
import numpy as np
import time

# The intervals are on [-1/4, 1/4] with 400 equal intervals.
# Each interval has width 0.5/400 = 1/800.
# The step function is constant on each interval.

def _score_step(signed):
    """
    Compute the upper bound on C3 for a signed step function.
    The autoconvolution R[k] = sum_i signed[i]*signed[i+k] (discrete).
    The upper bound on C3 relates to the ratio of peak to sidelobe energy,
    or specifically the ratio of the L1 norm of the sidelobes to the peak.
    
    Actually, for the "third autocorrelation inequality", we want to minimize
    the ratio of the sum of absolute sidelobe autocorrelation values to the 
    peak autocorrelation value. This gives an upper bound on C3.
    
    Let's define:
    - autocorr R[m] = sum_i signed[i] * signed[i+m] for m in 0..N-1 (circular)
    - peak = |R[0]| = sum_i signed[i]^2
    - sidelobe = max_{m != 0} |R[m]|
    
    The score is the upper bound on C3, which is proportional to sidelobe/peak.
    Lower is better.
    """
    # Use circular autocorrelation via FFT for efficiency
    # But for step functions with signed values, let's be careful.
    # The discrete autocorrelation for a periodic function:
    R = np.fft.ifft(np.abs(np.fft.fft(signed))**2).real
    peak = abs(R[0])
    if peak < 1e-12:
        return 1e12
    # Sidelobes: all m != 0
    sidelobe = np.max(np.abs(np.delete(R, 0)))
    return sidelobe / peak

def construct():
    deadline = time.time() + 110  # Safety margin before 120s kill
    
    # Start with a reasonable initial guess: alternating signs
    # This tends to reduce autocorrelation sidelobes
    x0 = np.ones(N)
    x0[1::2] = -1.0
    
    best_score = _score_step(x0)
    best_x = x0.copy()
    
    # Try several optimization strategies
    strategies = [
        "alt",      # alternating
        "random",   # random signed
        "grad",     # gradient-based (soft sign)
        "pattern",  # structured patterns
    ]
    
    # Strategy 1: Try different simple patterns
    patterns = [
        np.ones(N),
        np.where(np.arange(N) % 2 == 0, 1.0, -1.0),
        np.where(np.arange(N) % 3 == 0, 1.0, -1.0),
        np.where((np.arange(N) % 4 < 2), 1.0, -1.0),
        np.sin(np.arange(N) * np.pi / 10.0).sign() if np.sin(np.arange(N) * np.pi / 10.0).any() else np.ones(N),
    ]
    for p in patterns:
        p = np.sign(p)
        if np.all(p == 0):
            p = np.ones(N)
        s = _score_step(p)
        if s < best_score:
            best_score = s
            best_x = p.copy()
        if time.time() > deadline:
            break
    
    # Strategy 2: Random restarts with local search
    rng = np.random.RandomState(42)
    for restart in range(50):
        if time.time() > deadline:
            break
        
        x = np.where(rng.rand(N) < 0.5, 1.0, -1.0)
        s = _score_step(x)
        if s < best_score:
            best_score = s
            best_x = x.copy()
        
        # Local search: flip individual bits
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(N):
                if time.time() > deadline:
                    break
                x[i] = -x[i]
                s_new = _score_step(x)
                if s_new < best_score:
                    best_score = s_new
                    best_x = x.copy()
                    improved = True
                else:
                    x[i] = -x[i]  # revert
    
    # Strategy 3: Gradient-based optimization with soft signs
    # Use tanh to get smooth signed values, then threshold
    if time.time() < deadline - 20:
        def soft_score(z):
            """Score using soft signed values (tanh parameterization)."""
            signed = np.tanh(z)
            # Normalize to unit L2 norm
            norm = np.linalg.norm(signed)
            if norm < 1e-12:
                return 1e12
            signed = signed / norm
            R = np.fft.ifft(np.abs(np.fft.fft(signed))**2).real
            peak = abs(R[0])
            if peak < 1e-12:
                return 1e12
            sidelobe = np.max(np.abs(np.delete(R, 0)))
            return sidelobe / peak
        
        # Initial z
        z0 = np.where(best_x > 0, 1.0, -1.0) * 2.0
        
        try:
            res = minimize(
                soft_score,
                z0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if time.time() < deadline:
                z_opt = res.x
                signed_opt = np.tanh(z_opt)
                norm = np.linalg.norm(signed_opt)
                if norm > 1e-12:
                    signed_opt = signed_opt / norm
                # Threshold to hard signed
                hard_opt = np.sign(signed_opt)
                if np.all(hard_opt == 0):
                    hard_opt = np.sign(signed_opt + 1e-6)
                s_hard = _score_step(hard_opt)
                if s_hard < best_score:
                    best_score = s_hard
                    best_x = hard_opt.copy()
        except Exception:
            pass
    
    # Final: return the best hard signed step function
    # Ensure values are strictly ±1
    result = np.sign(best_x)
    if np.all(result == 0):
        result = np.ones(N)
    
    return result
# EVOLVE-BLOCK-END
