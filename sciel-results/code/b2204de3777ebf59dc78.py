"""Combined program: Fourier-phase representation optimization + Fast discrete refinement."""
import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.signal import fftconvolve

N = 400

# EVOLVE-BLOCK-START
def _score_sequence(s):
    """True score: max|autoconv| / (sum s)^2."""
    sq_int = (np.sum(s)) ** 2
    if sq_int < 1e-10:
        return np.inf
    ac = fftconvolve(s, s, mode='full')
    max_ac = np.max(np.abs(ac))
    return max_ac / sq_int

def _soft_papr_objective(phases, K, beta):
    """
    Objective: Minimize soft max|autoconv| / (sum h)^2.
    h is constructed from phases of a Fourier sum.
    """
    n = N
    # Construct h from phases
    # h[n] = sum_{k=0}^{K-1} cos(2*pi*k*n/N + phases[k])
    # Using matrix multiplication for speed
    t = np.arange(n)
    k = np.arange(K)
    # Matrix M[n, k] = 2*pi*k*n/N
    M = 2.0 * np.pi * np.outer(k, t) / n
    # h[n] = sum_k cos(M[n,k] + phases[k])
    # cos(a+b) = cos(a)cos(b) - sin(a)sin(b)
    cos_M = np.cos(M)
    sin_M = np.sin(M)
    cos_p = np.cos(phases)
    sin_p = np.sin(phases)
    
    h = cos_M @ cos_p - sin_M @ sin_p
    
    # Calculate autoconvolution
    # Use FFT for speed in the objective function
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    ac_valid = ac[:size]
    
    # Soft max of |ac|
    # We use log-sum-exp on ac^2 to approximate max|ac|^2, then sqrt
    ac_sq = ac_valid * ac_valid
    m = np.max(ac_sq)
    if m < 1e-10:
        soft_max_sq = 0.0
    else:
        # Top-K approximation for stability and speed? 
        # For N=400, size=799. Full sum is fine.
        lse = m + np.log(np.sum(np.exp((ac_sq - m) * beta)))
        soft_max_sq = lse / beta
    
    soft_max_abs = np.sqrt(soft_max_sq)
    
    s = np.sum(h)
    if abs(s) < 1e-6:
        return 1e10
        
    return soft_max_abs / (s * s)

def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin
    
    def _now():
        return time.time()
        
    rng = np.random.RandomState(42)
    
    best_score = np.inf
    best_h = None
    
    # Phase 1: Optimize Fourier Phases
    # We represent h as a sum of K complex exponentials.
    # This structure tends to produce signals with good autocorrelation properties (like Zadoff-Chu).
    
    K_max = 20  # Number of Fourier components to optimize
    beta = 20.0 # Sharpness of soft-max
    
    # Precompute M matrix for speed in objective
    t = np.arange(N)
    k = np.arange(K_max)
    M = 2.0 * np.pi * np.outer(k, t) / N
    cos_M = np.cos(M)
    sin_M = np.sin(M)
    
    def fast_objective(phases):
        cos_p = np.cos(phases)
        sin_p = np.sin(phases)
        h = cos_M @ cos_p - sin_M @ sin_p
        
        size = 2 * N - 1
        fsize = 1
        while fsize < size:
            fsize *= 2
        H = np.fft.rfft(h, fsize)
        ac = np.fft.irfft(H * H, fsize)
        ac_valid = ac[:size]
        
        ac_sq = ac_valid * ac_valid
        m = np.max(ac_sq)
        if m < 1e-10:
            soft_max_sq = 0.0
        else:
            lse = m + np.log(np.sum(np.exp((ac_sq - m) * beta)))
            soft_max_sq = lse / beta
        
        soft_max_abs = np.sqrt(soft_max_sq)
        
        s = np.sum(h)
        if abs(s) < 1e-6:
            return 1e10
            
        return soft_max_abs / (s * s)

    # Try multiple initializations of phases
    n_restarts = 10
    for r in range(n_restarts):
        if _now() > deadline - 30:
            break
            
        # Initialize phases randomly or with specific patterns
        if r == 0:
            phases0 = np.zeros(K_max)
        elif r == 1:
            phases0 = np.linspace(0, 2*np.pi, K_max, endpoint=False)
        else:
            phases0 = rng.rand(K_max) * 2 * np.pi
            
        try:
            res = minimize(
                fast_objective,
                phases0,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            # Reconstruct h from optimized phases
            phases_opt = res.x
            cos_p = np.cos(phases_opt)
            sin_p = np.sin(phases_opt)
            h_cont = cos_M @ cos_p - sin_M @ sin_p
            
            # Project to binary signs
            h_bin = np.sign(h_cont)
            h_bin[h_bin == 0] = 1.0
            
            # Ensure non-zero sum
            if np.sum(h_bin) == 0:
                h_bin[0] = 1.0
                if np.sum(h_bin) == 0: h_bin[1] = 1.0
                
            sc = _score_sequence(h_bin)
            if sc < best_score:
                best_score = sc
                best_h = h_bin.copy()
                
        except Exception:
            pass

    # If no good result from Fourier optimization, fallback to random/binary
    if best_h is None:
        best_h = np.ones(N)
        best_score = _score_sequence(best_h)
        
    # Phase 2: Discrete Local Search and Simulated Annealing
    # Use the fast O(N) flip update logic from the inspiration
    
    cur = best_h.copy()
    cur_ac = fftconvolve(cur, cur, mode='full')
    cur_sum = float(np.sum(cur))
    cur_score = best_score
    
    def _max_ac_after_flip(cur, cur_ac, cur_sum, j):
        """Vectorized single-flip score update."""
        val = cur[j]
        delta = -2.0 * val
        new_sum = cur_sum + delta
        nsq = new_sum ** 2
        if nsq < 1e-10:
            return np.inf
        new_ac = cur_ac.copy()
        new_ac[j:j+N] += 2.0 * delta * cur
        new_ac[j] += delta * delta
        return float(np.max(np.abs(new_ac)) / nsq)

    # Greedy Local Search
    # Iterate until no improvement or time limit
    max_passes = 500
    for _ in range(max_passes):
        if _now() > deadline - 20:
            break
            
        best_j = -1
        best_js = cur_score
        best_new_sum = cur_sum
        
        # Evaluate all possible single flips
        for j in range(N):
            ns = _max_ac_after_flip(cur, cur_ac, cur_sum, j)
            if ns < best_js:
                best_js = ns
                best_j = j
                best_new_sum = cur_sum - 2.0 * cur[j]
                
        if best_j == -1 or best_js >= cur_score - 1e-12:
            break
            
        # Apply the best flip
        val = cur[best_j]
        delta = -2.0 * val
        cur_ac = cur_ac.copy()
        cur_ac[best_j:best_j+N] += 2.0 * delta * cur
        cur_ac[best_j] += delta * delta
        cur[best_j] = -val
        cur_sum = best_new_sum
        cur_score = best_js
        
        if cur_score < best_score:
            best_score = cur_score
            best_h = cur.copy()

    # Simulated Annealing
    # Use remaining time for SA to escape local minima
    if _now() < deadline - 5:
        T_start = 0.05
        T_min = 1e-6
        sa_steps = 10000
        # Adjust steps based on remaining time
        time_left = deadline - 5 - _now()
        # Estimate time per step (approx 1ms for N=400 with vectorized update?)
        # Let's just run until deadline
        for it in range(sa_steps):
            if _now() > deadline - 2:
                break
                
            idx = rng.randint(N)
            new_score = _max_ac_after_flip(cur, cur_ac, cur_sum, idx)
            if new_score == np.inf:
                continue
                
            d = new_score - cur_score
            T = max(T_start * (0.995 ** it), T_min)
            
            if d < 0 or rng.rand() < np.exp(-d / T):
                val = cur[idx]
                delta = -2.0 * val
                cur_ac = cur_ac.copy()
                cur_ac[idx:idx+N] += 2.0 * delta * cur
                cur_ac[idx] += delta * delta
                cur[idx] = -val
                cur_sum += delta
                cur_score = new_score
                
                if cur_score < best_score:
                    best_score = cur_score
                    best_h = cur.copy()

    return best_h

# EVOLVE-BLOCK-END
