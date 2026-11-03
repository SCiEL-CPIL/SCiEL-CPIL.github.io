import numpy as np
import time
from scipy.optimize import minimize
import scipy.fft

N = 400

# EVOLVE-BLOCK-START
def _fft_size(n):
    """Find the next power of 2 greater than or equal to n."""
    m = 1
    while m < n:
        m *= 2
    return m

def _autoconv(h):
    """Compute linear autoconvolution using FFT."""
    fsize = _fft_size(2 * N - 1)
    H = scipy.fft.rfft(h, fsize)
    Ac = H * np.conjugate(H)
    ac = scipy.fft.irfft(Ac, fsize)
    return ac[:2*N-1]

def _score(h):
    """Compute the exact objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    ac = _autoconv(h)
    max_ac = np.max(np.abs(ac))
    return max_ac / (s * s)

def _objective_smooth(h, beta):
    """
    Smooth objective function using soft-max approximation of max|autoconv|.
    We minimize max|ac| / (sum h)^2.
    Let S = sum h. We normalize h to have sum 1 inside the objective to handle scaling invariance.
    """
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    
    # Normalize to sum=1 to remove scaling invariance
    h_norm = h / s
    
    ac = _autoconv(h_norm)
    abs_ac = np.abs(ac)
    
    # Soft max approximation
    # To avoid overflow in exp, subtract max
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0
    
    # Clip to prevent numerical issues with very large beta
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -50, 50)
    
    log_sum = np.log(np.sum(np.exp(scaled)))
    soft_max = m + log_sum / beta
    
    return soft_max

def _local_search(h, deadline, max_iters=100):
    """
    Local search using sign flips and small perturbations to refine the exact score.
    """
    h = h.copy()
    best_score = _score(h)
    
    # Normalize
    s = np.sum(h)
    if abs(s) > 1e-12:
        h = h / s
        best_score = _score(h)
    else:
        return h, best_score
        
    for i in range(max_iters):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try flipping signs of a few random elements
        # Or try coordinate-wise adjustments
        # Given the structure, sign flips are a strong heuristic for signed step functions
        # But continuous values might be better. Let's try small random perturbations.
        
        # Strategy: Random perturbation with acceptance if score improves
        for _ in range(20):
            if time.time() > deadline:
                break
            
            h_pert = h.copy()
            # Add small noise
            noise = np.random.normal(0, 0.05, N)
            h_pert = h_pert + noise
            
            # Re-normalize
            s_p = np.sum(h_pert)
            if abs(s_p) < 1e-12:
                continue
            h_pert = h_pert / s_p
            
            val = _score(h_pert)
            if val < best_score:
                best_score = val
                h = h_pert
                improved = True
                
        if not improved:
            # Try sign flips on random subset
            for _ in range(20):
                if time.time() > deadline:
                    break
                h_pert = h.copy()
                mask = np.random.random(N) < 0.1
                h_pert[mask] *= -1
                
                s_p = np.sum(h_pert)
                if abs(s_p) < 1e-12:
                    continue
                h_pert = h_pert / s_p
                
                val = _score(h_pert)
                if val < best_score:
                    best_score = val
                    h = h_pert
                    improved = True
            
            if not improved:
                break
                
    return h, best_score

def construct():
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin
    
    best_score = np.inf
    best_h = np.ones(N) / N
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Random Gaussian
    candidates.append(rng.standard_normal(N))
    
    # 2. Random Signs
    candidates.append(rng.choice([1.0, -1.0], N))
    
    # 3. Low-pass filtered noise
    noise = rng.standard_normal(N)
    H = scipy.fft.rfft(noise)
    cutoff = 5
    H[cutoff:] = 0
    candidates.append(scipy.fft.irfft(H, N))
    
    # 4. Cosine wave
    freq = 2.0
    phase = 0.0
    candidates.append(np.cos(2*np.pi*freq*np.arange(N)/N + phase))
    
    # 5. Two-tone
    f1, f2 = 1.5, 4.5
    p1, p2 = 0.1, 0.5
    candidates.append(np.cos(2*np.pi*f1*np.arange(N)/N + p1) + np.cos(2*np.pi*f2*np.arange(N)/N + p2))
    
    # 6. Chirp
    t = np.arange(N)
    candidates.append(np.cos(2*np.pi*(0.5 + 0.5*t/N)*t/N))
    
    # 7. Block signs
    h_block = np.zeros(N)
    block_size = 20
    for i in range(0, N, block_size):
        sign = rng.choice([1.0, -1.0])
        h_block[i:i+block_size] = sign
    candidates.append(h_block)
    
    # 8. Sparse random
    h_sparse = np.zeros(N)
    mask = rng.random(N) < 0.1
    h_sparse[mask] = rng.choice([1.0, -1.0], mask.sum())
    candidates.append(h_sparse)

    # Optimization loop
    for h_init in candidates:
        if time.time() > deadline - 10:
            break
            
        h = h_init.copy()
        # Initial normalization
        s = np.sum(h)
        if abs(s) < 1e-12:
            continue
        h = h / s
        
        # Annealing schedule for beta
        # Start low, increase to sharp
        betas = [10, 20, 50, 100, 200, 500]
        
        for beta in betas:
            if time.time() > deadline - 5:
                break
                
            # Define objective for this beta
            def obj(x):
                return _objective_smooth(x, beta)
            
            try:
                res = minimize(obj, h, method='L-BFGS-B', 
                               options={'maxiter': 50, 'disp': False, 'gtol': 1e-6})
                
                if res.x is not None and np.all(np.isfinite(res.x)):
                    h = res.x
                    # Normalize
                    s = np.sum(h)
                    if abs(s) > 1e-12:
                        h = h / s
                    else:
                        break
            except Exception:
                break
        
        # Local search refinement
        h_opt, score_opt = _local_search(h, deadline - 2)
        
        if score_opt < best_score:
            best_score = score_opt
            best_h = h_opt.copy()
            
        # If we have a very good score, we might stop early?
        # The theoretical minimum is related to the seed. 1.0 is all-ones.
        # Signed can be lower. Let's keep searching if time permits.
        
    # Final check
    if not np.all(np.isfinite(best_h)):
        return np.ones(N)
        
    # Ensure sum is not zero
    s_final = np.sum(best_h)
    if abs(s_final) < 1e-12:
        return np.ones(N)
        
    return best_h
# EVOLVE-BLOCK-END
