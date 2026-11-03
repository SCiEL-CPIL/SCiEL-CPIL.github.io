import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 100
STEP_WIDTH = 2.0 / N

def compute_overlaps_fast(h):
    """Compute overlap for all N discrete shifts using FFT for speed."""
    # overlap[k] = STEP_WIDTH * sum_i h[i] * (1 - h[(i+k)%N])
    #           = STEP_WIDTH * (sum_i h[i] - sum_i h[i]*h[(i+k)%N])
    #           = STEP_WIDTH * (N/2 - corr[k])  where corr is circular autocorrelation
    # Circular autocorrelation via FFT
    H = np.fft.fft(h)
    corr = np.fft.ifft(H * np.conj(H)).real
    overlaps = STEP_WIDTH * (N / 2.0 - corr)
    return overlaps

def logsumexp(x, beta=1.0):
    """Numerically stable log-sum-exp for smooth max approximation."""
    x_max = np.max(x)
    return x_max + np.log(np.sum(np.exp(x - x_max))) / beta

def h_from_binary(b, sigma=0.3):
    """Generate smooth h from binary signal via Gaussian smoothing, then project to sum=N/2."""
    # Smooth the binary signal
    kernel_size = max(1, int(sigma * N))
    if kernel_size % 2 == 0:
        kernel_size += 1
    x = np.arange(-kernel_size//2, kernel_size//2 + 1)
    kernel = np.exp(-x**2 / (2 * sigma**2 * (N/10.0)**2))
    kernel /= np.sum(kernel)
    
    h_smooth = np.convolve(b, kernel, mode='same')
    h_smooth = np.clip(h_smooth, 0, 1)
    
    # Project to sum = N/2
    target_sum = N / 2.0
    current_sum = np.sum(h_smooth)
    if current_sum > 1e-10:
        h_smooth = h_smooth * (target_sum / current_sum)
    h_smooth = np.clip(h_smooth, 0, 1)
    
    # Fixed-point projection
    for _ in range(50):
        current_sum = np.sum(h_smooth)
        diff = target_sum - current_sum
        if abs(diff) < 1e-10:
            break
        h_smooth = h_smooth + diff / N
        h_smooth = np.clip(h_smooth, 0, 1)
    
    return h_smooth

def objective_binary(b, beta=50.0, sigma=0.3):
    """Objective: smoothed max overlap, with b in [0,1] mapped to binary-like h."""
    # Map b to [0,1] and use as binary signal
    b_clipped = np.clip(b, 0, 1)
    h = h_from_binary(b_clipped, sigma)
    
    overlaps = compute_overlaps_fast(h)
    # Smooth max via log-sum-exp
    return logsumexp(overlaps, beta)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    best_h = np.full(N, 0.5)
    best_score = np.max(compute_overlaps_fast(best_h))
    
    np.random.seed(42)
    
    # Phase 1: Differential evolution on binary-like parameters
    # Parameters represent the binary switching pattern
    # Use a lower-dimensional representation: optimize transition positions
    
    # Try several sigma values and betas
    for sigma in [0.2, 0.3, 0.5, 0.8]:
        if time.time() > deadline:
            break
        
        for beta in [30.0, 50.0, 100.0]:
            if time.time() > deadline:
                break
            
            def obj_de(b):
                if time.time() > deadline:
                    return np.inf
                return objective_binary(b, beta=beta, sigma=sigma)
            
            try:
                bounds = [(0, 1)] * N
                result = differential_evolution(
                    obj_de,
                    bounds,
                    seed=42,
                    maxiter=100,
                    tol=1e-6,
                    popsize=15,
                    polish=True,
                    updating='immediate'
                )
                
                if time.time() > deadline:
                    break
                
                b_opt = np.clip(result.x, 0, 1)
                h_candidate = h_from_binary(b_opt, sigma)
                score = np.max(compute_overlaps_fast(h_candidate))
                
                if score < best_score:
                    best_score = score
                    best_h = h_candidate.copy()
            except Exception:
                continue
    
    # Phase 2: Local refinement with L-BFGS-B on smoothed parameters
    if time.time() < deadline - 20:
        # Start from best found
        # Invert: try to find b that produces best_h
        # Use the smoothed version as initial guess
        b_init = np.clip(best_h, 0, 1)
        
        for sigma in [0.2, 0.3, 0.5]:
            if time.time() > deadline - 10:
                break
            
            for beta in [50.0, 100.0]:
                if time.time() > deadline - 5:
                    break
                
                def obj_lbgfs(b):
                    if time.time() > deadline - 1:
                        return np.inf
                    return objective_binary(b, beta=beta, sigma=sigma)
                
                try:
                    bounds = [(0, 1)] * N
                    res = minimize(
                        obj_lbgfs,
                        b_init,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 200, 'ftol': 1e-10}
                    )
                    
                    if time.time() > deadline:
                        break
                    
                    b_opt = np.clip(res.x, 0, 1)
                    h_candidate = h_from_binary(b_opt, sigma)
                    score = np.max(compute_overlaps_fast(h_candidate))
                    
                    if score < best_score:
                        best_score = score
                        best_h = h_candidate.copy()
                except Exception:
                    continue
    
    # Phase 3: Try structured patterns (alternating blocks)
    if time.time() < deadline - 10:
        # Pattern: blocks of 0s and 1s with varying block lengths
        for block_frac in [0.25, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.75]:
            if time.time() > deadline:
                break
            
            # Create pattern with period based on block_frac
            # For a pattern with fraction f of 1s, use period = round(1/f) or similar
            # Try multiple periods
            for period in range(2, min(N, 20)):
                if time.time() > deadline:
                    break
                
                # Create binary pattern
                b_pattern = np.zeros(N)
                ones_per_period = max(1, round(period * block_frac))
                for i in range(N):
                    if (i % period) < ones_per_period:
                        b_pattern[i] = 1.0
                
                h_candidate = h_from_binary(b_pattern, sigma=0.3)
                score = np.max(compute_overlaps_fast(h_candidate))
                
                if score < best_score:
                    best_score = score
                    best_h = h_candidate.copy()
    
    # Phase 4: Final local refinement
    if time.time() < deadline - 5:
        # Direct optimization on h with constraints using projected gradient
        h = best_h.copy()
        
        for step in range(500):
            if time.time() > deadline - 2:
                break
            
            # Compute gradient of smooth max
            overlaps = compute_overlaps_fast(h)
            max_idx = np.argmax(overlaps)
            beta = 100.0
            weights = np.exp(overlaps - overlaps[max_idx]) / beta
            weights /= np.sum(weights)
            
            # Gradient of sum_i w_i * overlap_i w.r.t. h
            # overlap_k = STEP_WIDTH * (N/2 - corr_k)
            # corr_k = sum_i h[i] * h[(i+k)%N]
            # d(overlap_k)/dh[j] = -STEP_WIDTH * (h[(j-k)%N] + h[(j+k)%N]) / 2 (symmetric)
            # Actually: d(corr_k)/dh[j] = h[(j-k)%N] + (if k != 0, h[(j+k)%N]) but for circular...
            # corr_k = sum_i h[i]*h[(i+k)%N]
            # d(corr_k)/dh[j] = h[(j-k)%N] + h[(j+k)%N] (for k != 0, both terms)
            # For k=0: d(corr_0)/dh[j] = 2*h[j]
            
            grad = np.zeros(N)
            for k in range(N):
                if weights[k] > 1e-15:
                    if k == 0:
                        grad += -weights[k] * STEP_WIDTH * 2 * h
                    else:
                        shifted_neg = np.roll(h, k)  # h[(i-k)%N] for i -> j: h[(j-k)%N]
                        shifted_pos = np.roll(h, -k)  # h[(i+k)%N] for i -> j: h[(j+k)%N]
                        grad += -weights[k] * STEP_WIDTH * (shifted_neg + shifted_pos)
            
            # Project gradient step
            lr = 0.01
            h_new = h - lr * grad
            h_new = np.clip(h_new, 0, 1)
            
            # Project to sum constraint
            target_sum = N / 2.0
            for _ in range(20):
                current_sum = np.sum(h_new)
                diff = target_sum - current_sum
                if abs(diff) < 1e-12:
                    break
                h_new = h_new + diff / N
                h_new = np.clip(h_new, 0, 1)
            
            h = h_new
        
        score = np.max(compute_overlaps_fast(h))
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    return best_h
