"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _overlap_max(h):
    """Compute the maximum overlap of h with shifted 1-h over all circular shifts."""
    # h is an array of length N
    # overlap at shift k: sum_i h[i] * (1 - h[(i+k) % N])
    # = sum_i h[i] - sum_i h[i]*h[(i+k)%N]
    # = N/2 - sum_i h[i]*h[(i+k)%N]
    # So maximizing overlap is equivalent to minimizing the minimum of sum_i h[i]*h[(i+k)%N]
    # The overlap at shift k is N/2 - autocorr[k]
    # We want to minimize max_k (N/2 - autocorr[k]) = N/2 - min_k autocorr[k]
    
    # Compute circular autocorrelation
    # autocorr[k] = sum_i h[i] * h[(i+k) % N]
    # This is the circular cross-correlation of h with itself, which equals the circular autocorrelation
    
    H = np.fft.fft(h)
    power = np.abs(H)**2
    autocorr = np.fft.ifft(power).real
    
    # autocorr[0] should be sum(h^2), autocorr[k] for k>0 is the circular autocorrelation
    # The overlap at shift k is N/2 - autocorr[k]
    # We want to minimize max_k (N/2 - autocorr[k]) = N/2 - min_k autocorr[k]
    # So we want to maximize min_k autocorr[k]
    
    # Note: autocorr[0] = sum(h^2) which is typically larger than other values
    # The overlap includes shift 0? Let's re-read: "maximum overlap of h with shifted 1-h"
    # Shifted by k, for k = 0, 1, ..., N-1
    # At k=0: sum h[i]*(1-h[i]) = N/2 - sum h[i]^2
    # At k>0: sum h[i]*(1-h[(i+k)%N]) = N/2 - autocorr[k]
    
    # So max overlap = N/2 - min(autocorr[0], autocorr[1], ..., autocorr[N-1])
    # But autocorr[0] = sum h[i]^2 >= N/4 (by Cauchy-Schwarz, since sum h[i] = N/2)
    # And for k != 0, autocorr[k] can be negative
    
    # Actually wait, let me reconsider. The overlap function is:
    # overlap(k) = sum_i h[i] * (1 - h[(i+k) % N])
    # We want to minimize max_k overlap(k)
    
    # Let's just compute it directly
    overlaps = np.zeros(N)
    for k in range(N):
        shifted = np.roll(h, k)
        overlaps[k] = np.sum(h * (1 - shifted))
    
    return np.max(overlaps)


def _objective(params):
    """Objective function: maximize min autocorrelation (equivalently minimize max overlap)."""
    h = params
    # Ensure h is in [0, 1] and sums to N/2
    # We'll handle constraints separately
    return _overlap_max(h)


def _project(h):
    """Project h onto the feasible set: h[i] in [0,1], sum(h) = N/2."""
    # Simple projection: clip to [0,1], then adjust to meet sum constraint
    h = np.clip(h, 0, 1)
    current_sum = np.sum(h)
    target_sum = N / 2.0
    
    if current_sum == target_sum:
        return h
    
    # Distribute the difference
    diff = target_sum - current_sum
    if diff > 0:
        # Need to increase sum; increase elements that are < 1
        mask = h < 1.0
        if np.any(mask):
            n_mask = np.sum(mask)
            max_possible = n_mask * 1.0 - np.sum(h[mask])
            if diff <= max_possible:
                h[mask] += diff / n_mask
            else:
                h[mask] = 1.0
                remaining = diff - max_possible
                if remaining > 0:
                    # This shouldn't happen if we started with sum close to N/2
                    h += remaining / N
                    h = np.clip(h, 0, 1)
        else:
            h += diff / N
            h = np.clip(h, 0, 1)
    else:
        # Need to decrease sum; decrease elements that are > 0
        mask = h > 0.0
        if np.any(mask):
            n_mask = np.sum(mask)
            max_possible = np.sum(h[mask])
            if -diff <= max_possible:
                h[mask] += diff / n_mask
            else:
                h[mask] = 0.0
                remaining = diff + max_possible
                if remaining < 0:
                    h += remaining / N
                    h = np.clip(h, 0, 1)
        else:
            h += diff / N
            h = np.clip(h, 0, 1)
    
    return h


def construct():
    """Construct step heights to minimize maximum overlap."""
    start_time = time.time()
    deadline = 120.0  # seconds, with safety margin
    
    # Start with a good initial guess
    # The trivial solution is h = 0.5 everywhere, giving overlap = N/2 - N/4 = N/4 = 25
    # We want to do better.
    
    # Let's try a spectral approach: design h such that its autocorrelation is as flat as possible
    # (i.e., all nonzero lags have the same autocorrelation value)
    # This is like a Barker sequence or Zadoff-Chu sequence in the continuous domain.
    
    # Strategy: Use gradient descent on the max overlap objective
    # with projection onto the feasible set
    
    best_h = np.full(N, 0.5)
    best_score = _overlap_max(best_h)
    
    # Try multiple random initializations
    np.random.seed(42)
    
    # Method 1: Random initialization + projected gradient descent
    for init_iter in range(20):
        if time.time() - start_time > deadline - 10:
            break
        
        # Random initial h in [0,1] with sum N/2
        h = np.random.uniform(0, 1, N)
        h = _project(h)
        
        # Gradient descent with projection
        lr = 0.01
        for step in range(500):
            if time.time() - start_time > deadline - 5:
                break
            
            # Compute gradient of max overlap
            # overlap(k) = sum_i h[i] * (1 - h[(i+k)%N])
            # d(overlap(k))/d(h[j]) = (1 - h[(j+k)%N]) - h[(j-k)%N] for the term where i=j
            # Wait, let's be more careful.
            # overlap(k) = sum_i h[i] - sum_i h[i]*h[(i+k)%N]
            # d(overlap(k))/d(h[j]) = 1 - h[(j+k)%N] - h[(j-k)%N]
            # (the second term comes from when i=j in the first sum, and the third from when (i+k)%N=j i.e. i=(j-k)%N)
            
            # We want to minimize max_k overlap(k)
            # Use subgradient: if k* is the argmax, use gradient of overlap(k*)
            
            overlaps = np.zeros(N)
            for k in range(N):
                shifted = np.roll(h, k)
                overlaps[k] = np.sum(h * (1 - shifted))
            
            k_star = np.argmax(overlaps)
            
            # Gradient of overlap(k_star) w.r.t. h
            grad = np.zeros(N)
            for j in range(N):
                grad[j] = 1.0 - h[(j + k_star) % N] - h[(j - k_star) % N]
            
            # Projected gradient step
            h_new = h - lr * grad
            h_new = _project(h_new)
            
            new_score = _overlap_max(h_new)
            if new_score < best_score:
                best_score = new_score
                best_h = h_new.copy()
            
            h = h_new
        
        if time.time() - start_time > deadline - 5:
            break
    
    # Method 2: Try structured initializations
    # Sine-based: h[i] = 0.5 + A*sin(2*pi*i/N + phase)
    for A in [0.1, 0.2, 0.3, 0.4]:
        for phase in np.linspace(0, 2*np.pi, 8, endpoint=False):
            if time.time() - start_time > deadline - 10:
                break
            
            h = 0.5 + A * np.sin(2*np.pi*np.arange(N)/N + phase)
            h = np.clip(h, 0, 1)
            h = _project(h)
            
            # Local refinement with projected gradient descent
            lr = 0.005
            for step in range(200):
                if time.time() - start_time > deadline - 5:
                    break
                
                overlaps = np.zeros(N)
                for k in range(N):
                    shifted = np.roll(h, k)
                    overlaps[k] = np.sum(h * (1 - shifted))
                
                k_star = np.argmax(overlaps)
                
                grad = np.zeros(N)
                for j in range(N):
                    grad[j] = 1.0 - h[(j + k_star) % N] - h[(j - k_star) % N]
                
                h_new = h - lr * grad
                h_new = _project(h_new)
                
                new_score = _overlap_max(h_new)
                if new_score < best_score:
                    best_score = new_score
                    best_h = h_new.copy()
                
                h = h_new
            
            if time.time() - start_time > deadline - 10:
                break
    
    # Final polish: use scipy minimize with SLSQP
    if time.time() - start_time < deadline - 30:
        def obj(x):
            h = _project(x)
            return _overlap_max(h)
        
        def grad_obj(x):
            h = _project(x)
            overlaps = np.zeros(N)
            for k in range(N):
                shifted = np.roll(h, k)
                overlaps[k] = np.sum(h * (1 - shifted))
            k_star = np.argmax(overlaps)
            grad = np.zeros(N)
            for j in range(N):
                grad[j] = 1.0 - h[(j + k_star) % N] - h[(j - k_star) % N]
            return grad
        
        # Start from best_h
        res = minimize(obj, best_h, method='SLSQP', jac=grad_obj, 
                       options={'maxiter': 100, 'ftol': 1e-8})
        
        if res.success or (hasattr(res, 'fun') and res.fun < best_score):
            h_opt = _project(res.x)
            score_opt = _overlap_max(h_opt)
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt
    
    return best_h
# EVOLVE-BLOCK-END
