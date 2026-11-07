"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _correlate_values(h):
    """Compute all cross-correlation values between h and 1-h (full mode)."""
    return np.correlate(h, 1.0 - h, mode="full")

def _objective(h, beta):
    """Smooth surrogate for max(correlate(h, 1-h, 'full')) using log-sum-exp."""
    vals = _correlate_values(h)
    m = np.max(vals)
    if m < -1e10:
        return m
    # log-sum-exp for numerical stability
    shifted = vals - m
    s = np.sum(np.exp(beta * shifted))
    return m + np.log(s) / beta

def _objective_grad(h, beta):
    """Gradient of the smooth surrogate objective."""
    vals = _correlate_values(h)
    m = np.max(vals)
    shifted = vals - m
    weights = np.exp(beta * shifted)
    weights /= np.sum(weights)
    
    # Gradient of correlate(h, 1-h, 'full') w.r.t. h
    # correlate(h, g, 'full')[k] = sum_i h[i] * g[i + k - N + 1] for appropriate range
    # For g = 1 - h, we need d/dh of correlate(h, 1-h)
    # Let's compute it numerically via the structure:
    # correlate(h, 1-h, 'full')[j] for j in 0..2N-2
    # = sum_i h[i] * (1 - h[i + j - (N-1)]) where indices are valid
    # Actually, let's just use the fact that:
    # C[j] = sum_i h[i] * (1-h[i+j-N+1])
    # dC[j]/dh[k] = (1-h[k+j-N+1]) + h[k] * (-1) if k-N+1 <= k+j-N+1 < k+N
    # This is getting complex. Let me use a different approach.
    
    # Use the linearity: correlate(h, 1-h) = correlate(h, ones) - correlate(h, h)
    # correlate(h, ones, 'full') = cumulative sums
    # correlate(h, h, 'full') = autocorrelation
    
    # Let me just compute gradient numerically with finite differences for reliability
    # But that's slow. Let me derive it properly.
    
    # np.correlate(h, 1-h, mode='full')[j] = sum_{i} h[i] * (1-h[i + j - (N-1)])
    # where i ranges such that both indices are in [0, N-1]
    
    # Let k = j - (N-1), so j = k + N - 1, k ranges from -(N-1) to N-1
    # C[k] = sum_i h[i] * (1 - h[i+k]) for i such that 0<=i<N and 0<=i+k<N
    # = sum_i h[i] - sum_i h[i]*h[i+k]
    
    # dC[k]/dh[m] = 1 - (h[m+k] if 0<=m+k<N else 0) - (h[m-k] if 0<=m-k<N else 0)
    # Wait, that's not right. Let me be more careful.
    
    # C[k] = sum_{i: 0<=i<N, 0<=i+k<N} h[i] * (1 - h[i+k])
    # = sum_{i: 0<=i<N, 0<=i+k<N} h[i] - sum_{i: 0<=i<N, 0<=i+k<N} h[i]*h[i+k]
    
    # dC[k]/dh[m] = [1 if 0<=m<N and 0<=m+k<N else 0] - [h[m+k] if 0<=m+k<N else 0] - [h[m-k] if 0<=m-k<N else 0]
    # Wait, the second term: sum_i h[i]*h[i+k], derivative w.r.t. h[m]:
    # = h[m+k] if m is the first index (i=m, need 0<=m+k<N) 
    #   + h[m-k] if m is the second index (i+k=m => i=m-k, need 0<=m-k<N)
    
    # So dC[k]/dh[m] = I[0<=m+k<N] - I[0<=m+k<N]*h[m+k] - I[0<=m-k<N]*h[m-k]
    # where I is indicator. But also need 0<=m<N which is always true since m is index of h.
    # And for the first sum, need 0<=m<N and 0<=m+k<N.
    
    # Let me just compute the gradient matrix G where G[k,m] = dC[k]/dh[m]
    # k ranges from -(N-1) to N-1, mapped to j = k+N-1 in 0..2N-2
    
    grad = np.zeros_like(h)
    for m in range(N):
        for k in range(-(N-1), N):
            # C[k] = sum_i h[i]*(1-h[i+k])
            # dC[k]/dh[m] contribution:
            # From first sum: 1 if 0<=m<N and 0<=m+k<N
            # From second sum: -(h[m+k] if 0<=m+k<N else 0) - (h[m-k] if 0<=m-k<N else 0)
            idx = k + N - 1  # index in the full correlation array
            if idx < 0 or idx >= 2*N - 1:
                continue
            term1 = 1.0 if (0 <= m < N and 0 <= m + k < N) else 0.0
            term2 = h[m + k] if (0 <= m + k < N) else 0.0
            term3 = h[m - k] if (0 <= m - k < N) else 0.0
            dC = term1 - term2 - term3
            grad[m] += weights[idx] * dC
    
    return grad

def _project(h):
    """Project onto {h: 0<=h[i]<=1, sum(h)=N/2}."""
    h = np.clip(h, 0, 1)
    # Adjust to satisfy sum constraint
    diff = N / 2.0 - np.sum(h)
    if abs(diff) > 1e-12:
        # Distribute the difference, respecting bounds
        # Simple approach: add diff/N to all, then clip and iterate
        for _ in range(100):
            h = h + diff / N
            h = np.clip(h, 0, 1)
            new_diff = N / 2.0 - np.sum(h)
            if abs(new_diff) < 1e-12:
                break
            diff = new_diff
    return h

def _true_objective(h):
    """The actual objective value (lower is better)."""
    vals = _correlate_values(h)
    return np.max(vals) / N * 2

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    best_h = np.full(N, 0.5)
    best_val = _true_objective(best_h)
    
    # Try multiple initializations and solve
    inits = [
        np.full(N, 0.5),
        np.linspace(0.1, 0.9, N),
        np.sin(np.linspace(0, 4*np.pi, N)) * 0.5 + 0.5,
        np.random.default_rng(42).uniform(0, 1, N),
        np.tile([0.2, 0.8], N // 2),
        np.tile([0.1, 0.9, 0.3, 0.7], N // 4),
    ]
    
    # Normalize initializations to sum to N/2
    for i in range(len(inits)):
        h_init = inits[i].copy()
        h_init = np.clip(h_init, 0, 1)
        diff = N / 2.0 - np.sum(h_init)
        if abs(diff) > 1e-10:
            h_init = h_init + diff / N
            h_init = np.clip(h_init, 0, 1)
        
        if time.time() > deadline:
            break
        
        # Use projected gradient descent with log-sum-exp smoothing
        h = h_init.copy()
        beta = 10.0  # Smoothing parameter
        
        for step in range(5000):
            if time.time() > deadline:
                break
            
            # Compute objective and gradient
            obj = _objective(h, beta)
            grad = _objective_grad(h, beta)
            
            # Adaptive step size
            step_size = 0.01 / (1.0 + step * 0.001)
            
            h_new = h - step_size * grad
            h_new = _project(h_new)
            
            # Check improvement
            new_obj = _objective(h_new, beta)
            if new_obj < obj:
                h = h_new
            else:
                step_size *= 0.5
            
            # Increase beta over time (less smoothing)
            beta *= 1.01
            if beta > 1000:
                beta = 1000
        
        # Evaluate true objective
        val = _true_objective(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
        
        # Also try with the current h as new init for further refinement
        inits.append(h.copy())
    
    # Final polish with high beta
    if time.time() < deadline - 10:
        h = best_h.copy()
        beta = 1000.0
        for step in range(2000):
            if time.time() > deadline:
                break
            obj = _objective(h, beta)
            grad = _objective_grad(h, beta)
            step_size = 0.001 / (1.0 + step * 0.001)
            h_new = h - step_size * grad
            h_new = _project(h_new)
            new_obj = _objective(h_new, beta)
            if new_obj < obj:
                h = h_new
            else:
                step_size *= 0.5
        
        val = _true_objective(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    return best_h
# EVOLVE-BLOCK-END
