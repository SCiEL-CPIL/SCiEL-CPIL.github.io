"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import linear_sum_assignment
import time

def _overlap_value(h, shifts):
    """Compute the maximum overlap of h with shifted 1-h at given shifts."""
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    max_ov = 0.0
    for s in shifts:
        # shifted h: h_shifted[i] = h[(i+s) % n]
        h_shifted = np.roll(h, -s)
        # overlap = sum(h[i] * (1 - h_shifted[i]))
        ov = np.sum(h * (1.0 - h_shifted))
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _smooth_overlap(h, shifts, beta=10.0):
    """Smooth surrogate for max overlap using log-sum-exp."""
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    vals = []
    for s in shifts:
        h_shifted = np.roll(h, -s)
        ov = np.sum(h * (1.0 - h_shifted))
        vals.append(ov)
    vals = np.array(vals)
    # soft-max via log-sum-exp
    m = np.max(vals)
    if m > 1e-10:
        return m + np.log(np.sum(np.exp(beta * (vals - m)))) / beta
    else:
        return m

def _project_to_simplex(h, target_sum):
    """Project h onto {x: 0 <= x_i <= 1, sum(x) = target_sum} using water-filling."""
    h = np.clip(h, 0.0, 1.0)
    s = np.sum(h)
    if abs(s - target_sum) < 1e-12:
        return h
    # Adjust: add uniform offset, then clip
    diff = target_sum - s
    n = len(h)
    # Try adding diff/n uniformly
    h2 = h + diff / n
    h2 = np.clip(h2, 0.0, 1.0)
    # Iterate a few times for better projection
    for _ in range(50):
        s2 = np.sum(h2)
        diff2 = target_sum - s2
        if abs(diff2) < 1e-12:
            break
        h2 = h2 + diff2 / n
        h2 = np.clip(h2, 0.0, 1.0)
    return h2

def _construct_candidate(seed, n=N, target_sum=N/2.0):
    """Generate a random candidate with the correct sum."""
    rng = np.random.RandomState(seed)
    # Start with random values and project
    h = rng.uniform(0, 1, n)
    h = _project_to_simplex(h, target_sum)
    return h

def _construct_block_candidate(block_len, high_val, low_val, n=N, target_sum=N/2.0):
    """Construct a block-structured candidate."""
    n_blocks = n // block_len
    remainder = n % block_len
    h = np.empty(n)
    idx = 0
    for i in range(n_blocks):
        h[idx:idx+block_len] = high_val if i % 2 == 0 else low_val
        idx += block_len
    if remainder > 0:
        h[idx:] = high_val
    # Project to fix sum
    h = _project_to_simplex(h, target_sum)
    return h

def _optimize_from_start(h0, shifts, time_limit, beta_start=5.0, beta_end=50.0):
    """Run smooth optimization from a starting point."""
    target_sum = N / 2.0
    h = h0.copy()
    n = len(h)
    
    # Adaptive beta schedule
    best_h = h.copy()
    best_val = _overlap_value(h, shifts)
    
    n_iter = 0
    max_iter = 200
    
    for it in range(max_iter):
        if time.time() > time_limit:
            break
        beta = beta_start * (beta_end / beta_start) ** (it / max(1, max_iter - 1))
        
        def objective(x):
            x_proj = _project_to_simplex(x, target_sum)
            return _smooth_overlap(x_proj, shifts, beta)
        
        # Use L-BFGS-B with bounds
        bounds = [(0.0, 1.0)] * n
        result = minimize(objective, h, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8})
        
        h = _project_to_simplex(result.x, target_sum)
        val = _overlap_value(h, shifts)
        if val < best_val:
            best_val = val
            best_h = h.copy()
        n_iter += 1
    
    return best_h, best_val

def _local_polish(h, shifts, time_limit):
    """High-precision local polish using gradient descent on the max overlap."""
    target_sum = N / 2.0
    h = h.copy()
    n = len(h)
    best_h = h.copy()
    best_val = _overlap_value(h, shifts)
    
    lr = 0.01
    for it in range(500):
        if time.time() > time_limit:
            break
        
        # Compute gradient of max overlap
        # Find the shift that gives max overlap
        max_ov = 0.0
        max_shift = 0
        for s in shifts:
            h_shifted = np.roll(h, -s)
            ov = np.sum(h * (1.0 - h_shifted))
            if ov > max_ov:
                max_ov = ov
                max_shift = s
        
        # Gradient of ov = sum(h[i] * (1 - h[(i+s)%n]))
        # d/dh[i] = (1 - h[(i+s)%n]) - sum_j h[(j+s)%n] * delta_{i,j-s}
        # More carefully: ov = sum_i h[i] - sum_i h[i]*h[(i+s)%n]
        # d/dh[i] = 1 - h[(i+s)%n] - h[(i-s)%n]  (for the cross term)
        # Actually: sum_i h[i]*h[(i+s)%n], d/dh[i] = h[(i+s)%n] + h[(i-s)%n] (when i != i+s and i != i-s)
        # But when s=0 it's different. Let's handle generally.
        
        h_shifted_fwd = np.roll(h, -max_shift)  # h[(i+s)%n]
        h_shifted_bwd = np.roll(h, max_shift)   # h[(i-s)%n]
        
        # gradient of sum h[i] is 1
        # gradient of sum h[i]*h[(i+s)%n] w.r.t. h[i]:
        # = h[(i+s)%n] + (if max_shift != 0) h[(i-s)%n] * (correction for double counting when 2s=0 mod n)
        # For simplicity, use: grad_cross[i] = h[(i+s)%n] + h[(i-s)%n] - (1 if 2*max_shift % n == 0 else 0) * h[i]
        grad_cross = h_shifted_fwd + h_shifted_bwd
        if 2 * max_shift % n == 0:
            grad_cross -= h  # subtract double count
        
        grad = 1.0 - grad_cross
        
        # Project gradient to satisfy sum constraint (remove mean)
        grad = grad - np.mean(grad)
        
        # Gradient step
        h = h - lr * grad
        h = _project_to_simplex(h, target_sum)
        
        val = _overlap_value(h, shifts)
        if val < best_val:
            best_val = val
            best_h = h.copy()
        
        # Adaptive learning rate
        if val > best_val * 1.001:
            lr *= 0.9
    
    return best_h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Shifts to consider: all shifts from 1 to N-1
    shifts = list(range(1, N))
    
    target_sum = N / 2.0
    
    best_h = np.full(N, 0.5)
    best_val = _overlap_value(best_h, shifts)
    
    # Phase 1: Generate diverse initial candidates
    candidates = []
    
    # Random candidates
    for seed in range(20):
        if time.time() > deadline - 30.0:
            break
        h = _construct_candidate(seed)
        candidates.append(h)
    
    # Block candidates
    for block_len in [2, 3, 4, 5, 7, 10, 13, 20, 25]:
        if time.time() > deadline - 30.0:
            break
        # Try various high/low values
        for high_val in [0.0, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0]:
            low_val = 1.0 - high_val  # keep average 0.5
            h = _construct_block_candidate(block_len, high_val, low_val)
            candidates.append(h)
    
    # Sawtooth / linear patterns
    for period in [2, 3, 4, 5, 7, 10]:
        if time.time() > deadline - 30.0:
            break
        h = np.zeros(N)
        for i in range(N):
            frac = (i % period) / period
            h[i] = 0.5 + 0.5 * np.sin(2 * np.pi * frac)
        h = _project_to_simplex(h, target_sum)
        candidates.append(h)
    
    # Phase 2: Optimize each candidate
    for idx, h0 in enumerate(candidates):
        if time.time() > deadline - 20.0:
            break
        
        h_opt, val = _optimize_from_start(h0, shifts, deadline - 10.0, 
                                           beta_start=5.0, beta_end=30.0)
        if val < best_val:
            best_val = val
            best_h = h_opt.copy()
    
    # Phase 3: Local polish on the best
    if time.time() < deadline - 5.0:
        h_polished, val_polished = _local_polish(best_h, shifts, deadline - 2.0)
        if val_polished < best_val:
            best_val = val_polished
            best_h = h_polished.copy()
    
    # Final projection to ensure exact sum
    best_h = _project_to_simplex(best_h, target_sum)
    
    return best_h
# EVOLVE-BLOCK-END
