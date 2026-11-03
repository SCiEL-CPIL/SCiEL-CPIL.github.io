"""Combined program: peak-targeting local search with fast FFT-based scoring."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.signal import fftconvolve

def _compute_score(x):
    """Compute the C3' metric: max |autoconvolution| / (sum(x))^2"""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = fftconvolve(x, x, mode='full')
    max_abs = np.max(np.abs(c))
    return max_abs / (s * s)

def _compute_conv(x):
    """Compute full autoconvolution using FFT."""
    return fftconvolve(x, x, mode='full')

def _peak_targeting_search(x, deadline, max_rounds=200):
    """
    Peak-targeting local search: identify which single-bit flip most reduces
    the maximum absolute autoconvolution value.
    """
    best_seq = x.copy()
    best_score = _compute_score(best_seq)
    current = x.copy()
    current_score = best_score
    
    for round_idx in range(max_rounds):
        if time.time() > deadline:
            break
        
        # Compute current autoconvolution
        c = _compute_conv(current)
        max_abs = np.max(np.abs(c))
        
        # Find the lag(s) that achieve the maximum
        abs_c = np.abs(c)
        max_val = abs_c.max()
        
        # For each position i, compute the effect of flipping x[i] on the autoconvolution.
        # Flipping x[i] changes x[i] from v to -v, so the change in autoconvolution at lag k
        # is: delta_c[k] = -2*v*x[i] * (sum of x over the shifted window)
        # More precisely, c_new[k] = c_old[k] - 2*v*(x[i]*x[i-k] + ... ) 
        # Actually, the autoconvolution c[k] = sum_j x[j]*x[j+k].
        # If we flip x[i] (change by -2*v), then:
        #   c_new[k] = c_old[k] - 2*v * (x[i-k] + x[i+k])  ... no, that's not right either.
        # 
        # Let's think more carefully. c[k] = sum_j x[j] * x[j+k].
        # When we flip x[i], x[i] becomes -x[i], so the change is -2*x[i] at position i.
        # The new autoconvolution: c_new[k] = sum_j x'_j * x'_{j+k}
        # where x'_j = x_j except x'_i = -x_i.
        # 
        # The terms that change are those where j=i or j+k=i:
        #   j=i: term is x'_i * x'_{i+k} = -x_i * x_{i+k} (if i+k != i)
        #   j+k=i, i.e., j=i-k: term is x'_{i-k} * x'_i = x_{i-k} * (-x_i)
        # 
        # So delta_c[k] = (-x_i * x_{i+k}) - (x_i * x_{i+k}) + (x_{i-k} * (-x_i)) - (x_{i-k} * x_i)
        #                = -2*x_i*x_{i+k} - 2*x_{i-k}*x_i
        #                = -2*x_i*(x_{i+k} + x_{i-k})
        # 
        # But we need to be careful about boundary effects and the case where i+k=i (k=0)
        # or i-k=i (k=0).
        # 
        # For k=0: delta_c[0] = -2*x_i*x_i - 2*x_i*x_i = -4*x_i^2 = -4 (since x_i is ±1)
        # Wait, that doesn't seem right. Let me re-derive.
        # 
        # c[0] = sum_j x_j^2 = N (since all x_j are ±1)
        # After flipping x_i: c_new[0] = sum_j x'_j^2 = N (still, since (-x_i)^2 = x_i^2)
        # So delta_c[0] = 0. My formula gave -4, which is wrong.
        # 
        # The issue: when k=0, j=i and j+k=i are the same term, so I'm double-counting.
        # 
        # Let me redo this. The autoconvolution is c[k] = sum_j x_j * x_{j+k}.
        # Flipping x_i changes x_i to -x_i. The affected terms in c[k] are:
        #   - Terms where j = i: x_i * x_{i+k} becomes -x_i * x_{i+k}, change = -2*x_i*x_{i+k}
        #   - Terms where j+k = i, i.e., j = i-k: x_{i-k} * x_i becomes x_{i-k} * (-x_i), change = -2*x_{i-k}*x_i
        # 
        # But when k=0, these are the same term (j=i=j+k), so we only count it once:
        #   Change = -2*x_i*x_i = -2
        # 
        # For k != 0:
        #   Change = -2*x_i*x_{i+k} - 2*x_{i-k}*x_i = -2*x_i*(x_{i+k} + x_{i-k})
        # 
        # We need to handle boundary conditions: x[j] is 0 if j < 0 or j >= N.
        
        # For efficiency, let's precompute the effect for each i.
        # But with N=400, we can afford O(N^2) per round if we're careful.
        
        # Actually, let's just try flipping each position and see which gives the best improvement.
        # With N=400, that's 400 evaluations per round, each taking ~O(N log N) with FFT.
        # 400 * 200 rounds = 80,000 FFT evaluations. Each FFT is ~O(N log N) ≈ 400*9 ≈ 3600 ops.
        # Total: ~288M ops. This might be too slow in Python.
        
        # Alternative: compute the delta analytically.
        # For each i, the new max_abs can be computed from the old c and the delta.
        
        best_improvement = 0
        best_flip_idx = -1
        
        for i in range(N):
            if time.time() > deadline:
                break
            
            v = current[i]
            # Compute delta_c for all k
            # delta_c[k] = -2*v*(x[i+k] + x[i-k]) for k != 0, and -2 for k=0
            # with boundary handling (x[j]=0 if j out of range)
            
            # Efficient approach: the change in the autoconvolution is a shifted version
            # of the sequence itself.
            # delta_c = -2*v * (shifted_x + reversed_shifted_x) where shifted_x has x[i] removed
            # 
            # More precisely:
            # delta_c[k] = -2*v * (x[i+k] + x[i-k]) for valid indices
            # 
            # Let's construct this vector.
            delta = np.zeros(len(c))
            # c has length 2*N - 1, indexed from -(N-1) to N-1
            # c[k] corresponds to lag k - (N-1)
            
            for k in range(N):
                # x[i+k] contribution
                if i + k < N:
                    delta[k + (N-1)] += -2 * v * current[i + k]
                # x[i-k] contribution  
                if i - k >= 0:
                    delta[k + (N-1)] += -2 * v * current[i - k]
                # Also negative lags
                if k > 0:
                    # For lag -k: x[i-k] and x[i+k]
                    if i - k >= 0:
                        delta[(N-1) - k] += -2 * v * current[i - k]
                    if i + k < N:
                        delta[(N-1) - k] += -2 * v * current[i + k]
            
            # The lag-0 term: c[0] doesn't change (sum of squares is constant)
            delta[N-1] = 0  # Override any incorrect computation for k=0
            
            new_c = c + delta
            new_max_abs = np.max(np.abs(new_c))
            new_score = new_max_abs / (np.sum(current) ** 2)
            
            if new_score < current_score - 1e-12:
                if new_score < best_score:
                    best_improvement = current_score - new_score
                    best_flip_idx = i
                    best_seq_candidate = None  # We'll recompute
        
        if best_flip_idx >= 0:
            current[best_flip_idx] *= -1
            new_score = _compute_score(current)
            if new_score < best_score:
                best_score = new_score
                best_seq = current.copy()
                current_score = new_score
            else:
                current[best_flip_idx] *= -1  # Revert
        
        # Check for convergence
        if best_flip_idx < 0:
            break
    
    return best_seq, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s limit
    
    # Start with all ones
    x = np.ones(N)
    best_seq = x.copy()
    best_score = _compute_score(x)
    
    # Try some diverse initializations
    rng = np.random.RandomState(42)
    
    # Random with various biases
    for frac in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]:
        if time.time() > deadline:
            break
        n_neg = int(frac * N)
        idx = rng.choice(N, n_neg, replace=False)
        x_try = np.ones(N)
        x_try[idx] = -1
        s = _compute_score(x_try)
        if s < best_score:
            best_score = s
            best_seq = x_try.copy()
    
    # Block patterns
    for block_len in [2, 3, 4, 5, 7, 8, 10, 16]:
        if time.time() > deadline:
            break
        pattern = np.ones(block_len)
        if block_len > 1:
            pattern[:block_len // 2] = -1
        x_try = np.tile(pattern, N // block_len)
        if len(x_try) < N:
            x_try = np.concatenate([x_try, np.ones(N - len(x_try))])
        s = _compute_score(x_try)
        if s < best_score:
            best_score = s
            best_seq = x_try.copy()
    
    # Peak-targeting local search
    if time.time() < deadline:
        best_seq, best_score = _peak_targeting_search(best_seq, deadline)
    
    # Final polish: try individual flips one more time
    if time.time() < deadline:
        x = best_seq.copy()
        for i in range(N):
            if time.time() > deadline:
                break
            x[i] *= -1
            s = _compute_score(x)
            if s < best_score:
                best_score = s
                best_seq = x.copy()
            else:
                x[i] *= -1  # Revert
    
    return best_seq
# EVOLVE-BLOCK-END
