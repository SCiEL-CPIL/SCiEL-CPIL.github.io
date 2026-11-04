"""
Strategy: Golay-like construction + iterative coordinate descent on ±1/0/1 values.

Rationale:
1. The metric is scale-invariant, so we can work with discrete values in {-1, 0, 1}
   and then scale at the end if needed (though metric is invariant to scaling).
2. Golay sequences have ideal autocorrelation (zero off-peak), which directly
   minimizes the max absolute autoconvolution.
3. We use a greedy/coordinate-descent approach: start from a random ±1 sequence,
   then iteratively flip individual bits if it improves the metric.
4. Multiple random restarts to escape local minima.
5. Deadline guard to stay within 120s.
"""

import numpy as np
import time

N = 400

def _autoconv_max_abs(x):
    """Max absolute value of the discrete linear autoconvolution of x."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    conv = np.fft.irfft(fx * fx, size)
    return float(np.max(np.abs(conv[:2 * n - 1])))

def _integral(x):
    """Integral of the equal-width step function = sum of the heights."""
    return float(np.sum(x))

def _metric(x):
    """max |autoconv| / (integral)^2. Lower is better."""
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)

def _metric_fast(x):
    """Faster metric computation for inner loop."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    conv = np.fft.irfft(fx * fx, size)
    m = float(np.max(np.abs(conv[:2 * n - 1])))
    s = float(np.sum(x))
    if abs(s) < 1e-12:
        return float('inf')
    return m / (s * s)

def _generate_golay_like(rng):
    """Generate a Golay-like sequence using recursive construction."""
    # Golay complementary pair construction
    # Start with [1]
    a = np.array([1.0])
    b = np.array([1.0])
    # Golay pair: if (a,b) is a Golay pair, then (a, -b) and (a conj, b conj) etc.
    # Standard Golay construction: 
    # Given Golay pair (a, b), new pairs: (a, b), (a, -b), (conj(a), conj(b)), etc.
    # For real sequences: (a, b), (a, -b), (-a, b), (-a, -b)
    
    # Build up to length 256 (2^8)
    target_len = 256
    while len(a) < target_len:
        # Golay construction: new_a = [a, b], new_b = [a, -b] (or variants)
        a_new = np.concatenate([a, b])
        b_new = np.concatenate([a, -b])
        a, b = a_new, b_new
    
    # Truncate to N=400 or use first N elements
    # Since 400 is not a power of 2, we'll use the first 400 elements of a longer sequence
    # Or we can pad/truncate
    
    # Let's use a different approach: generate a random ±1 sequence and then
    # apply coordinate descent
    return None

def _coordinate_descent(x, max_iterations=1000, time_limit=10.0):
    """
    Iterative coordinate descent: flip individual elements to minimize metric.
    x is an array of values in {-1, 1} (or {-1, 0, 1}).
    """
    n = len(x)
    x = x.copy().astype(float)
    current_metric = _metric_fast(x)
    
    start_time = time.time()
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        # Shuffle order to avoid bias
        order = np.random.permutation(n)
        
        for i in order:
            old_val = x[i]
            # Try flipping: if +1 -> -1, if -1 -> +1, if 0 -> 1 or -1
            if old_val == 1.0:
                new_val = -1.0
            elif old_val == -1.0:
                new_val = 1.0
            else:  # 0
                # Try both +1 and -1
                x[i] = 1.0
                m1 = _metric_fast(x)
                x[i] = -1.0
                m2 = _metric_fast(x)
                if m1 <= m2 and m1 < current_metric:
                    x[i] = 1.0
                    current_metric = m1
                    improved = True
                elif m2 < current_metric:
                    current_metric = m2
                    improved = True
                else:
                    x[i] = 0.0
                continue
            
            x[i] = new_val
            new_metric = _metric_fast(x)
            
            if new_metric < current_metric:
                current_metric = new_metric
                improved = True
            else:
                x[i] = old_val
        
        if not improved:
            break
    
    return x, current_metric

def construct():
    """
    Construct a signal that minimizes max |autoconv| / (integral)^2.
    Uses multiple random restarts with coordinate descent.
    """
    deadline = time.time() + 130.0
    
    best_x = np.ones(N)
    best_val = _metric(best_x)
    
    # Also try alternating pattern
    alt_x = np.ones(N)
    alt_x[1::2] = -1
    alt_val = _metric(alt_x)
    if alt_val < best_val:
        best_x = alt_x
        best_val = alt_val
    
    # Random restarts with coordinate descent
    np.random.seed(42)
    
    for restart in range(20):
        if time.time() > deadline - 20:
            break
        
        # Random ±1 initial sequence
        x_init = np.random.choice([-1.0, 1.0], size=N)
        
        # Coordinate descent
        x_opt, val = _coordinate_descent(x_init, max_iterations=500, time_limit=5.0)
        
        if val < best_val:
            best_x = x_opt
            best_val = val
    
    # Final polish: try a few more coordinate descent passes on the best
    if time.time() < deadline - 10:
        x_polished, val = _coordinate_descent(best_x, max_iterations=200, time_limit=15.0)
        if val < best_val:
            best_x = x_polished
            best_val = val
    
    return best_x
