import numpy as np
import time

N = 400

def _compute_convolution(x):
    """Compute linear autoconvolution of x."""
    return np.convolve(x, x, mode='full')

def _compute_metric(x, conv):
    """Compute the upper bound metric. Assuming L1 norm of autoconvolution."""
    return np.sum(np.abs(conv))

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initialize with random signs
    x = np.random.choice([-1, 1], size=N).astype(np.float64)
    conv = _compute_convolution(x)
    best_metric = _compute_metric(x, conv)
    best_x = x.copy()
    
    # Simulated Annealing Parameters
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    # Precompute shifts for fast update? 
    # Update formula: 
    # If we flip x[i] to -x[i]:
    # x_new = x - 2*x[i]*e_i
    # conv_new = conv - 4*x[i]*(x shifted by i) + 4*(e_i shifted by i)
    # Note: conv is length 2N-1.
    # x shifted by i in the context of convolution:
    # The term (x * e_i) contributes to conv indices i ... i+N-1.
    # Specifically, conv[k] = sum_j x[j] x[k-j].
    # The change in conv due to flipping x[i]:
    # Let delta = -2 * x[i].
    # x_new[j] = x[j] + delta * (j==i)
    # conv_new[k] = conv[k] + delta * (x[k-i] + x[i-(k-N+1)]?) -- careful with bounds.
    # Easier: 
    # conv_new = conv + delta * (x shifted right by i + x shifted left by i) + delta^2 * (impulse at 2i)
    # where shifts are valid within the convolution range.
    
    # Let's just recompute if N is small enough? 
    # N=400. conv length 799.
    # np.convolve is C-optimized. 
    # 10,000 iterations * 0.1ms per conv = 1s.
    # We can afford ~50,000 iterations.
    
    iterations = 0
    while time.time() < deadline:
        # Pick random index
        i = np.random.randint(N)
        old_val = x[i]
        new_val = -old_val
        
        # Calculate new convolution
        # x_new = x with x[i] flipped
        # Instead of copying x, we can compute the change
        # But for simplicity and safety with N=400, let's use the O(N) update logic properly.
        
        # Let's stick to full recomputation if it's fast enough, or use the update.
        # Full recomputation is safer to implement correctly.
        # Let's test speed mentally: 400x400 dot products. NumPy does this in <1ms.
        # Let's try to do 1000 iterations per second. 110s -> 110,000 iterations.
        
        x[i] = new_val
        conv_new = _compute_convolution(x)
        metric_new = _compute_metric(x, conv_new)
        
        delta = metric_new - best_metric
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            best_x = x.copy()
            best_metric = metric_new
            conv = conv_new
        else:
            x[i] = old_val # revert
            
        T *= cooling
        if T < T_min:
            T = T_min
            # Optional: restart with new random if stuck? 
            # For now, keep going.
            
        iterations += 1
        # Check time more frequently if iterations are fast
        if iterations % 100 == 0:
            if time.time() > deadline - 5.0:
                break

    return best_x

# EVOLVE-BLOCK-START
import numpy as np
import time

N = 400

def _compute_convolution(x):
    return np.convolve(x, x, mode='full')

def _compute_metric(x, conv):
    return np.sum(np.abs(conv))

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initialize with random signs
    x = np.random.choice([-1, 1], size=N).astype(np.float64)
    conv = _compute_convolution(x)
    best_metric = _compute_metric(x, conv)
    best_x = x.copy()
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    iterations = 0
    while time.time() < deadline:
        i = np.random.randint(N)
        old_val = x[i]
        new_val = -old_val
        
        x[i] = new_val
        conv_new = _compute_convolution(x)
        metric_new = _compute_metric(x, conv_new)
        
        delta = metric_new - best_metric
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            best_x = x.copy()
            best_metric = metric_new
            conv = conv_new
        else:
            x[i] = old_val
            
        T *= cooling
        if T < T_min:
            T = T_min
            
        iterations += 1
        if iterations % 100 == 0:
            if time.time() > deadline - 5.0:
                break

    return best_x
# EVOLVE-BLOCK-END
