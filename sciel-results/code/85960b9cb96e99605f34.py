"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.optimize as opt
import scipy.fft as fft

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.fft as fft

def _autoconv_lin(x):
    """Linear autoconvolution via FFT."""
    n = len(x)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    X = np.fft.rfft(x, fsize)
    c = np.fft.irfft(X * X, fsize)[:size]
    return c

def _metric_offdiag(x):
    """Compute max |c_k| for k != 0 divided by (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e9
    c = _autoconv_lin(x)
    max_abs = np.max(np.abs(c[1:]))
    return max_abs / (s * s)

def _soft_max_abs(c, temp=0.01):
    """Smooth approximation of max |c| using log-sum-exp."""
    abs_c = np.abs(c)
    abs_c = np.maximum(abs_c, 1e-12)
    scaled = abs_c / temp
    max_scaled = np.max(scaled)
    result = temp * (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled))))
    return result

def _soft_max_abs_grad(c, temp=0.01):
    """
    Compute soft max of |c| and its gradient w.r.t. c.
    Returns (value, grad_dc) where grad_dc has same shape as c.
    """
    abs_c = np.abs(c)
    abs_c = np.maximum(abs_c, 1e-12)
    scaled = abs_c / temp
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    sum_exp = np.sum(exp_scaled)
    value = temp * (max_scaled + np.log(sum_exp))
    # d/d|c| of soft_max = exp_scaled / sum_exp
    weights = exp_scaled / sum_exp
    # d|c|/dc = sign(c)
    grad_dc = weights * np.sign(c)
    return value, grad_dc

def _objective_and_grad(x, temp=0.01):
    """
    Compute smooth objective and its gradient w.r.t. x.
    Objective: soft_max(|c[1:]|) / (sum x)^2
    Uses chain rule through autoconvolution.
    """
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e9, np.zeros_like(x)
    
    c = _autoconv_lin(x)
    
    # Soft max on off-diagonal terms
    val, grad_dc = _soft_max_abs(c[1:], temp)
    
    # Objective = val / s^2
    obj = val / (s * s)
    
    # Gradient w.r.t. c[1:]: grad_dc / s^2
    # Gradient w.r.t. x from autoconvolution:
    # c_k = sum_i x_i * x_{i+k} for k >= 0
    # dc_k/dx_j = x_{j+k} (if j+k < N) + x_{j-k} (if j-k >= 0, k>0)
    
    # Build gradient w.r.t. x
    grad_x = np.zeros(N)
    for k in range(1, N):
        w = grad_dc[k-1] / (s * s)
        if w == 0:
            continue
        grad_x[:N-k] += w * x[k:]
        grad_x[k:] += w * x[:N-k]
    
    # Gradient from the 1/s^2 term
    obj_grad = grad_x - 2.0 * val * x / (s * s * s)
    
    return obj, obj_grad

def _project_to_sum_one(x):
    """Project x onto the hyperplane sum(x) = 1."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.ones(N) / N
    return x / s

def _subgradient_descent(x0, max_iters=50000, lr_init=0.001, temp_init=0.02,
                         deadline=None, momentum=0.9):
    """
    Subgradient descent with momentum and annealing on the smooth surrogate.
    """
    x = x0.copy()
    x = _project_to_sum_one(x)
    
    v = np.zeros_like(x)
    best_val = _metric_offdiag(x)
    best_x = x.copy()
    
    temp = temp_init
    lr = lr_init
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        temp = max(0.001, temp_init * (0.9995 ** it))
        lr = lr_init * (0.9999 ** it)
        
        obj, grad = _objective_and_grad(x, temp)
        
        v = momentum * v - lr * grad
        x_new = x + v
        x_new = _project_to_sum_one(x_new)
        
        true_val = _metric_offdiag(x_new)
        if true_val < best_val:
            best_val = true_val
            best_x = x_new.copy()
        
        x = x_new
    
    return best_x, best_val

def _random_signed_sequence(n=N):
    """Generate a random signed sequence with sum close to 1."""
    x = np.random.randn(n)
    x = _project_to_sum_one(x)
    return x

def _structured_sequence(type_str, n=N):
    """Generate structured initial sequences."""
    i = np.arange(n)
    if type_str == "alternating":
        x = np.where(i % 2 == 0, 1.0, -0.5)
    elif type_str == "block4":
        x = np.tile([1, 1, -1, -1], n // 4 + 1)[:n]
    elif type_str == "sine":
        x = np.sin(2 * np.pi * i / n)
        x = x - np.mean(x) + 0.1
    elif type_str == "random_block":
        x = np.random.choice([1.0, -1.0], size=n)
    else:
        x = np.random.randn(n)
    return _project_to_sum_one(x)

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_A = np.ones(N)
    best_val = _metric_offdiag(best_A)
    
    def time_up():
        return time.time() > deadline
    
    candidates = []
    candidates.append(np.ones(N))
    candidates.append(_structured_sequence("alternating"))
    candidates.append(_structured_sequence("block4"))
    for _ in range(8):
        candidates.append(_random_signed_sequence())
    candidates.append(_structured_sequence("sine"))
    candidates.append(_structured_sequence("random_block"))
    
    for idx, x0 in enumerate(candidates):
        if time_up():
            break
        
        try:
            remaining = deadline - time.time()
            if remaining < 5:
                break
            
            max_iters = int(min(50000, remaining * 1500))
            max_iters = max(max_iters, 1000)
            
            x_opt, val_opt = _subgradient_descent(
                x0,
                max_iters=max_iters,
                lr_init=0.0005,
                temp_init=0.05,
                deadline=deadline,
                momentum=0.9
            )
            
            if val_opt < best_val:
                best_val = val_opt
                best_A = x_opt.copy()
                
        except Exception as e:
            continue
    
    restart_count = 0
    while not time_up() and restart_count < 20:
        remaining = deadline - time.time()
        if remaining < 3:
            break
        
        x0 = _random_signed_sequence()
        try:
            max_iters = int(min(30000, remaining * 1000))
            max_iters = max(max_iters, 500)
            
            x_opt, val_opt = _subgradient_descent(
                x0,
                max_iters=max_iters,
                lr_init=0.0003,
                temp_init=0.03,
                deadline=deadline,
                momentum=0.9
            )
            
            if val_opt < best_val:
                best_val = val_opt
                best_A = x_opt.copy()
                
        except Exception:
            break
        restart_count += 1
    
    return best_A
# EVOLVE-BLOCK-END
