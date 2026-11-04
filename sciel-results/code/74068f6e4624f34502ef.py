import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _compute_ratio(h):
    # h is non-negative
    l2_sq = np.sum(h**2)
    if l2_sq < 1e-15:
        return 0.0
    l1 = np.sum(h)
    conv = np.convolve(h, h, mode='full')
    l_inf = np.max(conv)
    if l1 * l_inf < 1e-15:
        return 0.0
    return l2_sq / (l1 * l_inf)

def _objective_and_grad(x):
    # x is unbounded, we use y = x - max(x) to prevent overflow
    y = x - np.max(x)
    
    # Compute S1, S2
    exp_y = np.exp(y)
    S1 = np.sum(exp_y)
    S2 = np.sum(exp_y**2)
    
    # Compute C_k = sum_{i+j=k} exp(y_i) exp(y_j)
    # Use convolution
    C = np.convolve(exp_y, exp_y, mode='full')
    
    # Soft log max of C
    # alpha for soft-max
    alpha = 0.1 
    # To avoid overflow in exp(C/alpha), normalize C
    C_max = np.max(C)
    C_shifted = C - C_max
    soft_log_max = alpha * np.log(np.sum(np.exp(C_shifted / alpha))) + C_max
    
    # Objective: ln(S1) + soft_log_max - 1.5 * ln(S2)
    # Note: ln(S1) = ln(sum exp(y))
    # ln(S2) = ln(sum exp(2y))
    
    # Gradient
    # d/dy ln(S1) = exp(y) / S1
    grad_S1 = exp_y / S1
    
    # d/dy ln(S2) = 2 * exp(2y) / S2
    grad_S2 = 2 * exp_y**2 / S2
    
    # d/dy soft_log_max
    # p_k = exp(C_k / alpha) / sum(exp(C_j / alpha))
    exp_C = np.exp(C_shifted / alpha)
    p = exp_C / np.sum(exp_C)
    
    # d(C_k)/dy_i = 2 * exp(y_i) * exp(y_{k-i})
    # We need sum_k p_k * 2 * exp(y_i) * exp(y_{k-i})
    # This is 2 * exp(y_i) * sum_k p_k * exp(y_{k-i})
    # Let Q_i = sum_k p_k * exp(y_{k-i})
    # C has length 2N-1. y has length N.
    # k-i must be in [0, N-1].
    Q = np.zeros(N)
    # Efficient way:
    # Q is cross-correlation of p and exp_y?
    # p is on 0..2N-2. exp_y on 0..N-1.
    # Q_i = sum_{k} p_k exp_y[k-i]
    # This is convolution of p and reversed exp_y?
    # Let's just do it directly or with FFT.
    # N=50, direct is fine.
    # But we can use np.correlate or similar.
    # Q_i = (p * exp_y)_i? No.
    # Let's use FFT for safety and speed.
    # Size 2N-1 is small.
    
    # Direct loop is O(N^2) = 2500. Very fast.
    # But let's try to vectorize.
    # p is shape (2N-1,). exp_y is shape (N,).
    # We want Q[i] = sum_{k} p[k] * exp_y[k-i] where 0 <= k-i < N.
    # This is the cross-correlation of p and exp_y.
    # np.correlate(p, exp_y, mode='valid')?
    # p length 2N-1, exp_y length N.
    # correlate(p, exp_y) gives length 2N-1.
    # We want the part where exp_y is fully inside p?
    # Actually, Q_i corresponds to shifting exp_y by i.
    # Let's just use the property:
    # Q = np.convolve(p, exp_y[::-1], mode='full')[N-1:3*N-2] ?
    # Let's stick to direct computation for correctness in this small case, or just use the analytical derivative logic with a simple loop.
    
    # Actually, Q_i = sum_{j=0}^{N-1} p_{i+j} exp_y[j]
    # This is the cross-correlation.
    Q = np.zeros(N)
    for i in range(N):
        # k ranges from i to i+N-1
        # p[k] * exp_y[k-i]
        Q[i] = np.dot(p[i:i+N], exp_y)
        
    grad_soft = 2 * exp_y * Q
    
    grad = grad_S1 + grad_soft - 1.5 * grad_S2
    
    val = np.log(S1) + soft_log_max - 1.5 * np.log(S2)
    
    return val, grad

def construct():
    start_time = time.time()
    deadline = 140.0 # 10s margin
    
    best_h = None
    best_score = -1e9
    
    # Initializations
    inits = []
    # 1. Random normal
    for _ in range(20):
        inits.append(np.random.randn(N))
    # 2. Gaussian
    x = np.linspace(-1, 1, N)
    inits.append(np.exp(-x**2 / 0.1))
    inits.append(np.exp(-x**2 / 0.5))
    inits.append(np.exp(-x**2 / 2.0))
    # 3. Sinc
    inits.append(np.sinc(x))
    # 4. Uniform
    inits.append(np.zeros(N))
    # 5. Sparse
    for _ in range(10):
        h = np.zeros(N)
        idx = np.random.choice(N, 5, replace=False)
        h[idx] = 1.0
        inits.append(h)
    # 6. Cosine
    inits.append(np.cos(np.pi * x / 2)**2)
    
    # Shuffle inits
    np.random.shuffle(inits)
    
    for x0 in inits:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(_objective_and_grad, x0, method='L-BFGS-B', jac=True, options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
            x_opt = res.x
            
            # Convert to h
            y = x_opt - np.max(x_opt)
            exp_y = np.exp(y)
            S2 = np.sum(exp_y**2)
            h = exp_y / np.sqrt(S2)
            
            # Ensure non-negative (should be)
            h = np.maximum(h, 0)
            
            score = _compute_ratio(h)
            
            if score > best_score:
                best_score = score
                best_h = h.copy()
                
        except Exception as e:
            pass
            
    if best_h is None:
        return [1.0] * N
        
    return list(best_h)
