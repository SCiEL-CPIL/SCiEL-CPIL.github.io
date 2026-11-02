import numpy as np
import time
import scipy.optimize

N = 600

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initial z
    z = np.zeros(N)
    
    # Objective and gradient
    def objective_and_grad(z):
        x = np.exp(z - np.max(z))
        x_sum = np.sum(x)
        x = x / x_sum
        
        # Convolution
        c = np.convolve(x, x, mode='full')
        
        # LogSumExp for stability
        beta = 10.0 # Start with moderate beta
        max_c = np.max(c)
        log_sum_exp = max_c + np.log(np.sum(np.exp(beta * (c - max_c))))
        val = log_sum_exp / beta
        
        # Gradient
        # p_k = exp(beta * (c_k - max_c)) / sum(exp(...))
        exp_term = np.exp(beta * (c - max_c))
        p = exp_term / np.sum(exp_term)
        
        # dL/dc = p
        # dc/dx_i = 2 * sum_{k} p_k x_{k-i} (cross correlation)
        # This is 2 * (p * x) reversed? 
        # c_k = sum_i x_i x_{k-i}
        # d c_k / d x_i = x_{k-i} + x_i (if k-i == i? No, just x_{k-i} from first term and x_{k-i} from second term if indices match)
        # Actually, c_k = sum_{j=0}^k x_j x_{k-j}.
        # d c_k / d x_i = x_{k-i} + x_i * (1 if k-i == i? No, the term is x_i x_{k-i}. If we vary x_i, we get x_{k-i}. The term x_{k-i} x_i is the same. So derivative is 2 x_{k-i} if k-i != i, and 2 x_i if k-i == i.
        # Basically d c_k / d x_i = 2 x_{k-i} for valid indices.
        # So dL/dx_i = 2 sum_k p_k x_{k-i}.
        # This is 2 * cross_correlation(p, x) at lag i?
        # np.correlate(p, x, 'full')?
        # Let's use FFT for speed or just np.convolve if N is small.
        # p is length 2N-1. x is length N.
        # We need sum_{k} p_k x_{k-i}.
        # This is (p * x)_i where * is convolution?
        # (p * x)_i = sum_k p_k x_{i-k}.
        # We have x_{k-i}. Let j = k-i => k = i+j.
        # sum_j p_{i+j} x_j.
        # This is cross-correlation.
        # np.correlate(p, x, mode='valid' or 'full')?
        # p length 2N-1. x length N.
        # We want result for i in 0..N-1.
        # sum_{j=0}^{N-1} p_{i+j} x_j.
        # This is the convolution of p and reversed x?
        # Let's just compute it directly or use FFT.
        # FFT is O(N log N).
        # P = fft(p, 2*N)
        # X = fft(x, 2*N)
        # Prod = P * conj(X) ? No.
        # Cross correlation r_i = sum_j p_{i+j} x_j.
        # This is (x * p)_i? No.
        # Let's use the property: sum_j p_{i+j} x_j = (x * p)_{i} ?
        # (x * p)_k = sum_j x_j p_{k-j}.
        # If we set k = -i? No.
        # Let's stick to direct calculation for N=600. 600*600 = 360k ops. Very fast.
        # grad_x = 2 * np.array([np.dot(p[i:i+N], x) for i in range(N)])
        
        grad_x = 2.0 * np.array([np.dot(p[i:i+N], x) for i in range(N)])
        
        # Chain rule for softmax
        # dx_i/dz_j = x_i (delta_ij - x_j)
        # dL/dz_j = sum_i dL/dx_i dx_i/dz_j
        # = sum_i grad_x_i x_i (delta_ij - x_j)
        # = grad_x_j x_j - x_j sum_i grad_x_i x_i
        
        sum_grad_x_x = np.sum(grad_x * x)
        grad_z = x * (grad_x - sum_grad_x_x)
        
        return val, grad_z

    # Optimize
    best_z = z.copy()
    best_val = np.inf
    
    # Multi-start or just one good start?
    # Uniform is a local max for max(conv)? No, uniform minimizes max(conv) for fixed sum?
    # For sum=1, uniform x=1/N.
    # conv is triangular. Peak = N * (1/N)^2 * N?
    # c_0 = sum x_i^2 = N * (1/N)^2 = 1/N.
    # Max is 1/N.
    # If we concentrate mass, e.g. x_0=1, c_0=1. Max=1.
    # 1/N < 1. So uniform is better.
    # So we start at uniform (z=0) and try to improve?
    # Wait, if uniform is the global minimum, why did the seed score 1.0 and we want to go lower?
    # The seed score is 1.0.
    # My metric: Score = N * max(conv(x,x)) for sum=1 x.
    # For uniform x=1/N: max(conv) = 1/N. Score = N * (1/N) = 1.
    # So uniform gives 1.0.
    # Can we get lower than 1.0?
    # The task is "First autocorrelation inequality".
    # C1 is a constant.
    # Maybe the uniform distribution is NOT the minimizer?
    # Or maybe the metric is different?
    # "normalized maximum of its autoconvolution".
    # If I can get < 1.0, I win.
    
    # Let's run L-BFGS-B
    try:
        res = scipy.optimize.minimize(objective_and_grad, z, method='L-BFGS-B', jac=True, options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        if res.success:
            z = res.x
            val, _ = objective_and_grad(z)
            if val < best_val:
                best_val = val
                best_z = z.copy()
    except Exception:
        pass
    
    # Convert back to heights
    x = np.exp(best_z - np.max(best_z))
    x = x / np.sum(x)
    
    # Scale to match seed sum (N)
    heights = x * N
    
    return list(heights)
