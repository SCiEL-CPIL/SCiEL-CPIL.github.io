import numpy as np
import scipy.optimize
import time

N = 600
DEADLINE = 145.0  # 5 seconds margin

def compute_score(heights):
    """
    Computes the normalized maximum of the autoconvolution.
    We assume the metric is max(c) / (sum(p))^2.
    If this doesn't improve, the metric might exclude k=0.
    However, I will calculate the standard one first.
    """
    p = np.asarray(heights)
    if np.sum(p) == 0:
        return 1e9
    c = np.fft.irfft(np.fft.rfft(p) ** 2, n=2 * N)
    # The convolution length is 2N-1, but irfft with n=2N gives zero padding.
    # We only care about the valid part or the whole?
    # Autoconvolution of length N sequence has length 2N-1.
    # c[k] for k in 0..2N-2.
    # The peak is usually at k=N-1 (center).
    # Let's take the max of the whole thing.
    max_c = np.max(c)
    norm = np.sum(p) ** 2
    return max_c / norm

def smooth_max_surrogate(values, beta=100.0):
    """
    Smooth approximation of max(values).
    """
    # Subtract max for numerical stability
    v_max = np.max(values)
    return v_max + np.log(np.sum(np.exp(beta * (values - v_max)))) / beta

def objective_and_grad(x):
    """
    x: latent variables for softmax parameterization.
    Returns objective value and gradient.
    """
    # Softmax
    x_shifted = x - np.max(x)
    p = np.exp(x_shifted)
    p /= np.sum(p)
    
    # Autoconvolution
    P = np.fft.rfft(p)
    C = P ** 2
    c = np.fft.irfft(C, n=2 * N)
    
    # We want to minimize max(c) / (sum(p))^2.
    # Since sum(p)=1, we minimize max(c).
    # But we need the gradient of the smooth max.
    # Let L = smooth_max(c, beta).
    # dL/dc_k = exp(beta*c_k) / sum(exp(beta*c_j))
    
    beta = 50.0
    c_max = np.max(c)
    exp_c = np.exp(beta * (c - c_max))
    sum_exp = np.sum(exp_c)
    w = exp_c / sum_exp  # weights for the max
    
    # Gradient of L w.r.t c
    # dL/dc = w
    
    # Gradient of c w.r.t p
    # c = p * p
    # dc/dp = 2 * (w * p_rev) ? 
    # Let's use FFT for gradient.
    # dL/dP = 2 * P * IFFT(w)
    # Note: IFFT(w) gives the time domain gradient of the convolution wrt the frequency domain?
    # L = f(c), c = IFFT(P^2)
    # dL/dP = 2 P * FFT(dL/dc)  <-- Check this.
    # c_k = sum_j P_j e^{i ...}
    # Actually, c = IFFT(P^2).
    # dL/dP = IFFT(2 * P * FFT(w))?
    # Let's derive:
    # L = sum_k w_k c_k
    # c_k = sum_j p_j p_{k-j}
    # dL/dp_i = sum_k w_k (p_{k-i} + p_{k-i}) = 2 (w * p)_i ?
    # Using FFT:
    # FFT(dL/dp) = 2 * FFT(w) * P
    # So dL/dp = IFFT(2 * FFT(w) * P)
    
    W = np.fft.rfft(w)
    dL_dp = np.fft.irfft(2 * W * P, n=N)
    
    # Gradient of L w.r.t x
    # p = softmax(x)
    # dp_i/dx_j = p_i (delta_ij - p_j)
    # dL/dx_j = sum_i dL/dp_i p_i (delta_ij - p_j)
    #          = p_j dL/dp_j - p_j sum_i p_i dL/dp_i
    #          = p_j (dL/dp_j - sum(p * dL/dp))
    
    grad_p = dL_dp
    dot = np.sum(p * grad_p)
    grad_x = p * (grad_p - dot)
    
    # Objective value
    # We return the actual max / sum^2 for the score, but use smooth max for optimization.
    # The optimizer minimizes the smooth max.
    obj_val = c_max + np.log(sum_exp) / beta
    # Normalize by sum(p)^2 = 1
    return obj_val, grad_x

def construct():
    start_time = time.time()
    
    # Initial guess: Uniform
    # x = log(p). For uniform p, x is constant.
    x0 = np.zeros(N)
    
    # We want to minimize the smooth max.
    # But we suspect the uniform is optimal for max including 0.
    # If the metric excludes 0, we should modify the objective.
    # Let's try a "U-shaped" initial guess to see if it improves.
    # p_i = 1 + alpha * cos(2 pi i / N)
    # This creates a dip in the center.
    
    # Let's use a few different initializations and pick the best one.
    best_x = x0
    best_score = 1e9
    
    # Optimization function
    def opt(x_init):
        # L-BFGS-B
        res = scipy.optimize.minimize(
            objective_and_grad,
            x_init,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        return res.x, res.fun

    # Try uniform
    x_opt, _ = opt(x0)
    p_opt = np.exp(x_opt - np.max(x_opt))
    p_opt /= np.sum(p_opt)
    score = compute_score(p_opt)
    best_x = x_opt
    best_score = score
    
    # Try U-shaped (low in middle)
    # p_i proportional to 1 + 0.5 * cos(2*pi*i/N)
    x_init_u = np.log(1 + 0.5 * np.cos(2 * np.pi * np.arange(N) / N) + 1e-6)
    x_opt_u, _ = opt(x_init_u)
    p_opt_u = np.exp(x_opt_u - np.max(x_opt_u))
    p_opt_u /= np.sum(p_opt_u)
    score_u = compute_score(p_opt_u)
    if score_u < best_score:
        best_x = x_opt_u
        best_score = score_u

    # Try Gaussian (high in middle)
    # p_i proportional to exp(-(i-N/2)^2 / (2*sigma^2))
    sigma = N / 4
    i = np.arange(N) - N / 2
    x_init_g = np.log(np.exp(-i**2 / (2 * sigma**2)) + 1e-6)
    x_opt_g, _ = opt(x_init_g)
    p_opt_g = np.exp(x_opt_g - np.max(x_opt_g))
    p_opt_g /= np.sum(p_opt_g)
    score_g = compute_score(p_opt_g)
    if score_g < best_score:
        best_x = x_opt_g
        best_score = score_g

    # Time check
    if time.time() - start_time > DEADLINE:
        # Return best so far
        p_final = np.exp(best_x - np.max(best_x))
        p_final /= np.sum(p_final)
        return list(p_final)
    
    # If we have time, try to refine the best one further
    # Or try a random perturbation
    x_init_rand = x0 + 0.1 * np.random.randn(N)
    x_opt_r, _ = opt(x_init_rand)
    p_opt_r = np.exp(x_opt_r - np.max(x_opt_r))
    p_opt_r /= np.sum(p_opt_r)
    score_r = compute_score(p_opt_r)
    if score_r < best_score:
        best_x = x_opt_r
        best_score = score_r

    p_final = np.exp(best_x - np.max(best_x))
    p_final /= np.sum(p_final)
    return list(p_final)

# To ensure the function returns a list of floats
def construct():
    start_time = time.time()
    
    N = 600
    x0 = np.zeros(N)
    
    def objective_and_grad(x):
        x_shifted = x - np.max(x)
        p = np.exp(x_shifted)
        p /= np.sum(p)
        
        P = np.fft.rfft(p)
        C = P ** 2
        c = np.fft.irfft(C, n=2 * N)
        
        beta = 50.0
        c_max = np.max(c)
        exp_c = np.exp(beta * (c - c_max))
        sum_exp = np.sum(exp_c)
        w = exp_c / sum_exp
        
        W = np.fft.rfft(w)
        dL_dp = np.fft.irfft(2 * W * P, n=N)
        
        grad_p = dL_dp
        dot = np.sum(p * grad_p)
        grad_x = p * (grad_p - dot)
        
        obj_val = c_max + np.log(sum_exp) / beta
        return obj_val, grad_x

    def compute_score(heights):
        p = np.asarray(heights)
        if np.sum(p) == 0:
            return 1e9
        c = np.fft.irfft(np.fft.rfft(p) ** 2, n=2 * N)
        max_c = np.max(c)
        norm = np.sum(p) ** 2
        return max_c / norm

    def opt(x_init):
        try:
            res = scipy.optimize.minimize(
                objective_and_grad,
                x_init,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            return res.x
        except Exception:
            return x_init

    best_x = x0
    best_score = 1e9

    # 1. Uniform
    x_opt = opt(x0)
    p_opt = np.exp(x_opt - np.max(x_opt))
    p_opt /= np.sum(p_opt)
    score = compute_score(p_opt)
    best_x = x_opt
    best_score = score

    # 2. U-shaped
    x_init_u = np.log(1 + 0.5 * np.cos(2 * np.pi * np.arange(N) / N) + 1e-6)
    x_opt_u = opt(x_init_u)
    p_opt_u = np.exp(x_opt_u - np.max(x_opt_u))
    p_opt_u /= np.sum(p_opt_u)
    score_u = compute_score(p_opt_u)
    if score_u < best_score:
        best_x = x_opt_u
        best_score = score_u

    # 3. Gaussian
    sigma = N / 4
    i = np.arange(N) - N / 2
    x_init_g = np.log(np.exp(-i**2 / (2 * sigma**2)) + 1e-6)
    x_opt_g = opt(x_init_g)
    p_opt_g = np.exp(x_opt_g - np.max(x_opt_g))
    p_opt_g /= np.sum(p_opt_g)
    score_g = compute_score(p_opt_g)
    if score_g < best_score:
        best_x = x_opt_g
        best_score = score_g

    # 4. Random
    if time.time() - start_time < DEADLINE - 5:
        x_init_r = x0 + 0.1 * np.random.randn(N)
        x_opt_r = opt(x_init_r)
        p_opt_r = np.exp(x_opt_r - np.max(x_opt_r))
        p_opt_r /= np.sum(p_opt_r)
        score_r = compute_score(p_opt_r)
        if score_r < best_score:
            best_x = x_opt_r
            best_score = score_r

    p_final = np.exp(best_x - np.max(best_x))
    p_final /= np.sum(p_final)
    return list(p_final)
