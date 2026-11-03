"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Direct convex optimization over the 600 step heights using FISTA
with a smooth approximation of the max function. The objective is to minimize
the log-sum-exp of the autoconvolution normalized by the square of the L1 norm.
We use a spectral initialization based on the idea that a flat autoconvolution
corresponds to a flat power spectrum in the Fourier domain.
"""

import numpy as np
import time

N = 600
TIME_LIMIT = 130  # Safety margin before 150s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

def lse(x, beta):
    """Numerically stable Log-Sum-Exp."""
    x_max = np.max(x)
    if x_max < 1e-15:
        return 0.0
    return x_max + np.log(np.sum(np.exp(beta * (x - x_max)))) / beta

def objective_and_grad(f, beta):
    """
    Compute objective and gradient for minimizing the smooth max of autoconvolution.
    Objective: LSE(beta * (conv / s^2))
    """
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return 1e10, np.zeros_like(f)
    
    conv = np.convolve(f, f, mode='full')
    z = conv / (s * s)
    
    # LSE
    z_max = np.max(z)
    if z_max < 1e-15:
        return 0.0, np.zeros_like(f)
    
    exp_z = np.exp(beta * (z - z_max))
    sum_exp = np.sum(exp_z)
    obj = z_max + np.log(sum_exp) / beta
    
    # Gradient of LSE w.r.t z: p_i = exp(z_i) / sum_exp
    p = exp_z / sum_exp
    
    # Gradient w.r.t f:
    # d obj / d f_j = sum_i p_i * d z_i / d f_j
    # z_i = conv_i / s^2
    # d z_i / d f_j = (2 * f[i-j] * s^2 - 2 * s * conv_i) / s^4
    #                = 2 * (f[i-j] * s - conv_i) / s^3
    # Note: conv_i is the autoconvolution at lag i.
    # The indices for f[i-j] are valid when 0 <= i-j < N.
    
    grad = np.zeros(N)
    # We need to sum over all i in [0, 2N-2]
    # p is a vector of length 2N-1.
    
    # For a given j, the term is sum_{i} p_i * 2 * (f[i-j] * s - conv_i) / s^3
    # = (2 / s^3) * (s * sum_{i} p_i * f[i-j] - sum_{i} p_i * conv_i)
    
    # sum_{i} p_i * conv_i is a scalar (let's call it c1)
    c1 = np.sum(p * conv)
    
    # sum_{i} p_i * f[i-j] is a convolution of p and f.
    # p is length 2N-1, f is length N.
    # conv_pf = np.convolve(p, f, mode='full')
    # We need the terms where the index of f matches j.
    # The index k in conv_pf corresponds to sum_{i} p_i * f_{k-i}.
    # We want f_j, so we need k-i = j => i = k-j.
    # The valid i are in [0, 2N-2].
    # So for a fixed j, we sum p_{k-j} * f_j over k.
    # This is equivalent to a cross-correlation or a shifted sum.
    
    # Let's compute the full convolution of p and f.
    # p is length 2N-1, f is length N. Result length 3N-2.
    # This might be expensive if done naively every step, but N=600 is small.
    # However, we can be smarter.
    # sum_i p_i f_{i-j}
    # Let v = f. We are computing (p * v) where * is convolution?
    # Actually, it's simpler to just loop over j and use slicing, or use FFT.
    # Given N=600, a simple loop or vectorized slice is fine.
    
    # Let's compute the vector S where S[j] = sum_i p_i * f[i-j]
    # This is the cross-correlation of p and f.
    # scipy.signal.correlate(p, f, mode='same')? No, p is longer.
    # Let's just use np.correlate(f, p, mode='full')?
    # np.correlate(v, w) computes sum_i v[i] * conj(w[i-k]).
    # We want sum_i p_i * f[i-j].
    # This is np.correlate(f, p, mode='full')[2*N-2 - j]?
    
    # Let's verify.
    # corr = np.correlate(f, p, mode='full')
    # corr[k] = sum_i f[i] * p[i-k]
    # We want sum_i p_i * f[i-j].
    # Let i' = i-j => i = i'+j.
    # sum_{i'} p_{i'+j} * f[i']
    # This is not a standard simple correlation index without shifting.
    
    # Let's just compute it directly for N=600.
    # S = np.zeros(N)
    # for j in range(N):
    #     valid_i = np.arange(max(0, j), min(2*N-1, N+j))
    #     S[j] = np.sum(p[valid_i] * f[valid_i - j])
    
    # Vectorized approach:
    # We can pad f and p and use FFT, or just use the explicit loop which is 600 * 600 = 360k ops.
    # 360k ops is very fast in Python/NumPy if vectorized, but a Python loop is slow.
    # Let's use the correlation trick.
    
    # We want S[j] = sum_i p_i f_{i-j}.
    # Let's define g = f[::-1] (reversed f).
    # Then conv(p, g)[k] = sum_i p_i g[k-i] = sum_i p_i f[N-1-(k-i)] = sum_i p_i f[N-1-k+i].
    # Let m = N-1-k+i => i = m - (N-1-k).
    # This is getting confusing.
    
    # Let's stick to the definition:
    # grad_j = (2/s^3) * (s * S_j - c1)
    # where S_j = sum_i p_i f_{i-j}.
    
    # We can compute S using FFT.
    # S is the cross-correlation of p and f.
    # S[j] = (p \star f)[j]
    # FFT method:
    size = 2 * N
    F = np.fft.rfft(f, n=size)
    P = np.fft.rfft(p, n=size)
    # Correlation is ifft(conj(F) * P) or ifft(F * conj(P))?
    # np.correlate(f, p) corresponds to ifft(conj(F) * P).
    # Let's check: corr[k] = sum_n f[n] conj(p[n-k]).
    # We want sum_i p_i f[i-j].
    # Let n = i-j => i = n+j.
    # sum_n p[n+j] f[n].
    # This is the correlation of p and f, but shifted.
    
    # Actually, let's just use the explicit loop with NumPy slicing. It's O(N^2) but with N=600, it's ~0.36M operations.
    # A Python loop of 600 iterations, each doing a vectorized dot product of length ~600.
    # This should be fast enough.
    
    S = np.zeros(N)
    for j in range(N):
        i_start = max(0, j)
        i_end = min(2*N-1, N+j)
        if i_end > i_start:
            S[j] = np.dot(p[i_start:i_end], f[i_start-j:i_end-j])
            
    grad = (2.0 / (s**3)) * (s * S - c1)
    
    return obj, grad

def fista(f0, beta, max_iter=1000, tol=1e-8):
    """
    FISTA for minimizing f(x) subject to x >= 0.
    f(x) = LSE(beta * (conv(x) / sum(x)^2))
    """
    f = f0.copy()
    y = f0.copy()
    t = 1.0
    L = 10.0  # Initial Lipschitz constant estimate (needs to be large enough)
    
    best_f = f.copy()
    best_obj = objective_and_grad(f, beta)[0]
    
    for k in range(max_iter):
        # 1. Gradient step
        obj_y, grad_y = objective_and_grad(y, beta)
        
        # Update Lipschitz constant L
        # L = max(L, ||f(y) - f(x) - grad_y^T (y-x)|| / ||y-x||^2)
        # For simplicity and speed, we can just use a fixed large L or update it conservatively.
        # The Hessian of LSE is bounded. For N=600, the values are small.
        # Let's try to adapt L.
        # Actually, let's just use a fixed L that is likely sufficient.
        # The second derivative of LSE is bounded by beta.
        # The mapping x -> conv(x)/s^2 is quadratic-ish.
        # Let's start with L = 100.0 and increase if it fails (oscillates).
        
        step_size = 1.0 / L
        
        # z = y - step_size * grad_y
        z = y - step_size * grad_y
        
        # 2. Proximal step (projection onto x >= 0)
        f_new = np.maximum(z, 0.0)
        
        # 3. Update y for next iteration
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t**2)) / 2.0
        y = f_new + (t - 1.0) / t_new * (f_new - f)
        t = t_new
        f = f_new
        
        # Check convergence
        obj_f, _ = objective_and_grad(f, beta)
        if obj_f < best_obj:
            best_obj = obj_f
            best_f = f.copy()
            
        if np.linalg.norm(f - f_new) < tol:
            break
            
        # Adaptive L: if the objective increases, increase L
        if obj_f > obj_y + 0.1 * obj_y: 
             L *= 2.0
        elif k > 10 and L > 10.0:
             L *= 0.95
             
    return best_f, best_obj

def construct():
    start_time = time.time()
    
    # Initial guess: Constant function
    f0 = np.ones(N)
    
    # Spectral Initialization:
    # We want a flat power spectrum.
    # Let's try a function that is a sum of sinusoids or a flat-top window?
    # Actually, the constant function has a delta spectrum.
    # A "white noise" like function has a flat spectrum, but we need non-negative.
    # Let's start with a smoothed version of white noise, or just the constant.
    # The constant function is a local minimum? No, it has a huge peak at lag 0.
    # We want to spread the mass.
    
    # Let's try a few different initializations and pick the best one after a short FISTA run.
    candidates = []
    
    # 1. Constant
    candidates.append(np.ones(N))
    
    # 2. Cosine bump in the middle
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    f_cos = np.cos(np.pi * x / 0.25) ** 2
    f_cos = np.maximum(f_cos, 0.0)
    candidates.append(f_cos)
    
    # 3. Two bumps
    f_two = np.zeros(N)
    centers = [-0.1, 0.1]
    for c in centers:
        mask = np.abs(x - c) < 0.1
        f_two[mask] += np.cos(np.pi * (x[mask] - c) / 0.2) ** 2
    f_two = np.maximum(f_two, 0.0)
    candidates.append(f_two)
    
    # 4. Random smooth
    rng = np.random.RandomState(42)
    f_rand = rng.rand(N)
    # Smooth it
    kernel = np.ones(10) / 10
    f_rand = np.convolve(f_rand, kernel, mode='same')
    f_rand = np.maximum(f_rand, 0.0)
    candidates.append(f_rand)
    
    best_f = None
    best_score = np.inf
    
    for f_init in candidates:
        if time.time() - start_time > TIME_LIMIT - 20:
            break
            
        # Normalize to sum 1 for consistent scaling in optimization
        s_init = np.sum(f_init)
        if s_init < 1e-15:
            continue
        f_norm = f_init / s_init
        
        # Run FISTA
        try:
            f_opt, obj_opt = fista(f_norm, beta=50.0, max_iter=500)
            
            # Compute actual score
            score = compute_score(f_opt)
            
            if score < best_score:
                best_score = score
                best_f = f_opt
        except Exception:
            continue
            
    # If no candidate worked, return constant
    if best_f is None:
        best_f = np.ones(N)
        best_score = compute_score(best_f)
        
    # Final polish: Run FISTA again with higher beta and more iterations on the best result
    if time.time() - start_time < TIME_LIMIT - 5:
        try:
            f_polished, obj_polished = fista(best_f, beta=100.0, max_iter=500)
            score_polished = compute_score(f_polished)
            if score_polished < best_score:
                best_f = f_polished
                best_score = score_polished
        except Exception:
            pass
            
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
