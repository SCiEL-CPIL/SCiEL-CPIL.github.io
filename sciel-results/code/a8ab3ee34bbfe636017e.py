import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _calc_score(h):
    """
    Calculate the score (upper bound on C3) for a given step function h.
    Score = max_{k != 0} |(h*h)_k| / ||h||_2^2
    For binary +/- 1, ||h||_2^2 = N.
    """
    # np.correlate(h, h, 'full') gives lags from -(N-1) to N-1.
    # Index N-1 corresponds to lag 0.
    corr = np.correlate(h, h, mode='full')
    mask = np.ones(len(corr), dtype=bool)
    mask[N-1] = False
    peak = np.max(np.abs(corr[mask]))
    norm = np.sum(h**2)
    if norm == 0:
        return 0.0
    return peak / norm

def _soft_max(x, k):
    """
    Soft maximum approximation using log-sum-exp.
    """
    if len(x) == 0:
        return 0.0
    max_val = np.max(x)
    if max_val > 1e10:
        return max_val
    x_scaled = k * (x - max_val)
    x_scaled = np.clip(x_scaled, -50, 50)
    sum_exp = np.sum(np.exp(x_scaled))
    if sum_exp == 0:
        return max_val
    return max_val + np.log(sum_exp) / k

def _objective_and_grad(x, k=10.0):
    """
    Objective function for continuous optimization.
    Returns value only; gradient is computed numerically by the optimizer.
    """
    M = 2 * N
    x_padded = np.zeros(M)
    x_padded[:N] = x
    
    X = np.fft.fft(x_padded)
    Y = np.fft.ifft(X * np.conj(X)).real
    
    indices_pos = np.arange(1, N)
    indices_neg = np.arange(M - (N - 1), M)
    
    vals = np.concatenate([Y[indices_pos], Y[indices_neg]])
    abs_vals = np.abs(vals)
    
    f_val = _soft_max(abs_vals, k)
    return f_val

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    best_h = None
    best_score = float('inf')

    num_restarts = 20
    
    for i in range(num_restarts):
        if time.time() > deadline - 30:
            break
            
        if i % 2 == 0:
            x0 = np.random.choice([-1, 1], size=N).astype(float)
        else:
            x0 = np.random.randn(N)
            
        bounds = [(-1.0, 1.0) for _ in range(N)]
        res = minimize(_objective_and_grad, x0, method='L-BFGS-B',
                       bounds=bounds,
                       options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-5})
        
        x_opt = res.x
        h = np.sign(x_opt)
        h[h == 0] = 1
        
        score = _calc_score(h)
        
        improved = True
        max_iter_local = 500
        while improved and time.time() < deadline - 10:
            improved = False
            for _ in range(3):
                if time.time() > deadline - 10:
                    break
                for j in range(N):
                    h[j] *= -1
                    new_score = _calc_score(h)
                    if new_score < score:
                        score = new_score
                        improved = True
                    else:
                        h[j] *= -1
                        
                if not improved:
                    break
            
        if score < best_score:
            best_score = score
            best_h = h.copy()
            
        if best_score < 0.001:
            break

    if best_h is None:
        best_h = np.array([1 if i % 2 == 0 else -1 for i in range(N)])
        best_score = _calc_score(best_h)

    return best_h
