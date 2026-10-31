import numpy as np
import time

N = 600
DEADLINE = 140.0  # Safety margin

def construct():
    start_time = time.time()
    
    # Initialize
    h = np.ones(N) / N
    best_h = h.copy()
    
    # Compute initial metric
    M = 1024
    num_lags = 2 * N - 1
    
    def get_true_max(h_vec):
        h_pad = np.zeros(M)
        h_pad[:N] = h_vec
        H = np.fft.fft(h_pad)
        c = np.real(np.fft.ifft(H * H))
        return np.max(c[:num_lags])
        
    best_val = get_true_max(h)
    
    def project_to_simplex(v, s=1.0):
        v = np.asarray(v, dtype=np.float64)
        n = len(v)
        if n == 0:
            return v
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
        theta = (cssv[rho] - s) / (rho + 1.0)
        w = np.maximum(v - theta, 0)
        w_sum = np.sum(w)
        if w_sum > 0:
            w *= s / w_sum
        return w

    def compute_grad_and_max(h_curr, beta):
        h_pad = np.zeros(M)
        h_pad[:N] = h_curr
        H = np.fft.fft(h_pad)
        C_freq = H * H
        c = np.real(np.fft.ifft(C_freq))
        c_lags = c[:num_lags]
        true_max = np.max(c_lags)
        
        # Softmax
        c_max = true_max
        exp_c = np.exp(beta * (c_lags - c_max))
        sum_exp = np.sum(exp_c)
        if sum_exp == 0:
            p = np.ones(num_lags) / num_lags
        else:
            p = exp_c / sum_exp
            
        p_pad = np.zeros(M)
        p_pad[:num_lags] = p
        P = np.fft.fft(p_pad)
        
        G_freq = P * H
        G = 2.0 * np.real(np.fft.ifft(G_freq))
        
        return G[:N], true_max

    beta = 10.0
    eta = 0.05
    
    while time.time() - start_time < DEADLINE:
        grad, curr_max = compute_grad_and_max(h, beta)
        
        # If current is better than best, update best
        if curr_max < best_val:
            best_val = curr_max
            best_h = h.copy()
            
        # Update h
        h_new = h - eta * grad
        h_new = project_to_simplex(h_new, 1.0)
        
        # Optional: Line search or simple acceptance
        # We can check if the new point has a lower soft-max or true max.
        # For speed, just accept and rely on projection/step size.
        h = h_new
        
        # Adaptivity
        beta *= 1.01
        if beta > 500.0:
            beta = 500.0
        eta *= 0.995
        if eta < 1e-6:
            eta = 1e-6

    return best_h.tolist()
