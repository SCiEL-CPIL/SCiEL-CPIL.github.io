import numpy as np
import time

N = 600
dx = 0.5 / N

def construct():
    start_time = time.time()
    # 120s limit, use 110s to be safe
    deadline = start_time + 110.0
    
    # Initialize with box function
    # Normalized to L1=1
    f = np.ones(N) * (1.0 / (N * dx))
    
    beta = 50.0
    eta = 0.1
    
    best_score = np.inf
    best_f = f.copy()
    
    def calc_score(f_curr):
        norm = np.sum(f_curr) * dx
        if norm < 1e-12:
            return np.inf
        f_norm = f_curr / norm
        F = np.fft.fft(f_norm)
        P = np.abs(F)**2
        g = np.fft.ifft(P).real
        return np.max(g)
    
    current_score = calc_score(f)
    best_score = current_score
    best_f = f.copy()
    
    iter_count = 0
    while time.time() < deadline:
        iter_count += 1
        
        F = np.fft.fft(f)
        P = np.abs(F)**2
        g = np.fft.ifft(P).real
        
        # Softmax
        max_g = np.max(g)
        exp_g = np.exp(beta * (g - max_g))
        softmax_g = exp_g / np.sum(exp_g)
        grad_g = beta * softmax_g
        
        grad_P = np.fft.fft(grad_g).real
        
        P_new = P - eta * grad_P
        P_new = np.maximum(0, P_new)
        
        phase_F = np.angle(F)
        mag_F = np.sqrt(P_new)
        F_new = mag_F * np.exp(1j * phase_F)
        
        f_new = np.fft.ifft(F_new).real
        f_new = np.maximum(0, f_new)
        
        norm = np.sum(f_new) * dx
        if norm > 1e-12:
            f_new = f_new / norm
        else:
            break
            
        score = calc_score(f_new)
        if score < best_score:
            best_score = score
            best_f = f_new.copy()
            
        f = f_new
        
        # Decay eta
        eta *= 0.99
        
    return list(best_f)
