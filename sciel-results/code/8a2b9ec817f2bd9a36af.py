import numpy as np
import time

N = 600
dx = 0.5 / N

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    f = np.ones(N) / N
    best_score = np.inf
    best_f = f.copy()
    
    L_fft = 4096
    tau = 0.05
    lr = 0.05
    
    for step in range(20000):
        if time.time() > deadline:
            break
            
        F = np.fft.fft(f, L_fft)
        C = np.fft.ifft(F * F).real
        c = C[:2*N-1]
        
        w = np.exp(c / tau)
        w_sum = np.sum(w)
        if w_sum > 0:
            w /= w_sum
        else:
            w = np.zeros_like(c)
            w[0] = 1.0
            
        W = np.fft.fft(w, L_fft)
        # corr(f, w)
        Q = np.fft.ifft(F * np.conj(W)).real
        # g_i = 2 * Q[(L_fft - i) % L_fft]
        indices = (L_fft - np.arange(N)) % L_fft
        g = 2.0 * Q[indices]
        
        f_new = f - lr * g
        
        # Project to simplex
        f_sorted = np.sort(f_new)[::-1]
        css = np.cumsum(f_sorted)
        k_vals = np.arange(1, N+1)
        sums = css[k_vals-1]
        vals = f_sorted[k_vals-1]
        cond = (k_vals * vals - sums + 1) > 0
        if np.any(cond):
            k_max = np.max(np.where(cond)[0]) + 1
        else:
            k_max = 0
            
        if k_max > 0:
            theta = (css[k_max-1] - 1.0) / k_max
            f_proj = np.maximum(f_new - theta, 0.0)
        else:
            f_proj = np.ones(N) / N
            
        s = np.sum(f_proj)
        if s > 0:
            f_proj /= s
        else:
            f_proj = np.ones(N) / N
            
        f = f_proj
        
        c_max = np.max(c)
        curr_score = c_max / dx
        
        if curr_score < best_score:
            best_score = curr_score
            best_f = f.copy()
            
        if step % 100 == 0:
            lr *= 0.98
            tau *= 1.01 # Increase tau to smooth out? Or decrease to sharpen?
            # We want to minimize the max, so we want the gradient to focus on the peak.
            # Smaller tau focuses more on the peak.
            # But small tau makes gradient noisy.
            # Let's keep tau constant or decrease it slightly.
            tau = max(0.01, tau * 0.99)

    return best_f.tolist()
