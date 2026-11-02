"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.signal

def _compute_autocorr(x):
    size = 2 * N
    fx = np.fft.fft(x, size)
    acf = np.fft.ifft(fx * np.conj(fx)).real
    return acf[:N]

def _compute_objective(x):
    r = _compute_autocorr(x)
    return np.abs(r[0]) + 2.0 * np.sum(np.abs(r[1:]))

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_x = None
    best_val = float('inf')
    
    # Try a few random restarts
    max_restarts = 5
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        x = np.random.choice([-1, 1], size=N)
        
        # Continuous Gradient Descent
        eta = 0.05
        max_steps = 100
        H_len = 2 * N - 1
        center = N - 1
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
                
            r = _compute_autocorr(x)
            s = np.sign(r)
            
            H = np.zeros(H_len)
            H[center] = s[0]
            H[center-1 : center-N : -1] = s[1:]
            H[center+1 : center+N] = s[1:]
            
            G = 2.0 * scipy.signal.convolve(x, H, mode='same')
            
            x = x - eta * G
            x = np.clip(x, -1.0, 1.0)
        
        # Round to discrete
        x_disc = np.sign(x)
        x_disc[x_disc == 0] = 1.0
        
        val = _compute_objective(x_disc)
        if val < best_val:
            best_val = val
            best_x = x_disc.copy()
            
        # Local Search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            r = _compute_autocorr(x_disc)
            
            for i in range(N):
                if time.time() > deadline:
                    break
                    
                u = np.zeros(N)
                if i >= 0:
                    u[0:i+1] = x_disc[i::-1]
                if i < N:
                    u[0:N-i] += x_disc[i:N]
                
                delta_r = -2.0 * x_disc[i] * u
                
                dJ = np.abs(r[0] + delta_r[0]) - np.abs(r[0])
                dJ += 2.0 * (np.sum(np.abs(r[1:] + delta_r[1:])) - np.sum(np.abs(r[1:])))
                
                if dJ < -1e-9:
                    x_disc[i] *= -1
                    r = r + delta_r
                    improved = True
                    
        if time.time() > deadline:
            break

    if best_x is None:
        return np.ones(N)
        
    return best_x
# EVOLVE-BLOCK-END
