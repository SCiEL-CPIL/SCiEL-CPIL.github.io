import numpy as np
import time

N = 600
# FFT size, must be at least 2*N - 1 for linear convolution
# 1024 is a power of 2 and >= 1199
M = 2048 

def _autoconv(A):
    """Compute linear autoconvolution of A using FFT."""
    F = np.fft.fft(A, M)
    C = np.fft.ifft(F * np.conj(F)).real
    return C[:2*N-1]

def _compute_gradient(C, A, tau):
    """Compute gradient of the smooth surrogate objective."""
    # Compute weights for log-sum-exp: w_k = exp(tau * C_k) / sum(exp(tau * C_j))
    # Subtract max for numerical stability
    C_max = np.max(C)
    exp_C = np.exp(tau * (C - C_max))
    sum_exp = np.sum(exp_C)
    w = exp_C / sum_exp
    
    # Gradient G = 2 * IFFT(FFT(w) * FFT(A))
    F = np.fft.fft(A, M)
    W = np.fft.fft(w, M)
    G = 2.0 * np.fft.ifft(W * F).real
    return G[:N]

def construct():
    start_time = time.time()
    # Safety margin: stop 10 seconds before the 150s limit
    deadline = start_time + 140.0
    
    best_A = np.ones(N)
    best_score = np.inf
    
    # Initialize with a few diverse candidates
    candidates = [
        np.ones(N),
        np.random.rand(N),
        np.concatenate([np.ones(N//2), np.zeros(N//2)]),
        np.concatenate([np.zeros(N//4), np.ones(N//2), np.zeros(N//4)])
    ]
    
    for init_A in candidates:
        A = init_A.copy()
        A = np.maximum(A, 1e-9) # Ensure strictly positive for multiplicative updates
        A /= np.linalg.norm(A)
        
        # Parameters for optimization
        tau = 1.0
        eta = 0.05
        last_best_score = np.inf
        no_improve_count = 0
        max_tau = 1000.0
        
        while time.time() < deadline:
            C = _autoconv(A)
            current_score = np.max(C)
            
            if current_score < best_score:
                best_score = current_score
                best_A = A.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > 50:
                    # Increase sharpness of max approximation
                    tau *= 2.0
                    eta *= 0.5
                    no_improve_count = 0
                    if tau > max_tau:
                        break
                elif no_improve_count > 10:
                    eta *= 0.95

            # Compute gradient
            G = _compute_gradient(C, A, tau)
            
            # Multiplicative Weights Update (Mirror Descent)
            # A_new = A * exp(-eta * G)
            # Shift G to avoid underflow/overflow in exp
            G_shifted = G - np.min(G)
            update_factor = np.exp(-eta * G_shifted)
            
            A = A * update_factor
            
            # Normalize to unit L2 norm
            norm_A = np.linalg.norm(A)
            if norm_A == 0 or not np.isfinite(norm_A):
                break
            A /= norm_A
            
            # Check time
            if time.time() > deadline:
                break

    # Final polish: run a few more steps with high tau and small eta
    if time.time() < deadline - 5.0:
        A = best_A.copy()
        A = np.maximum(A, 1e-9)
        A /= np.linalg.norm(A)
        
        tau = max(100.0, tau)
        eta = 0.001
        
        for _ in range(100):
            if time.time() > deadline - 2.0:
                break
            C = _autoconv(A)
            current_score = np.max(C)
            if current_score < best_score:
                best_score = current_score
                best_A = A.copy()
            
            G = _compute_gradient(C, A, tau)
            G_shifted = G - np.min(G)
            update_factor = np.exp(-eta * G_shifted)
            A = A * update_factor
            norm_A = np.linalg.norm(A)
            if norm_A == 0 or not np.isfinite(norm_A):
                break
            A /= norm_A

    return list(best_A)
