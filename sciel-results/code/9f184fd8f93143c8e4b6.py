import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 400
DELTA = 1.0 / 800.0 # Width of each interval

def _conv_linear(h):
    """Compute linear convolution h * h using FFT."""
    size = 2 * N
    H = np.fft.fft(h, size)
    G = H * H
    g = np.fft.ifft(G).real
    return g[:2*N-1]

def _norm3(g):
    """Compute L3 norm of g."""
    return np.sum(np.abs(g)**3)**(1/3)

def _objective_and_grad(u):
    """
    Objective: Minimize S(u) = ||u||_2^3 / ||u*u||_3 + penalty(||u||_2 - 1)^2
    u: vector of length N
    """
    norm2 = np.linalg.norm(u)
    if norm2 == 0:
        return 1e10, np.zeros_like(u)
    
    h = u / norm2
    g = _conv_linear(h)
    norm3 = _norm3(g)
    
    if norm3 == 0:
        return 1e10, np.zeros_like(u)
    
    # Base score
    score = (norm2**3) / norm3
    
    # Penalty
    penalty = 1000.0 * (norm2 - 1.0)**2
    total_score = score + penalty
    
    # Gradient
    # grad_score = (3 * norm2 * u * norm3 - norm2**3 * grad_norm3) / norm3**2
    # Wait, u is the variable. h = u/norm2.
    # Let's differentiate S(u) = A(u)/B(u) where A=||u||^3, B=||u*u||_3.
    # dA/du = 3 ||u|| u
    # dB/du = ?
    # We know d(-B)/dh = -2 DELTA corr(w, h) where w = sign(g)|g|^2/B^3.
    # So dB/dh = 2 DELTA corr(w, h).
    # By chain rule, dB/du = dB/dh * dh/du.
    # h = u / ||u||.
    # dh/du = (1/||u||) I - (u u^T) / ||u||^3.
    # This is getting complex.
    
    # Alternative: Optimize on the sphere directly using parameterization?
    # Or just use numerical gradient?
    # N=400. Numerical gradient is 400 evaluations.
    # Each eval is FFT (fast).
    # 400 * 500 steps * 400 evals = 80M FFTs? Too slow.
    
    # Let's stick to Projected Gradient Descent which is simpler and robust.
    # We already have the gradient for h on the sphere.
    # We will implement a custom loop in construct().
    
    return total_score, np.zeros_like(u) # Placeholder

def _gradient_on_sphere(h):
    """
    Compute gradient of -||h*h||_3 with respect to h, assuming ||h||_2=1.
    """
    g = _conv_linear(h)
    norm3_cubed = np.sum(np.abs(g)**3)
    if norm3_cubed == 0:
        return np.zeros_like(h)
    norm3 = norm3_cubed ** (1/3)
    
    # w_j = sign(g_j) |g_j|^2 / norm3^3
    w = np.sign(g) * (np.abs(g)**2) / norm3_cubed
    
    # grad = -2 * DELTA * corr(w, h)
    # corr(w, h)[k] = sum_j w_j h_{j-k}
    # Using FFT:
    size = 2 * N
    W = np.fft.fft(w, size)
    H = np.fft.fft(h, size)
    # corr(w, h) = conv(w, h_rev)
    h_rev = h[::-1]
    H_rev = np.fft.fft(h_rev, size)
    C = np.fft.ifft(W * H_rev).real
    
    # Extract relevant part
    # C[N-1 + k] corresponds to k
    grad = -2.0 * DELTA * C[N-1 : 2*N-1]
    
    return grad

def construct():
    start_time = time.time()
    best_h = None
    best_score = 1e10
    
    # Initializations
    inits = []
    inits.append(np.ones(N))
    inits.append(np.random.choice([-1, 1], N).astype(float))
    inits.append(np.random.randn(N))
    # Block pattern
    h_block = np.zeros(N)
    h_block[:N//2] = 1
    h_block[N//2:] = -1
    inits.append(h_block)
    
    # Add a few random Gaussian inits
    for _ in range(5):
        inits.append(np.random.randn(N))

    for h0 in inits:
        if time.time() - start_time > 140:
            break
            
        h = h0.copy()
        norm = np.linalg.norm(h)
        if norm == 0:
            continue
        h = h / norm
        
        # Gradient Descent
        eta = 0.05
        for step in range(1000):
            if time.time() - start_time > 140:
                break
                
            # Compute current score
            g = _conv_linear(h)
            norm3 = _norm3(g)
            if norm3 > 0:
                current_score = 1.0 / norm3 # Since ||h||=1
            else:
                current_score = 1e10
                
            if current_score < best_score:
                best_score = current_score
                best_h = h.copy()
                
            # Gradient step
            grad = _gradient_on_sphere(h)
            h_new = h - eta * grad
            
            # Project back to sphere
            norm_new = np.linalg.norm(h_new)
            if norm_new == 0:
                break
            h = h_new / norm_new
            
            # Adaptive eta: if score increases, reduce eta
            # (Simple heuristic: if we didn't improve in last few steps, reduce)
            # For now, fixed eta or slow decay
            eta *= 0.999

    # Final polish: try to improve the best found
    if best_h is not None:
        h = best_h.copy()
        eta = 0.01
        for step in range(500):
            if time.time() - start_time > 145:
                break
            g = _conv_linear(h)
            norm3 = _norm3(g)
            if norm3 > 0:
                current_score = 1.0 / norm3
            else:
                current_score = 1e10
                
            if current_score < best_score:
                best_score = current_score
                best_h = h.copy()
                
            grad = _gradient_on_sphere(h)
            h_new = h - eta * grad
            norm_new = np.linalg.norm(h_new)
            if norm_new == 0:
                break
            h = h_new / norm_new
            eta *= 0.995

    if best_h is None:
        return np.ones(N)
        
    return best_h
