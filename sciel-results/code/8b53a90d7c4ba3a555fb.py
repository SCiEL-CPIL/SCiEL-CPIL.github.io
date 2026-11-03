import numpy as np
import time

N = 600
TIME_LIMIT = 115

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    conv = np.convolve(f, f, mode='full')
    return np.max(conv) / (s * s)

def mirror_descent(f_init, start_time, max_time, eta0=0.01, eta_decay=0.999, restart_iter=500):
    """
    Mirror Descent (Exponentiated Gradient) to minimize max autoconvolution.
    """
    rng = np.random.RandomState(42)
    f = f_init.copy()
    f = np.maximum(f, 0)
    s = np.sum(f)
    if s > 0:
        f = f / s
    else:
        f = np.ones(N) / N
        
    best_f = f.copy()
    best_val = compute_score(f)
    current_val = best_val
    
    eta = eta0
    no_improve_count = 0
    
    for t in range(100000):
        if time.time() - start_time > max_time:
            break
            
        # Compute autoconvolution
        conv = np.convolve(f, f, mode='full')
        max_idx = np.argmax(conv)
        max_val = conv[max_idx]
        current_val = max_val / (np.sum(f)**2)
        
        if current_val < best_val:
            best_val = current_val
            best_f = f.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # Restart if stuck
        if no_improve_count > restart_iter:
            # Add noise to escape local minimum
            f = f * (1 + rng.normal(0, 0.1, N))
            f = np.maximum(f, 0)
            s = np.sum(f)
            if s > 0:
                f = f / s
            no_improve_count = 0
            eta = eta0 # Reset step size
            
        # Compute gradient of C[max_idx]
        g = np.zeros(N)
        k = max_idx
        for j in range(N):
            # Contribution from f[j] * f[k-j]
            # Derivative wrt f[j]:
            # If k-j is valid, add f[k-j]
            # Add f[j] (from the other side of the product, or if k-j==j it's 2*f[j])
            # Actually, d/dj sum_i f_i f_{k-i}
            # Term i=j: f_j f_{k-j} -> f_{k-j}
            # Term i=k-j: f_{k-j} f_j -> f_j
            # So g[j] = f[k-j] (if valid) + f[j]
            
            # Optimization: Vectorized gradient calculation
            pass # We will do vectorized below
            
        # Vectorized Gradient Calculation
        # g[j] = f[k-j] + f[j]
        # f[k-j] is a reversed and shifted version of f.
        
        # Create a buffer for f[k-j]
        # If k-j >= 0 and k-j < N, val = f[k-j]
        # This is equivalent to correlating f with a delta at k?
        
        # Let's use FFT for gradient?
        # Grad of C_k is essentially 2 * (f * delta_k) in some sense?
        # No, let's just do the loop or vectorized indexing.
        # N=600 is small. Loop is fine, but vectorized is faster.
        
        # Vectorized:
        # Indices j in 0..N-1
        # idx1 = k - j
        # valid1 = (idx1 >= 0) & (idx1 < N)
        # g[valid1] += f[idx1[valid1]]
        # g[all] += f[all]
        
        idx1 = k - np.arange(N)
        valid1 = (idx1 >= 0) & (idx1 < N)
        g[valid1] += f[idx1[valid1]]
        g += f
        
        # Update
        # f_j <- f_j * exp(-eta * g_j)
        # Subtract max g to prevent underflow
        g_max = np.max(g)
        if g_max > 0:
            g_shifted = g - g_max
        else:
            g_shifted = g
            
        # Clip g_shifted to avoid extreme values if eta is large
        # exp(-eta * g)
        update = np.exp(-eta * g_shifted)
        f_new = f * update
        
        s_new = np.sum(f_new)
        if s_new < 1e-15:
            # Reset
            f_new = np.ones(N) / N
        else:
            f_new = f_new / s_new
            
        f = f_new
        eta *= eta_decay
        
    return best_f, best_val

def construct():
    start_time = time.time()
    rng = np.random.RandomState(123)
    
    # Initial guesses
    # Uniform
    f1 = np.ones(N) / N
    # Gaussian-like
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    f2 = np.exp(-10 * x**2)
    f2 = f2 / np.sum(f2)
    
    best_f = None
    best_score = np.inf
    
    for f_init in [f1, f2]:
        if time.time() - start_time > TIME_LIMIT - 20:
            break
        refined_f, refined_score = mirror_descent(
            f_init, start_time, TIME_LIMIT - 10, 
            eta0=0.05, eta_decay=0.9995, restart_iter=200
        )
        if refined_score < best_score:
            best_score = refined_score
            best_f = refined_f

    if best_f is None:
        best_f = f1
        
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
