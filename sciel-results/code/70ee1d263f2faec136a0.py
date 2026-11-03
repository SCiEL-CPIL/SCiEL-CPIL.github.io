import numpy as np
import time
import random

N = 600
TIME_LIMIT = 115  # Safety margin

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    # Use FFT for fast convolution
    conv = np.fft.irfft(np.fft.rfft(f) ** 2, n=2*N-1)
    return np.max(conv) / (s * s)

def compute_conv(f):
    """Compute autoconvolution of f."""
    return np.fft.irfft(np.fft.rfft(f) ** 2, n=2*N-1)

def coordinate_descent_step(f, idx, conv):
    """
    For coordinate idx, find the best value given all others fixed.
    The max of autoconvolution is piecewise linear in f[idx].
    We try a few candidate values.
    """
    f_orig = f[idx]
    s_other = np.sum(f) - f_orig
    
    # The autoconvolution at lag k is:
    # C_k = sum_j f[j]*f[k-j]
    # Terms involving f[idx]:
    # When j=idx: f[idx]*f[k-idx]
    # When k-j=idx: f[j]*f[idx]
    # If idx == k-idx (i.e., k=2*idx), these are the same term: f[idx]^2
    
    # For each lag k, the contribution from f[idx] is:
    # a_k * f[idx] + b_k * f[idx]^2
    # where a_k = f[k-idx] (if valid) + f[idx + N - k] (if valid, for circular? no, linear conv)
    # Actually for linear convolution of length N:
    # C_k = sum_{j=max(0,k-N+1)}^{min(N-1,k)} f[j]*f[k-j]
    # Terms with f[idx]:
    #   j=idx: f[idx]*f[k-idx] if 0<=k-idx<N
    #   k-j=idx => j=k-idx: f[k-idx]*f[idx] if 0<=k-idx<N
    # If both apply and idx != k-idx, we get 2*f[idx]*f[k-idx]
    # If idx == k-idx, we get f[idx]^2
    
    # Let's precompute coefficients for each lag k
    # a_k = sum of f[k-idx] terms (linear in f[idx])
    # b_k = 1 if idx == k-idx else 0 (quadratic)
    
    # For simplicity, let's just try a few candidate values for f[idx]
    # and pick the one that minimizes the max autoconvolution.
    
    candidates = [0.0, f_orig * 0.5, f_orig, f_orig * 1.5, 2.0 * s_other / N]
    # Also try values based on the current max
    best_val = f_orig
    best_score = np.inf
    
    for cand in candidates:
        cand = max(0.0, cand)
        f_new = f.copy()
        f_new[idx] = cand
        # Normalize to sum=1
        s_new = np.sum(f_new)
        if s_new < 1e-15:
            continue
        f_new = f_new / s_new
        conv_new = compute_conv(f_new)
        score = np.max(conv_new) / (np.sum(f_new) ** 2)
        if score < best_score:
            best_score = score
            best_val = cand
    
    f[idx] = best_val
    return f, best_score

def local_search(f, time_limit=10, max_rounds=100):
    """
    Coordinate descent with randomized ordering.
    """
    f = f.copy()
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return f, np.inf
    f = f / s
    
    current_score = compute_score(f)
    start_time = time.time()
    
    for round_num in range(max_rounds):
        if time.time() - start_time > time_limit:
            break
        
        # Random permutation of indices
        perm = np.random.permutation(N)
        
        improved = False
        for idx in perm:
            if time.time() - start_time > time_limit:
                break
            f_new, score_new = coordinate_descent_step(f, idx, None)
            if score_new < current_score - 1e-10:
                f = f_new
                current_score = score_new
                improved = True
        
        if not improved:
            break
    
    return f, current_score

def simulated_annealing(f, time_limit=15):
    """
    Simulated annealing on the step heights.
    """
    f = f.copy()
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return f, np.inf
    f = f / s
    
    current_score = compute_score(f)
    best_f = f.copy()
    best_score = current_score
    
    start_time = time.time()
    T = 0.1  # Initial temperature
    cooling = 0.995
    
    while time.time() - start_time < time_limit:
        # Randomly pick a coordinate and perturb it
        idx = np.random.randint(N)
        delta = np.random.randn() * 0.1 * max(f[idx], 1.0/N)
        f_new = f.copy()
        f_new[idx] = max(0.0, f[idx] + delta)
        
        # Normalize
        s_new = np.sum(f_new)
        if s_new < 1e-15:
            T *= cooling
            continue
        f_new = f_new / s_new
        
        new_score = compute_score(f_new)
        delta_score = new_score - current_score
        
        if delta_score < 0 or np.random.rand() < np.exp(-delta_score / max(T, 1e-10)):
            f = f_new
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_f = f.copy()
        
        T *= cooling
    
    return best_f, best_score

def construct():
    start_time = time.time()
    best_f = None
    best_score = np.inf
    
    # Multiple restarts with different initializations
    for restart in range(8):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        
        # Generate initial point
        if restart == 0:
            # Uniform
            f_init = np.ones(N) / N
        elif restart == 1:
            # Triangular
            x = np.linspace(0, 1, N)
            f_init = 1 - np.abs(2*x - 1)
            f_init = f_init / np.sum(f_init)
        elif restart == 2:
            # Cosine
            x = np.linspace(0, 1, N)
            f_init = 0.5 * (1 + np.cos(2*np.pi*x))
            f_init = f_init / np.sum(f_init)
        elif restart == 3:
            # Plateau
            f_init = np.zeros(N)
            f_init[N//4:3*N//4] = 1.0
            f_init = f_init / np.sum(f_init)
        elif restart == 4:
            # Gaussian
            x = np.linspace(0, 1, N)
            f_init = np.exp(-((x-0.5)**2) / (2*0.15**2))
            f_init = f_init / np.sum(f_init)
        else:
            # Random smooth
            r = np.random.rand(N)
            r = np.convolve(r, np.ones(15)/15, mode='same')
            f_init = r / np.sum(r)
        
        # Phase 1: Local search (coordinate descent)
        f_ls, score_ls = local_search(f_init, time_limit=8)
        
        # Phase 2: Simulated annealing to escape local minima
        f_sa, score_sa = simulated_annealing(f_ls, time_limit=10)
        
        # Phase 3: Final local search polish
        f_final, score_final = local_search(f_sa, time_limit=5)
        
        if score_final < best_score:
            best_score = score_final
            best_f = f_final
    
    # Ensure valid
    best_f = np.maximum(best_f, 0.0)
    s = np.sum(best_f)
    if s < 1e-15:
        best_f = np.ones(N) / N
    else:
        best_f = best_f / s
    
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
