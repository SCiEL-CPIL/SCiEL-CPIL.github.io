import time
import numpy as np
from scipy.optimize import minimize

N = 600

def compute_metric(h):
    """Compute normalized maximum autoconvolution."""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    s = h.sum()
    if s < 1e-15:
        return 1e10
    hn = h / s
    # Linear autoconvolution via FFT
    fft_size = 2 * N
    F = np.fft.rfft(hn, fft_size)
    conv = np.fft.irfft(F * F, fft_size)[:2 * N - 1]
    return conv.max()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    def elapsed():
        return time.time() - start_time
    
    # Initialize with uniform
    h_best = np.ones(N)
    best_val = compute_metric(h_best)
    
    # Phase 1: Try various structured candidates
    candidates = []
    
    # Uniform
    candidates.append(np.ones(N))
    
    # Triangular
    center = N // 2
    tri = np.array([max(0, 1.0 - abs(i - center) / center) for i in range(N)])
    candidates.append(tri)
    
    # Gaussian
    x = np.linspace(-3, 3, N)
    gauss = np.exp(-x**2)
    candidates.append(gauss)
    
    # Raised cosine
    rc = 0.5 * (1 + np.cos(np.pi * np.linspace(-1, 1, N)))
    candidates.append(rc)
    
    # Inverse Gaussian (peaks at edges)
    inv_gauss = 1.0 - 0.3 * np.exp(-3.0 * np.linspace(-1, 1, N)**2)
    inv_gauss = np.maximum(inv_gauss, 0)
    candidates.append(inv_gauss)
    
    # Two-peak
    two = np.zeros(N)
    two[N//4] = 1.0
    two[3*N//4] = 1.0
    candidates.append(two)
    
    # Plateau
    for w_frac in [0.3, 0.5, 0.7, 0.9]:
        w = int(N * w_frac)
        plat = np.zeros(N)
        plat[N//2 - w//2 : N//2 + w//2 + 1] = 1.0
        candidates.append(plat)
    
    # Evaluate candidates
    for c in candidates:
        if elapsed() > deadline:
            break
        c = np.maximum(c, 0)
        if c.sum() < 1e-15:
            continue
        val = compute_metric(c)
        if val < best_val:
            best_val = val
            h_best = c.copy()
    
    # Phase 2: Log-parameterized optimization with coordinate descent
    # h_i = exp(x_i) / sum(exp(x_j))
    def softmax(x):
        x_shifted = x - np.max(x)
        e = np.exp(x_shifted)
        return e / e.sum()
    
    def metric_from_params(x):
        h = softmax(x)
        return compute_metric(h)
    
    # Start from best candidate's log
    h_init = h_best.copy()
    h_init = np.maximum(h_init, 1e-10)
    x0 = np.log(h_init / h_init.sum())
    
    # Coordinate descent on logits
    x = x0.copy()
    current_val = metric_from_params(x)
    
    # Try small steps in each coordinate
    step_sizes = [0.01, 0.05, 0.1, 0.2]
    
    for epoch in range(3):
        if elapsed() > deadline - 20:
            break
        for _ in range(3):  # Passes
            if elapsed() > deadline - 10:
                break
            for i in range(N):
                if elapsed() > deadline - 5:
                    break
                for step in step_sizes:
                    for sign in [1, -1]:
                        x_trial = x.copy()
                        x_trial[i] += sign * step
                        h_trial = softmax(x_trial)
                        val_trial = compute_metric(h_trial)
                        if val_trial < current_val:
                            x = x_trial
                            current_val = val_trial
                            if current_val < best_val:
                                best_val = current_val
                                h_best = h_trial.copy()
                # Early exit if no improvement possible with larger steps
                if elapsed() > deadline - 5:
                    break
    
    # Phase 3: Cosine series refinement
    # h_i = a0 + sum_{k=1}^{M} a_k * cos(2*pi*k*i/N)
    # Ensure h_i >= 0
    i = np.arange(N)
    M_max = 8
    
    # Start from current best
    h_cos = h_best.copy()
    
    for M in range(1, M_max + 1):
        if elapsed() > deadline - 5:
            break
        
        # Build cosine basis
        cos_basis = np.column_stack([np.cos(2 * np.pi * k * i / N) for k in range(1, M + 1)])
        
        # Try small perturbations
        for k in range(M):
            if elapsed() > deadline - 3:
                break
            for amp in [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]:
                for sign in [1, -1]:
                    h_test = h_cos + sign * amp * cos_basis[:, k]
                    h_test = np.maximum(h_test, 0)
                    if h_test.sum() < 1e-15:
                        continue
                    val = compute_metric(h_test)
                    if val < best_val:
                        best_val = val
                        h_best = h_test.copy()
                        h_cos = h_test.copy()
    
    # Final: ensure nonnegative and return
    h_best = np.maximum(h_best, 0.0)
    if h_best.sum() < 1e-15:
        h_best = np.ones(N)
    
    return list(h_best)
