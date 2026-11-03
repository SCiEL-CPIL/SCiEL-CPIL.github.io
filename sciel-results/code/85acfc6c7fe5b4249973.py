"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize_scalar

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def construct():
    t_start = time.time()
    deadline = t_start + 130.0  # Safety margin

    best_val = float('inf')
    best_h = None

    # 1. Parametric Window Optimization (Positive/Smooth)
    # We assume the optimal shape is positive and smooth.
    # We will optimize a few families of windows.
    
    # Family 1: All Ones
    h_ones = np.ones(N)
    val_ones = _score(h_ones)
    if val_ones < best_val:
        best_val = val_ones
        best_h = h_ones.copy()

    # Family 2: Hanning-like
    def make_hanning(alpha):
        i = np.arange(N)
        if N > 1:
            phase = 2 * np.pi * i / (N - 1)
        else:
            phase = 0
        h = 0.5 * (1 + alpha) + 0.5 * (1 - alpha) * np.cos(phase)
        return h

    res_hann = minimize_scalar(lambda a: _score(make_hanning(a)), bounds=(0, 1), method='bounded')
    if res_hann.success:
        h_opt = make_hanning(res_hann.x)
        val_opt = _score(h_opt)
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt.copy()

    # Family 3: Gaussian
    def make_gaussian(sigma):
        i = np.arange(N)
        center = (N - 1) / 2.0
        h = np.exp(-((i - center) ** 2) / (2 * sigma * sigma))
        return h

    res_gauss = minimize_scalar(lambda s: _score(make_gaussian(s)), bounds=(1, N), method='bounded')
    if res_gauss.success:
        h_opt = make_gaussian(res_gauss.x)
        val_opt = _score(h_opt)
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt.copy()

    # Family 4: Power Law (Waterfall)
    def make_power(p):
        i = np.arange(N)
        center = (N - 1) / 2.0
        dist = np.abs(i - center) / (N / 2.0)
        h = np.power(np.maximum(1 - dist, 0.0001), p)
        return h

    res_pow = minimize_scalar(lambda p: _score(make_power(p)), bounds=(0.1, 10), method='bounded')
    if res_pow.success:
        h_opt = make_power(res_pow.x)
        val_opt = _score(h_opt)
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt.copy()

    # 2. Local Search on Real Values (Refinement of Positive Shape)
    h_cur = best_h.copy()
    val_cur = best_val
    
    if time.time() > deadline - 10:
        return h_cur

    rng = random.Random(42)
    step_size = 0.01
    iterations = 2000
    
    for _ in range(iterations):
        if time.time() > deadline - 5:
            break
        
        i = rng.randint(0, N - 1)
        old_h = h_cur[i]
        new_h_val = old_h + rng.uniform(-step_size, step_size)
        h_cur[i] = new_h_val
        
        new_val = _score(h_cur)
        
        if new_val < val_cur:
            val_cur = new_val
            best_h = h_cur.copy()
            step_size *= 0.95
        else:
            h_cur[i] = old_h
            step_size = min(step_size * 1.05, 0.5)

    # 3. Signed Binary Sequence Search (Barker-like)
    # The task allows "signed step heights". Binary sequences (+1/-1) are a strong candidate
    # for minimizing max|autoconv| / (sum)^2 because they maximize the "spread" of energy
    # while keeping the sum potentially large (or balanced).
    # Note: For binary sequences, sum is integer. If sum is 0, score is inf.
    # We want sequences with large |sum| and small max|autoconv|.
    
    # Initialize with random signs
    h_bin = np.random.choice([-1.0, 1.0], size=N)
    val_bin = _score(h_bin)
    
    # Local Search: Flip signs to reduce score
    # We iterate through indices and flip if it improves the score.
    # To make this faster, we can precompute the autoconvolution and update it locally?
    # For N=400, FFT is fast enough for full recompute in a few ms.
    # 400 flips * ~2ms = 0.8s. This is feasible.
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(N):
            if time.time() > deadline - 5:
                break
            old_sign = h_bin[i]
            h_bin[i] = -old_sign
            new_val = _score(h_bin)
            if new_val < val_bin:
                val_bin = new_val
                improved = True
            else:
                h_bin[i] = old_sign
        
        if val_bin < best_val:
            best_val = val_bin
            best_h = h_bin.copy()

    # 4. Gradient Descent on Continuous Relaxation of Signed Problem
    # Start from the best binary sequence found (or random)
    # Objective: f(x) = max_k |ac_k| / (sum x)^2
    # Soft surrogate: L(x) = (1/T) * log(sum_k exp(T * |ac_k|)) / (sum x)^2
    # We can optimize x in R^N.
    
    # Let's try optimizing the binary sequence further using a soft-max gradient step.
    # This is effectively "simulated annealing" or "gradient flow" on the sign variables.
    
    # Actually, let's just do a more aggressive local search on the binary sequence.
    # Random restarts for binary search.
    for restart in range(5):
        if time.time() > deadline - 10:
            break
        h_bin = np.random.choice([-1.0, 1.0], size=N)
        val_bin = _score(h_bin)
        if val_bin >= best_val:
            continue
            
        # Local search with random order to escape local minima
        for _ in range(1000): # Max 1000 flips per restart
            if time.time() > deadline - 10:
                break
            i = rng.randint(0, N - 1)
            old_sign = h_bin[i]
            h_bin[i] = -old_sign
            new_val = _score(h_bin)
            if new_val < val_bin:
                val_bin = new_val
                if val_bin < best_val:
                    best_val = val_bin
                    best_h = h_bin.copy()
            else:
                h_bin[i] = old_sign

    return best_h

# EVOLVE-BLOCK-END
