import numpy as np
import time

N = 400
START_TIME = time.time()
TIME_LIMIT = 145
SAFETY_MARGIN = 5

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective(x):
    """Compute max(|autoconv|) / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _objective_grad_smooth(x, beta=50.0):
    """
    Compute soft-max surrogate objective and its gradient.
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(x)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    max_val = np.max(abs_ac)
    
    if max_val < 1e-15:
        return 1e10, np.zeros_like(x)
    
    # Soft-max via log-sum-exp
    shifted = beta * (abs_ac - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    soft_max = max_val + (1.0 / beta) * np.log(sum_exp)
    val = soft_max / (s * s)
    
    # Gradient computation
    # d(soft_max)/dx[j] = sum_n weights[n] * sign(ac[n]) * (x[j+n] + x[j-n])
    
    g = np.zeros(2*N-1)
    g[:len(ac)] = weights * np.sign(ac)
    
    L_fft = 1
    while L_fft < 3*N:
        L_fft *= 2
    
    G = np.fft.rfft(g, n=L_fft)
    Xf = np.fft.rfft(x, n=L_fft)
    
    # term1[j] = sum_n g[n] * x[j+n]
    term1 = np.fft.irfft(np.conj(G) * Xf, n=L_fft)[:N]
    
    # term2[j] = sum_n g[n] * x[j-n]
    term2 = np.fft.irfft(G * Xf, n=L_fft)[:N]
    
    d_softmax_dx = term1 + term2
    
    grad = (1.0 / (s * s)) * d_softmax_dx - (2.0 * soft_max / (s ** 3)) * np.ones(N)
    
    return val, grad

def _game_theory_optimize(x0, max_iters=200, lr_x=0.01, lr_p=0.1, temperature=0.1):
    """
    Zero-sum game formulation:
    Minimize over x: max over p (distribution over lags) of E_p[ac] / s^2
    Actually, we want to minimize max |ac|/s^2.
    The adversary chooses a lag k to maximize |ac[k]|/s^2.
    This is equivalent to minimizing t such that |ac[k]| <= t*s^2 for all k.
    
    We use an iterative scheme where:
    1. Given current x, the adversary picks a distribution p over lags proportional to exp(|ac[k]|/temperature).
    2. We update x to minimize the expected value under p, or more simply, we use the gradient of the soft-max.
    
    This is essentially the same as the soft-max gradient descent, but framed as a game.
    However, we can add a "re-weighting" step or use a different update rule.
    
    Let's implement a simpler variant:
    Iterate:
    1. Compute ac.
    2. Compute weights p[k] = exp(beta * |ac[k]|) / sum(exp(beta * |ac[k]|))
    3. Compute gradient of sum(p[k] * |ac[k]|) w.r.t x.
       This is the same as _objective_grad_smooth but with the specific p.
    4. Update x.
    
    To make it distinct and potentially more robust, we can alternate between:
    - A step that reduces the current max peak (coordinate-like).
    - A step that flattens the distribution of peaks (game-like).
    
    Let's stick to the gradient of the soft-max but allow the temperature (beta) to anneal.
    """
    x = x0.copy()
    s = np.sum(x)
    if abs(s) < 1e-10:
        return x0, _objective(x0)
    
    # Normalize to sum 1 for stability during optimization
    x = x / s
    
    for i in range(max_iters):
        if _check_deadline():
            break
            
        # Anneal beta: start low (smooth), end high (sharp)
        progress = i / max_iters
        beta = 10.0 + 90.0 * progress # 10 to 100
        
        val, grad = _objective_grad_smooth(x, beta=beta)
        
        # Line search with backtracking
        step = lr_x
        success = False
        for _ in range(20):
            trial = x - step * grad
            trial_s = np.sum(trial)
            if abs(trial_s) < 1e-10:
                step *= 0.5
                continue
            # Objective is scale invariant? No, max|ac|/s^2 is invariant to scaling x by c.
            # So we can just evaluate _objective(trial).
            trial_val = _objective(trial)
            if trial_val < val * (1 - 1e-5): # Armijo condition
                x = trial
                success = True
                break
            step *= 0.5
        
        if not success:
            # If no improvement, try a smaller fixed step or break
            x = x - 0.001 * grad
            # x = x / np.sum(x) # Keep sum stable? The objective handles it.
            
    return x, _objective(x)

def _generate_random_phase_init(rng, N):
    """
    Generate an initial guess based on random Fourier phases.
    """
    L_fft = 1
    while L_fft < 2*N:
        L_fft *= 2
    
    phases = rng.uniform(0, 2*np.pi, size=L_fft//2 + 1)
    
    k = np.arange(L_fft//2 + 1)
    t = k / (L_fft/2)
    
    profile_type = rng.integers(0, 3)
    if profile_type == 0:
        amps = np.ones(L_fft//2 + 1)
    elif profile_type == 1:
        sigma = rng.uniform(0.2, 0.5)
        amps = np.exp(-t**2 / (2*sigma**2))
    else:
        amps = 1.0 - 0.5 * t
    
    X_half = amps * np.exp(1j * phases)
    
    X = np.zeros(L_fft, dtype=complex)
    X[:L_fft//2 + 1] = X_half
    X[L_fft//2 + 1:] = np.conj(X_half[L_fft//2-1:0:-1])
    
    x = np.fft.irfft(X, n=L_fft)[:N]
    
    s = np.sum(x)
    if abs(s) > 1e-10:
        x = x / s
    else:
        x = np.ones(N) / N
        
    return x

def _polish_peak_reduction(x, max_iters=100):
    """
    Targeted polish: identify the peak with max |ac| and try to reduce it
    by moving in the direction of its gradient.
    """
    best_x = x.copy()
    best_val = _objective(best_x)
    
    for _ in range(max_iters):
        if _check_deadline():
            break
            
        ac = _autoconv_fft(best_x)
        abs_ac = np.abs(ac)
        max_idx = np.argmax(abs_ac)
        
        # Gradient of ac[max_idx]
        k = max_idx
        grad_peak = np.zeros(N)
        for j in range(N):
            val = 0.0
            if 0 <= k - j < N:
                val += best_x[k-j]
            if 0 <= j - k < N:
                val += best_x[j-k]
            grad_peak[j] = val
            
        norm_g = np.linalg.norm(grad_peak)
        if norm_g < 1e-10:
            break
        grad_peak /= norm_g
        
        improved = False
        for step in [0.05, 0.02, 0.01, -0.05, -0.02]:
            trial = best_x - step * grad_peak
            val = _objective(trial)
            if val < best_val:
                best_val = val
                best_x = trial.copy()
                improved = True
                break
        
        if not improved:
            break
            
    return best_x, best_val

def construct():
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    if _check_deadline():
        return best_x
    
    rng = np.random.default_rng(42)
    
    inits = []
    t = np.linspace(0, 1, N)
    
    # Structured inits
    inits.append(np.ones(N))
    inits.append(np.exp(-((t-0.5)**2)/0.05))
    inits.append(1.0 + 0.5*np.cos(2*np.pi*t))
    inits.append(1.0 + 0.5*np.sin(2*np.pi*t))
    inits.append(np.linspace(1, 0.5, N))
    
    # Random Phase Inits
    for _ in range(20):
        inits.append(_generate_random_phase_init(rng, N))
        
    # Random Smooth
    for _ in range(10):
        freqs = rng.uniform(0.1, 5.0, size=3)
        phases = rng.uniform(0, 2*np.pi, size=3)
        amps = rng.uniform(0.1, 0.5, size=3)
        sig = 1.0 + sum(a * np.cos(2*np.pi*f*t + p) for a, f, p in zip(amps, freqs, phases))
        sig = np.maximum(sig, 0.1)
        inits.append(sig / np.sum(sig))
        
    for x0 in inits:
        if _check_deadline():
            break
            
        s0 = np.sum(x0)
        if abs(s0) < 1e-10:
            continue
        x0 = x0 / s0
        
        # Strategy: Game Theory Optimization (Annealed Soft-Max Gradient Descent)
        # This is the structural change: framing the max reduction as a game
        # and using annealing to move from global to local structure.
        x_opt, val_opt = _game_theory_optimize(x0, max_iters=150)
        
        if val_opt < best_val:
            best_val = val_opt
            best_x = x_opt.copy()
            
        if _check_deadline():
            break
            
        # Polish with peak-specific reduction
        x_pol, val_pol = _polish_peak_reduction(best_x, max_iters=50)
        if val_pol < best_val:
            best_val = val_pol
            best_x = x_pol.copy()

    return best_x
