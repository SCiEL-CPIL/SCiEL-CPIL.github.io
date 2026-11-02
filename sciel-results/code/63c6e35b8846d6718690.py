"""Model-and-optimize: Zero-sum game (minimax) formulation for autoconvolution minimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 138  # Safety margin before 150s kill

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - 5)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective_only(x):
    """Compute the raw objective: max|ac| / (sum(x))^2."""
    ac = _autoconv_fft(x)
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    return np.max(np.abs(ac)) / (s * s)

def _soft_max_and_grad(x, beta):
    """
    Compute soft-max surrogate for max|ac| and its gradient using pure FFT.
    Returns (soft_max_value, gradient_array)
    """
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    m = np.max(abs_ac)
    exps = np.exp(beta * (abs_ac - m))
    log_sum = np.log(np.sum(exps))
    soft_val = m + log_sum / beta
    
    weights = exps / np.sum(exps)
    signs = np.sign(ac)
    signs[ac == 0] = 1.0
    g = weights * signs  # length 2N-1, this is d(soft_max)/d(ac_k)
    
    # Gradient w.r.t. x: grad_x[i] = sum_k g[k] * d(ac[k])/dx[i]
    # d(ac[k])/dx[i] = x[k-i] + x[i-k] (for valid indices)
    # grad_x[i] = sum_k g[k] * (x[k-i] + x[i-k])
    # = A[i] + B[i] where A[i] = sum_m x[m]*g[m+i] and B[i] = sum_m x[m]*g[i-m]
    
    L_fft2 = 1
    while L_fft2 < 2 * (2*N - 1):
        L_fft2 *= 2
    
    X_f2 = np.fft.rfft(x, n=L_fft2)
    G_f2 = np.fft.rfft(g, n=L_fft2)
    
    # A[i] = sum_m x[m]*g[m+i] = cross-correlation
    A = np.fft.irfft(X_f2 * np.conj(G_f2), n=L_fft2)[:N]
    # B[i] = sum_m x[m]*g[i-m] = convolution
    B = np.fft.irfft(X_f2 * G_f2, n=L_fft2)[:N]
    
    grad_x = A + B
    return soft_val, grad_x

def _objective_and_grad(x, beta):
    """Compute objective and gradient."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N)
    soft_val, grad_soft = _soft_max_and_grad(x, beta)
    val = soft_val / (s * s)
    grad = grad_soft / (s * s) - 2.0 * soft_val * np.ones(N) / (s * s * s)
    return val, grad

def _game_solve(x_init, beta, n_iters=200, lr_x=0.01, lr_w=0.001):
    """
    Solve the zero-sum game:
    min_x max_w E_w[|ac_k|] / s^2
    
    Using iterative gradient updates:
    - x update: gradient descent on soft_max|ac|/s^2
    - w update: multiplicative weights on lags based on |ac_k|
    
    Returns (best_x, best_val)
    """
    x = x_init.copy()
    s = np.sum(x)
    if abs(s) < 1e-10:
        s = 1e-10
    
    L_ac = 2 * N - 1
    w = np.ones(L_ac) / L_ac  # probability distribution over lags
    
    best_x = x.copy()
    best_val = _objective_only(x)
    
    for it in range(n_iters):
        if _check_deadline():
            break
        
        # Compute autoconvolution
        ac = _autoconv_fft(x)
        abs_ac = np.abs(ac)
        
        # Compute weighted objective: E_w[|ac_k|] / s^2
        weighted_abs = np.dot(w, abs_ac)
        val = weighted_abs / (s * s)
        
        # Update x: gradient descent
        # grad_x[i] = sum_k w[k] * sign(ac_k) * d(ac_k)/dx[i]
        g = w * np.sign(ac)
        g[ac == 0] = 1.0
        
        L_fft2 = 1
        while L_fft2 < 2 * (2*N - 1):
            L_fft2 *= 2
        
        X_f2 = np.fft.rfft(x, n=L_fft2)
        G_f2 = np.fft.rfft(g, n=L_fft2)
        
        A = np.fft.irfft(X_f2 * np.conj(G_f2), n=L_fft2)[:N]
        B = np.fft.irfft(X_f2 * G_f2, n=L_fft2)[:N]
        grad_x = A + B
        
        # Full gradient of val = weighted_abs / s^2
        grad_val = grad_x / (s * s) - 2.0 * weighted_abs * np.ones(N) / (s * s * s)
        
        x = x - lr_x * grad_val
        
        # Update w: multiplicative weights based on |ac_k|
        # Increase weight on lags with high |ac_k|
        m = np.max(abs_ac)
        if m > 1e-10:
            exps = np.exp(lr_w * (abs_ac - m))
            w = exps / np.sum(exps)
        
        # Track best
        current_val = _objective_only(x)
        if current_val < best_val:
            best_val = current_val
            best_x = x.copy()
        
        # Adaptive learning rate
        if it % 20 == 0 and it > 0:
            lr_x *= 0.9
            lr_w *= 0.95
    
    return best_x, best_val

def _spectral_init(rng):
    """
    Generate initial points via spectral amplitude shaping.
    """
    L_fft = 1
    while L_fft < 2*N:
        L_fft *= 2
    
    k = np.arange(L_fft // 2 + 1)
    t = k / (L_fft // 2)
    
    windows = [
        np.ones(L_fft // 2 + 1),
        np.exp(-0.5 * (t - 0.5)**2 / (0.15**2)),
        np.exp(-0.5 * (t - 0.5)**2 / (0.25**2)),
        1.0 / (1.0 + (t / 0.3)**2),
        np.sinc(0.5 * t) ** 2,
        np.exp(-t),
        1.0 / (1.0 + t),
    ]
    
    inits = []
    for w in windows:
        w = np.maximum(w, 0)
        phase = rng.uniform(0, 2*np.pi, size=L_fft // 2 + 1)
        spec = w * np.exp(1j * phase)
        spec_full = np.zeros(L_fft, dtype=complex)
        spec_full[:L_fft//2+1] = spec
        spec_full[L_fft//2+1:] = np.conj(spec[1:L_fft//2][::-1])
        x = np.fft.ifft(spec_full).real[:N]
        x = x - np.min(x) + 0.1
        inits.append(x)
    
    t_arr = np.linspace(0, 1, N)
    inits.append(np.ones(N))
    inits.append(1.0 + 0.5 * np.sin(2 * np.pi * t_arr))
    inits.append(1.0 + 0.5 * np.cos(2 * np.pi * t_arr))
    inits.append(np.exp(t_arr))
    inits.append(np.exp(-t_arr))
    inits.append(1.0 + 0.8 * (t_arr - 0.5)**2)
    inits.append(1.0 + 0.5 * (t_arr - 0.5))
    
    for _ in range(8):
        inits.append(rng.normal(1.0, 0.3, size=N))
    
    for _ in range(4):
        x = np.ones(N) * 0.1
        n_spikes = rng.integers(3, 10)
        idx = rng.choice(N, size=n_spikes, replace=False)
        x[idx] = 1.0
        inits.append(x)
    
    for _ in range(4):
        x = np.ones(N)
        for freq in rng.integers(1, 5, size=3):
            x += 0.3 * np.sin(2 * np.pi * freq * t_arr + rng.uniform(0, 2*np.pi))
        inits.append(x)
    
    return inits

def _polish(x, beta, max_iter=300):
    """
    Polish a solution using L-BFGS-B with soft-max surrogate.
    """
    try:
        def obj_and_grad_func(x):
            return _objective_and_grad(x, beta)
        
        result = minimize(obj_and_grad_func, x, method='L-BFGS-B',
                          jac=True, options={'maxiter': max_iter, 'gtol': 1e-10, 'ftol': 1e-12})
        
        if result.fun < _objective_only(x):
            return result.x.copy(), result.fun
    except Exception:
        pass
    
    return x, _objective_only(x)

def construct():
    best_x = np.ones(N)
    best_val = _objective_only(best_x)
    
    if _check_deadline():
        return best_x
    
    rng = np.random.default_rng(12345)
    
    # Phase 1: Multi-start with game-based optimization
    initial_points = _spectral_init(rng)
    
    for x0 in initial_points:
        if _check_deadline():
            break
        
        x_start = x0 + rng.normal(0, 0.01, size=N)
        
        # Try game-based optimization
        x_game, val_game = _game_solve(x_start, beta=50.0, n_iters=100, lr_x=0.01, lr_w=0.001)
        if val_game < best_val:
            best_val = val_game
            best_x = x_game.copy()
        
        # Also try direct L-BFGS
        if not _check_deadline():
            x_lf, val_lf = _polish(x_start, beta=50.0, max_iter=200)
            if val_lf < best_val:
                best_val = val_lf
                best_x = x_lf.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 2: Polish with higher beta
    for beta in [100.0, 200.0, 500.0]:
        if _check_deadline():
            break
        best_x, best_val = _polish(best_x, beta, max_iter=300)
    
    if _check_deadline():
        return best_x
    
    # Phase 3: Game-based refinement on the polished solution
    if not _check_deadline():
        best_x, best_val = _game_solve(best_x, beta=200.0, n_iters=200, lr_x=0.005, lr_w=0.0005)
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Final L-BFGS polish
    if not _check_deadline():
        best_x, best_val = _polish(best_x, beta=500.0, max_iter=400)
    
    if _check_deadline():
        return best_x
    
    # Phase 5: Structured random polish
    step_size = 0.01
    n_steps = 0
    max_steps = 200000
    
    t_arr = np.linspace(0, 1, N)
    freqs = [1, 2, 3, 5, 8]
    
    while n_steps < max_steps:
        if _check_deadline():
            break
        
        r = rng.random()
        if r < 0.3:
            f = freqs[rng.integers(0, len(freqs))]
            perturbation = step_size * np.sin(2 * np.pi * f * t_arr + rng.uniform(0, 2*np.pi))
        elif r < 0.5:
            perturbation = np.zeros(N)
            n_spikes = rng.integers(1, 4)
            idx = rng.choice(N, size=n_spikes, replace=False)
            perturbation[idx] = rng.normal(0, step_size * 3, size=n_spikes)
        else:
            perturbation = rng.normal(0, step_size, size=N)
        
        x_trial = best_x + perturbation
        val = _objective_only(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
            step_size *= 0.999
        else:
            step_size = min(0.1, step_size * 1.01)
        n_steps += 1
    
    # Phase 6: Coordinate descent polish
    if not _check_deadline():
        coord_steps = 0
        max_coord_steps = 50000
        while coord_steps < max_coord_steps:
            if _check_deadline():
                break
            improved = False
            for i in range(0, N, max(1, N // 100)):
                if _check_deadline():
                    break
                for delta in [0.01, -0.01, 0.005, -0.005, 0.02, -0.02]:
                    x_trial = best_x.copy()
                    x_trial[i] += delta
                    val = _objective_only(x_trial)
                    if val < best_val:
                        best_val = val
                        best_x = x_trial
                        improved = True
                        break
            coord_steps += 1
            if not improved:
                break
    
    return best_x
# EVOLVE-BLOCK-END
