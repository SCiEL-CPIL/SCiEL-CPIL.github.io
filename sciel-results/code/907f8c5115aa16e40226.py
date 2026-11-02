import numpy as np
import time

N = 400

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
        return np.inf
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _game_step(x, q, lr_x, lr_q):
    """
    One step of the zero-sum game:
    x minimizes E_q[|ac|] / (sum x)^2
    q maximizes E_q[|ac|]
    
    Returns updated x, q, and the current objective value.
    """
    L = 2 * N - 1
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    s = np.sum(x)
    
    if abs(s) < 1e-10:
        return x, q, np.inf
    
    # Expected absolute autoconvolution
    expected_abs = np.dot(q, abs_ac)
    
    # Gradient w.r.t. x of E_q[|ac|] / s^2
    # d/dx_k [E_q[|ac|]] = sum_i q_i * sign(ac_i) * (x_{i-k} + x_{k-i})
    # Let v[i] = q[i] * sign(ac[i])
    # grad_sm_k = sum_i v[i] * x[i-k] + sum_i v[i] * x[k-i]
    
    v = q * np.sign(ac)
    
    # Compute gradient using FFT
    L_fft = 1
    while L_fft < L + N:
        L_fft *= 2
    
    v_pad = np.zeros(L_fft)
    v_pad[:L] = v
    
    x_pad = np.zeros(L_fft)
    x_pad[:N] = x
    
    V = np.fft.rfft(v_pad, n=L_fft)
    X = np.fft.rfft(x_pad, n=L_fft)
    
    # Term 1: sum_i v[i] * x[i-k] (cross-correlation)
    corr1 = np.fft.irfft(V * np.conj(X), n=L_fft)
    
    # Term 2: sum_i v[i] * x[k-i] (convolution)
    conv2 = np.fft.irfft(V * X, n=L_fft)
    
    grad_sm = corr1[:N] + conv2[:N]
    
    # Gradient of objective = grad_sm / s^2 - 2 * expected_abs / s^3
    grad_obj = grad_sm / (s * s) - 2.0 * expected_abs / (s * s * s)
    
    # Update x
    x_new = x - lr_x * grad_obj
    
    # Update q: q_i proportional to exp(|ac_i| / tau_q)
    # Use a temperature for the q update
    tau_q = 0.5 / max(np.max(abs_ac), 1e-10)
    shifted = abs_ac / tau_q
    shifted = shifted - np.max(shifted)
    q_new = np.exp(shifted)
    q_sum = np.sum(q_new)
    if q_sum > 0:
        q_new /= q_sum
    else:
        q_new = q.copy()
    
    # Mix with old q for stability
    q_new = 0.3 * q + 0.7 * q_new
    
    return x_new, q_new, expected_abs / (s * s)

def construct():
    START_TIME = time.time()
    TIME_LIMIT = 140  # safety margin
    
    def _time_ok():
        return (time.time() - START_TIME) < TIME_LIMIT
    
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Structured initializations
    initializations = [
        np.ones(N),
        np.linspace(2.0, 1.0, N),
        np.linspace(1.0, 2.0, N),
    ]
    for r in [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4]:
        initializations.append(r ** np.arange(N))
    for r in [1.05, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0]:
        initializations.append(r ** np.arange(N))
    
    t = np.linspace(0, 1, N)
    initializations.append(1.0 + 0.5 * np.cos(np.pi * t))
    initializations.append(1.0 + np.cos(np.pi * t))
    initializations.append(1.0 + t**2)
    initializations.append(1.0 + 4.0 * (t - 0.5)**2)
    initializations.append(np.exp(2.0 * t))
    initializations.append(np.exp(-2.0 * t))
    initializations.append(1.0 + 0.3 * np.sin(2 * np.pi * t))
    initializations.append(1.0 + 0.5 * np.sin(np.pi * t))
    
    for freq in [1, 2, 3, 4, 5, 7, 10]:
        initializations.append(np.sign(np.sin(2 * np.pi * freq * t)))
        initializations.append(1.0 + 0.5 * np.sign(np.sin(2 * np.pi * freq * t)))
    
    # Random initializations
    for _ in range(20):
        initializations.append(rng.normal(1.0, 0.5, size=N))
    
    for x0 in initializations:
        if not _time_ok():
            break
        # Ensure sum > 0
        if np.sum(x0) < 0:
            x0 = -x0
        val = _objective(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()
    
    # Phase 2: Zero-sum game optimization
    L = 2 * N - 1
    q = np.ones(L) / L  # uniform distribution over lags
    
    lr_x = 0.05
    lr_q = 0.1
    
    for game_iter in range(50000):
        if not _time_ok():
            break
        
        x_new, q_new, game_val = _game_step(best_x, q, lr_x, lr_q)
        
        # Ensure sum is positive
        if np.sum(x_new) < 0:
            x_new = -x_new
        
        actual_val = _objective(x_new)
        
        if actual_val < best_val:
            best_val = actual_val
            best_x = x_new.copy()
            q = q_new.copy()
        else:
            # If no improvement, try smaller step
            lr_x *= 0.95
            if lr_x < 1e-6:
                break
        
        # Adaptively adjust learning rate
        if game_iter % 1000 == 0 and game_iter > 0:
            # Decay learning rate slowly
            lr_x = max(0.001, lr_x * 0.9)
    
    # Phase 3: Gradient descent with smooth surrogate for fine-tuning
    tau = 0.05
    lr = 0.05
    for gd_iter in range(30000):
        if not _time_ok():
            break
        
        # Compute smooth objective and gradient
        s = np.sum(best_x)
        if abs(s) < 1e-10:
            break
        
        ac = _autoconv_fft(best_x)
        abs_ac = np.abs(ac)
        
        m = np.max(abs_ac)
        if m < 1e-15:
            break
        
        shifted = (abs_ac - m) / tau
        shifted = np.clip(shifted, -50, 0)
        weights = np.exp(shifted)
        w_sum = np.sum(weights)
        if w_sum < 1e-15:
            break
        weights /= w_sum
        
        soft_max = m + tau * np.log(w_sum)
        val_s = soft_max / (s * s)
        
        v = weights * np.sign(ac)
        
        L_fft = 1
        while L_fft < L + N:
            L_fft *= 2
        
        v_pad = np.zeros(L_fft)
        v_pad[:L] = v
        
        x_pad = np.zeros(L_fft)
        x_pad[:N] = best_x
        
        V = np.fft.rfft(v_pad, n=L_fft)
        X = np.fft.rfft(x_pad, n=L_fft)
        
        corr1 = np.fft.irfft(V * np.conj(X), n=L_fft)
        conv2 = np.fft.irfft(V * X, n=L_fft)
        
        grad_sm = corr1[:N] + conv2[:N]
        
        grad_obj = grad_sm / (s * s) - 2.0 * soft_max / (s * s * s)
        
        actual_val = _objective(best_x)
        
        norm_grad = np.linalg.norm(grad_obj)
        if norm_grad < 1e-10:
            break
        
        step = grad_obj / (norm_grad + 1e-10)
        
        alpha = lr
        improved = False
        for _ in range(15):
            x_trial = best_x - alpha * step
            trial_val = _objective(x_trial)
            if trial_val < actual_val * (1 - 1e-12):
                best_x = x_trial.copy()
                best_val = trial_val
                improved = True
                break
            alpha *= 0.5
        
        if not improved:
            lr *= 0.7
            if lr < 1e-10:
                break
        else:
            lr = min(0.5, lr * 1.15)
        
        if gd_iter % 200 == 0 and tau > 0.005:
            tau *= 0.9
    
    # Phase 4: Fine-tuning with small perturbations
    step_size = 0.01
    stall_count = 0
    for iteration in range(200000):
        if not _time_ok():
            break
        
        perturbation = rng.normal(0, step_size, size=N)
        x_trial = best_x + perturbation
        
        val = _objective(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
            stall_count = 0
            step_size = max(0.0005, step_size * 0.9995)
        else:
            stall_count += 1
            if stall_count % 5000 == 0:
                step_size = min(0.3, step_size * 1.03)
    
    return best_x
