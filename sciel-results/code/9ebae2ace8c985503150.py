"""
Combined spectral-continuous optimization + discrete polish for signed step functions
minimizing max|autoconv|/(sum x)^2.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START
START_TIME = time.time()
TIME_LIMIT = 148

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - 3)

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
    """Compute objective value only (fast, no gradient)."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    return m / (s * s)

def _soft_max_and_grad(x, beta):
    """
    Compute soft-max surrogate for max|ac| and its gradient w.r.t. x.
    """
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    m = np.max(abs_ac)
    exps = np.exp(beta * (abs_ac - m))
    sum_exps = np.sum(exps)
    soft_val = m + np.log(sum_exps) / beta
    
    weights = exps / sum_exps
    signs = np.sign(ac)
    signs[ac == 0] = 1.0
    grad_ac = weights * signs  # length 2N-1
    
    # grad_x[n] = 2 * sum_k grad_ac[k] * x[k-n] for 0<=k-n<N
    L_corr = 1
    while L_corr < 3 * N:
        L_corr *= 2
    
    g_padded = np.zeros(L_corr)
    g_padded[:2*N-1] = grad_ac
    
    X_f = np.fft.rfft(x, n=L_corr)
    G_f = np.fft.rfft(g_padded, n=L_corr)
    
    corr = np.fft.irfft(G_f * np.conj(X_f), n=L_corr)
    grad_x = 2.0 * corr[:N]
    
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

def _spectral_init(rng, mode=0):
    """
    Generate initial x via spectral phase retrieval with various magnitude profiles.
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    n_freq = L_fft // 2 + 1
    
    if mode == 0:
        freqs = np.fft.rfftfreq(L_fft)
        mag = np.ones(n_freq)
        taper = np.minimum(1.0, np.abs(freqs) * L_fft / 5)
        mag *= taper
    elif mode == 1:
        freqs = np.fft.rfftfreq(L_fft)
        mag = 0.5 + 0.5 * np.cos(2 * np.pi * freqs * L_fft * 0.5)
    elif mode == 2:
        mag = 1.0 + 0.3 * rng.normal(0, 1, size=n_freq)
        mag = np.maximum(0.1, mag)
    elif mode == 3:
        freqs = np.fft.rfftfreq(L_fft)
        mag = np.exp(-20 * (freqs * L_fft / 2) ** 2)
    elif mode == 4:
        freqs = np.fft.rfftfreq(L_fft)
        cutoff = 0.4
        transition = 0.05
        mag = np.where(np.abs(freqs) < cutoff - transition, 1.0,
                       np.where(np.abs(freqs) < cutoff + transition,
                                0.5 * (1 + np.cos(np.pi * (np.abs(freqs) - cutoff) / transition)),
                                0.1))
    else:
        mag = np.ones(n_freq)
    
    phase = rng.uniform(0, 2 * np.pi, size=n_freq)
    phase[0] = 0
    
    spectrum = mag * np.exp(1j * phase)
    spectrum[0] = mag[0]
    
    x_full = np.fft.irfft(spectrum, n=L_fft)
    x = x_full[:N]
    
    s = np.sum(x)
    if s < 0:
        x = -x
    
    return x

def _continuous_optimize(x0, beta=50.0, max_iters=3000):
    """
    Optimize in continuous domain using L-BFGS with soft-max surrogate.
    """
    def obj_grad(x):
        val, grad = _objective_and_grad(x, beta)
        return val, grad
    
    try:
        res = minimize(obj_grad, x0, method='L-BFGS-B', 
                      options={'maxiter': max_iters, 'gtol': 1e-4, 'disp': False})
        return res.x, res.fun
    except Exception:
        return x0, _objective_only(x0)

def _project_to_binary(x):
    """Project continuous vector to ±1 based on sign."""
    x_bin = np.sign(x)
    x_bin[x_bin == 0] = 1.0
    s = np.sum(x_bin)
    if s < 0:
        x_bin = -x_bin
    return x_bin

def _local_search(x, max_iters=3000):
    """
    Greedy local search: flip single bits if it improves the objective.
    """
    current_val = _objective_only(x)
    best_x = x.copy()
    best_val = current_val
    
    indices = np.arange(N)
    
    for _ in range(max_iters):
        if _check_deadline():
            break
        
        improved = False
        np.random.shuffle(indices)
        
        for i in indices:
            x[i] *= -1
            val = _objective_only(x)
            
            if val < best_val:
                best_val = val
                best_x = x.copy()
                improved = True
            else:
                x[i] *= -1
        
        if not improved:
            break
            
    return best_x, best_val

def _simulated_annealing(x0, T0=0.5, T_end=0.001, steps=15000):
    """
    Simulated Annealing on ±1 vectors.
    """
    x = x0.copy()
    current_val = _objective_only(x)
    best_x = x.copy()
    best_val = current_val
    
    rng = np.random.default_rng()
    
    for step in range(steps):
        if _check_deadline():
            break
            
        T = T0 * (T_end / T0) ** (step / steps)
        
        i = rng.integers(0, N)
        x[i] *= -1
        
        new_val = _objective_only(x)
        
        delta = new_val - current_val
        
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_x = x.copy()
        else:
            x[i] *= -1
            
    return best_x, best_val

def _spectral_phase_optimize(x0, rng, max_iters=100, lr=0.01):
    """
    Optimize the phase of the spectrum while keeping magnitude fixed.
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    n_freq = L_fft // 2 + 1
    
    X = np.fft.rfft(x0, n=L_fft)
    mag = np.abs(X).copy()
    phase = np.angle(X).copy()
    phase[0] = 0
    
    best_x = x0.copy()
    best_val = _objective_only(x0)
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        val, grad = _objective_and_grad(best_x, 50.0)
        
        G = np.fft.rfft(grad, n=L_fft)
        
        X_curr = mag * np.exp(1j * phase)
        phase_grad = np.imag(X_curr * np.conj(G)) / (mag + 1e-10)
        phase_grad[0] = 0
        
        phase -= lr * phase_grad
        
        spectrum = mag * np.exp(1j * phase)
        spectrum[0] = mag[0]
        x_trial = np.fft.irfft(spectrum, n=L_fft)[:N]
        
        s = np.sum(x_trial)
        if s < 0:
            x_trial = -x_trial
            phase = -phase
            phase[0] = 0
        
        val_trial = _objective_only(x_trial)
        
        if val_trial < best_val:
            best_val = val_trial
            best_x = x_trial.copy()
            if it % 20 == 0 and it > 0:
                lr = max(0.001, lr * 0.95)
        else:
            phase += lr * phase_grad
            lr = max(0.001, lr * 0.8)
    
    return best_x, best_val

def construct():
    best_x = None
    best_val = float('inf')
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Continuous optimization from multiple spectral initializations
    for mode in range(5):
        if _check_deadline():
            break
        
        x0 = _spectral_init(rng, mode=mode)
        x_cont, val_cont = _continuous_optimize(x0, beta=50.0, max_iters=2000)
        
        # Project to binary
        x_bin = _project_to_binary(x_cont)
        val_bin = _objective_only(x_bin)
        
        if val_bin < best_val:
            best_val = val_bin
            best_x = x_bin.copy()
        
        # Also try the continuous value directly (in case it's close to ±1)
        if _check_deadline():
            break
    
    if best_x is None:
        best_x = np.ones(N)
        best_val = _objective_only(best_x)
    
    # Phase 2: Polish the best binary candidate with local search
    if not _check_deadline():
        x_ls, val_ls = _local_search(best_x, max_iters=2000)
        if val_ls < best_val:
            best_val = val_ls
            best_x = x_ls.copy()
    
    # Phase 3: More spectral phase optimization from the best solution
    if not _check_deadline():
        x_sp, val_sp = _spectral_phase_optimize(best_x, rng, max_iters=80)
        x_bin2 = _project_to_binary(x_sp)
        val_bin2 = _objective_only(x_bin2)
        if val_bin2 < best_val:
            best_val = val_bin2
            best_x = x_bin2.copy()
            x_ls2, val_ls2 = _local_search(best_x, max_iters=1500)
            if val_ls2 < best_val:
                best_val = val_ls2
                best_x = x_ls2.copy()
    
    # Phase 4: SA restarts from different random initializations
    for r in range(15):
        if _check_deadline():
            break
        
        x0 = rng.choice([-1, 1], size=N)
        s = np.sum(x0)
        if abs(s) < 1:
            x0[0] *= -1
        
        x_sa, val_sa = _simulated_annealing(x0, T0=0.5, T_end=0.001, steps=10000)
        
        if val_sa < best_val:
            best_val = val_sa
            best_x = x_sa.copy()
            
            if not _check_deadline():
                x_ls, val_ls = _local_search(best_x, max_iters=1000)
                if val_ls < best_val:
                    best_val = val_ls
                    best_x = x_ls.copy()
    
    # Phase 5: Final polish
    if not _check_deadline():
        x_final, val_final = _local_search(best_x, max_iters=2000)
        if val_final < best_val:
            best_val = val_final
            best_x = x_final.copy()
    
    return best_x
# EVOLVE-BLOCK-END
