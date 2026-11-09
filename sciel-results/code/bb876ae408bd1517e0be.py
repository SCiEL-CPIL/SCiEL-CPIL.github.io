import numpy as np
from scipy.optimize import minimize
import time

N = 50
_DEADLINE = 140.0  # safety margin before 150s kill
_start_time = None

def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _ratio(h):
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-12):
        return -1.0
    h = np.clip(h, 0, None)
    l2_sq = np.sum(h * h)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return -1.0
    # autoconvolution via FFT for speed
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    ac = np.fft.irfft(H * H.conj(), n=n_fft)[:N]
    # full autoconvolution includes negative lags too; use full convolution
    ac_full = np.fft.irfft(H * H.conj(), n=n_fft)
    sup_ac = np.max(np.abs(ac_full))
    if sup_ac < 1e-15:
        return -1.0
    return l2_sq / (l1 * sup_ac)

def _neg_ratio(h):
    r = _ratio(h)
    if r < 0:
        return 1.0
    return -r

def construct():
    global _start_time
    _start_time = time.time()
    
    best_h = np.ones(N)
    best_val = _ratio(best_h)
    
    # Try several initial shapes
    initial_guesses = [
        np.ones(N),
        np.exp(-0.5 * (np.arange(N) - N//2)**2 / (N//6)**2),  # Gaussian
        np.ones(N) * (1 + 0.5 * np.cos(2*np.pi*np.arange(N)/N)),  # modulated
        np.array([1.0]*10 + [0.1]*30 + [1.0]*10),  # two peaks
        np.array([0.0]*5 + [1.0]*40 + [0.0]*5),  # plateau
    ]
    
    for guess in initial_guesses:
        if _elapsed() > _DEADLINE - 20:
            break
        guess = np.clip(guess, 0, None)
        if np.sum(guess) < 1e-10:
            continue
        # Normalize to avoid scale issues (ratio is scale-invariant)
        guess = guess / np.sum(guess)
        
        # Use Nelder-Mead for robustness
        res = minimize(_neg_ratio, guess, method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10,
                                'adaptive': True})
        if _elapsed() > _DEADLINE - 5:
            break
        h_opt = np.clip(res.x, 0, None)
        val = _ratio(h_opt)
        if val > best_val:
            best_val = val
            best_h = h_opt
    
    # Try a few more random restarts if time permits
    rng = np.random.default_rng(42)
    for _ in range(20):
        if _elapsed() > _DEADLINE - 5:
            break
        # Random log-normal-ish shapes
        base = rng.random(N)
        base = base ** rng.uniform(0.5, 3.0)
        base = np.clip(base, 0, None)
        if np.sum(base) < 1e-10:
            continue
        base = base / np.sum(base)
        res = minimize(_neg_ratio, base, method='Nelder-Mead',
                       options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10,
                                'adaptive': True})
        h_opt = np.clip(res.x, 0, None)
        val = _ratio(h_opt)
        if val > best_val:
            best_val = val
            best_h = h_opt
    
    return list(best_h)
