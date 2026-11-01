import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _autocorr_fft(h):
    """Compute full autocorrelation via FFT."""
    n = len(h)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    H = np.fft.rfft(h, size)
    ac = np.fft.irfft(H * np.conj(H), size)[:2 * n - 1]
    return ac

def _calc_score(h):
    """
    Score = max_{k != 0} |(h*h)_k| / ||h||_2^2
    """
    corr = _autocorr_fft(h)
    # corr[0] is lag -(N-1), corr[N-1] is lag 0, corr[2N-2] is lag N-1
    mask = np.ones(len(corr), dtype=bool)
    mask[N - 1] = False
    peak = np.max(np.abs(corr[mask]))
    norm = np.sum(h ** 2)
    if norm == 0:
        return 0.0
    return peak / norm

def _continuous_objective(x):
    """Objective for continuous optimization: peak sidelobe of continuous h."""
    # x is in [-1, 1] range, reshape to N
    h = x
    corr = _autocorr_fft(h)
    mask = np.ones(len(corr), dtype=bool)
    mask[N - 1] = False
    peak = np.max(np.abs(corr[mask]))
    norm = np.sum(h ** 2) + 1e-10
    return peak / norm

def _continuous_gradient(x):
    """
    Gradient of the peak sidelobe objective.
    We use a soft-max approximation: f(x) = (1/beta) * log(sum exp(beta * |corr_k|)) for k != 0
    But for simplicity and speed, we just do a finite-difference gradient if needed,
    or use the fact that the gradient of max |corr_k| w.r.t. h is tricky.
    
    Actually, L-BFGS-B can do finite differences. Let's just provide the function.
    """
    # We'll let scipy do finite differences, but we can provide a gradient for speed.
    # Let's use a simple approach: the gradient of the peak is the derivative of the max component.
    # This is non-smooth. A better approach is to use a differentiable proxy.
    # Let's stick with finite differences for now, or a simple gradient.
    # For speed, let's compute the gradient of the sum of squares of sidelobes as a proxy,
    # or just let scipy handle it.
    pass

def _discretize(h):
    """Convert continuous h to +/- 1."""
    return np.where(h >= 0, 1.0, -1.0)

def _hill_climb(h, deadline):
    """Fast FFT-based hill climbing with single and block flips."""
    h = h.copy()
    current_score = _calc_score(h)
    best_h = h.copy()
    best_score = current_score
    
    # Precompute correlation for fast updates? 
    # For single flip at j:
    # corr_new[k] = corr_old[k] - 2*h[j]*h[j+k] + 2*h[j]*h[j+k] ... wait.
    # h[j] flips from v to -v.
    # corr[k] = sum_i h[i]h[i+k]
    # The terms involving h[j] are h[j]h[j+k] and h[j-k]h[j].
    # This is complex to update incrementally for all k.
    # FFT is fast enough. N=400. _autocorr_fft is ~10us.
    # 120s / 10us = 12 million flips. Plenty.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try all single flips
        for j in range(N):
            if time.time() > deadline:
                break
            old_val = h[j]
            h[j] = -h[j]
            new_score = _calc_score(h)
            if new_score < current_score:
                current_score = new_score
                improved = True
                if new_score < best_score:
                    best_score = new_score
                    best_h = h.copy()
            else:
                h[j] = old_val
        
        # Try some random block flips to escape local minima
        for _ in range(10):
            if time.time() > deadline:
                break
            block_size = np.random.choice([2, 4, 8, 16])
            start = np.random.randint(0, N - block_size + 1)
            h[start:start+block_size] *= -1
            new_score = _calc_score(h)
            if new_score < current_score:
                current_score = new_score
                improved = True
                if new_score < best_score:
                    best_score = new_score
                    best_h = h.copy()
            else:
                h[start:start+block_size] *= -1  # revert
    
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_h = None
    best_score = float('inf')
    
    # Phase 1: Continuous relaxation with L-BFGS-B
    # Start from a few random initializations
    n_init = 5
    for i in range(n_init):
        if time.time() > deadline - 30:
            break
        
        # Random initial point in [-1, 1]
        x0 = np.random.uniform(-1, 1, N)
        
        # Optimize
        try:
            res = minimize(
                _continuous_objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-6, 'maxls': 50}
            )
            x_opt = res.x
            h_disc = _discretize(x_opt)
            score = _calc_score(h_disc)
            
            if score < best_score:
                best_score = score
                best_h = h_disc.copy()
        except Exception:
            continue
    
    # Phase 2: If no good result, try deterministic sequences
    if best_h is None or best_score > 0.01:
        # Van der Corput
        def _van_der_corput(n, base=2):
            seq = []
            for i in range(1, n + 1):
                f = 0.0
                d = 1.0
                temp = i
                while temp > 0:
                    temp, rem = divmod(temp, base)
                    d /= base
                    f += rem * d
                seq.append(f)
            return np.array(seq)
        
        vdc = _van_der_corput(N)
        med = np.median(vdc)
        h = np.where(vdc > med, 1, -1).astype(float)
        score = _calc_score(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
        
        # Random sequences
        for _ in range(10):
            if time.time() > deadline - 20:
                break
            h = np.random.choice([-1, 1], size=N).astype(float)
            score = _calc_score(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
        
        if best_h is None:
            best_h = np.ones(N)
            best_score = _calc_score(best_h)
    
    # Phase 3: Hill climbing refinement
    if time.time() < deadline - 5:
        best_h, best_score = _hill_climb(best_h, deadline - 2)
    
    return best_h
