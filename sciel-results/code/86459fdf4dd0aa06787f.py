import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import remez

N = 400
N_FFT = 2 * N - 1

def _autoconv(f):
    """Computes the full linear autoconvolution of f using FFT."""
    F = np.fft.rfft(f, n=N_FFT)
    return np.fft.irfft(F * F, n=N_FFT)

def _soft_max_abs_ac(f, beta=50.0):
    """
    Smooth approximation of max |ac| using log-sum-exp.
    Returns (value, gradient) using vectorized computation.
    """
    F = np.fft.rfft(f, n=N_FFT)
    ac = np.fft.irfft(F * F, n=N_FFT)
    abs_ac = np.abs(ac)
    max_ac = np.max(abs_ac)
    if max_ac < 1e-30:
        return max_ac, np.zeros_like(f)
    
    shifted = beta * (abs_ac - max_ac)
    exp_shifted = np.exp(shifted)
    log_sum = np.log(np.sum(exp_shifted)) / beta + max_ac
    
    weights = exp_shifted / np.sum(exp_shifted)
    signs = np.sign(ac)
    
    g = weights * signs
    grad = np.array([2.0 * np.dot(g[i:i+N], f) for i in range(N)])
    
    return log_sum, grad

def _objective_and_grad(f):
    """
    Objective: log(soft_max_abs_ac) - 2*log(|sum f|)
    This is scale invariant and smooth.
    """
    smax, grad_smax = _soft_max_abs_ac(f)
    s = np.sum(f)
    
    if s == 0 or smax < 1e-30:
        return 1e10, np.ones(N) * 1e-6
    
    g1 = grad_smax / smax
    g2 = -2.0 * np.ones(N) / s
    
    obj = np.log(smax) - 2.0 * np.log(np.abs(s))
    
    return obj, g1 + g2

def _score(f):
    """Compute the actual C3' score."""
    ac = _autoconv(f)
    s = np.sum(f)
    if abs(s) < 1e-12:
        return np.inf
    return np.max(np.abs(ac)) / (s * s)

def _greedy_peak_shave(f, max_iters=2000, deadline=None):
    """
    Greedy peak shaving: iteratively flip the sign of the element most
    correlated with the current maximum autoconvolution peak.
    """
    best = f.copy()
    best_score = _score(best)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        ac = _autoconv(best)
        abs_ac = np.abs(ac)
        peak_idx = np.argmax(abs_ac)
        peak_val = ac[peak_idx]
        
        valid_i = []
        if peak_idx < N:
            i_start = max(0, peak_idx - (N - 1))
            i_end = min(N - 1, peak_idx)
            for i in range(i_start, i_end + 1):
                valid_i.append(i)
        else:
            i_start = max(0, peak_idx - N + 1)
            i_end = min(N - 1, peak_idx)
            for i in range(i_start, i_end + 1):
                valid_i.append(i)
        
        if not valid_i:
            break
        
        valid_i = np.array(valid_i)
        k_vals = peak_idx - valid_i
        
        products = best[valid_i] * best[k_vals]
        
        same_sign_mask = np.sign(products) == np.sign(peak_val)
        
        if not np.any(same_sign_mask):
            candidate = best.copy()
            idx = np.random.randint(N)
            candidate[idx] *= -1
            sc = _score(candidate)
            if sc < best_score:
                best = candidate
                best_score = sc
            continue
        
        candidates_idx = valid_i[same_sign_mask]
        candidates_prod = products[same_sign_mask]
        
        best_local = np.argmax(np.abs(candidates_prod))
        flip_idx = candidates_idx[best_local]
        
        candidate = best.copy()
        candidate[flip_idx] *= -1
        sc = _score(candidate)
        
        if sc < best_score:
            best = candidate
            best_score = sc
        else:
            if len(candidates_idx) > 1:
                order = np.argsort(np.abs(candidates_prod))[::-1]
                for trial in range(min(5, len(order))):
                    fi = candidates_idx[order[trial]]
                    candidate = best.copy()
                    candidate[fi] *= -1
                    sc = _score(candidate)
                    if sc < best_score:
                        best = candidate
                        best_score = sc
                        break
    
    return best, best_score

def _spectral_barker_init(seed=0):
    """
    Barker-like initialization: inverse FFT of phase-only mask with flat magnitude.
    Then apply greedy peak shaving.
    """
    rng = np.random.RandomState(seed)
    n_freqs = N_FFT // 2 + 1
    
    phases = rng.uniform(0, 2 * np.pi, size=n_freqs)
    mags = np.ones(n_freqs)
    mags[0] = 0
    
    F = mags * np.exp(1j * phases)
    F_full = np.zeros(N_FFT, dtype=complex)
    F_full[:n_freqs] = F
    if n_freqs > 1:
        F_full[n_freqs:] = np.conj(F[1:n_freqs][::-1])
    
    f = np.fft.irfft(F_full, n=N_FFT)[:N]
    
    s = np.sum(f)
    if s < 0:
        f = -f
    elif abs(s) < 1e-12:
        f = rng.randn(N)
    
    f, _ = _greedy_peak_shave(f, max_iters=100)
    
    return f

def _remez_init(n_taps, num_ri, num_rl, seed=0, deadline=None):
    """
    Generate equiripple FIR filter coefficients using Parks-McClellan (remez).
    Design a lowpass filter whose magnitude response is flat (equiripple),
    then use the time-domain coefficients as initial signed step heights.
    The equiripple property distributes energy uniformly across lags.
    """
    if deadline is not None and time.time() > deadline:
        return None
    
    rng = np.random.RandomState(seed)
    
    # Design various equiripple filters as diverse initializations
    # num_ri = number of ripple intervals (odd for type I FIR)
    # We want the impulse response to have mixed signs (signed step function)
    
    # Strategy: design a bandstop or notch filter to get oscillatory impulse response
    # A bandstop filter has a non-monotonic frequency response, leading to oscillatory h[n]
    
    candidates = []
    
    # Type 1: Lowpass with specific cutoff to create oscillatory tail
    try:
        # remez requires odd number of taps for type I
        n_taps_odd = n_taps if n_taps % 2 == 1 else n_taps + 1
        # Adjust to N
        if n_taps_odd > N:
            n_taps_odd = N if N % 2 == 1 else N - 1
        
        # Design a bandstop filter: [0, f1] pass, [f1, f2] stop, [f2, 1] pass
        # This creates oscillatory impulse response
        f1 = 0.15 + 0.05 * (seed % 5)  # Vary the stopband location
        f2 = f1 + 0.1 + 0.03 * (seed % 7)
        if f2 > 0.85:
            f2 = 0.85
        
        # remez: bands in [0, 0.5] normalized frequency
        # weights: equal weight to all bands
        bands = [0.0, f1, f2, 0.5]
        desired = [1.0, 0.0, 1.0]  # Pass, Stop, Pass
        weights = [1.0, 1.0, 1.0]
        
        h = remez(n_taps_odd - 1, bands, desired, weights)
        
        # Pad or trim to length N
        if len(h) < N:
            h = np.pad(h, (0, N - len(h)))
        else:
            h = h[:N]
        
        # Ensure non-zero sum
        s = np.sum(h)
        if abs(s) < 1e-12:
            h += rng.randn(N) * 0.01
        elif s < 0:
            h = -h
        
        candidates.append(h)
    except Exception:
        pass
    
    # Type 2: Lowpass with sharp cutoff
    try:
        n_taps_odd = N if N % 2 == 1 else N - 1
        cutoff = 0.2 + 0.04 * (seed % 10)
        h = remez(n_taps_odd - 1, [0.0, cutoff], [1.0, 0.0])
        if len(h) < N:
            h = np.pad(h, (0, N - len(h)))
        else:
            h = h[:N]
        
        s = np.sum(h)
        if abs(s) < 1e-12:
            h += rng.randn(N) * 0.01
        elif s < 0:
            h = -h
        
        candidates.append(h)
    except Exception:
        pass
    
    # Type 3: Highpass filter (oscillatory response)
    try:
        n_taps_odd = N if N % 2 == 1 else N - 1
        cutoff = 0.2 + 0.03 * (seed % 8)
        h = remez(n_taps_odd - 1, [0.0, cutoff], [0.0, 1.0])
        if len(h) < N:
            h = np.pad(h, (0, N - len(h)))
        else:
            h = h[:N]
        
        s = np.sum(h)
        if abs(s) < 1e-12:
            h += rng.randn(N) * 0.01
        elif s < 0:
            h = -h
        
        candidates.append(h)
    except Exception:
        pass
    
    if not candidates:
        return None
    
    # Pick the candidate with best initial score
    best_h = None
    best_sc = np.inf
    for h in candidates:
        sc = _score(h)
        if sc < best_sc:
            best_sc = sc
            best_h = h
    
    return best_h

def _local_search(f, deadline, max_iters=50000):
    """Local search: try flipping signs and small perturbations"""
    best = f.copy()
    best_score = _score(best)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Try random sign flips
        n_flips = np.random.randint(1, 5)
        indices = np.random.choice(N, n_flips, replace=False)
        candidate = best.copy()
        candidate[indices] *= -1
        
        sc = _score(candidate)
        if sc < best_score:
            best = candidate
            best_score = sc
        
        # Try single flip of the most correlated element with peak
        if it % 5 == 0:
            ac = _autoconv(best)
            abs_ac = np.abs(ac)
            peak_idx = np.argmax(abs_ac)
            peak_val = ac[peak_idx]
            
            if peak_idx < N:
                i_start = max(0, peak_idx - (N - 1))
                i_end = min(N - 1, peak_idx)
            else:
                i_start = max(0, peak_idx - N + 1)
                i_end = min(N - 1, peak_idx)
            
            if i_start <= i_end:
                valid_i = np.arange(i_start, i_end + 1)
                k_vals = peak_idx - valid_i
                products = best[valid_i] * best[k_vals]
                same_sign_mask = np.sign(products) == np.sign(peak_val)
                
                if np.any(same_sign_mask):
                    candidates_idx = valid_i[same_sign_mask]
                    candidates_prod = products[same_sign_mask]
                    best_local = np.argmax(np.abs(candidates_prod))
                    flip_idx = candidates_idx[best_local]
                    
                    candidate = best.copy()
                    candidate[flip_idx] *= -1
                    sc = _score(candidate)
                    if sc < best_score:
                        best = candidate
                        best_score = sc
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin (5s buffer)
    
    best_f = None
    best_score = np.inf
    
    np.random.seed(42)
    
    # Phase 1: Parks-McClellan (remez) equiripple initializations
    # These generate oscillatory impulse responses with flat magnitude response,
    # distributing autoconvolution energy more uniformly across lags
    n_remez = 12
    for r in range(n_remez):
        if time.time() > deadline - 40.0:
            break
        f0 = _remez_init(N, 1, 1, seed=r * 131 + 7, deadline=deadline - 40.0)
        if f0 is not None:
            sc = _score(f0)
            if sc < best_score:
                best_f = f0.copy()
                best_score = sc
    
    # Phase 1b: Spectral Barker initializations with greedy shaving
    n_spectral = 6
    for r in range(n_spectral):
        if time.time() > deadline - 35.0:
            break
        f0 = _spectral_barker_init(seed=r * 137 + 1)
        sc = _score(f0)
        if sc < best_score:
            best_f = f0.copy()
            best_score = sc
    
    # Phase 2: Gradient optimization from best and other starts
    starts = []
    if best_f is not None:
        starts.append(best_f.copy())
    
    # Add diverse starts including remez-based
    starts.append(np.ones(N))
    starts.append(np.random.randn(N))
    starts.append(np.random.choice([-1.0, 1.0], size=N))
    s = np.sum(starts[2])
    if s < 0:
        starts[2] = -starts[2]
    
    t = np.linspace(0, 10, N)
    starts.append(np.cos(t**2))
    
    starts.append(np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]))
    
    # Add a remez-based start if available
    f_remez_extra = _remez_init(N, 1, 1, seed=999, deadline=deadline - 25.0)
    if f_remez_extra is not None:
        starts.append(f_remez_extra)
    
    for f0 in starts:
        if time.time() > deadline - 20.0:
            break
        
        try:
            res = minimize(
                lambda x: _objective_and_grad(x)[0],
                f0,
                jac=lambda x: _objective_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 30}
            )
            
            candidate = res.x
            sc = _score(candidate)
            if sc < best_score:
                best_f = candidate.copy()
                best_score = sc
                
        except Exception:
            pass
    
    # Phase 3: Greedy peak shaving polish
    if best_f is not None and time.time() < deadline - 10.0:
        polished, polished_score = _greedy_peak_shave(best_f, max_iters=5000, deadline=deadline - 10.0)
        if polished_score < best_score:
            best_f = polished
            best_score = polished_score
    
    # Phase 4: Local search polish
    if best_f is not None and time.time() < deadline - 3.0:
        polished, polished_score = _local_search(best_f, deadline - 3.0, max_iters=30000)
        if polished_score < best_score:
            best_f = polished
            best_score = polished_score
    
    # Final: try a few more remez inits with remaining time
    if time.time() < deadline - 5.0:
        for r in range(4):
            if time.time() > deadline - 5.0:
                break
            f0 = _remez_init(N, 1, 1, seed=2000 + r * 89, deadline=deadline - 5.0)
            if f0 is not None:
                sc = _score(f0)
                if sc < best_score:
                    best_f = f0.copy()
                    best_score = sc
                    # Quick polish
                    if time.time() < deadline - 3.0:
                        polished, polished_score = _greedy_peak_shave(best_f, max_iters=1000, deadline=deadline - 3.0)
                        if polished_score < best_score:
                            best_f = polished
                            best_score = polished_score
    
    if best_f is None:
        best_f = np.ones(N)
        
    return best_f
