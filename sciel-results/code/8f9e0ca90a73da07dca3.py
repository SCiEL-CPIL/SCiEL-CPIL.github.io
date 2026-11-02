"""Signed autocorrelation minimization via spectral PSD shaping + phase retrieval."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint

N = 400
DEADLINE = 115.0  # safety margin before 120s kill

def _check_deadline():
    return time.time() > DEADLINE

def _autoconv_fft(f):
    """Full autoconvolution via FFT."""
    n = len(f)
    size = 2 * n
    F = np.fft.rfft(f, size=size)
    P = F * np.conj(F)
    ac = np.fft.irfft(P, size=size)
    return ac[:n]  # lags 0..n-1 (positive lags only, ac[0] is full sum^2)

def _score(f):
    """Return max |autoconv[k]| for k>=1 divided by (sum f)^2, or inf if sum is 0."""
    s = np.sum(f)
    if abs(s) < 1e-9:
        return np.inf
    ac = _autoconv_fft(f)
    # Exclude lag 0 (which is sum(f_i*f_i) = N for ±1, or ||f||^2)
    # Actually autoconv[0] = sum f_i^2 = N for ±1 signs
    # We want max over k=1..N-1 of |ac[k]|
    peak = np.max(np.abs(ac[1:]))
    return peak / (s * s)

def _round_to_signed(f):
    """Map continuous f to ±1 signs."""
    return np.where(f >= 0, 1.0, -1.0)

def _random_signs(rng):
    return rng.choice([-1.0, 1.0], size=N)

def _spectral_init(rng, num_iter=30):
    """
    Initialize via spectral shaping: start with random PSD, 
    use iterative amplitude/phase projection to reduce autoconv peak.
    """
    f = _random_signs(rng)
    best_score = _score(f)
    
    for _ in range(num_iter):
        if _check_deadline():
            break
        # Compute current PSD
        F = np.fft.rfft(f, size=2*N)
        # Perturb phase randomly
        phase = np.angle(F)
        phase += rng.normal(0, 0.5, size=len(F))
        new_F = np.abs(F) * np.exp(1j * phase)
        # Inverse FFT and round
        new_f = np.fft.irfft(new_F, size=2*N)[:N]
        new_f = _round_to_signed(new_f)
        new_score = _score(new_f)
        if new_score < best_score:
            best_score = new_score
            f = new_f
    return f, best_score

def _coordinate_descent(f, max_iter=500, rng=None):
    """
    Coordinate descent on signs: try flipping each element, accept if score improves.
    Uses approximate gradient via finite differences on a subset.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    current_score = _score(f)
    f = f.copy()
    
    # Precompute ac for efficiency
    ac = _autoconv_fft(f)
    s = np.sum(f)
    
    # We'll do random coordinate descent
    order = rng.permutation(N)
    for idx in range(min(max_iter, N * 5)):
        if _check_deadline():
            break
        i = order[idx % N]
        # Try flipping f[i]
        f[i] *= -1
        new_ac = _autoconv_fft(f)
        new_s = np.sum(f)
        if abs(new_s) < 1e-9:
            new_score = np.inf
        else:
            new_score = np.max(np.abs(new_ac[1:])) / (new_s * new_s)
        
        if new_score < current_score:
            current_score = new_score
            ac = new_ac
            s = new_s
        else:
            f[i] *= -1  # revert
    
    return f, current_score

def _psd_optimization(rng):
    """
    Optimize the power spectral density to minimize peak autoconvolution.
    The autoconvolution is the inverse FFT of |F|^2.
    We want to minimize max_k |IFFT(|F|^2)[k]| for k>=1.
    
    This is non-convex in general, but we can use a greedy/iterative approach:
    - Start with flat PSD (random signs)
    - Iteratively adjust PSD to reduce the largest autoconv peak
    
    Since |F|^2 >= 0 and sum |F|^2 = N^2 (Parseval for ±1), 
    we can work in the PSD domain.
    """
    # Start with random signs
    f = _random_signs(rng)
    best_score = _score(f)
    best_f = f.copy()
    
    num_rounds = 50
    for r in range(num_rounds):
        if _check_deadline():
            break
        
        F = np.fft.rfft(f, size=2*N)
        psd = np.abs(F) ** 2
        
        # Find the lag with maximum |autoconv|
        ac = _autoconv_fft(f)
        lag_idx = np.argmax(np.abs(ac[1:])) + 1
        target_lag = lag_idx
        
        # The autoconv at lag k is related to the PSD via:
        # ac[k] = sum_j psd[j] * exp(2*pi*i*j*k/N') 
        # where N' is the FFT size
        # We want to reduce |ac[target_lag]|
        
        # Greedy: reduce PSD at frequencies that constructively interfere at target_lag
        N_fft = 2 * N
        freqs = np.arange(len(F))
        # Contribution to ac[k] from frequency j: psd[j] * cos(2*pi*j*k/N_fft)
        # (since ac is real, imaginary parts cancel)
        phase_term = np.cos(2 * np.pi * freqs * target_lag / N_fft)
        
        # We want to reduce the total. Reduce psd where phase_term has same sign as ac[target_lag]
        sign_ac = np.sign(ac[target_lag])
        
        # Scale down PSD where it contributes constructively
        reduction_mask = (sign_ac * phase_term) > 0
        psd_new = psd.copy()
        # Reduce by 10% at those frequencies
        psd_new[reduction_mask] *= 0.9
        # Normalize to maintain total power
        if np.sum(psd_new) > 0:
            psd_new *= np.sum(psd) / np.sum(psd_new)
        
        # New phase: keep original phase (or randomize slightly)
        new_F = np.sqrt(psd_new) * np.exp(1j * np.angle(F))
        new_f = np.fft.irfft(new_F, size=N_fft)[:N]
        new_f = _round_to_signed(new_f)
        
        new_score = _score(new_f)
        if new_score < best_score:
            best_score = new_score
            best_f = new_f.copy()
            f = new_f.copy()
    
    return best_f, best_score

def _l_bfgs_refine(f0, rng):
    """
    Use L-BFGS on a relaxed version: minimize max_k |ac[k]| subject to ||f||^2 = N.
    Then project back to ±1.
    """
    def objective(x):
        f = np.tanh(x * 3)  # smooth ±1 constraint
        ac = _autoconv_fft(f)
        peak = np.max(np.abs(ac[1:]))
        return peak
    
    x0 = f0 * 0.5  # start near ±1 in tanh space
    result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'gtol': 1e-4})
    
    f_refined = _round_to_signed(np.tanh(result.x * 3))
    return f_refined

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_f = None
    best_score = np.inf
    
    # Phase 1: Spectral initialization with multiple restarts
    for restart in range(8):
        if _check_deadline():
            break
        f, score = _spectral_init(rng, num_iter=20)
        if score < best_score:
            best_score = score
            best_f = f.copy()
    
    # Phase 2: PSD optimization from best start
    if best_f is not None and not _check_deadline():
        f_psd, score_psd = _psd_optimization(rng)
        if score_psd < best_score:
            best_score = score_psd
            best_f = f_psd.copy()
    
    # Phase 3: Coordinate descent refinement
    if best_f is not None and not _check_deadline():
        f_cd, score_cd = _coordinate_descent(best_f, max_iter=N * 3, rng=rng)
        if score_cd < best_score:
            best_score = score_cd
            best_f = f_cd.copy()
    
    # Phase 4: More coordinate descent iterations
    if best_f is not None and not _check_deadline():
        f_cd2, score_cd2 = _coordinate_descent(best_f, max_iter=N * 2, rng=rng)
        if score_cd2 < best_score:
            best_score = score_cd2
            best_f = f_cd2.copy()
    
    # Final: ensure we return ±1
    if best_f is None:
        best_f = np.ones(N)
    
    return best_f
