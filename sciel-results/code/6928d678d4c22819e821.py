"""
Signed autocorrelation minimization via spectral shaping and fast delta local search.
Objective: minimize max|autoconv(h)| / (sum h)^2 over signed step heights h of length N.
Structural idea: use band-limited phase-mask spectral init (inverse DFT of random unit-amplitude
phases) to produce low-peak time-domain signals, then refine with incremental delta updates
to avoid full FFT recomputation at every step.
"""
import time
import numpy as np

N = 400
_DEADLINE = 128.0  # safety margin under 150s kill


def _fft_autoconv(h):
    """Linear autoconvolution via FFT."""
    n = len(h)
    size = 2 * n - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    C = np.fft.irfft(H * H, n=fft_size)
    return C[:size]


def _eval(h):
    """Return (value, c) where value = max|c| / (sum h)^2."""
    c = _fft_autoconv(h)
    m = float(np.max(np.abs(c)))
    s = float(np.sum(h))
    denom = s * s
    if denom <= 1e-12:
        denom = 1e-12
    return m / denom, c


def _golay_init():
    """Generate a Golay complementary pair and return the better truncated sequence."""
    a = np.array([1.0])
    b = np.array([1.0])
    while len(a) < N:
        a = np.concatenate([a, b])
        b = np.concatenate([a, -b])
    a = a[:N]
    b = b[:N]
    val_a, _ = _eval(a)
    val_b, _ = _eval(b)
    return a if val_a <= val_b else b


def _rudin_shapiro_init():
    """Generate a Rudin-Shapiro sequence of length >= N."""
    r = np.array([1.0])
    s = np.array([1.0])
    while len(r) < N:
        r, s = np.concatenate([r, s]), np.concatenate([r, -s])
    return r[:N]


def _spectral_phase_init(seed):
    """
    Spectral shaping: construct h as sign of inverse DFT of a random unit-amplitude
    phase mask (band-limited). This exploits the uncertainty principle to reduce
    time-domain peaks, which directly reduces autoconvolution sidelobes.
    """
    np.random.seed(seed)
    size = 1
    while size < 2 * N - 1:
        size *= 2
    half = size // 2 + 1
    # Random phases with unit amplitude
    phases = np.exp(1j * 2 * np.pi * np.random.rand(half))
    # Enforce Hermitian symmetry for real output
    H = np.zeros(size, dtype=np.complex128)
    H[:half] = phases
    H[half:] = np.conj(H[1:half][::-1])
    # Zero out DC to allow sign changes (optional; keep for now)
    # h_cont = np.fft.irfft(H, n=size)[:N]
    h_cont = np.fft.ifft(H, n=size).real[:N]
    h_spec = np.sign(h_cont)
    h_spec[h_spec == 0] = 1.0
    return h_spec


def _delta_local_search(h, c, max_sweeps=100, deadline=128.0):
    """
    Fast local search using incremental delta updates.
    Flipping h[idx] by d = ±2 changes c[L] by d*(h[idx+L] + h[idx-L]) for valid indices.
    This avoids full FFT recomputation, enabling many more iterations in the time budget.
    """
    start = time.time()
    h = h.copy()
    c = c.copy()
    s = float(np.sum(h))
    denom = s * s if abs(s) > 1e-12 else 1e-12
    current_val = float(np.max(np.abs(c))) / denom
    
    np.random.seed(123)
    Ls = np.arange(-(N - 1), N)
    zero_pos = N - 1
    
    for sweep in range(max_sweeps):
        if time.time() > deadline:
            break
        
        improved = False
        order = np.random.permutation(N)
        
        for idx in order:
            if time.time() > deadline:
                break
            
            old_val = h[idx]
            target = -old_val
            d = target - old_val  # ±2
            
            # Compute delta_c for all lags
            valid1 = (idx + Ls >= 0) & (idx + Ls < N)
            valid2 = (idx - Ls >= 0) & (idx - Ls < N)
            term1 = np.zeros(N * 2 - 1)
            term2 = np.zeros(N * 2 - 1)
            term1[valid1] = h[idx + Ls[valid1]]
            term2[valid2] = h[idx - Ls[valid2]]
            delta = d * (term1 + term2)
            delta[zero_pos] = 0.0  # L=0 autoconv term doesn't change for ±1
            
            new_c = c + delta
            new_s = s + d
            new_denom = new_s * new_s if abs(new_s) > 1e-12 else 1e-12
            new_val = float(np.max(np.abs(new_c))) / new_denom
            
            if new_val < current_val - 1e-15:
                h[idx] = target
                c = new_c
                s = new_s
                denom = new_denom
                current_val = new_val
                improved = True
            # else: revert (h unchanged since we didn't modify it yet)
        
        if not improved:
            break
    
    return h, current_val


def _simulated_annealing(h_init, c_init, max_steps=20000, deadline=128.0):
    """
    Simulated annealing with delta updates. Uses incremental c updates for speed.
    """
    start = time.time()
    h = h_init.copy()
    c = c_init.copy()
    s = float(np.sum(h))
    denom = s * s if abs(s) > 1e-12 else 1e-12
    current_val = float(np.max(np.abs(c))) / denom
    best_val = current_val
    best_h = h.copy()
    best_c = c.copy()
    
    T_start = 0.5
    T_end = 0.005
    np.random.seed(7)
    
    Ls = np.arange(-(N - 1), N)
    zero_pos = N - 1
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        progress = step / max_steps
        T = T_start * (T_end / T_start) ** progress
        
        idx = np.random.randint(N)
        old_val = h[idx]
        target = -old_val
        d = target - old_val
        
        valid1 = (idx + Ls >= 0) & (idx + Ls < N)
        valid2 = (idx - Ls >= 0) & (idx - Ls < N)
        term1 = np.zeros(N * 2 - 1)
        term2 = np.zeros(N * 2 - 1)
        term1[valid1] = h[idx + Ls[valid1]]
        term2[valid2] = h[idx - Ls[valid2]]
        delta = d * (term1 + term2)
        delta[zero_pos] = 0.0
        
        new_c = c + delta
        new_s = s + d
        new_denom = new_s * new_s if abs(new_s) > 1e-12 else 1e-12
        new_val = float(np.max(np.abs(new_c))) / new_denom
        
        if new_val <= current_val:
            h[idx] = target
            c = new_c
            s = new_s
            denom = new_denom
            current_val = new_val
            if new_val < best_val - 1e-15:
                best_val = new_val
                best_h = h.copy()
                best_c = c.copy()
        else:
            delta_val = new_val - current_val
            if np.random.random() < np.exp(-delta_val / (T + 1e-12)):
                h[idx] = target
                c = new_c
                s = new_s
                denom = new_denom
                current_val = new_val
            # else: revert
    
    return best_h, best_val, best_c


def construct():
    t_start = time.time()
    deadline = t_start + _DEADLINE
    
    best_h = None
    best_val = float('inf')
    best_c = None
    
    # Generate diverse initial candidates
    candidates = []
    
    # Golay
    h_golay = _golay_init()
    val_g, c_g = _eval(h_golay)
    candidates.append((val_g, h_golay.copy(), c_g.copy()))
    
    # Rudin-Shapiro
    h_rs = _rudin_shapiro_init()
    val_rs, c_rs = _eval(h_rs)
    candidates.append((val_rs, h_rs.copy(), c_rs.copy()))
    
    # Spectral phase inits with multiple seeds
    spectral_seeds = [42, 1, 7, 13, 17, 19, 23, 29, 31, 37, 41, 43]
    for seed in spectral_seeds:
        if time.time() > deadline - 40:
            break
        h_spec = _spectral_phase_init(seed)
        val_s, c_s = _eval(h_spec)
        candidates.append((val_s, h_spec.copy(), c_s.copy()))
    
    # Sort by initial value
    candidates.sort(key=lambda x: x[0])
    
    # Run local search on top candidates
    for val_init, h_init, c_init in candidates[:6]:
        if time.time() > deadline - 25:
            break
        h_opt, val_opt = _delta_local_search(h_init, c_init, max_sweeps=40, deadline=deadline - 10)
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt.copy()
            best_c = _fft_autoconv(h_opt)
    
    # Run simulated annealing on the best result
    if time.time() < deadline - 15 and best_h is not None:
        h_sa, val_sa, c_sa = _simulated_annealing(best_h.copy(), best_c.copy(), max_steps=15000, deadline=deadline - 5)
        if val_sa < best_val:
            best_val = val_sa
            best_h = h_sa
            best_c = c_sa
    
    # Final local search polish
    if time.time() < deadline - 3 and best_h is not None:
        h_final, val_final = _delta_local_search(best_h.copy(), best_c.copy(), max_sweeps=30, deadline=deadline)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final
    
    if best_h is None:
        best_h = np.ones(N)
    
    return best_h
