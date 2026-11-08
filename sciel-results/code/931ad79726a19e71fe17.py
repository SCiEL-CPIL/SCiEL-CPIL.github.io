"""
Signed autocorrelation minimization via simulated annealing on ±1 sequences.
Objective: minimize max|autoconv(h)| / (sum h)^2 over signed step heights h of length N.
Uses simulated annealing to escape local minima, with FFT-based evaluation.
"""
import time
import numpy as np

N = 400
_DEADLINE = 130.0  # safety margin under 150s kill


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
    """
    Generate a Golay complementary pair of length >= N and truncate to N.
    """
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
    """
    Generate a Rudin-Shapiro sequence of length >= N.
    """
    length = 1
    while length < N:
        length *= 2
    rs = np.zeros(length)
    rs[0] = 1.0
    rs[1] = 1.0
    for i in range(2, length):
        if i % 2 == 0:
            rs[i] = rs[i // 2]
        else:
            rs[i] = -rs[(i + 1) // 2]
    return rs[:N]


def _spectral_init(seed):
    """
    Spectral initialization: random phase in frequency domain, threshold to ±1.
    """
    np.random.seed(seed)
    size = 1
    while size < 2 * N - 1:
        size *= 2
    H = np.random.randn(size // 2 + 1) + 1j * np.random.randn(size // 2 + 1)
    H[0] = float(np.real(H[0]))
    if size > 1:
        H[size // 2] = float(np.real(H[size // 2]))
    h_cont = np.fft.irfft(H, n=size)[:N]
    h_spec = np.sign(h_cont)
    h_spec[h_spec == 0] = 1.0
    return h_spec


def _simulated_annealing(h_init, max_steps=50000, deadline=130.0):
    """
    Simulated annealing on ±1 sequences.
    Flip random bits, accept if it improves or with probability exp(-delta/T).
    """
    start = time.time()
    h = h_init.copy()
    current_val, current_c = _eval(h)
    best_val = current_val
    best_h = h.copy()
    
    # Temperature schedule
    T_start = 1.0
    T_end = 0.01
    T = T_start
    
    # Count steps to determine cooling rate
    # We'll do adaptive steps based on time
    np.random.seed(42)
    
    step_count = 0
    no_improve_count = 0
    
    while step_count < max_steps:
        if time.time() > deadline:
            break
        
        # Cool temperature
        progress = step_count / max_steps
        T = T_start * (T_end / T_start) ** progress
        
        # Pick random index to flip
        idx = np.random.randint(N)
        old_val = h[idx]
        h[idx] = -old_val
        
        # Evaluate new state
        new_val, new_c = _eval(h)
        
        # Accept if better, or with probability exp(-(new_val - current_val) / T)
        if new_val <= current_val:
            current_val = new_val
            current_c = new_c
            if new_val < best_val - 1e-15:
                best_val = new_val
                best_h = h.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            delta = new_val - current_val
            # Metropolis criterion
            if np.random.random() < np.exp(-delta / (T + 1e-12)):
                current_val = new_val
                current_c = new_c
                no_improve_count += 1
            else:
                # Revert
                h[idx] = old_val
                no_improve_count += 1
        
        step_count += 1
        
        # Early termination if stuck for too long
        if no_improve_count > 5000 and T < 0.05:
            break
    
    return best_h, best_val


def _local_search(h, c, max_sweeps=50, deadline=130.0):
    """
    Fast local search: greedily flip bits that reduce the objective.
    """
    start = time.time()
    h = h.copy()
    current_val, _ = _eval(h)
    best_val = current_val
    best_h = h.copy()
    
    np.random.seed(42)
    
    for sweep in range(max_sweeps):
        if time.time() > deadline:
            break
        
        improved = False
        order = np.random.permutation(N)
        
        for idx in order:
            if time.time() > deadline:
                break
            
            old_val = h[idx]
            h[idx] = -old_val
            new_val, _ = _eval(h)
            
            if new_val < current_val - 1e-15:
                current_val = new_val
                improved = True
                if new_val < best_val:
                    best_val = new_val
                    best_h = h.copy()
            else:
                h[idx] = old_val
        
        if not improved:
            break
    
    return best_h, best_val


def construct():
    t_start = time.time()
    deadline = t_start + _DEADLINE
    
    best_h = None
    best_val = float('inf')
    
    # Generate initial candidates
    candidates = []
    
    # Golay init
    h_golay = _golay_init()
    val_g, c_g = _eval(h_golay)
    candidates.append((val_g, h_golay))
    
    # Rudin-Shapiro init
    h_rs = _rudin_shapiro_init()
    val_rs, c_rs = _eval(h_rs)
    candidates.append((val_rs, h_rs))
    
    # Multiple spectral inits
    for seed in [42, 7, 123, 456, 789, 1000, 2000, 3000, 4000, 5000]:
        if time.time() > deadline - 30:
            break
        h_spec = _spectral_init(seed)
        val_s, c_s = _eval(h_spec)
        candidates.append((val_s, h_spec))
    
    # Sort by initial value
    candidates.sort(key=lambda x: x[0])
    
    # Run local search on top candidates first (fast)
    for val_init, h_init in candidates[:5]:
        if time.time() > deadline - 20:
            break
        h_copy = h_init.copy()
        h_opt, val_opt = _local_search(h_copy, None, max_sweeps=30, deadline=deadline - 5)
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt.copy()
    
    # Run simulated annealing on the best local search result
    if time.time() < deadline - 10 and best_h is not None:
        h_sa, val_sa = _simulated_annealing(best_h.copy(), max_steps=30000, deadline=deadline - 3)
        if val_sa < best_val:
            best_val = val_sa
            best_h = h_sa
    
    # Final local search polish
    if time.time() < deadline - 2 and best_h is not None:
        h_final, val_final = _local_search(best_h.copy(), None, max_sweeps=20, deadline=deadline)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final
    
    if best_h is None:
        best_h = np.ones(N)
    
    return best_h
