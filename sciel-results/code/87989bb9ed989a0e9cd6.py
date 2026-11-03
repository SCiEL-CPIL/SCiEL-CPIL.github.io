"""
Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Uses simulated annealing with constraint-aware moves and spectral initialization.
"""
import time
import numpy as np

N = 100
_T_START = time.time()
_DEADLINE = _T_START + 145


def _linear_overlap_values(h):
    """Compute C_k = (2/N)*(N/2 - r_k) where r_k = sum_{i=k}^{N-1} h_i h_{i-k}."""
    n = len(h)
    n_fft = 1
    while n_fft < 2 * n:
        n_fft *= 2
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _project(h, target_sum):
    """
    Project h onto the set {x : 0 <= x_i <= 1, sum(x) = target_sum}.
    Uses bisection on the Lagrange multiplier for the sum constraint.
    """
    h = np.asarray(h, dtype=np.float64)
    low = -np.max(h) - 1.0
    high = 1.0 - np.min(h) + 1.0
    
    current_sum = np.sum(h)
    if abs(current_sum - target_sum) < 1e-12 and np.all(h >= -1e-12) and np.all(h <= 1 + 1e-12):
        return np.clip(h, 0.0, 1.0)

    def sum_of_clipped(offset):
        return np.sum(np.clip(h + offset, 0.0, 1.0))

    for _ in range(60):
        mid = (low + high) / 2.0
        s_mid = sum_of_clipped(mid)
        if s_mid > target_sum:
            high = mid
        else:
            low = mid
    
    offset = (low + high) / 2.0
    h_proj = np.clip(h + offset, 0.0, 1.0)
    
    diff = target_sum - np.sum(h_proj)
    if abs(diff) > 1e-14:
        if diff > 0:
            idx = np.argmin(h_proj)
            h_proj[idx] += diff
        else:
            idx = np.argmax(h_proj)
            h_proj[idx] += diff
        h_proj = np.clip(h_proj, 0.0, 1.0)
        
    return h_proj


def _spectral_init(frequencies, amplitudes, phase_seed):
    """
    Generate an initial h based on spectral components.
    frequencies: array of frequency indices
    amplitudes: array of amplitudes
    phase_seed: random seed for phases
    """
    rng = np.random.RandomState(phase_seed)
    h = np.zeros(N)
    for freq, amp in zip(frequencies, amplitudes):
        phase = rng.uniform(0, 2 * np.pi)
        h += amp * np.cos(2 * np.pi * freq * np.arange(N) / N + phase)
    h = np.clip(h, 0.0, 1.0)
    h = _project(h, N / 2.0)
    return h


def _simulated_annealing(h0, t_start, t_end, n_iter, seed=42):
    """
    Simulated annealing with constraint-aware moves.
    """
    rng = np.random.RandomState(seed)
    h = np.array(h0, dtype=np.float64)
    h = _project(h, N / 2.0)
    
    best_h = h.copy()
    best_score = _score(h)
    current_score = best_score
    
    # Temperature schedule: geometric cooling
    cooling_rate = (t_end / t_start) ** (1.0 / n_iter)
    
    for i in range(n_iter):
        if time.time() > _DEADLINE - 5:
            break
        
        temp = t_start * (cooling_rate ** i)
        
        # Choose move type
        move_type = rng.random()
        
        if move_type < 0.3:
            # Single coordinate adjustment
            idx = rng.randint(N)
            delta = rng.uniform(-0.1, 0.1)
            h_new = h.copy()
            h_new[idx] += delta
            h_new = _project(h_new, N / 2.0)
            
        elif move_type < 0.6:
            # Pair swap: adjust two coordinates to maintain sum
            idx1, idx2 = rng.choice(N, size=2, replace=False)
            delta = rng.uniform(-0.05, 0.05)
            h_new = h.copy()
            h_new[idx1] += delta
            h_new[idx2] -= delta
            h_new = _project(h_new, N / 2.0)
            
        elif move_type < 0.85:
            # Local neighborhood: adjust a small window
            center = rng.randint(N)
            window = 3
            start = max(0, center - window)
            end = min(N, center + window + 1)
            deltas = rng.uniform(-0.05, 0.05, size=end - start)
            h_new = h.copy()
            h_new[start:end] += deltas
            h_new = _project(h_new, N / 2.0)
            
        else:
            # Large jump: random perturbation
            mask = rng.random(N) < 0.1
            h_new = h.copy()
            h_new[mask] += rng.uniform(-0.2, 0.2, size=np.sum(mask))
            h_new = _project(h_new, N / 2.0)
        
        new_score = _score(h_new)
        
        # Acceptance criterion
        if new_score < current_score:
            h = h_new
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = h_new.copy()
        else:
            # Accept with probability exp(-(new_score - current_score) / temp)
            delta_e = new_score - current_score
            if delta_e < 100 * temp:  # Avoid overflow
                prob = np.exp(-delta_e / temp)
                if rng.random() < prob:
                    h = h_new
                    current_score = new_score
    
    return best_h, best_score


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    best_h = None
    best_score = float('inf')
    
    np.random.seed(42)
    
    # Phase 1: Spectral initialization with diverse frequency content
    # The overlap structure suggests that certain frequency patterns may be beneficial
    # Try different spectral compositions
    
    spectral_configs = [
        # Low frequency dominant
        ([1, 2, 3], [0.2, 0.15, 0.1], 42),
        # Medium frequency
        ([5, 7, 9], [0.15, 0.1, 0.08], 43),
        # High frequency
        ([10, 15, 20], [0.1, 0.08, 0.05], 44),
        # Mixed
        ([2, 8, 16], [0.15, 0.12, 0.08], 45),
        # Near-constant with small perturbation
        ([1, 5], [0.1, 0.05], 46),
    ]
    
    inits = []
    for freqs, amps, seed in spectral_configs:
        h_init = _spectral_init(freqs, amps, seed)
        inits.append(h_init)
    
    # Add some random inits
    for seed in range(5):
        h_init = np.random.rand(N)
        h_init = _project(h_init, N / 2.0)
        inits.append(h_init)
    
    # Constant init
    inits.append(np.full(N, 0.5))
    
    # Phase 2: Simulated annealing from each initialization
    t_start = 0.1
    t_end = 0.001
    n_iter = 300
    
    for h_init in inits:
        if time.time() > _DEADLINE - 20:
            break
        
        h_opt, s_opt = _simulated_annealing(h_init, t_start, t_end, n_iter)
        
        if s_opt < best_score:
            best_score = s_opt
            best_h = h_opt.copy()
    
    # Phase 3: Local refinement using coordinate descent
    if best_h is not None and time.time() < _DEADLINE - 5:
        h_ref = best_h.copy()
        for outer in range(30):
            if time.time() > _DEADLINE - 2:
                break
            improved = False
            for i in range(N):
                if time.time() > _DEADLINE - 2:
                    break
                old_val = h_ref[i]
                # Try several values
                for new_val in [0.0, 1.0, old_val - 0.05, old_val + 0.05, 
                               old_val - 0.01, old_val + 0.01,
                               old_val - 0.001, old_val + 0.001]:
                    if new_val < 0 or new_val > 1:
                        continue
                    h_test = h_ref.copy()
                    h_test[i] = new_val
                    h_test = _project(h_test, N / 2.0)
                    s_test = _score(h_test)
                    if s_test < best_score - 1e-12:
                        best_score = s_test
                        best_h = h_test.copy()
                        h_ref = h_test.copy()
                        improved = True
            if not improved:
                break
    
    if best_h is None:
        best_h = np.full(N, 0.5)
        
    best_h = _project(best_h, N / 2.0)
    return best_h
