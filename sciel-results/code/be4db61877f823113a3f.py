"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _fft_max_overlap(h):
    """Fast computation of max overlap using FFT.
    
    overlap[k] = sum_i h[i] * (1-h[(i+k) % N])
               = sum_i h[i] - sum_i h[i]*h[(i+k)%N]
               = N/2 - autocorr[k]
    """
    H = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(H * np.conj(H)))
    # autocorr[s] = sum_i h[i] * h[(i-s)%N]
    # We need sum_i h[i]*h[(i+k)%N] which is autocorr[-k] = autocorr[N-k]
    # So overlap[k] = N/2 - autocorr[N-k]
    overlaps = N/2.0 - autocorr[(-np.arange(N)) % N]
    return np.max(overlaps)

def _fft_overlaps(h):
    """Return all overlaps."""
    H = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(H * np.conj(H)))
    return N/2.0 - autocorr[(-np.arange(N)) % N]

def _project(h):
    """Project h to valid space: [0,1] and sum N/2 exactly."""
    h = np.asarray(h, dtype=np.float64).copy()
    for _ in range(20):
        h = np.clip(h, 0, 1)
        diff = N/2.0 - np.sum(h)
        if abs(diff) < 1e-15:
            break
        mask = (h > 1e-10) & (h < 1.0 - 1e-10)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
        else:
            h += diff / N
    diff = N/2.0 - np.sum(h)
    if abs(diff) > 1e-15:
        idx = np.argmin(np.abs(h - 0.5))
        h[idx] += diff
        h = np.clip(h, 0, 1)
    return h

def _sinusoidal_candidate(A, phi, freq=1):
    """Generate a sinusoidal candidate h_i = 0.5 + A*sin(2*pi*freq*i/N + phi), clipped to [0,1]."""
    i = np.arange(N)
    h = 0.5 + A * np.sin(2 * np.pi * freq * i / N + phi)
    h = np.clip(h, 0, 1)
    h = _project(h)
    return h

def _grid_search_sin():
    """Grid search over sinusoidal parameters A, phi, freq."""
    best_h = None
    best_val = np.inf
    
    for freq in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for A in np.linspace(0.1, 0.5, 15):
            for phi in np.linspace(0, 2*np.pi, 30, endpoint=False):
                h = _sinusoidal_candidate(A, phi, freq)
                val = _fft_max_overlap(h)
                if val < best_val:
                    best_val = val
                    best_h = h.copy()
    
    return best_h, best_val

def _local_search_binary(h_start, time_limit):
    """Local search on binary-like sequences via pairwise swaps."""
    best_h = h_start.copy()
    best_val = _fft_max_overlap(best_h)
    deadline = time.time() + time_limit
    
    no_improve = 0
    max_no_improve = 3000
    
    while time.time() < deadline and no_improve < max_no_improve:
        idx_1 = np.where(best_h > 0.5)[0]
        idx_0 = np.where(best_h < 0.5)[0]
        
        if len(idx_1) == 0 or len(idx_0) == 0:
            break
        
        improved = False
        # Try random pairwise swaps
        for _ in range(100):
            if time.time() > deadline:
                break
            i = idx_1[np.random.randint(len(idx_1))]
            j = idx_0[np.random.randint(len(idx_0))]
            
            h_new = best_h.copy()
            h_new[i] = 0
            h_new[j] = 1
            
            val_new = _fft_max_overlap(h_new)
            if val_new < best_val - 1e-12:
                best_val = val_new
                best_h = h_new
                improved = True
                no_improve = 0
                break
        
        if not improved:
            no_improve += 100
    
    return best_h, best_val

def _smooth_refine(h_start, time_limit):
    """Refine using smooth optimization with log-sum-exp surrogate."""
    deadline = time.time() + time_limit
    best_h = h_start.copy()
    best_val = _fft_max_overlap(best_h)
    
    for beta in [5.0, 10.0, 20.0]:
        if time.time() > deadline - 5:
            break
        
        def objective(x):
            h = np.clip(x, 0, 1)
            overlaps = _fft_overlaps(h)
            max_ov = np.max(overlaps)
            scaled = beta * (overlaps - max_ov)
            return max_ov + np.log(np.sum(np.exp(scaled))) / beta
        
        def jac(x):
            h = np.clip(x, 0, 1)
            overlaps = _fft_overlaps(h)
            max_ov = np.max(overlaps)
            scaled = beta * (overlaps - max_ov)
            weights = np.exp(scaled) / np.sum(np.exp(scaled))
            
            # Numerical gradient approximation via FFT
            # dO_s/dh_k: O_s = N/2 - sum_i h[i]h[(i-s)%N]
            # Use finite differences for simplicity
            eps = 1e-6
            grad = np.zeros(N)
            for k in range(N):
                h_p = h.copy()
                h_p[k] += eps
                h_p = np.clip(h_p, 0, 1)
                ov_p = _fft_overlaps(h_p)
                max_p = np.max(ov_p)
                scaled_p = beta * (ov_p - max_p)
                obj_p = max_p + np.log(np.sum(np.exp(scaled_p))) / beta
                
                h_m = h.copy()
                h_m[k] -= eps
                h_m = np.clip(h_m, 0, 1)
                ov_m = _fft_overlaps(h_m)
                max_m = np.max(ov_m)
                scaled_m = beta * (ov_m - max_m)
                obj_m = max_m + np.log(np.sum(np.exp(scaled_m))) / beta
                
                grad[k] = (obj_p - obj_m) / (2 * eps)
            return grad
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0}]
        bounds = [(0, 1)] * N
        
        try:
            res = minimize(objective, best_h, jac=jac, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 200, 'ftol': 1e-8})
            if res.success:
                h_opt = _project(res.x)
                val = _fft_max_overlap(h_opt)
                if val < best_val:
                    best_val = val
                    best_h = h_opt
        except Exception:
            pass
    
    return best_h, best_val

def _generate_binary_candidates():
    """Generate diverse binary initial candidates."""
    candidates = []
    np.random.seed(42)
    
    # Thue-Morse
    tm = np.array([bin(i).count('1') % 2 for i in range(N)], dtype=float)
    if np.sum(tm) != N/2:
        idx_1 = np.where(tm == 1)[0]
        idx_0 = np.where(tm == 0)[0]
        curr_sum = np.sum(tm)
        target = N/2.0
        if curr_sum > target:
            to_flip = int(curr_sum - target)
            tm[idx_1[:to_flip]] = 0
        else:
            to_flip = int(target - curr_sum)
            tm[idx_0[:to_flip]] = 1
    candidates.append(tm.copy())
    
    # Alternating
    alt = np.array([0 if (i % 4) < 2 else 1 for i in range(N)], dtype=float)
    candidates.append(alt.copy())
    
    # Random binary
    for _ in range(20):
        h_rand = np.zeros(N)
        ones_idx = np.random.choice(N, N//2, replace=False)
        h_rand[ones_idx] = 1.0
        candidates.append(h_rand)
    
    # Block patterns
    for block_size in [2, 3, 4, 5]:
        pattern = []
        for i in range(N):
            cycle_pos = i % (2 * block_size)
            pattern.append(1.0 if cycle_pos < block_size else 0.0)
        h_block = np.array(pattern)
        if np.sum(h_block) != N/2:
            idx_1 = np.where(h_block == 1)[0]
            idx_0 = np.where(h_block == 0)[0]
            curr_sum = np.sum(h_block)
            target = N/2.0
            if curr_sum > target:
                to_flip = int(curr_sum - target)
                h_block[idx_1[:to_flip]] = 0
            else:
                to_flip = int(target - curr_sum)
                h_block[idx_0[:to_flip]] = 1
        candidates.append(h_block)
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Start with the known optimal constant solution
    best_h = np.full(N, 0.5)
    best_score = _fft_max_overlap(best_h)
    
    # Phase 1: Grid search over sinusoidal parameters
    if time.time() < deadline - 60:
        sin_h, sin_val = _grid_search_sin()
        if sin_val < best_score:
            best_score = sin_val
            best_h = sin_h
    
    # Phase 2: Generate binary candidates and evaluate
    np.random.seed(42)
    binary_candidates = _generate_binary_candidates()
    
    for h_cand in binary_candidates:
        if time.time() > deadline - 40:
            break
        val = _fft_max_overlap(h_cand)
        if val < best_score:
            best_score = val
            best_h = h_cand.copy()
    
    # Phase 3: Local search from best binary candidate
    if time.time() < deadline - 30:
        # Find best binary starting point
        best_binary = best_h.copy()
        if np.all((best_binary < 0.1) | (best_binary > 0.9)):
            ls_h, ls_val = _local_search_binary(best_binary, time_limit=25)
            if ls_val < best_score:
                best_score = ls_val
                best_h = ls_h
    
    # Phase 4: Smooth refinement
    if time.time() < deadline - 10:
        refine_h, refine_val = _smooth_refine(best_h, time_limit=8)
        if refine_val < best_score:
            best_score = refine_val
            best_h = refine_h
    
    # Final projection
    best_h = _project(best_h)
    
    return best_h
# EVOLVE-BLOCK-END
