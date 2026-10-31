"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

N = 100

def _fft_overlaps(h):
    """Compute all shift overlaps using FFT."""
    H = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(H * np.conj(H)))
    C = autocorr[(-np.arange(N)) % N]
    return N / 2.0 - C

def _fft_max_overlap(h):
    """Compute maximum overlap using FFT."""
    return np.max(_fft_overlaps(h))

def _project(h):
    """Project h to valid space: [0,1] and sum exactly N/2."""
    h = np.asarray(h, dtype=np.float64).copy()
    for _ in range(20):
        h = np.clip(h, 0.0, 1.0)
        diff = N / 2.0 - np.sum(h)
        if abs(diff) < 1e-15:
            break
        mask = (h > 1e-10) & (h < 1.0 - 1e-10)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
    diff = N / 2.0 - np.sum(h)
    if abs(diff) > 1e-14:
        idx = np.argmin(np.abs(h - 0.5))
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
    return h

def _soft_max_overlap(h, beta=50.0):
    """Smooth surrogate for max overlap using log-sum-exp."""
    overlaps = _fft_overlaps(h)
    m = np.max(overlaps)
    scaled = beta * (overlaps - m)
    # Clip for numerical stability
    scaled = np.clip(scaled, -500, 500)
    return m + np.log(np.sum(np.exp(scaled))) / beta

def _binary_swap_search(h_start, time_limit, rng):
    """Local search using binary swaps (1->0 and 0->1 simultaneously).
    
    This preserves the sum constraint exactly and is very fast.
    """
    best_h = h_start.copy()
    best_val = _fft_max_overlap(best_h)
    deadline = time.time() + time_limit
    
    no_improve = 0
    max_no_improve = 2000
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Identify indices near 1 and near 0
        idx_1 = np.where(best_h > 0.7)[0]
        idx_0 = np.where(best_h < 0.3)[0]
        
        if len(idx_1) == 0 or len(idx_0) == 0:
            break
        
        improved = False
        # Try random swaps
        n_tries = min(500, len(idx_1) * len(idx_0))
        for _ in range(n_tries):
            if time.time() > deadline:
                break
            i = idx_1[rng.integers(len(idx_1))]
            j = idx_0[rng.integers(len(idx_0))]
            
            h_new = best_h.copy()
            h_new[i] = 0.0
            h_new[j] = 1.0
            
            val_new = _fft_max_overlap(h_new)
            
            if val_new < best_val - 1e-12:
                best_val = val_new
                best_h = h_new
                improved = True
                no_improve = 0
                break
        
        if not improved:
            no_improve += 1
    
    return best_h, best_val

def _continuous_refine(h_start, time_limit):
    """Refine using continuous optimization with soft-max surrogate."""
    best_h = h_start.copy()
    best_val = _fft_max_overlap(best_h)
    deadline = time.time() + time_limit
    
    # Multi-scale beta annealing
    betas = [5.0, 10.0, 20.0, 50.0, 100.0]
    
    for beta in betas:
        if time.time() > deadline - 5:
            break
        
        def objective(x):
            h = np.clip(x, 0, 1)
            return _soft_max_overlap(h, beta=beta)
        
        try:
            res = minimize(objective, best_h, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10})
            h_cand = _project(res.x)
            val = _fft_max_overlap(h_cand)
            if val < best_val:
                best_val = val
                best_h = h_cand.copy()
        except Exception:
            pass
    
    return best_h, best_val

def _generate_binary_candidates(rng):
    """Generate diverse binary initial candidates with sum exactly N/2."""
    candidates = []
    
    # 1. Thue-Morse sequence
    tm = np.array([bin(i).count('1') % 2 for i in range(N)], dtype=float)
    if np.sum(tm) != N / 2:
        idx_1 = np.where(tm == 1)[0]
        idx_0 = np.where(tm == 0)[0]
        curr_sum = np.sum(tm)
        target = N / 2.0
        if curr_sum > target:
            to_flip = int(curr_sum - target)
            tm[idx_1[:to_flip]] = 0
        else:
            to_flip = int(target - curr_sum)
            tm[idx_0[:to_flip]] = 1
    candidates.append(tm.copy())
    
    # 2. Alternating 0011 pattern
    alt = np.array([0.0 if (i % 4) < 2 else 1.0 for i in range(N)])
    candidates.append(alt.copy())
    
    # 3. Random binary sequences
    for _ in range(40):
        h_rand = np.zeros(N)
        ones_idx = rng.choice(N, N // 2, replace=False)
        h_rand[ones_idx] = 1.0
        candidates.append(h_rand.copy())
    
    # 4. Thresholded sine waves at various frequencies
    for freq in [1, 2, 3, 4, 5, 7]:
        h_sine = np.sin(2 * np.pi * freq * np.arange(N) / N)
        h_bin = (h_sine > 0).astype(float)
        if np.sum(h_bin) != N / 2:
            idx_1 = np.where(h_bin == 1)[0]
            idx_0 = np.where(h_bin == 0)[0]
            curr_sum = np.sum(h_bin)
            target = N / 2.0
            if curr_sum > target:
                to_flip = int(curr_sum - target)
                h_bin[idx_1[:to_flip]] = 0
            else:
                to_flip = int(target - curr_sum)
                h_bin[idx_0[:to_flip]] = 1
        candidates.append(h_bin.copy())
    
    # 5. Block patterns
    for block_size in [2, 3, 4, 5, 7]:
        pattern = np.array([1.0 if (i % (2 * block_size)) < block_size else 0.0 
                          for i in range(N)])
        if np.sum(pattern) != N / 2:
            idx_1 = np.where(pattern == 1)[0]
            idx_0 = np.where(pattern == 0)[0]
            curr_sum = np.sum(pattern)
            target = N / 2.0
            if curr_sum > target:
                to_flip = int(curr_sum - target)
                pattern[idx_1[:to_flip]] = 0
            else:
                to_flip = int(target - curr_sum)
                pattern[idx_0[:to_flip]] = 1
        candidates.append(pattern.copy())
    
    return candidates

def construct():
    deadline = time.time() + 115.0  # 5s safety margin from 120s limit
    rng = np.random.default_rng(42)
    
    # Start with uniform
    best_h = np.full(N, 0.5)
    best_val = _fft_max_overlap(best_h)
    
    # Phase 1: Generate and evaluate binary candidates
    candidates = _generate_binary_candidates(rng)
    
    for h_cand in candidates:
        if time.time() > deadline - 60:
            break
        h_proj = _project(h_cand)
        val = _fft_max_overlap(h_proj)
        if val < best_val:
            best_val = val
            best_h = h_proj.copy()
    
    # Phase 2: Binary swap local search from best candidate
    if time.time() < deadline - 40:
        time_left = min(30, deadline - time.time() - 40)
        best_h, best_val = _binary_swap_search(best_h, time_left, rng)
    
    # Phase 3: Try binary swap search from a few other good candidates
    if time.time() < deadline - 25:
        # Re-evaluate top candidates for swap search
        scored = []
        for h_cand in candidates[:20]:
            h_proj = _project(h_cand)
            val = _fft_max_overlap(h_proj)
            scored.append((val, h_proj))
        scored.sort(key=lambda x: x[0])
        
        for val, h_cand in scored[:5]:
            if time.time() > deadline - 20:
                break
            if val >= best_val:
                continue
            time_left = min(5, deadline - time.time() - 20)
            h_new, val_new = _binary_swap_search(h_cand, time_left, rng)
            if val_new < best_val:
                best_val = val_new
                best_h = h_new.copy()
    
    # Phase 4: Continuous refinement
    if time.time() < deadline - 10:
        time_left = min(20, deadline - time.time() - 10)
        best_h, best_val = _continuous_refine(best_h, time_left)
    
    # Phase 5: Final binary polish
    if time.time() < deadline - 5:
        time_left = min(5, deadline - time.time() - 5)
        best_h, best_val = _binary_swap_search(best_h, time_left, rng)
    
    # Phase 6: Final continuous polish
    if time.time() < deadline - 2:
        time_left = min(3, deadline - time.time() - 2)
        best_h, best_val = _continuous_refine(best_h, time_left)
    
    # Ensure exact sum
    best_h = _project(best_h)
    return best_h
# EVOLVE-BLOCK-END
