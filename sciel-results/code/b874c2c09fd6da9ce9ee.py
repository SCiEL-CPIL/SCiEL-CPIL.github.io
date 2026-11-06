"""Combined program: DFT-inspired initialization + histogram-based incremental local search."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

N = 256

def _precompute_aps():
    """Precompute all APs (length >= 2) and their containing indices."""
    aps_indices = []
    aps_containing = [[] for _ in range(N)]
    
    for d in range(1, N + 1):
        for start in range(0, N - d + 1):
            terms = list(range(start, N, d))
            if len(terms) > 1:
                ap_id = len(aps_indices)
                aps_indices.append(terms)
                for idx in terms:
                    aps_containing[idx].append(ap_id)
    
    return aps_indices, aps_containing

def _init_histogram(ap_sums, num_aps):
    """Initialize histogram from AP sums."""
    HIST_SIZE = 2 * N + 1
    OFFSET = N
    hist = np.zeros(HIST_SIZE, dtype=np.int32)
    for s in ap_sums:
        hist[s + OFFSET] += 1
    return hist

def _get_max_from_hist(hist):
    """Get maximum absolute value from histogram."""
    HIST_SIZE = 2 * N + 1
    OFFSET = N
    for val in range(N, -1, -1):
        if hist[val + OFFSET] > 0 or (val > 0 and hist[-val + OFFSET] > 0):
            return val
    return 0

def _dft_init():
    """Initialize signs using DFT-based approach to minimize low-frequency energy."""
    # Start with a random phase vector
    rng = random.Random(12345)
    phases = np.array([rng.uniform(0, 2 * math.pi) for _ in range(N)])
    
    # Use a few iterations of "gradient descent" on the continuous relaxation
    # Minimize sum of |DFT|^2 for low frequencies
    for _ in range(100):
        x_cont = np.cos(phases) + 1j * np.sin(phases)
        # Compute DFT
        X = np.fft.fft(x_cont)
        # Focus on low frequencies (first 16 and last 16)
        low_freq = list(range(1, 17)) + list(range(N - 16, N - 1))
        # Gradient: we want to reduce |X[k]| for low k
        # Simple update: adjust phases to reduce energy
        grad = np.zeros(N, dtype=complex)
        for k in low_freq:
            # Contribution from frequency k
            Xk = X[k]
            # d|Xk|^2/dtheta_j = -2 * Re(Xk * conj(x_j) * exp(2*pi*i*k*j/N))
            for j in range(N):
                grad[j] += -2 * np.real(Xk * np.conj(x_cont[j]) * np.exp(2j * math.pi * k * j / N))
        # Update phases
        phases -= 0.1 * np.angle(grad)
    
    # Project to +/- 1
    signs = np.where(np.cos(phases) >= 0, 1, -1).astype(np.int8)
    return signs

def _greedy_init(aps_indices, aps_containing):
    """Greedy initialization: place signs one by one to minimize current max."""
    num_aps = len(aps_indices)
    HIST_SIZE = 2 * N + 1
    OFFSET = N
    
    signs = np.zeros(N, dtype=np.int8)
    ap_sums = np.zeros(num_aps, dtype=np.int16)
    hist = np.zeros(HIST_SIZE, dtype=np.int32)
    hist[OFFSET] = num_aps  # All sums start at 0
    
    for i in range(N):
        best_sign = 1
        best_max = N + 1
        best_sums = None
        best_hist = None
        
        for s in [1, -1]:
            new_sums = ap_sums.copy()
            new_hist = hist.copy()
            
            for ap_idx in aps_containing[i]:
                old_sum = new_sums[ap_idx]
                new_sum = old_sum + s
                new_sums[ap_idx] = new_sum
                new_hist[old_sum + OFFSET] -= 1
                new_hist[new_sum + OFFSET] += 1
            
            m = _get_max_from_hist(new_hist)
            if m < best_max:
                best_max = m
                best_sign = s
                best_sums = new_sums
                best_hist = new_hist
        
        signs[i] = best_sign
        ap_sums = best_sums
        hist = best_hist
    
    return signs, ap_sums, hist, best_max

def _local_search(signs, ap_sums, hist, current_max, aps_containing, time_limit):
    """Fast local search with histogram-based incremental updates."""
    HIST_SIZE = 2 * N + 1
    OFFSET = N
    num_aps = len(aps_containing[0]) if aps_containing else 0
    
    best_signs = signs.copy()
    best_ap_sums = ap_sums.copy()
    best_hist = hist.copy()
    best_max = current_max
    
    rng = random.Random(42)
    
    while time.time() < time_limit:
        # Pick 1 or 2 random indices to flip
        n_flips = 1 if rng.random() < 0.6 else 2
        indices = rng.sample(range(N), n_flips)
        
        # Store old state
        old_signs = signs.copy()
        old_ap_sums = ap_sums.copy()
        old_hist = hist.copy()
        
        # Apply flips
        affected_aps = set()
        for idx in indices:
            signs[idx] = -signs[idx]
            for ap_idx in aps_containing[idx]:
                affected_aps.add(ap_idx)
        
        # Update sums and histogram for affected APs
        for ap_idx in affected_aps:
            old_sum = old_ap_sums[ap_idx]
            new_sum = ap_sums[ap_idx]
            hist[old_sum + OFFSET] -= 1
            hist[new_sum + OFFSET] += 1
            ap_sums[ap_idx] = new_sum
        
        # Get new max
        new_max = _get_max_from_hist(hist)
        
        if new_max < best_max:
            best_max = new_max
            best_signs = signs.copy()
            best_ap_sums = ap_sums.copy()
            best_hist = hist.copy()
        else:
            # Revert
            signs = old_signs
            ap_sums = old_ap_sums
            hist = old_hist
    
    return best_signs, best_ap_sums, best_hist, best_max

def _max_abs_aps(x):
    """Compute the maximum absolute signed sum over all arithmetic progressions."""
    max_abs = 0
    for d in range(1, N + 1):
        for r in range(d):
            chain = []
            pos = r
            while pos < N:
                chain.append(x[pos])
                pos += d
            if len(chain) == 0:
                continue
            max_so_far = 0
            min_so_far = 0
            current_max = 0
            current_min = 0
            for v in chain:
                current_max = max(0, current_max + v)
                current_min = min(0, current_min + v)
                if current_max > max_so_far:
                    max_so_far = current_max
                if abs(current_min) > abs(min_so_far):
                    min_so_far = current_min
            cand = max(max_so_far, abs(min_so_far))
            if cand > max_abs:
                max_abs = cand
    return max_abs

def construct():
    start_time = time.time()
    deadline = 55.0
    
    # Precompute APs
    aps_indices, aps_containing = _precompute_aps()
    num_aps = len(aps_indices)
    
    # Try multiple initialization strategies
    candidates = []
    
    # 1. DFT-based initialization
    signs_dft = _dft_init()
    ap_sums_dft = np.zeros(num_aps, dtype=np.int16)
    for ap_id, terms in enumerate(aps_indices):
        for idx in terms:
            ap_sums_dft[ap_id] += signs_dft[idx]
    hist_dft = _init_histogram(ap_sums_dft, num_aps)
    max_dft = _get_max_from_hist(hist_dft)
    candidates.append((signs_dft, ap_sums_dft, hist_dft, max_dft))
    
    # 2. Greedy initialization
    signs_greedy, ap_sums_greedy, hist_greedy, max_greedy = _greedy_init(aps_indices, aps_containing)
    candidates.append((signs_greedy, ap_sums_greedy, hist_greedy, max_greedy))
    
    # 3. Random initialization
    rng = random.Random(42)
    signs_rand = np.array([rng.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
    ap_sums_rand = np.zeros(num_aps, dtype=np.int16)
    for ap_id, terms in enumerate(aps_indices):
        for idx in terms:
            ap_sums_rand[ap_id] += signs_rand[idx]
    hist_rand = _init_histogram(ap_sums_rand, num_aps)
    max_rand = _get_max_from_hist(hist_rand)
    candidates.append((signs_rand, ap_sums_rand, hist_rand, max_rand))
    
    # Pick the best candidate
    best_idx = min(range(len(candidates)), key=lambda i: candidates[i][3])
    best_signs, best_ap_sums, best_hist, best_max = candidates[best_idx]
    
    # Run local search with remaining time
    local_search_deadline = start_time + deadline - 5.0  # Leave 5s for final eval
    best_signs, best_ap_sums, best_hist, best_max = _local_search(
        best_signs, best_ap_sums, best_hist, best_max, aps_containing, local_search_deadline
    )
    
    # Final evaluation to verify
    final_score = _max_abs_aps(best_signs)
    
    return list(best_signs)
# EVOLVE-BLOCK-END
