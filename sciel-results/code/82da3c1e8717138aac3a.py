import numpy as np
import time

N = 100
HALF = N / 2.0

def _max_overlap_from_binary(x):
    """x is binary {0,1} with sum = N/2. Returns max overlap value."""
    # O_k = sum_i x[i] * (1 - x[(i+k)%N])
    #     = sum_i x[i] - sum_i x[i] x[(i+k)%N]
    #     = HALF - ac[k]
    # where ac[k] = circular autocorrelation
    X = np.fft.fft(x.astype(np.float64))
    ac = np.fft.ifft(X * np.conj(X)).real
    return HALF - np.min(ac)

def _max_overlap_general(h):
    """h in [0,1], sum = N/2. Returns max overlap."""
    H = np.fft.fft(h)
    ac = np.fft.ifft(H * np.conj(H)).real
    return HALF - np.min(ac)

def _construct_binary_sequence():
    """
    Search for a binary sequence with N/2 ones that has a flat autocorrelation.
    Use simulated annealing with bit-swap moves.
    """
    np.random.seed(12345)
    
    # Start with alternating pattern (good baseline)
    x = np.zeros(N)
    x[0::2] = 1.0  # 50 ones
    # Verify sum
    assert np.sum(x) == HALF, f"Sum is {np.sum(x)}, expected {HALF}"
    
    cur_overlap = _max_overlap_from_binary(x)
    best_x = x.copy()
    best_overlap = cur_overlap
    
    start_time = time.time()
    deadline = 130.0
    
    T = 2.0
    T_min = 0.001
    cooling = 0.9995
    
    # Precompute for speed
    # We'll use bit-swap: pick a 1 and a 0, swap them.
    
    it = 0
    while time.time() - start_time < deadline:
        it += 1
        
        # Pick random 1-index and 0-index
        ones_idx = np.where(x == 1.0)[0]
        zeros_idx = np.where(x == 0.0)[0]
        if len(ones_idx) == 0 or len(zeros_idx) == 0:
            break
        i = ones_idx[np.random.randint(len(ones_idx))]
        j = zeros_idx[np.random.randint(len(zeros_idx))]
        
        # Swap
        x[i] = 0.0
        x[j] = 1.0
        
        new_overlap = _max_overlap_from_binary(x)
        
        delta = new_overlap - cur_overlap
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            cur_overlap = new_overlap
            if new_overlap < best_overlap:
                best_overlap = new_overlap
                best_x = x.copy()
        else:
            # Revert
            x[i] = 1.0
            x[j] = 0.0
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_x, best_overlap

def _refine_continuous(x0, time_budget):
    """
    Take a binary sequence and refine in continuous space using local search
    to potentially squeeze out a bit more.
    """
    h = x0.astype(np.float64).copy()
    cur_overlap = _max_overlap_general(h)
    best_h = h.copy()
    best_overlap = cur_overlap
    
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Local search: perturb small groups of entries
    np.random.seed(777)
    step = 0.05
    patience = 0
    max_patience = 500
    
    while time.time() - start_time < deadline:
        patience += 1
        if patience > max_patience:
            break
        
        # Perturb a few entries
        n_mod = np.random.randint(1, 5)
        idxs = np.random.choice(N, n_mod, replace=False)
        old_vals = h[idxs].copy()
        
        h[idxs] += np.random.uniform(-step, step, n_mod)
        h[idxs] = np.clip(h[idxs], 0.0, 1.0)
        
        # Re-normalize sum
        residual = HALF - np.sum(h)
        if abs(residual) > 1e-12:
            # Distribute to entries with room
            if residual > 0:
                mask = h < 1.0
            else:
                mask = h > 0.0
            if np.sum(mask) > 0:
                h[mask] += residual / np.sum(mask)
                h = np.clip(h, 0.0, 1.0)
            # Final fix
            residual = HALF - np.sum(h)
            if abs(residual) > 1e-12:
                h += residual / N
                h = np.clip(h, 0.0, 1.0)
        
        new_overlap = _max_overlap_general(h)
        
        if new_overlap < cur_overlap - 1e-12:
            cur_overlap = new_overlap
            if new_overlap < best_overlap:
                best_overlap = new_overlap
                best_h = h.copy()
            patience = 0
            step = min(step * 1.1, 0.2)
        else:
            h[idxs] = old_vals
            # Re-normalize
            residual = HALF - np.sum(h)
            if abs(residual) > 1e-12:
                h += residual / N
                h = np.clip(h, 0.0, 1.0)
            step = max(step * 0.98, 0.005)
    
    return best_h, best_overlap

def construct():
    start_time = time.time()
    total_deadline = 145.0  # safety margin
    
    # Phase 1: Binary search (most of the time)
    binary_time = 90.0
    x0, binary_overlap = _construct_binary_sequence()
    
    # Phase 2: Continuous refinement
    elapsed = time.time() - start_time
    remaining = total_deadline - elapsed - 3.0
    if remaining > 10.0:
        h_refined, cont_overlap = _refine_continuous(x0, remaining)
        if cont_overlap < binary_overlap:
            best_h = h_refined
            best_overlap = cont_overlap
        else:
            best_h = x0.astype(np.float64)
            best_overlap = binary_overlap
    else:
        best_h = x0.astype(np.float64)
        best_overlap = binary_overlap
    
    # Final: ensure exact sum and bounds
    best_h = np.clip(best_h, 0.0, 1.0)
    residual = HALF - np.sum(best_h)
    if abs(residual) > 1e-14:
        # Distribute to entries with room
        if residual > 0:
            mask = best_h < 1.0 - 1e-9
        else:
            mask = best_h > 1e-9
        n_mask = np.sum(mask)
        if n_mask > 0:
            best_h[mask] += residual / n_mask
        best_h = np.clip(best_h, 0.0, 1.0)
        # Tiny residual fix
        residual = HALF - np.sum(best_h)
        if abs(residual) > 1e-14:
            best_h += residual / N
            best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h
