"""Combined program: structured pattern search + FFT-accelerated local search for Erdos minimum overlap."""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
def _max_overlap_fft(h):
    """Compute max over all circular shifts of sum(h[i] * (1-h[(i+s)%N])) using FFT."""
    N = len(h)
    inv_h = 1.0 - h
    # Circular cross-correlation: corr[s] = sum_i h[i] * inv_h[(i+s) % N]
    # This is the circular correlation of h and inv_h
    # Using FFT: corr = ifft(fft(h) * conj(fft(inv_h)))
    H = np.fft.fft(h)
    IH = np.fft.fft(inv_h)
    corr = np.real(np.fft.ifft(H * np.conj(IH)))
    # corr[s] = sum_i h[(i-s) % N] * inv_h[i] = sum_i h[i] * inv_h[(i+s) % N]
    # Actually let's verify: ifft(fft(a)*conj(fft(b)))[s] = sum_k a[k] * b[(k+s)%N]
    # So corr[s] = sum_k h[k] * inv_h[(k+s)%N] which is what we want
    return float(np.max(corr))

def _max_overlap_direct(h):
    """Direct computation for verification."""
    N = len(h)
    inv_h = 1.0 - h
    best = 0.0
    for s in range(N):
        ov = np.dot(h, np.roll(inv_h, s))
        if ov > best:
            best = ov
    return best

def _score(h):
    """Score function: returns the maximum overlap, or inf if invalid."""
    s = h.sum()
    if abs(s - N / 2.0) > 1e-8:
        return float('inf')
    if h.min() < -1e-10 or h.max() > 1.0 + 1e-10:
        return float('inf')
    return _max_overlap_fft(h)

def _normalize_to_sum(h, target=N/2.0):
    """Adjust h to have sum exactly target, staying in [0,1]."""
    h = h.copy()
    current_sum = h.sum()
    diff = target - current_sum
    
    if abs(diff) < 1e-15:
        return h
    
    # Iteratively adjust: add diff/N to all, clip, repeat
    for _ in range(100):
        if abs(diff) < 1e-15:
            break
        h += diff / N
        h = np.clip(h, 0.0, 1.0)
        diff = target - h.sum()
    
    # Final exact fix
    diff = target - h.sum()
    if abs(diff) > 1e-15:
        if diff > 0:
            # Need to increase sum: add to elements with room
            idxs = np.where(h < 1.0)[0]
            if len(idxs) > 0:
                # Distribute among elements with most room
                room = 1.0 - h[idxs]
                order = np.argsort(-room)
                for i in order:
                    add = min(diff, room[i])
                    h[idxs[i]] += add
                    diff -= add
                    if abs(diff) < 1e-15:
                        break
        else:
            # Need to decrease sum: subtract from elements with value > 0
            idxs = np.where(h > 0.0)[0]
            if len(idxs) > 0:
                order = np.argsort(h[idxs])
                for i in order:
                    sub = min(-diff, h[idxs[i]])
                    h[idxs[i]] -= sub
                    diff += sub
                    if abs(diff) < 1e-15:
                        break
    
    return h

def _try_pattern(pattern, deadline):
    """Expand pattern to length N, normalize, score."""
    if time.time() > deadline:
        return None, None
    
    base = np.array(pattern, dtype=float)
    if len(base) == 0:
        return None, None
    
    # Tile to length N
    reps = N // len(base) + 1
    tiled = np.tile(base, reps)[:N]
    
    # Normalize to sum N/2
    h = _normalize_to_sum(tiled)
    
    sc = _score(h)
    if np.isinf(sc):
        return None, None
    
    return h, sc

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    rng = np.random.default_rng(42)
    
    # Start with constant solution
    best_h = np.full(N, 0.5)
    best_score = _score(best_h)  # Should be 25.0
    
    # Phase 1: Try structured patterns
    patterns_to_try = []
    
    # Sine-based patterns with various frequencies and phases
    for freq in range(1, N + 1):
        for phase in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            x = np.arange(N)
            pat = 0.5 + 0.5 * np.sin(2 * np.pi * freq * x / N + phase)
            patterns_to_try.append(pat)
    
    # Cosine-based patterns
    for freq in range(1, N + 1):
        for phase in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            x = np.arange(N)
            pat = 0.5 + 0.5 * np.cos(2 * np.pi * freq * x / N + phase)
            patterns_to_try.append(pat)
    
    # Block patterns: alternating high/low blocks
    for block_len in range(1, N + 1):
        for offset in range(block_len):
            pat = np.zeros(N)
            for i in range(N):
                block_idx = (i + offset) // block_len
                if block_idx % 2 == 0:
                    pat[i] = 1.0
                else:
                    pat[i] = 0.0
            patterns_to_try.append(pat)
    
    # Random patterns
    for _ in range(500):
        patterns_to_try.append(rng.random(N))
    
    # Binary patterns (0/1)
    for _ in range(200):
        pat = (rng.random(N) > 0.5).astype(float)
        patterns_to_try.append(pat)
    
    for pat in patterns_to_try:
        if time.time() > deadline:
            break
        h, sc = _try_pattern(pat, deadline)
        if h is not None and sc < best_score:
            best_score = sc
            best_h = h.copy()
    
    # Phase 2: Local search with sum-preserving moves
    h = best_h.copy()
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        # Pick two random indices
        i = rng.integers(0, N)
        j = rng.integers(0, N)
        if i == j:
            no_improve_count += 1
            continue
        
        # Try moving eps from j to i
        eps = rng.uniform(0.01, 0.1)
        
        # Check bounds
        if h[i] + eps > 1.0 or h[j] - eps < 0.0:
            no_improve_count += 1
            continue
        
        h_new = h.copy()
        h_new[i] += eps
        h_new[j] -= eps
        
        sc = _score(h_new)
        if np.isinf(sc):
            no_improve_count += 1
            continue
        
        if sc < best_score:
            best_score = sc
            best_h = h_new.copy()
            h = h_new.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
    
    # Phase 3: High-precision polish - try smaller eps
    h = best_h.copy()
    no_improve_count = 0
    
    while time.time() < deadline and no_improve_count < 200:
        i = rng.integers(0, N)
        j = rng.integers(0, N)
        if i == j:
            no_improve_count += 1
            continue
        
        eps = rng.uniform(0.001, 0.01)
        
        if h[i] + eps > 1.0 or h[j] - eps < 0.0:
            no_improve_count += 1
            continue
        
        h_new = h.copy()
        h_new[i] += eps
        h_new[j] -= eps
        
        sc = _score(h_new)
        if np.isinf(sc):
            no_improve_count += 1
            continue
        
        if sc < best_score:
            best_score = sc
            best_h = h_new.copy()
            h = h_new.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
    
    # Final: ensure exact sum
    best_h = _normalize_to_sum(best_h)
    
    # Verify
    final_sc = _score(best_h)
    if np.isinf(final_sc):
        # Fallback to constant
        return np.full(N, 0.5)
    
    return best_h
# EVOLVE-BLOCK-END
