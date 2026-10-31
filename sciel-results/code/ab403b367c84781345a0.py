"""Erdos minimum overlap - find step heights minimizing max overlap with shifted complement."""
import numpy as np
import time

N = 100
WIDTH = 2.0 / N

def compute_max_overlap(h):
    """Compute the maximum overlap of h with shifted (1-h) over all shifts (circular)."""
    N = len(h)
    width = 2.0 / N
    max_overlap = 0.0
    for s in range(1, N):
        h_shifted = np.roll(h, -s)
        overlap = np.sum(h * (1.0 - h_shifted)) * width
        if overlap > max_overlap:
            max_overlap = overlap
    return max_overlap

def _soft_max(x, tau):
    """Smooth max via log-sum-exp."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    m = np.max(x)
    if m == -np.inf:
        return m
    return m + tau * np.log(np.sum(np.exp((x - m) / tau)))

def _overlaps(h):
    """Return array of overlaps for shifts 1..N-1."""
    N = len(h)
    width = 2.0 / N
    out = np.empty(N - 1)
    for s in range(1, N):
        hs = np.roll(h, -s)
        out[s - 1] = np.sum(h * (1.0 - hs)) * width
    return out

def _build_h_from_freq(coeffs, N):
    """Build h from frequency coefficients: h[i] = 0.5 + sum_k c_k * cos(2*pi*k*i/N)."""
    i = np.arange(N)
    h = np.full(N, 0.5)
    for k, c in enumerate(coeffs, start=1):
        h += c * np.cos(2.0 * np.pi * k * i / N)
    return h

def _project(h, N):
    """Project h to [0,1] with sum exactly N/2. Returns clipped array."""
    h = np.clip(h, 0.0, 1.0)
    # Adjust sum to be exactly N/2
    target = N / 2.0
    diff = target - np.sum(h)
    if abs(diff) < 1e-15:
        return h
    # Distribute adjustment: add to positions where we have room
    if diff > 0:
        # Need to increase; add to positions with h < 1
        room = 1.0 - h
        total_room = np.sum(room)
        if total_room < 1e-15:
            return h
        h = h + diff * room / total_room
    else:
        # Need to decrease; subtract from positions with h > 0
        room = h
        total_room = np.sum(room)
        if total_room < 1e-15:
            return h
        h = h + diff * room / total_room
    return np.clip(h, 0.0, 1.0)

def _coordinate_descent(h, max_iters=2000, tol=1e-10):
    """Coordinate descent: minimize max overlap by adjusting individual h[i]."""
    N = len(h)
    h = h.copy()
    best = compute_max_overlap(h)
    for it in range(max_iters):
        improved = False
        for i in range(N):
            h_old = h[i]
            # Try a few candidate values
            for h_new in [0.0, 0.25, 0.5, 0.75, 1.0]:
                if abs(h_new - h_old) < 1e-12:
                    continue
                h[i] = h_new
                # Check sum constraint
                s = np.sum(h)
                if abs(s - N/2) > 1e-8:
                    # Skip if violates sum
                    h[i] = h_old
                    continue
                score = compute_max_overlap(h)
                if score < best - 1e-12:
                    best = score
                    h_old = h_new
                    improved = True
            h[i] = h_old
        if not improved:
            break
    return h, best

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline with safety margin before 150s kill
    
    best_h = np.full(N, 0.5)
    best_score = compute_max_overlap(best_h)
    
    # Try some initial structured patterns
    initial_patterns = []
    # 1100 pattern
    if N % 4 == 0:
        p = np.tile(np.array([1.0, 1.0, 0.0, 0.0]), N // 4)
        initial_patterns.append(p)
    # 1010 pattern
    if N % 2 == 0:
        p = np.tile(np.array([1.0, 0.0]), N // 2)
        initial_patterns.append(p)
    # Block pattern: half ones, half zeros
    p = np.zeros(N)
    p[:N//2] = 1.0
    initial_patterns.append(p)
    
    for p in initial_patterns:
        if np.sum(p) == N/2:
            s = compute_max_overlap(p)
            if s < best_score:
                best_score = s
                best_h = p.copy()
    
    # Low-rank frequency optimization
    # Use K frequency components: h[i] = 0.5 + sum_{k=1}^{K} c_k cos(2*pi*k*i/N)
    # The mean is 0.5, so sum is automatically N/2 (cos has zero mean over full period)
    # This guarantees the sum constraint!
    
    K_max = 10  # Use up to 10 frequency components
    
    for K in range(1, K_max + 1):
        if time.time() > deadline - 20:
            break
        
        # Initialize coefficients
        rng = np.random.RandomState(42 + K)
        for trial in range(3):
            if time.time() > deadline - 15:
                break
            
            # Initialize with small random coefficients
            coeffs = rng.randn(K) * 0.1
            # Or try structured init: dominant low frequency
            if trial == 0:
                coeffs = np.zeros(K)
                coeffs[0] = 0.3  # Dominant first harmonic
            elif trial == 1:
                coeffs = np.zeros(K)
                coeffs[0] = 0.2
                if K > 1:
                    coeffs[1] = 0.15
            # trial == 2: random
            
            h = _build_h_from_freq(coeffs, N)
            h = _project(h, N)
            
            # Projected gradient descent on coefficients
            lr = 0.01
            tau = 0.01  # Soft-max temperature
            for step in range(500):
                if time.time() > deadline - 10:
                    break
                
                h = _build_h_from_freq(coeffs, N)
                h = _project(h, N)
                
                ov = _overlaps(h)
                obj = _soft_max(ov, tau)
                
                # Compute gradient w.r.t. coeffs
                # d(obj)/d(coeffs[k]) = sum_s [exp(ov[s]/tau) / sum(exp(ov[s]/tau))] * d(ov[s])/d(coeffs[k])
                # d(ov[s])/d(h[i]) = (1 - h[(i+s)%N]) * WIDTH
                # d(h[i])/d(coeffs[k]) = -2*pi*k/N * sin(2*pi*k*i/N)
                
                weights = np.exp(ov / tau)
                weights /= np.sum(weights)
                
                grad = np.zeros(K)
                i = np.arange(N)
                for k in range(K):
                    dh_dc = -2.0 * np.pi * k / N * np.sin(2.0 * np.pi * k * i / N)
                    # d(ov[s])/d(coeffs[k]) = sum_i (1-h[(i+s)%N]) * WIDTH * dh_dc[i]
                    # = WIDTH * sum_i (1-h[(i+s)%N]) * dh_dc[i]
                    # For efficiency, compute for all s at once
                    # d(ov[s]) = WIDTH * sum_i (1-h[(i+s)%N]) * dh_dc[i]
                    # = WIDTH * (sum_i dh_dc[i]) - WIDTH * sum_i h[(i+s)%N] * dh_dc[i]
                    # sum_i dh_dc[i] = 0 (integral of sin over full period)
                    # So d(ov[s]) = -WIDTH * sum_i h[(i+s)%N] * dh_dc[i]
                    # = -WIDTH * sum_i h[i] * dh_dc[(i-s)%N]  (by substitution)
                    # Let's just compute directly for each s
                    for s in range(1, N):
                        hs = np.roll(h, -s)
                        dov_dck = -WIDTH * np.sum(hs * dh_dc)
                        grad[k] += weights[s-1] * dov_dck
                
                # Update
                coeffs -= lr * grad
                # Keep coefficients small to stay in [0,1] after projection
                coeffs = np.clip(coeffs, -0.5, 0.5)
                
                # Adaptive learning rate
                if step % 50 == 0 and step > 0:
                    lr *= 0.95
            
            h = _build_h_from_freq(coeffs, N)
            h = _project(h, N)
            score = compute_max_overlap(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
        
        # Coordinate descent polish on best_h
        if time.time() < deadline - 15:
            h_cd, score_cd = _coordinate_descent(best_h, max_iters=500)
            if score_cd < best_score:
                best_score = score_cd
                best_h = h_cd.copy()
    
    # Final coordinate descent polish
    if time.time() < deadline - 5:
        h_cd, score_cd = _coordinate_descent(best_h, max_iters=1000)
        if score_cd < best_score:
            best_score = score_cd
            best_h = h_cd.copy()
    
    # Ensure sum is exactly N/2
    if abs(np.sum(best_h) - N/2) > 1e-10:
        # Fallback
        best_h = np.full(N, 0.5)
    
    return best_h
