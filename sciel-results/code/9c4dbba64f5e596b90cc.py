import time
import numpy as np

N = 100
_DEADLINE = time.time() + 115

def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs

def _score(h):
    """Compute the maximum linear overlap."""
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))

def _project(h):
    """
    Project h onto the polytope: 0 <= h_i <= 1, sum(h) = N/2.
    """
    h = np.asarray(h, dtype=np.float64).copy()
    target = N / 2.0
    
    # Bisection on lambda: h_i = clip(h_i + lambda, 0, 1)
    lo = -max(h) - 1.0
    hi = 1.0 - min(h) + 1.0
    
    for _ in range(80):
        mid = (lo + hi) / 2.0
        h_proj = np.clip(h + mid, 0.0, 1.0)
        s = np.sum(h_proj)
        if s > target:
            hi = mid
        else:
            lo = mid
    
    h_proj = np.clip(h + (lo + hi) / 2.0, 0.0, 1.0)
    # Final correction
    diff = target - np.sum(h_proj)
    if abs(diff) > 1e-14:
        mask = (h_proj > 1e-10) & (h_proj < 1 - 1e-10)
        if np.any(mask):
            h_proj[mask] += diff / np.sum(mask)
            h_proj = np.clip(h_proj, 0.0, 1.0)
        else:
            if diff > 0:
                idx = np.argmin(h_proj)
            else:
                idx = np.argmax(h_proj)
            h_proj[idx] += diff
            h_proj = np.clip(h_proj, 0.0, 1.0)
    return h_proj

def construct():
    """
    Simulated Annealing approach.
    """
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    # Initial candidates
    candidates = []
    i = np.arange(N)
    
    # 1. Constant
    candidates.append(np.full(N, 0.5))
    
    # 2. Cosine/Sine variations
    for freq in [1, 2, 3, 5]:
        candidates.append(0.5 + 0.5 * np.cos(2.0 * np.pi * freq * i / N))
        candidates.append(0.5 + 0.5 * np.sin(2.0 * np.pi * freq * i / N))
        
    # 3. Random
    for _ in range(10):
        candidates.append(np.random.uniform(0, 1, N))
        
    # 4. Binary-like (0 or 1) with sum N/2
    # This is a strong heuristic because the problem is quadratic.
    for _ in range(10):
        h_bin = np.zeros(N)
        indices = np.random.choice(N, N//2, replace=False)
        h_bin[indices] = 1.0
        candidates.append(h_bin)

    # Evaluate initial candidates
    for h in candidates:
        h_proj = _project(h)
        s = _score(h_proj)
        if s < best_score:
            best_score = s
            best_h = h_proj.copy()

    if best_h is None:
        best_h = np.full(N, 0.5)
        best_score = _score(best_h)

    # Simulated Annealing
    # Objective: Max Overlap
    # We want to minimize this.
    
    current_h = best_h.copy()
    current_score = best_score
    
    # Parameters
    T_start = 0.5
    T_end = 0.001
    alpha = 0.995  # Cooling rate
    T = T_start
    
    # We will run until deadline
    while time.time() < _DEADLINE - 2:
        # Generate a move
        # Strategy: 
        # 1. Pick two indices i, j
        # 2. Swap values? No, sum constraint makes swap tricky if we just swap. 
        #    Actually, swapping preserves sum.
        #    But we also need to respect [0,1].
        #    Better: Perturb two elements.
        
        # Move 1: Swap two elements
        if np.random.rand() < 0.5:
            i, j = np.random.choice(N, 2, replace=False)
            new_h = current_h.copy()
            new_h[i], new_h[j] = current_h[j], current_h[i]
        else:
            # Move 2: Perturb two elements in opposite directions to preserve sum
            i, j = np.random.choice(N, 2, replace=False)
            delta = np.random.uniform(-0.1, 0.1)
            new_h = current_h.copy()
            new_h[i] += delta
            new_h[j] -= delta
        
        # Project new_h
        new_h = _project(new_h)
        
        # Evaluate
        new_score = _score(new_h)
        
        # Acceptance probability
        if new_score < current_score:
            # Always accept improvement
            current_h = new_h
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_h = current_h.copy()
        else:
            # Accept with probability exp(-delta_E / T)
            delta_E = new_score - current_score
            if delta_E < 1e-10:
                prob = 1.0
            else:
                prob = np.exp(-delta_E / T)
            if np.random.rand() < prob:
                current_h = new_h
                current_score = new_score
        
        # Cool down
        T *= alpha
        if T < T_end:
            # Restart with some noise to escape local minima?
            # Or just stop?
            # Let's try a "reheat" strategy:
            # If we are stuck, inject noise.
            # Simple reheat:
            T = T_end * 10
            # Add small noise to current_h
            noise = np.random.normal(0, 0.05, N)
            current_h = _project(current_h + noise)
            current_score = _score(current_h)
            if current_score < best_score:
                best_score = current_score
                best_h = current_h.copy()

    return best_h
