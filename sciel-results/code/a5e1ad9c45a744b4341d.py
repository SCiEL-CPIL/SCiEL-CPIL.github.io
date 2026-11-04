import numpy as np
import time

N = 50
W = 1.0 / 25.0  # width of each interval

def _autoconv_max(h):
    """Compute the supremum norm of the autoconvolution of piecewise-constant h."""
    n = len(h)
    n_pad = 2 * n + 1
    h_pad = np.zeros(n_pad)
    h_pad[:n] = h
    h_conv = np.fft.ifft(np.fft.fft(h_pad) * np.fft.fft(h_pad))
    h_conv = np.real(h_conv)
    return np.max(h_conv) * W

def _ratio(h):
    """Compute the ratio ||h||^2_2 / (||h||_1 * ||h*h||_inf)."""
    l2_sq = np.sum(h**2) * W
    l1 = np.sum(h) * W
    autoconv_max = _autoconv_max(h)
    if l1 * autoconv_max < 1e-30:
        return 0.0
    return l2_sq / (l1 * autoconv_max)

def _local_search(h, max_iters=2000, tol=1e-10):
    """Coordinate descent + random perturbations to refine a candidate."""
    h = h.copy()
    h = np.maximum(h, 0.0)
    best_score = _ratio(h)
    rng = np.random.RandomState(123)
    
    for it in range(max_iters):
        improved = False
        # Try perturbing each coordinate
        for i in range(N):
            old_val = h[i]
            # Try several perturbation sizes
            for scale in [0.5, 0.1, 0.01, 0.001]:
                for direction in [-1, 1]:
                    new_val = max(0.0, old_val + direction * scale * max(old_val, 1.0))
                    if new_val == old_val:
                        continue
                    h[i] = new_val
                    s = _ratio(h)
                    if s > best_score + tol:
                        best_score = s
                        improved = True
                    else:
                        h[i] = old_val
                # Also try random values
                rand_val = rng.exponential(1.0) * max(old_val, 0.1)
                h[i] = rand_val
                s = _ratio(h)
                if s > best_score + tol:
                    best_score = s
                    improved = True
                else:
                    h[i] = old_val
        if not improved:
            break
    return h, best_score

def construct():
    deadline = time.time() + 115.0  # safety margin before 120s limit
    best_h = None
    best_score = 0.0
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Single spike at various positions
    for pos in range(N):
        h = np.zeros(N)
        h[pos] = 1.0
        candidates.append(h)
    
    # 2. Two-point spikes
    for pos1 in range(0, N, 5):
        for pos2 in range(pos1+1, min(pos1+10, N)):
            h = np.zeros(N)
            h[pos1] = 1.0
            h[pos2] = 1.0
            candidates.append(h)
    
    # 3. Concentrated triangular shapes
    for center in range(5, N-5, 3):
        for width in [1, 2, 3, 4, 5]:
            h = np.zeros(N)
            for i in range(max(0, center-width), min(N, center+width+1)):
                h[i] = 1.0 - abs(i - center) / (width + 1)
            candidates.append(h)
    
    # 4. Random concentrated shapes
    rng = np.random.RandomState(42)
    for _ in range(20):
        h = rng.exponential(1.0, N)
        # Concentrate: keep only top few
        top_k = rng.randint(1, 8)
        idx = np.argsort(h)[-top_k:]
        h2 = np.zeros(N)
        h2[idx] = h[idx]
        candidates.append(h2)
    
    # 5. Some structured patterns
    h = np.zeros(N)
    h[25] = 1.0
    candidates.append(h)
    
    h = np.zeros(N)
    h[24:26] = 1.0
    candidates.append(h)
    
    h = np.zeros(N)
    h[23:27] = [0.5, 1.0, 1.0, 0.5]
    candidates.append(h)
    
    # Evaluate all candidates and pick top ones for refinement
    scored = []
    for h in candidates:
        s = _ratio(h)
        scored.append((s, h))
    
    scored.sort(key=lambda x: -x[0])
    
    # Refine top candidates with local search
    top_k = min(15, len(scored))
    for i in range(top_k):
        if time.time() > deadline - 10:
            break
        s, h = scored[i]
        if s > best_score:
            best_score = s
            best_h = h.copy()
        
        # Local refinement
        h_refined, s_refined = _local_search(h, max_iters=1000)
        if s_refined > best_score:
            best_score = s_refined
            best_h = h_refined.copy()
    
    # Additional: try optimizing a parameterized family
    if time.time() < deadline - 20:
        # Parameterized: center spike with asymmetric wings
        for center in [20, 25, 30]:
            for left_w in [0, 1, 2, 3]:
                for right_w in [0, 1, 2, 3]:
                    for peak in [1.0, 2.0, 5.0, 10.0]:
                        h = np.zeros(N)
                        h[center] = peak
                        for d in range(1, left_w+1):
                            if center-d >= 0:
                                h[center-d] = peak * (1.0/d)**2
                        for d in range(1, right_w+1):
                            if center+d < N:
                                h[center+d] = peak * (1.0/d)**2
                        s = _ratio(h)
                        if s > best_score:
                            best_score = s
                            best_h = h.copy()
                        
                        # Quick local refine
                        if s > best_score * 0.9:
                            h_ref, s_ref = _local_search(h, max_iters=500)
                            if s_ref > best_score:
                                best_score = s_ref
                                best_h = h_ref.copy()
    
    # Final polish
    if best_h is not None and time.time() < deadline - 5:
        h_final, s_final = _local_search(best_h, max_iters=3000)
        if s_final > best_score:
            best_score = s_final
            best_h = h_final
    
    if best_h is None:
        best_h = np.zeros(N)
        best_h[25] = 1.0
    
    return list(best_h)
