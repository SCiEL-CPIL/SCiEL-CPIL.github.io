import numpy as np
import time

N = 50

def compute_ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf) for nonnegative h."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    conv = np.convolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return 0.0
    return h2_sq / conv_inf

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    best_h = np.zeros(N)
    best_h[0] = 1.0
    best_score = compute_ratio(best_h)
    
    # Key insight: sparse configurations with one dominant peak and smaller
    # well-separated peaks can achieve ratio > 1.
    # For 3 peaks (1/2, 1/4, 1/4) at well-separated positions, ratio = 1.5
    
    def make_sparse_peaks(heights, positions):
        """Create a vector with given heights at given positions."""
        h = np.zeros(N)
        for ht, pos in zip(heights, positions):
            h[pos] = ht
        return h
    
    # Try various sparse peak configurations
    candidates = []
    
    # Well-separated position sets to avoid convolution collisions
    pos_sets = [
        [0, 10, 30],
        [0, 15, 35],
        [0, 12, 33],
        [0, 10, 25, 40],
        [0, 8, 20, 35],
        [0, 10, 20, 30, 40],
        [0, 7, 18, 32, 45],
        [0, 5, 15, 25, 35, 45],
    ]
    
    # Height patterns
    height_patterns = []
    
    # One dominant, rest equal
    for k in range(1, 7):
        for a in np.linspace(0.3, 0.95, 15):
            if k == 1:
                height_patterns.append([1.0])
            else:
                b = (1 - a) / (k - 1)
                height_patterns.append([a] + [b] * (k - 1))
    
    # Geometric
    for k in range(2, 7):
        for r in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            heights = [1.0]
            for i in range(1, k):
                heights.append(heights[-1] * r)
            total = sum(heights)
            height_patterns.append([h / total for h in heights])
    
    # Specific promising configurations
    height_patterns.append([0.5, 0.25, 0.25])
    height_patterns.append([0.6, 0.2, 0.2])
    height_patterns.append([0.7, 0.15, 0.15])
    height_patterns.append([0.8, 0.1, 0.1])
    height_patterns.append([0.5, 0.3, 0.2])
    height_patterns.append([0.4, 0.3, 0.3])
    height_patterns.append([0.5, 0.25, 0.125, 0.125])
    height_patterns.append([0.4, 0.2, 0.2, 0.2])
    height_patterns.append([0.3, 0.2, 0.2, 0.2, 0.1])
    
    # Generate candidates
    for heights in height_patterns:
        k = len(heights)
        # Use position sets that match or exceed k
        for pos_set in pos_sets:
            if len(pos_set) >= k:
                positions = pos_set[:k]
                h = make_sparse_peaks(heights, positions)
                candidates.append(h)
    
    # Also try random sparse configurations
    np.random.seed(42)
    for _ in range(200):
        k = np.random.randint(1, 8)
        positions = np.random.choice(N, k, replace=False)
        heights = np.random.dirichlet(np.ones(k) * np.random.uniform(0.5, 5.0, k))
        h = make_sparse_peaks(heights, positions)
        candidates.append(h)
    
    # Evaluate all candidates
    for h in candidates:
        if time.time() > deadline - 15:
            break
        score = compute_ratio(h)
        if score > best_score:
            best_score = score
            best_h = h.copy()
    
    # Local search: coordinate descent
    def local_search(h, max_iter=5000):
        h = h.copy()
        h = h / np.sum(h)
        cur_score = compute_ratio(h)
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            i = np.random.randint(N)
            for delta in [0.005, 0.01, 0.05, 0.1, 0.2]:
                for sign in [1, -1]:
                    h_new = h.copy()
                    h_new[i] += sign * delta
                    if h_new[i] < 0:
                        continue
                    h_new = h_new / np.sum(h_new)
                    new_score = compute_ratio(h_new)
                    if new_score > cur_score:
                        cur_score = new_score
                        h = h_new
                        break
        return h, cur_score
    
    # Run local search from best candidate
    if time.time() < deadline - 15:
        h_local, score_local = local_search(best_h)
        if score_local > best_score:
            best_score = score_local
            best_h = h_local
    
    # Random restarts with local search
    if time.time() < deadline - 10:
        for restart in range(15):
            if time.time() > deadline - 5:
                break
            k = np.random.randint(2, 8)
            positions = np.random.choice(N, k, replace=False)
            heights = np.random.dirichlet(np.ones(k) * np.random.uniform(0.5, 3.0, k))
            h = make_sparse_peaks(heights, positions)
            h_local, score_local = local_search(h, max_iter=3000)
            if score_local > best_score:
                best_score = score_local
                best_h = h_local
    
    # Final refinement: try to sharpen
    if time.time() < deadline - 5:
        for _ in range(200):
            if time.time() > deadline - 3:
                break
            i = np.random.randint(N)
            h_new = best_h.copy()
            h_new[i] *= np.random.uniform(0.5, 2.0)
            h_new = h_new / np.sum(h_new)
            if compute_ratio(h_new) > best_score:
                best_score = compute_ratio(h_new)
                best_h = h_new
    
    return list(best_h)
