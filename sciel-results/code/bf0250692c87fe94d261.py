import numpy as np
import time
from scipy.optimize import linprog, minimize

N = 100

def _max_overlap_fast(h):
    """Compute max overlap of h with shifted (1-h) using FFT."""
    H = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(H * np.conj(H)))
    min_autocorr = np.min(autocorr)
    max_overlap = N / 2.0 - min_autocorr
    return max_overlap

def _project_to_feasible(h):
    """Project h to satisfy: sum = N/2, 0 <= h[i] <= 1."""
    h = h.copy()
    h = np.clip(h, 0, 1)
    target = N / 2.0
    current_sum = np.sum(h)
    diff = target - current_sum
    if abs(diff) > 1e-12:
        if diff > 0:
            room = 1.0 - h
            room_sum = np.sum(room)
            if room_sum > 1e-12:
                h += diff * room / room_sum
        else:
            h_sum = np.sum(h)
            if h_sum > 1e-12:
                h += diff * h / h_sum
        h = np.clip(h, 0, 1)
    # Final adjustment
    current_sum = np.sum(h)
    if abs(current_sum - target) > 1e-10:
        residual = target - current_sum
        if residual > 0:
            room = 1.0 - h
            mask = room > 1e-10
            if np.sum(mask) > 0:
                h[mask] += residual / np.sum(mask)
        else:
            mask = h > 1e-10
            if np.sum(mask) > 0:
                h[mask] += residual / np.sum(mask)
        h = np.clip(h, 0, 1)
    return h

def _smooth_max_overlap(h, t=10.0):
    """Smooth approximation of max overlap using softmax."""
    H = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(H * np.conj(H)))
    overlaps = N / 2.0 - autocorr
    # Softmax approximation
    max_val = np.max(overlaps)
    exp_overlaps = np.exp(t * (overlaps - max_val))
    smooth_max = max_val + np.log(np.mean(exp_overlaps)) / t
    return smooth_max

def _smooth_max_overlap_grad(h, t=10.0):
    """Gradient of smooth max overlap."""
    H = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(H * np.conj(H)))
    overlaps = N / 2.0 - autocorr
    
    max_val = np.max(overlaps)
    exp_overlaps = np.exp(t * (overlaps - max_val))
    weights = exp_overlaps / np.sum(exp_overlaps)
    
    # Gradient of autocorr_k with respect to h
    # autocorr_k = sum_i h[i] * h[(i-k)%N]
    # d(autocorr_k)/dh[j] = h[(j-k)%N] + h[(j+k)%N]
    # d(overlap_k)/dh[j] = -(h[(j-k)%N] + h[(j+k)%N])
    
    grad = np.zeros(N)
    for k in range(N):
        w = weights[k]
        if w > 1e-12:
            # autocorr_k = sum_i h[i]*h[(i-k)%N]
            # d(autocorr_k)/dh[j] = h[(j-k)%N] + (if k!=0) h[(j+k)%N]
            # Actually: autocorr_k = sum_i h[i] h[(i-k)%N]
            # Let j be fixed: term involving h[j] is h[j]*h[(j-k)%N] + h[(j+k)%N]*h[j] (when i=j+k)
            # Wait, let's be careful:
            # autocorr_k = sum_{i=0}^{N-1} h[i] * h[(i-k) mod N]
            # For a fixed j, h[j] appears when:
            #   i = j: term is h[j] * h[(j-k)%N]
            #   (i-k)%N = j => i = (j+k)%N: term is h[(j+k)%N] * h[j]
            # So d(autocorr_k)/dh[j] = h[(j-k)%N] + h[(j+k)%N] (but if k=0, these are the same, so it's 2*h[j])
            
            if k == 0:
                d_ac_k = 2.0 * h
            else:
                d_ac_k = np.roll(h, k) + np.roll(h, -k)
            # d(overlap_k)/dh = -d(autocorr_k)/dh
            grad -= w * d_ac_k
    
    return grad

def _coordinate_descent(h0, max_iters=5000, time_limit=20.0):
    """Coordinate descent with smooth max overlap."""
    h = h0.copy()
    t = 5.0
    start_time = time.time()
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        if time.time() - start_time > time_limit * 0.5:
            t = min(t * 1.5, 50.0)
        
        improved = False
        # Try a random subset of coordinates
        n_try = min(20, N)
        indices = np.random.choice(N, size=n_try, replace=False)
        
        for i in indices:
            grad_i = _smooth_max_overlap_grad(h, t)[i]
            step_size = 0.1 / max(1.0, abs(grad_i))
            
            best_val = _smooth_max_overlap(h, t)
            best_h_i = h[i]
            
            for delta in [step_size, -step_size, step_size*0.5, -step_size*0.5, step_size*2, -step_size*2]:
                new_h = h.copy()
                new_h[i] = np.clip(new_h[i] + delta, 0, 1)
                new_h = _project_to_feasible(new_h)
                val = _smooth_max_overlap(new_h, t)
                if val < best_val - 1e-10:
                    best_val = val
                    best_h_i = new_h[i]
                    h = new_h
                    improved = True
                    break
        
        if not improved and it % 100 == 0:
            # Try small random perturbations
            for _ in range(10):
                new_h = h + np.random.uniform(-0.05, 0.05, N)
                new_h = _project_to_feasible(new_h)
                val = _smooth_max_overlap(new_h, t)
                best_val = _smooth_max_overlap(h, t)
                if val < best_val - 1e-10:
                    h = new_h
                    improved = True
                    break
    
    return h

def _lp_solution():
    """
    Try to solve via linear programming.
    
    We want to minimize T such that for all k:
    overlap_k = sum_i h[i]*(1-h[(i-k)%N]) <= T
    
    This is non-linear. But we can try a linear relaxation.
    
    Alternative: Use the fact that overlap_k = N/2 - autocorr_k.
    We want to maximize min_k autocorr_k.
    
    For a given target autocorr value c, we need:
    autocorr_k >= c for all k
    0 <= h[i] <= 1
    sum h[i] = N/2
    
    autocorr_k = sum_i h[i] h[(i-k)%N] >= c
    
    This is still non-linear. Let's try a different approach.
    
    For small N or with clever reformulation...
    
    Actually, let's try: minimize T subject to
    sum_i h[i] - sum_i h[i]h[(i-k)%N] <= T for all k
    0 <= h[i] <= 1
    sum h[i] = N/2
    
    Still non-linear. Let's just use it as a heuristic starting point.
    """
    # Try a simple LP relaxation: assume autocorr_k >= c
    # For k=0: autocorr_0 = sum h[i]^2 >= c
    # For other k: sum_i h[i] h[(i-k)%N] >= c
    
    # Let's just generate a few good candidates
    candidates = []
    
    # Alternating pattern
    h_alt = np.zeros(N)
    h_alt[::2] = 1.0
    h_alt = _project_to_feasible(h_alt)
    candidates.append(h_alt)
    
    # Sine-based candidates with various frequencies
    for freq in range(1, N//2):
        for phase in np.linspace(0, 2*np.pi, 8, endpoint=False):
            x = np.arange(N)
            h = 0.5 + 0.48 * np.sin(2*np.pi*freq*x/N + phase)
            h = _project_to_feasible(h)
            candidates.append(h)
    
    # Random binary-like
    rng = np.random.RandomState(123)
    for _ in range(30):
        h = (rng.rand(N) > 0.5).astype(float)
        h = _project_to_feasible(h)
        candidates.append(h)
    
    # Smooth random
    for _ in range(30):
        h = 0.5 + 0.3 * np.sin(2*np.pi*rng.rand()*N + rng.rand()*2*np.pi)
        h = 0.5 + 0.2 * np.sin(2*np.pi*rng.rand()*N + rng.rand()*2*np.pi)
        h = _project_to_feasible(h)
        candidates.append(h)
    
    best_h = None
    best_score = np.inf
    for h in candidates:
        score = _max_overlap_fast(h)
        if score < best_score:
            best_score = score
            best_h = h
    
    return best_h, best_score

def _cluster_move(h, n_clusters=5, cluster_size=None, time_limit=5.0):
    """Swendsen-Wang style cluster moves for near-binary solutions."""
    start_time = time.time()
    h = h.copy()
    current_score = _max_overlap_fast(h)
    
    if cluster_size is None:
        cluster_size = max(1, N // n_clusters)
    
    for _ in range(20):
        if time.time() - start_time > time_limit:
            break
        
        # Create random clusters
        idx = np.random.permutation(N)
        clusters = [idx[i:i+cluster_size] for i in range(0, N, cluster_size)]
        
        # For each cluster, try flipping or shifting the pattern
        new_h = h.copy()
        for cl in clusters:
            # Option 1: Flip values in cluster
            if np.random.rand() < 0.3:
                for i in cl:
                    new_h[i] = 1.0 - h[i]
            # Option 2: Shift cluster content
            elif np.random.rand() < 0.5:
                shift = np.random.randint(1, cluster_size)
                cluster_vals = new_h[cl].copy()
                new_h[cl] = np.roll(cluster_vals, shift)
            # Option 3: Smooth within cluster
            else:
                cluster_vals = new_h[cl].copy()
                # Apply a small smoothing
                kernel = np.array([0.3, 0.4, 0.3])
                if len(cl) >= 3:
                    smoothed = np.convolve(cluster_vals, kernel, mode='same')
                    new_h[cl] = np.clip(smoothed, 0, 1)
        
        new_h = _project_to_feasible(new_h)
        new_score = _max_overlap_fast(new_h)
        
        if new_score < current_score - 1e-10:
            h = new_h
            current_score = new_score
    
    return h, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    # Initial candidates
    h_init, score_init = _lp_solution()
    best_h = h_init.copy()
    best_score = score_init
    
    # Local search with coordinate descent
    if time.time() < deadline - 30:
        remaining = min(20.0, deadline - time.time() - 10)
        h_opt = _coordinate_descent(best_h, max_iters=3000, time_limit=remaining)
        score_opt = _max_overlap_fast(h_opt)
        if score_opt < best_score:
            best_score = score_opt
            best_h = h_opt
    
    # Cluster moves
    if time.time() < deadline - 20:
        remaining = min(15.0, deadline - time.time() - 10)
        h_cl, score_cl = _cluster_move(best_h, time_limit=remaining)
        if score_cl < best_score:
            best_score = score_cl
            best_h = h_cl
    
    # More coordinate descent from cluster result
    if time.time() < deadline - 15:
        remaining = min(10.0, deadline - time.time() - 5)
        h_opt = _coordinate_descent(best_h, max_iters=2000, time_limit=remaining)
        score_opt = _max_overlap_fast(h_opt)
        if score_opt < best_score:
            best_score = score_opt
            best_h = h_opt
    
    # Final cluster refinement
    if time.time() < deadline - 10:
        remaining = min(8.0, deadline - time.time() - 5)
        h_cl, score_cl = _cluster_move(best_h, time_limit=remaining)
        if score_cl < best_score:
            best_score = score_cl
            best_h = h_cl
    
    # Final coordinate descent polish
    if time.time() < deadline - 5:
        remaining = min(5.0, deadline - time.time() - 2)
        h_opt = _coordinate_descent(best_h, max_iters=1000, time_limit=remaining)
        score_opt = _max_overlap_fast(h_opt)
        if score_opt < best_score:
            best_score = score_opt
            best_h = h_opt
    
    # Ensure validity
    best_h = _project_to_feasible(best_h)
    
    return best_h
