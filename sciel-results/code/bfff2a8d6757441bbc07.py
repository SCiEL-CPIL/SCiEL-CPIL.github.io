import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_ratio(h):
    """
    Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf) for h >= 0.
    """
    h = np.asarray(h, dtype=np.float64)
    # Ensure non-negativity
    h = np.maximum(h, 0.0)
    
    l1 = np.sum(h)
    if l1 < 1e-300:
        return 0.0
    
    l2_sq = np.sum(h * h)
    
    # Efficient autoconvolution using FFT
    # h*h has length 2N-1
    # Zero pad to power of 2 for FFT efficiency
    n_fft = 1 << (2 * N - 1).bit_length()
    H = np.fft.rfft(h, n=n_fft)
    hh = np.fft.irfft(H * H, n=n_fft)
    hh_max = np.max(hh)
    
    if hh_max < 1e-300:
        return 0.0
        
    return l2_sq / (l1 * hh_max)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    rng = np.random.default_rng(42)
    
    best_h = None
    best_score = -np.inf
    
    # Strategy: 
    # 1. Discrete search for the optimal support set S (indices where h_i > 0).
    #    We prioritize "Sidon-like" sets where the number of collisions in the autoconvolution is minimized.
    #    For a fixed support S, the optimal weights can be found via continuous optimization.
    #    We use a stochastic search (simulated annealing style) to explore different supports.
    # 2. For each candidate support, we run a fast local optimization on the weights.
    #    Since the support size K is small (typically 5-15 for N=50), this is very fast.
    
    # Initialize with a random set of moderate size
    initial_K = 10
    init_indices = rng.choice(N, size=initial_K, replace=False)
    current_support = set(init_indices)
    
    # Initialize weights uniformly
    current_weights = np.ones(len(current_support)) / len(current_support)
    current_h = np.zeros(N)
    current_h[list(current_support)] = current_weights
    current_score = compute_ratio(current_h)
    
    if current_score > best_score:
        best_score = current_score
        best_h = current_h.copy()
    
    # Simulated Annealing parameters
    T_start = 0.1
    T_end = 1e-6
    max_steps = 50000  # We will do fewer steps if each is expensive, but weight opt is fast
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
            
        progress = step / max_steps
        T = T_start * (T_end / T_start) ** progress
        
        # Generate a new support by mutating the current one
        new_support = set(current_support)
        
        # Randomly change the support size or composition
        # With some probability, change the size
        if rng.random() < 0.1:
            # Change size
            new_K = max(3, min(N, len(current_support) + rng.integers(-3, 4)))
            # Resample support of size new_K, keeping some old elements
            old_list = list(current_support)
            if new_K <= len(old_list):
                # Keep a random subset of old, add new randoms? 
                # Just resample entirely for simplicity if size changes significantly
                # Or keep most and swap some
                keep_prob = len(old_list) / max(1, new_K)
                new_candidates = []
                for idx in old_list:
                    if rng.random() < keep_prob:
                        new_candidates.append(idx)
                # Fill rest with random
                while len(new_candidates) < new_K:
                    cand = rng.integers(0, N)
                    if cand not in new_candidates:
                        new_candidates.append(cand)
                new_support = set(new_candidates[:new_K])
            else:
                # Increase size: keep all old, add new
                new_candidates = list(old_list)
                while len(new_candidates) < new_K:
                    cand = rng.integers(0, N)
                    if cand not in new_candidates:
                        new_candidates.append(cand)
                new_support = set(new_candidates)
        else:
            # Same size: swap one element
            if len(new_support) > 0:
                old_elem = rng.choice(list(new_support))
                new_elem = rng.integers(0, N)
                new_support.remove(old_elem)
                new_support.add(new_elem)
        
        # If support is empty, skip
        if len(new_support) == 0:
            continue
            
        # Optimize weights for this specific support
        # We want to maximize R(h) subject to h_i >= 0 and sum(h_i) = 1 for i in support.
        # This is a non-convex problem, but for small K, we can use a simple local search or Nelder-Mead.
        # However, Nelder-Mead might be too slow if called 50k times.
        # Let's try a fast heuristic: 
        # 1. Start with uniform weights.
        # 2. Perform a few iterations of gradient-free coordinate ascent or random perturbation.
        
        K = len(new_support)
        w = np.ones(K) / K
        
        # Fast local optimization on weights
        # We will do a few steps of a simple randomized coordinate search
        # Since K is small, we can afford a slightly more robust search.
        
        # Define a fast objective function for weights only
        def neg_ratio_for_weights(w_vec, sup_list):
            # w_vec is already normalized? Let's assume yes.
            h_temp = np.zeros(N)
            h_temp[sup_list] = w_vec
            return -compute_ratio(h_temp)
        
        sup_list = list(new_support)
        
        # Simple hill climbing on weights
        best_w = w.copy()
        best_w_score = -neg_ratio_for_weights(best_w, sup_list)
        
        # Run a short simulated annealing on the weights
        # T_w from 0.05 to 0.001
        for w_step in range(50):
            w_T = 0.05 * (0.001/0.05) ** (w_step/50.0)
            
            # Perturb weights
            w_new = best_w.copy()
            idx = rng.integers(0, K)
            w_new[idx] += rng.normal(0, 0.1)
            w_new = np.maximum(w_new, 0)
            s = np.sum(w_new)
            if s < 1e-10:
                continue
            w_new /= s
            
            score_new = -neg_ratio_for_weights(w_new, sup_list)
            
            if score_new >= best_w_score:
                best_w = w_new
                best_w_score = score_new
            else:
                delta = score_new - best_w_score
                if delta < 0:
                    prob = np.exp(delta / w_T)
                    if rng.random() < prob:
                        best_w = w_new
                        best_w_score = score_new
        
        # Evaluate the best weight configuration for this support
        candidate_h = np.zeros(N)
        candidate_h[sup_list] = best_w
        candidate_score = best_w_score
        
        # Acceptance criterion for support change
        if candidate_score >= current_score:
            current_support = new_support
            current_weights = best_w
            current_h = candidate_h
            current_score = candidate_score
        else:
            delta = candidate_score - current_score
            if delta < 0:
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    current_support = new_support
                    current_weights = best_w
                    current_h = candidate_h
                    current_score = candidate_score
        
        # Track best
        if current_score > best_score:
            best_score = current_score
            best_h = current_h.copy()
            
    # Final Polish: 
    # Run a more rigorous optimization on the best support found.
    # Use scipy.optimize.minimize with Nelder-Mead on the full weight vector for the best support.
    
    if best_h is not None:
        support = np.where(best_h > 1e-8)[0]
        if len(support) > 0:
            K = len(support)
            w0 = best_h[support].copy()
            s0 = np.sum(w0)
            if s0 > 0:
                w0 /= s0
                
            def objective(w_vec):
                # w_vec are weights, must sum to 1 and be non-negative
                # We handle constraints by projection or just penalizing?
                # Nelder-Mead doesn't handle constraints well.
                # Let's stick to the simple hill climbing but with more steps if time permits.
                pass
            
            # Instead of complex constrained optimization, let's do a longer hill climb on the best support
            w_best = w0.copy()
            best_w_score = compute_ratio(best_h)
            
            # Time check
            if time.time() < deadline - 5.0:
                # Do 2000 steps of weight optimization
                for w_step in range(2000):
                    if time.time() > deadline - 2.0:
                        break
                    w_T = 0.02 * (0.0001/0.02) ** (w_step/2000.0)
                    
                    w_new = w_best.copy()
                    idx = rng.integers(0, K)
                    w_new[idx] += rng.normal(0, 0.05)
                    w_new = np.maximum(w_new, 0)
                    s = np.sum(w_new)
                    if s < 1e-10:
                        continue
                    w_new /= s
                    
                    h_temp = np.zeros(N)
                    h_temp[support] = w_new
                    score_new = compute_ratio(h_temp)
                    
                    if score_new >= best_w_score:
                        w_best = w_new
                        best_w_score = score_new
                    else:
                        delta = score_new - best_w_score
                        if delta < 0:
                            prob = np.exp(delta / w_T)
                            if rng.random() < prob:
                                w_best = w_new
                                best_w_score = score_new
                
                # Update best_h if improved
                if best_w_score > best_score:
                    best_score = best_w_score
                    best_h = np.zeros(N)
                    best_h[support] = w_best
                    
    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print(compute_ratio(h))
