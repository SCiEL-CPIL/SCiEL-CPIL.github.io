import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve
from itertools import combinations

N = 50

def compute_ratio(h):
    """Compute the ratio: (L2 norm squared) / (L1 norm * sup norm of autoconvolution)
    for a vector of nonnegative heights."""
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    conv = fftconvolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return 0.0
    return h2_sq / conv_inf

def is_sidon_set(indices):
    """Check if a set of indices forms a Sidon set (all pairwise sums are distinct)."""
    sums = set()
    for i in range(len(indices)):
        for j in range(i, len(indices)):
            s = indices[i] + indices[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def generate_sidon_sets(n_max, min_size, max_size, max_sets=500):
    """Generate Sidon sets of various sizes within [0, n_max)."""
    sidon_sets = []
    all_indices = list(range(n_max))
    
    for size in range(min_size, max_size + 1):
        if len(sidon_sets) >= max_sets:
            break
        # Try to find Sidon sets of this size
        # Use a greedy approach with backtracking
        def find_sidon_sets(remaining_indices, current_set, depth):
            if len(current_set) == size:
                if is_sidon_set(current_set):
                    sidon_sets.append(list(current_set))
                return
            if depth >= len(remaining_indices) or len(sidon_sets) >= max_sets:
                return
            for i in range(depth, len(remaining_indices)):
                candidate = remaining_indices[i]
                # Quick check: does adding candidate maintain Sidon property?
                test_set = list(current_set) + [candidate]
                if is_sidon_set(test_set):
                    current_set.append(candidate)
                    find_sidon_sets(remaining_indices, current_set, i + 1)
                    current_set.pop()
        
        find_sidon_sets(all_indices, [], 0)
    
    return sidon_sets

def optimize_weights_for_support(support_indices, h_init=None, max_iter=2000):
    """Optimize weights on a fixed support to maximize the ratio."""
    n_support = len(support_indices)
    if n_support == 0:
        return None, 0.0
    
    def objective(w):
        # w is the weight vector on the support
        w = np.maximum(w, 1e-12)
        w = w / np.sum(w)  # normalize to L1=1
        h = np.zeros(N)
        h[support_indices] = w
        h2 = np.sum(w**2)
        conv = fftconvolve(h, h, mode='full')
        cinf = np.max(conv)
        if cinf < 1e-15:
            return 1e10
        return -h2 / cinf
    
    if h_init is not None:
        w0 = h_init[support_indices].copy()
        w0 = np.maximum(w0, 1e-6)
        w0 = w0 / np.sum(w0)
    else:
        w0 = np.ones(n_support) / n_support
    
    try:
        res = minimize(objective, w0, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
        w_opt = np.maximum(res.x, 1e-12)
        w_opt = w_opt / np.sum(w_opt)
        h_opt = np.zeros(N)
        h_opt[support_indices] = w_opt
        score = compute_ratio(h_opt)
        return h_opt, score
    except Exception:
        return None, 0.0

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin: kill at 150s, return by 145s

    best_h = np.zeros(N, dtype=np.float64)
    best_h[0] = 1.0
    best_score = compute_ratio(best_h)

    def update_best(h, sc):
        nonlocal best_h, best_score
        if sc > best_score:
            best_score = sc
            best_h = h.copy()

    # Phase 1: Generate and test Sidon sets
    # Start with smaller sets and grow
    sidon_candidates = []
    
    # Try to find good Sidon sets of various sizes
    for target_size in [3, 4, 5, 6, 7, 8, 9, 10]:
        if time.time() > deadline - 30.0:
            break
        
        # Generate candidate sets using random sampling + verification
        np.random.seed(42 + target_size)
        found = 0
        max_attempts = 500
        
        for attempt in range(max_attempts):
            if found >= 20:  # Cap per size
                break
            if time.time() > deadline - 30.0:
                break
            
            # Random subset
            indices = sorted(np.random.choice(N, target_size, replace=False))
            
            if is_sidon_set(indices):
                sidon_candidates.append(indices)
                found += 1
    
    # Also try structured Sidon sets: powers of 2, Fibonacci-like
    # Powers of 2: 0, 1, 2, 4, 8, 16, 32 (within 50)
    for base in [2, 3]:
        powers = []
        val = 0
        while val < N:
            powers.append(val)
            val = val * base + 1 if base == 2 else val + base
        if len(powers) >= 3:
            for size in range(3, min(len(powers), 10) + 1):
                sidon_candidates.append(powers[:size])
    
    # Try arithmetic progressions with prime gaps (often Sidon-like)
    for gap in [2, 3, 5, 7, 11, 13]:
        for start in range(0, 5):
            seq = []
            val = start
            while val < N and len(seq) < 12:
                seq.append(val)
                val += gap
            for size in range(3, min(len(seq), 10) + 1):
                candidate = seq[:size]
                if is_sidon_set(candidate):
                    sidon_candidates.append(candidate)
    
    # Deduplicate and sort by size (larger sets tend to give better ratios)
    unique_sidon = []
    seen = set()
    for s in sidon_candidates:
        key = tuple(s)
        if key not in seen:
            seen.add(key)
            unique_sidon.append(s)
    
    unique_sidon.sort(key=lambda x: -len(x))
    
    # Optimize weights on each Sidon set
    for sidon_set in unique_sidon[:50]:  # Limit to 50 best candidates
        if time.time() > deadline - 15.0:
            break
        
        h_opt, score = optimize_weights_for_support(sidon_set, max_iter=3000)
        if h_opt is not None:
            update_best(h_opt, score)
    
    # Phase 2: Local search to improve the best solution
    if best_score > 0.9 and time.time() < deadline - 10.0:
        # Try swapping support indices
        best_support = np.where(best_h > 1e-6)[0]
        
        for _ in range(100):
            if time.time() > deadline - 5.0:
                break
            
            # Try moving one support element to a new location
            if len(best_support) == 0:
                break
            
            idx_to_move = np.random.randint(len(best_support))
            old_idx = best_support[idx_to_move]
            
            # Try new locations
            improved = False
            for new_idx in range(N):
                if new_idx == old_idx:
                    continue
                if new_idx in best_support:
                    continue
                
                new_support = list(best_support)
                new_support[idx_to_move] = new_idx
                new_support.sort()
                
                if not is_sidon_set(new_support):
                    continue
                
                h_test, score_test = optimize_weights_for_support(new_support, max_iter=1000)
                if h_test is not None and score_test > best_score:
                    best_h = h_test.copy()
                    best_score = score_test
                    best_support = np.array(new_support)
                    improved = True
                    break
            
            if not improved:
                # Try adding a new support element
                if len(best_support) < 12:
                    for new_idx in range(N):
                        if new_idx in best_support:
                            continue
                        new_support = list(best_support) + [new_idx]
                        new_support.sort()
                        
                        if is_sidon_set(new_support):
                            h_test, score_test = optimize_weights_for_support(new_support, max_iter=1000)
                            if h_test is not None and score_test > best_score:
                                best_h = h_test.copy()
                                best_score = score_test
                                best_support = np.array(new_support)
                                break
    
    # Phase 3: Final polish with tighter optimization
    if best_score > 0.9 and time.time() < deadline - 3.0:
        best_support = np.where(best_h > 1e-6)[0]
        if len(best_support) > 0:
            h_final, score_final = optimize_weights_for_support(best_support, max_iter=10000)
            if h_final is not None and score_final > best_score:
                best_h = h_final.copy()
                best_score = score_final
    
    return list(best_h)
