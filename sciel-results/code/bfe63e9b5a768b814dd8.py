"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 50

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 135  # safety margin before 150s kill
    
    def compute_score(heights):
        """Compute the ratio: ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
        h = np.asarray(heights, dtype=float)
        l2_sq = np.sum(h**2)
        l1 = np.sum(h)
        if l1 < 1e-12:
            return 0.0
        # Autoconvolution using FFT
        n_conv = 2 * N
        h_padded = np.zeros(n_conv)
        h_padded[:N] = h
        conv = np.fft.irfft(np.fft.rfft(h_padded) ** 2)
        linf = np.max(np.abs(conv[:N]))
        if linf < 1e-12:
            return 0.0
        return l2_sq / (l1 * linf)
    
    def find_sidon_set(n, target_size, seed=0):
        """Find a Sidon set of size target_size within range(n)."""
        rng = random.Random(seed)
        best_sidon = None
        best_max_conv = float('inf')
        
        for attempt in range(20):
            # Greedy construction with randomization
            sidon = set()
            candidates = list(range(n))
            rng.shuffle(candidates)
            
            for i in candidates:
                if len(sidon) >= target_size:
                    break
                # Check if adding i keeps it a Sidon set
                new_set = sidon | {i}
                lst = sorted(new_set)
                sums = set()
                valid = True
                for idx_a in range(len(lst)):
                    for idx_b in range(idx_a, len(lst)):
                        s = lst[idx_a] + lst[idx_b]
                        if s in sums:
                            valid = False
                            break
                        sums.add(s)
                    if not valid:
                        break
                if valid:
                    sidon.add(i)
            
            if len(sidon) >= target_size:
                # Evaluate this Sidon set
                h = np.zeros(n)
                for i in sidon:
                    h[i] = 1.0
                conv = np.fft.irfft(np.fft.rfft(np.zeros(2*n).__setitem__(slice(0,n), h)) ** 2)
                max_conv = np.max(np.abs(conv[:n]))
                if max_conv < best_max_conv:
                    best_max_conv = max_conv
                    best_sidon = list(sidon)
        
        return best_sidon
    
    def local_search_sidon(sidon_list, n, max_iters=500):
        """Local search: swap elements to minimize max convolution value."""
        current = list(sidon_list)
        current_set = set(current)
        
        def get_max_conv(s):
            h = np.zeros(n)
            for i in s:
                h[i] = 1.0
            h_padded = np.zeros(2*n)
            h_padded[:n] = h
            conv = np.fft.irfft(np.fft.rfft(h_padded) ** 2)
            return np.max(np.abs(conv[:n]))
        
        current_max_conv = get_max_conv(current_set)
        improved = True
        iters = 0
        
        while improved and iters < max_iters and time.time() < deadline - 15:
            improved = False
            iters += 1
            # Try swapping each element with each non-element
            for idx, elem in enumerate(current):
                for new_elem in range(n):
                    if new_elem in current_set:
                        continue
                    # Try the swap
                    trial_set = current_set.copy()
                    trial_set.remove(elem)
                    trial_set.add(new_elem)
                    trial_max_conv = get_max_conv(trial_set)
                    if trial_max_conv < current_max_conv - 1e-10:
                        current[idx] = new_elem
                        current_set = trial_set
                        current_max_conv = trial_max_conv
                        improved = True
                        break
                if improved:
                    break
        
        return current, current_max_conv
    
    # Start with uniform
    best_heights = np.ones(N)
    best_score = compute_score(best_heights)
    
    # Try various Sidon set sizes
    for target_size in [10, 12, 14, 16, 18, 20]:
        if time.time() > deadline - 30:
            break
        
        # Try multiple seeds for diversity
        for seed in range(5):
            if time.time() > deadline - 25:
                break
            
            sidon = find_sidon_set(N, target_size, seed=seed)
            if sidon is None or len(sidon) < 4:
                continue
            
            # Local search to improve
            improved_sidon, max_conv = local_search_sidon(sidon, N)
            
            h = np.zeros(N)
            for i in improved_sidon:
                h[i] = 1.0
            s = compute_score(h)
            
            if s > best_score:
                best_score = s
                best_heights = h.copy()
    
    # Refine: try continuous heights on the best support
    if time.time() < deadline - 20:
        support = np.where(best_heights > 0)[0]
        if len(support) > 0:
            # Try different height values on support
            for height_val in [1.0, 0.8, 1.2, 0.5, 1.5]:
                h = np.zeros(N)
                h[support] = height_val
                s = compute_score(h)
                if s > best_score:
                    best_score = s
                    best_heights = h.copy()
            
            # Try asymmetric heights: higher in center of support
            n_support = len(support)
            if n_support >= 3:
                for center_weight in [1.0, 1.5, 2.0]:
                    h = np.zeros(N)
                    for i, idx in enumerate(support):
                        # Triangular weight peaking at center
                        dist = abs(i - n_support // 2)
                        h[idx] = center_weight - dist * 0.1
                        h[idx] = max(h[idx], 0.1)
                    s = compute_score(h)
                    if s > best_score:
                        best_score = s
                        best_heights = h.copy()
    
    # Ensure nonnegative and valid
    best_heights = np.maximum(best_heights, 0)
    if np.sum(best_heights) < 1e-12:
        return [1.0] * N
    
    return list(best_heights)
# EVOLVE-BLOCK-END
