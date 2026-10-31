import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    best_score = -1
    best_set = [0]
    
    def compute_score(U):
        if not U or 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 0
        
        U_arr = np.array(U, dtype=np.int64)
        n = len(U_arr)
        
        # Use FFT-based method for larger sets, direct for small
        if n > 15:
            n_sum, n_diff = _fft_count_sums_diffs(U_arr, max_u)
        else:
            # Direct computation for small sets
            sums = U_arr[:, None] + U_arr[None, :]
            diffs = U_arr[:, None] - U_arr[None, :]
            n_sum = len(np.unique(sums))
            n_diff = len(np.unique(diffs))
        
        if n_sum == 0:
            return -1
        
        base = 2 * max_u + 1
        if base <= 1:
            return 0
        
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        score = 1 + math.log(ratio) / math.log(base)
        return score

    def _fft_count_sums_diffs(U_arr, max_u):
        """Count unique sums and differences using FFT convolution."""
        size = 2 * max_u + 2
        fft_size = 1
        while fft_size < size:
            fft_size *= 2
        
        indicator = np.zeros(fft_size, dtype=np.float64)
        indicator[U_arr] = 1.0
        
        fft_ind = np.fft.fft(indicator)
        autocorr = np.fft.ifft(fft_ind * np.conj(fft_ind)).real
        n_sum = np.sum(autocorr[:2*max_u+1] > 0.5)
        
        indicator_rev = np.zeros(fft_size, dtype=np.float64)
        indicator_rev[max_u - U_arr] = 1.0
        fft_rev = np.fft.fft(indicator_rev)
        cross_corr = np.fft.ifft(fft_ind * np.conj(fft_rev)).real
        n_diff = np.sum(cross_corr[max_u:max_u*2+1] > 0.5)
        
        return int(n_sum), int(n_diff)

    def try_set(cand):
        nonlocal best_score, best_set
        cand = sorted(set(cand))
        if not cand or 0 not in cand:
            return
        if max(cand) == 0:
            return
        s = compute_score(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # 1. Generate Golomb Rulers (Maximize |U-U| for given n)
    # Greedy construction of Golomb rulers
    def generate_golomb(n, max_val=10000):
        U = [0]
        used_diffs = set()
        for _ in range(1, n):
            found = False
            for x in range(1, max_val + 1):
                # Check if x can be added
                valid = True
                for u in U:
                    d = abs(x - u)
                    if d in used_diffs:
                        valid = False
                        break
                if valid:
                    U.append(x)
                    for u in U[:-1]:
                        used_diffs.add(abs(x - u))
                    found = True
                    break
            if not found:
                return None
        return U

    # Try various sizes of Golomb rulers
    for n in range(2, 15):
        g = generate_golomb(n, 2000)
        if g:
            try_set(g)

    # 2. Generate Sidon Sets (Minimize |U+U| for given n)
    # A Sidon set has all pairwise sums distinct (except x+y=y+x)
    # This minimizes the denominator in the ratio.
    def generate_sidon(n, max_val=10000):
        U = [0, 1]
        used_sums = set()
        for u in U:
            used_sums.add(2*u)
        for u1 in U:
            for u2 in U:
                if u1 != u2:
                    used_sums.add(u1+u2)
        
        for _ in range(2, n):
            found = False
            for x in range(2, max_val + 1):
                if x in U:
                    continue
                valid = True
                # Check sums
                for u in U:
                    s = x + u
                    if s in used_sums:
                        valid = False
                        break
                if valid:
                    U.append(x)
                    for u in U[:-1]:
                        used_sums.add(x + u)
                    used_sums.add(2*x)
                    found = True
                    break
            if not found:
                return None
        return U

    for n in range(2, 15):
        s = generate_sidon(n, 2000)
        if s:
            try_set(s)

    # 3. Hybrid: Local Search on Best Candidates
    # Start from the best found so far
    current = list(best_set)
    
    # If best_set is small or score is low, try a specific known good structure
    # E.g. {0, 1, 4, 9, 16} (Squares are Sidon, but diffs are not unique)
    try_set([0, 1, 4, 9, 16, 25, 36])
    try_set([0, 1, 4, 9, 16, 25, 36, 49])
    
    current = list(best_set)
    no_improve = 0
    max_no_improve = 200
    
    while time.time() < deadline - 10 and no_improve < max_no_improve:
        improved = False
        max_cur = max(current)
        n_cur = len(current)
        
        # 1. Try swapping elements to reduce max or improve ratio
        # Focus on the largest element first
        if len(current) > 1:
            # Try moving the largest element to a smaller unused position
            largest_idx = len(current) - 1
            old_val = current[largest_idx]
            
            # Search for a better position for the largest element
            # We want to keep it distinct but maybe smaller?
            # Or maybe we can replace it with something that creates fewer sum collisions.
            
            # Try all positions for the last element
            for new_val in range(1, old_val + 10):
                if new_val in current:
                    continue
                cand = list(current)
                cand[-1] = new_val
                cand = sorted(cand)
                s = compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_set = list(cand)
                    current = list(cand)
                    improved = True
                    break
            
            # Also try replacing the second largest
            if not improved and len(current) > 2:
                second_largest_idx = len(current) - 2
                old_val2 = current[second_largest_idx]
                for new_val in range(1, old_val2 + 10):
                    if new_val in current:
                        continue
                    cand = list(current)
                    cand[second_largest_idx] = new_val
                    cand = sorted(cand)
                    s = compute_score(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        current = list(cand)
                        improved = True
                        break

        # 2. Try adding a new element if n is small
        if not improved and n_cur < 20:
            # Find a spot that adds unique diffs and minimal sum collisions
            best_new_val = -1
            best_new_score = -1
            
            # Sample candidate positions
            candidates = set()
            for i in range(1, max_cur + 5):
                candidates.add(i)
            for _ in range(50):
                candidates.add(random.randint(1, max_cur + 20))
            
            for new_val in candidates:
                if new_val in current:
                    continue
                cand = sorted(current + [new_val])
                s = compute_score(cand)
                if s > best_new_score:
                    best_new_score = s
                    best_new_val = new_val
            
            if best_new_val > 0 and best_new_score > best_score:
                best_score = best_new_score
                best_set = sorted(current + [best_new_val])
                current = list(best_set)
                improved = True

        # 3. Try removing an element if it hurts the score (e.g. creates many sum collisions)
        if not improved and n_cur > 3:
            for i in range(1, n_cur):
                cand = [x for j, x in enumerate(current) if j != i]
                s = compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_set = list(cand)
                    current = list(cand)
                    improved = True
                    break

        if not improved:
            no_improve += 1
            # Random restart/perturbation
            if random.random() < 0.2:
                # Perturb a random element
                idx = random.randint(1, len(current)-1)
                old = current[idx]
                new = random.randint(1, max_cur + 20)
                if new not in current:
                    cand = list(current)
                    cand[idx] = new
                    cand = sorted(cand)
                    try_set(cand)
                    current = list(best_set)

    return list(best_set)
