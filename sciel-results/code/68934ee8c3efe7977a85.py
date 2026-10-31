import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
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
        if n > 20:
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
        # For sums: convolve indicator with itself
        # For differences: convolve indicator with reversed indicator
        size = 2 * max_u + 2
        # Next power of 2 for efficient FFT
        fft_size = 1
        while fft_size < size:
            fft_size *= 2
        
        # Create indicator vector
        indicator = np.zeros(fft_size, dtype=np.float64)
        indicator[U_arr] = 1.0
        
        # Sum set: autocorrelation at positive lags
        fft_ind = np.fft.fft(indicator)
        autocorr = np.fft.ifft(fft_ind * np.conj(fft_ind)).real
        # Sums range from 0 to 2*max_u
        n_sum = np.sum(autocorr[:2*max_u+1] > 0.5)
        
        # Difference set: cross-correlation
        # Reverse the indicator (for differences)
        indicator_rev = np.zeros(fft_size, dtype=np.float64)
        indicator_rev[max_u - U_arr] = 1.0
        fft_rev = np.fft.fft(indicator_rev)
        cross_corr = np.fft.ifft(fft_ind * np.conj(fft_rev)).real
        # Differences range from -max_u to max_u
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

    # Phase 1: Try structured sets
    # Geometric progressions
    for base in [2, 3, 4, 5, 7, 10]:
        g = [0, 1]
        while g[-1] * base < 5000:
            g.append(g[-1] * base)
        try_set(g)
    
    # Golomb ruler-like sets (greedy construction)
    def generate_golomb_like(n, max_val=10000):
        """Generate a set that tries to have unique differences"""
        U = [0, 1]
        for _ in range(2, n):
            best_pos = -1
            best_score = -1
            for pos in range(2, max_val + 1):
                if pos in U:
                    continue
                # Check if this creates new differences
                new_diffs = set()
                for u in U:
                    new_diffs.add(pos - u)
                    new_diffs.add(u - pos)
                # We want to maximize unique differences
                all_diffs = set()
                for i in range(len(U)):
                    for j in range(len(U)):
                        all_diffs.add(U[i] - U[j])
                new_unique = len(new_diffs - all_diffs)
                if new_unique > best_score:
                    best_score = new_unique
                    best_pos = pos
                if best_score >= len(U):  # Perfect: all new
                    break
            if best_pos > 0:
                U.append(best_pos)
            else:
                break
        return U
    
    for n in [5, 6, 7, 8, 10, 12, 15]:
        g = generate_golomb_like(n, 5000)
        try_set(g)
    
    # Phase 2: Exhaustive search for very small sets
    for M in range(1, 12):
        if time.time() > deadline - 20:
            break
        for mask in range(1, 1 << M):
            if time.time() > deadline - 20:
                break
            cand = [0]
            for i in range(M):
                if mask & (1 << i):
                    cand.append(i + 1)
            try_set(cand)
    
    # Phase 3: Local search with FFT acceleration for larger sets
    current = list(best_set)
    no_improve = 0
    max_no_improve = 500
    
    while time.time() < deadline - 10 and no_improve < max_no_improve:
        improved = False
        max_cur = max(current)
        n_cur = len(current)
        
        # 1. Try adding an element
        candidates_to_add = set()
        # Small integers near existing
        for i in range(1, min(max_cur + 30, 2000)):
            if i not in current:
                candidates_to_add.add(i)
        # Gaps in the set
        for i in range(1, n_cur - 1):
            gap_start = current[i] + 1
            gap_end = current[i + 1] - 1
            for j in range(gap_start, min(gap_end + 1, gap_start + 10)):
                if j not in current:
                    candidates_to_add.add(j)
        # Large jumps
        for _ in range(30):
            candidates_to_add.add(random.randint(max_cur + 1, max_cur + 200))
        
        for new_el in candidates_to_add:
            if new_el in current or new_el < 1:
                continue
            cand = sorted(current + [new_el])
            s = compute_score(cand)
            if s > best_score:
                best_score = s
                best_set = list(cand)
                current = list(cand)
                improved = True
                no_improve = 0
                break
        
        # 2. Try replacing an element
        if len(current) >= 3:
            for i in range(len(current)):
                if current[i] == 0:
                    continue
                old_el = current[i]
                # Try replacing with neighbors and gaps
                for delta in range(-20, 21):
                    new_el = old_el + delta
                    if new_el < 1 or new_el in current or new_el == old_el:
                        continue
                    cand = list(current)
                    cand[i] = new_el
                    cand = sorted(set(cand))
                    if 0 not in cand:
                        continue
                    s = compute_score(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        current = list(cand)
                        improved = True
                        no_improve = 0
                        break
                if improved:
                    break
        
        # 3. Try removing an element
        if len(current) > 3:
            for i in range(len(current)):
                if current[i] == 0:
                    continue
                cand = [x for j, x in enumerate(current) if j != i]
                s = compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_set = list(cand)
                    current = list(cand)
                    improved = True
                    no_improve = 0
                    break
        
        # 4. Try growing the set significantly when n is small
        if len(current) < 30 and not improved:
            # Try adding multiple random elements at once
            for _ in range(5):
                new_els = random.sample(range(1, max_cur + 100), min(5, max_cur + 99))
                cand = sorted(set(current + new_els))
                s = compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_set = list(cand)
                    current = list(cand)
                    improved = True
                    no_improve = 0
                    break
        
        if not improved:
            no_improve += 1
            # Random perturbation to escape local minima
            if random.random() < 0.4:
                if len(current) > 2:
                    # Swap two elements
                    i = random.randint(1, len(current) - 1)
                    j = random.randint(1, len(current) - 1)
                    if i != j:
                        cand = list(current)
                        cand[i], cand[j] = cand[j], cand[i]
                        cand = sorted(set(cand))
                        if 0 in cand:
                            try_set(cand)
                            current = list(best_set)
                # Add a random element
                new_el = random.randint(1, max_cur + 100)
                if new_el not in current:
                    cand = sorted(current + [new_el])
                    try_set(cand)
                    current = list(best_set)

    return list(best_set)
