import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit
    
    best_score = -1
    best_set = [0]
    
    def compute_score(U):
        if not U or 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 0
        
        n = len(U)
        
        # FFT-based computation for both sumset and diffset
        max_u = max(U)
        
        # Sumset via FFT: convolve indicator with itself
        M_sum = 2 * max_u + 1
        M_sum_fft = 1
        while M_sum_fft < M_sum:
            M_sum_fft *= 2
        
        ind = np.zeros(M_sum_fft)
        for u in U:
            ind[u] = 1.0
        
        fft_ind = np.fft.fft(ind)
        conv = np.fft.ifft(fft_ind * fft_ind)
        conv = np.abs(conv)
        n_sum = int(np.sum(conv[:M_sum] > 0.5))
        
        # Diffset: U-U = {a-b : a,b in U}
        # Use FFT: shift U by max_u so all elements are non-negative
        # Then diffset corresponds to positions where cross-correlation is positive
        # Alternative: compute directly using the fact that d is in U-U iff
        # there exists a in U with a+d in U (for d>=0) or a-d in U (for d<0)
        # Better: use FFT on shifted array
        M_diff = 2 * max_u + 1
        M_diff_fft = 1
        while M_diff_fft < M_diff:
            M_diff_fft *= 2
        
        # Shift U by max_u: V = {u + max_u : u in U}
        # Then V - V gives differences in range [-(max_u - min(U)), max_u - min(U)]
        # Actually, let's just compute: for each d in [-max_u, max_u], check if d in U-U
        # Using FFT: autocorrelation of the indicator gives sum over a of f(a)*f(a-d)
        # which is the number of pairs (a, a-d) both in U, i.e., d is in U-U
        
        # Shift indicator so that U is centered
        ind_shifted = np.zeros(M_diff_fft)
        for u in U:
            ind_shifted[u + max_u] = 1.0
        
        fft_shifted = np.fft.fft(ind_shifted)
        # Autocorrelation: ifft(|fft|^2) gives sum_a f(a) * f(a - k) for each k
        auto = np.fft.ifft(fft_shifted * np.conj(fft_shifted))
        auto = np.abs(auto)
        
        # The autocorrelation at lag k tells us about differences
        # For a set with elements shifted by max_u, the autocorrelation at position 
        # max_u + d corresponds to the number of pairs (a, b) with a - b = d
        # Actually, let me think more carefully.
        # ind_shifted[i] = 1 if i - max_u in U, i.e., if i in U + max_u
        # autocorr[k] = sum_i ind_shifted[i] * ind_shifted[i-k] (with circular wrap)
        # This counts pairs (i, i-k) both in U+max_u, i.e., (i-max_u, i-k-max_u) in U
        # so difference = k. But we need to handle the circular wrap.
        # For |k| <= max_u, there's no wrap issue if M_diff_fft >= 2*max_u+1
        
        # Positions in autocorr: index j corresponds to lag j (or j - M_diff_fft for negative)
        # We want lags from -max_u to max_u
        # In the FFT output, lag 0 is at index 0, lag 1 at index 1, ..., lag max_u at index max_u
        # lag -1 at index M_diff_fft-1, lag -max_u at index M_diff_fft-max_u
        
        n_diff = 0
        for d in range(-max_u, max_u + 1):
            if d >= 0:
                idx = d
            else:
                idx = M_diff_fft + d
            if auto[idx] > 0.5:
                n_diff += 1
        
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
    
    # Phase 1: Greedy construction based on marginal difference yield
    current = [0]
    
    # Build up the set greedily
    for step in range(200):
        if time.time() > deadline - 30:
            break
        
        max_cur = max(current)
        
        # Generate candidate elements to add
        candidates = set()
        # Nearby elements
        for i in range(1, max_cur + 20):
            if i not in current:
                candidates.add(i)
        # Some random larger elements
        for _ in range(20):
            candidates.add(random.randint(max_cur + 1, max_cur + 100))
        
        # For each candidate, compute marginal gain in |U-U|/|U+U|
        best_cand = None
        best_gain = 0
        
        # First compute current score components
        cur_score = compute_score(current)
        
        for new_el in candidates:
            cand = sorted(current + [new_el])
            s = compute_score(cand)
            # Marginal gain in the score
            gain = s - cur_score
            if gain > best_gain:
                best_gain = gain
                best_cand = new_el
        
        if best_cand is not None:
            current = sorted(current + [best_cand])
            try_set(current)
        else:
            break
    
    # Phase 2: Exhaustive search for small sets
    for M in range(1, 14):
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
    
    # Phase 3: Local search refinement
    current = list(best_set)
    
    no_improve = 0
    max_no_improve = 500
    
    while time.time() < deadline - 5 and no_improve < max_no_improve:
        improved = False
        max_cur = max(current)
        
        # 1. Try adding an element
        candidates_to_add = set()
        for i in range(1, max_cur + 30):
            if i not in current:
                candidates_to_add.add(i)
        for _ in range(30):
            candidates_to_add.add(random.randint(max_cur + 1, max_cur + 150))
            
        for new_el in candidates_to_add:
            if new_el in current:
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
                for delta in range(-20, 21):
                    new_el = old_el + delta
                    if new_el < 1 or new_el in current:
                        continue
                    if new_el == old_el:
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
        
        if not improved:
            no_improve += 1
            # Random perturbation
            if random.random() < 0.3:
                if len(current) > 2:
                    i = random.randint(1, len(current)-1)
                    j = random.randint(1, len(current)-1)
                    if i != j:
                        cand = list(current)
                        cand[i], cand[j] = cand[j], cand[i]
                        cand = sorted(set(cand))
                        if 0 in cand:
                            try_set(cand)
                            current = list(best_set)
                else:
                    new_el = random.randint(1, max_cur + 50)
                    if new_el not in current:
                        cand = sorted(current + [new_el])
                        try_set(cand)
                        current = list(best_set)

    return list(best_set)
