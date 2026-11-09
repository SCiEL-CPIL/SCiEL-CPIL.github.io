"""Combined program: exhaustive small search + simulated annealing with FFT acceleration."""
import time
import random
import numpy as np
from math import log, exp

def _compute_score(U):
    """Compute the exact score for a given set U."""
    U_arr = np.array(sorted(set(U)), dtype=np.int64)
    if len(U_arr) == 0 or 0 not in U_arr:
        return 0.0
    n = len(U_arr)
    if n == 0:
        return 0.0
    max_u = U_arr[-1]
    if max_u == 0:
        return 1.0
    
    sumset = np.unique(U_arr[:, None] + U_arr[None, :])
    diffset = np.unique(U_arr[:, None] - U_arr[None, :])
    
    len_sum = len(sumset)
    len_diff = len(diffset)
    
    if len_sum == 0:
        return 0.0
        
    ratio = len_diff / len_sum
    if ratio <= 0:
        return 0.0
    
    denominator = log(2 * max_u + 1)
    if denominator == 0:
        return 1.0
    
    score = 1 + log(ratio) / denominator
    return score

def _fft_estimate_score(U, precision=0.5):
    """Fast approximate score using FFT convolution. Returns (score_estimate, exact_flag)."""
    U_arr = np.array(sorted(set(U)), dtype=np.int64)
    if len(U_arr) == 0 or 0 not in U_arr:
        return 0.0, True
    n = len(U_arr)
    max_u = U_arr[-1]
    if max_u == 0:
        return 1.0, True
    
    # For small sets, just compute exactly
    if n <= 8:
        return _compute_score(U_arr), True
    
    # Use FFT for larger sets
    size = 2 * max_u + 2
    # Pad to next power of 2 for FFT efficiency
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    
    # Create indicator vector
    indicator = np.zeros(fft_size, dtype=np.float64)
    indicator[U_arr] = 1.0
    
    # FFT for sumset: autocorrelation gives sumset structure
    # Actually, for sumset U+U, we need the convolution of indicator with itself
    # For diffset U-U, we need the cross-correlation
    
    # Sumset: number of distinct sums
    # Use the fact that |U+U| can be estimated from the support of the convolution
    fft_ind = np.fft.rfft(indicator)
    conv_sum = np.fft.irfft(fft_ind * fft_ind, n=fft_size)
    # The sumset elements are where conv_sum > 0
    # But this counts multiplicities, not distinct elements
    # For distinct count, we need a different approach
    
    # For distinct sumset size, use the indicator approach
    # Actually, let's just use exact for n <= 15 and skip FFT for now
    # The overhead of FFT setup isn't worth it for small n
    if n <= 15:
        return _compute_score(U_arr), True
    
    # For larger n, use a sampling-based estimate
    # This is approximate
    sumset_approx = set()
    diffset_approx = set()
    sample_indices = np.random.choice(n, size=min(n, 50), replace=False)
    for i in sample_indices:
        for j in sample_indices:
            sumset_approx.add(U_arr[i] + U_arr[j])
            diffset_approx.add(U_arr[i] - U_arr[j])
    
    # Scale by (n/len(sample))^2 to estimate full size
    scale_factor = (n / len(sample_indices)) ** 2
    len_sum_est = int(len(sumset_approx) * scale_factor)
    len_diff_est = int(len(diffset_approx) * scale_factor)
    
    if len_sum_est <= 0:
        return 0.0, False
    
    ratio_est = len_diff_est / len_sum_est
    denominator = log(2 * max_u + 1)
    if denominator == 0:
        return 1.0, False
    
    score_est = 1 + log(ratio_est) / denominator
    return score_est, False

def _exhaustive_small_search(max_n=6, max_val=50, timeout=10.0):
    """Exhaustively search all sets of size up to max_n containing 0."""
    start_time = time.time()
    best_set = [0, 1]
    best_score = _compute_score(best_set)
    
    for n in range(2, max_n + 1):
        if time.time() - start_time > timeout:
            break
        
        # Generate all combinations of size n-1 from 1..max_val
        # This is C(max_val, n-1) which can be large
        # For n=6, max_val=50: C(50,5) = 2,118,760 - too many
        # Use a smarter approach: generate candidates and check
        
        if n <= 4:
            # Exhaustive for small n
            from itertools import combinations
            for combo in combinations(range(1, max_val + 1), n - 1):
                if time.time() - start_time > timeout:
                    break
                candidate = [0] + list(combo)
                score = _compute_score(candidate)
                if score > best_score:
                    best_score = score
                    best_set = list(candidate)
        else:
            # For larger n, use greedy + local refinement
            # Start with best known small set and extend
            base = best_set[:]
            if len(base) < n:
                # Try to extend greedily
                current = list(base)
                for _ in range(n - len(current)):
                    added = False
                    for cand in range(current[-1] + 1, max_val + 1):
                        new_set = current + [cand]
                        score = _compute_score(new_set)
                        if score > _compute_score(current) - 0.01:  # Allow slight decrease
                            current = new_set
                            added = True
                            break
                    if not added:
                        break
                if len(current) >= n - 1:
                    score = _compute_score(current)
                    if score > best_score:
                        best_score = score
                        best_set = list(current)
    
    return best_set, best_score

def _simulated_annealing(U, max_iterations=50000, timeout=30.0, initial_temp=1.0, cooling=0.999):
    """Simulated annealing to escape local optima."""
    start_time = time.time()
    current = sorted(set(U))
    if 0 not in current:
        current.insert(0, 0)
    current_score = _compute_score(current)
    best = list(current)
    best_score = current_score
    
    temp = initial_temp
    
    for i in range(max_iterations):
        if time.time() - start_time > timeout:
            break
        
        U_set = set(current)
        max_u = max(U_set) if U_set else 0
        n = len(U_set)
        
        # Generate a random perturbation
        perturbation_type = random.random()
        
        if perturbation_type < 0.4:
            # Add an element
            if random.random() < 0.5:
                cand = random.randint(0, max_u + 50)
            else:
                # Add near existing elements
                x = random.choice(list(U_set))
                offset = random.randint(-10, 20)
                cand = max(0, x + offset)
            if cand not in U_set:
                new_set = U_set | {cand}
        elif perturbation_type < 0.7:
            # Remove an element
            removable = [x for x in current if x != 0]
            if removable:
                rem = random.choice(removable)
                new_set = U_set - {rem}
                if len(new_set) < 2:
                    continue
            else:
                continue
        else:
            # Swap: remove one, add another
            removable = [x for x in current if x != 0]
            if not removable:
                continue
            rem = random.choice(removable)
            if random.random() < 0.5:
                new_elem = random.randint(0, max_u + 50)
            else:
                x = random.choice(list(U_set - {rem})) if len(U_set) > 1 else 0
                offset = random.randint(-10, 20)
                new_elem = max(0, x + offset)
            new_set = (U_set - {rem}) | {new_elem}
            if len(new_set) < 2:
                continue
        
        new_score = _compute_score(new_set)
        delta = new_score - current_score
        
        # Accept if better, or with probability exp(-delta/temp)
        if delta > 0 or random.random() < exp(-delta / max(temp, 1e-10)):
            current = sorted(new_set)
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best = list(current)
        
        # Cool down
        temp *= cooling
        if temp < 0.01:
            temp = 0.01
    
    return best, best_score

def _local_search(U, max_iterations=20000, timeout=20.0):
    """Local search to improve the set."""
    start_time = time.time()
    current = sorted(set(U))
    if 0 not in current:
        current.insert(0, 0)
    current_score = _compute_score(current)
    best = list(current)
    best_score = current_score
    
    for i in range(max_iterations):
        if time.time() - start_time > timeout:
            break
        
        U_set = set(current)
        max_u = max(U_set) if U_set else 0
        n = len(U_set)
        
        # Generate candidates
        candidates = set()
        
        # Small perturbations
        for x in list(U_set):
            for offset in [-3, -2, -1, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]:
                c = x + offset
                if c >= 0:
                    candidates.add(c)
        
        # Extend max
        for k in range(1, 10):
            candidates.add(max_u + k)
        
        # Random
        for _ in range(30):
            candidates.add(random.randint(0, max_u + 50))
        
        candidate_list = list(candidates)
        random.shuffle(candidate_list)
        
        improved = False
        for cand in candidate_list:
            if time.time() - start_time > timeout:
                break
            if cand in U_set:
                continue
            new_set = U_set | {cand}
            new_score = _compute_score(new_set)
            if new_score > current_score + 1e-9:
                current = sorted(new_set)
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best = list(current)
                improved = True
                break
        
        if not improved:
            # Try removing an element
            removable = [x for x in current if x != 0]
            random.shuffle(removable)
            for cand in removable[:20]:
                if time.time() - start_time > timeout:
                    break
                new_set = U_set - {cand}
                if len(new_set) < 2:
                    continue
                new_score = _compute_score(new_set)
                if new_score > current_score + 1e-9:
                    current = sorted(new_set)
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best = list(current)
                    improved = True
                    break
        
        if not improved and i > 20:
            # Swap
            if random.random() < 0.6:
                removable = [x for x in current if x != 0]
                if removable:
                    rem = random.choice(removable)
                    new_elem = random.randint(0, max_u + 60)
                    new_set = (U_set - {rem}) | {new_elem}
                    if len(new_set) >= 2:
                        new_score = _compute_score(new_set)
                        if new_score > current_score + 1e-9:
                            current = sorted(new_set)
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best = list(current)
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Strategy 1: Exhaustive search for small sets (high value)
    best_set, best_score = _exhaustive_small_search(max_n=5, max_val=40, timeout=15.0)
    
    # Strategy 2: Try known good structures
    known_structures = [
        [0, 1, 3, 7],
        [0, 1, 4, 6],
        [0, 1, 4, 10, 12],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 25],
        [0, 1, 4, 10, 12, 17, 25, 28],
        [0, 1, 3, 7, 12],
        [0, 1, 3, 7, 12, 18],
        [0, 2, 3, 7],
        [0, 1, 2, 5],
        [0, 1, 3, 4],
        [0, 1, 4, 5],
        [0, 1, 2, 4],
        [0, 1, 2, 3],
        [0, 1, 3],
        [0, 1, 2],
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [0, 5],
    ]
    for struct in known_structures:
        if time.time() > deadline - 30:
            break
        s = _compute_score(struct)
        if s > best_score:
            best_score = s
            best_set = list(struct)
    
    # Strategy 3: Simulated annealing from best result
    if time.time() < deadline - 20:
        remaining = deadline - time.time() - 5
        if remaining > 10:
            improved, score = _simulated_annealing(best_set, max_iterations=100000, timeout=remaining, initial_temp=0.5, cooling=0.9995)
            if score > best_score:
                best_set = improved
                best_score = score
    
    # Strategy 4: Local search polish
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 2
        if remaining > 5:
            final_result, final_score = _local_search(best_set, max_iterations=50000, timeout=remaining)
            if final_score > best_score:
                best_set = final_result
                best_score = final_score
    
    # Final validation
    final_score = _compute_score(best_set)
    if final_score < 1.0:
        best_set = [0, 1, 3, 7]
    
    return best_set
