"""Optimized search for sumset/difference-set exponent lower bound."""
import time
import math
import random
import numpy as np
from numpy.fft import fft, ifft

def _score_set(U):
    """Compute the score for a finite set U using FFT for speed."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    maxU = max(U)
    if maxU == 0:
        return -1
    
    # Compute |U+U| using FFT
    sum_len = 2 * maxU + 1
    fft_len = 1
    while fft_len < 2 * sum_len:
        fft_len *= 2
    
    indicator = np.zeros(fft_len, dtype=np.float64)
    for u in U:
        indicator[u] = 1.0
    
    freq = fft(indicator)
    conv = ifft(freq * freq).real
    sumset_size = int(np.sum(conv[:sum_len] > 1e-9))
    
    # Compute |U-U| using autocorrelation
    if n <= 200:
        diffset = set()
        for a in U:
            for b in U:
                diffset.add(a - b)
        diffset_size = len(diffset)
    else:
        arr = np.array(U)
        diffs = arr[:, None] - arr[None, :]
        diffset_size = len(np.unique(diffs))
    
    if sumset_size == 0:
        return -1
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return -1
    denom = math.log(2 * maxU + 1)
    if denom <= 0:
        return -1
    return 1.0 + math.log(ratio) / denom

def _score_set_fft_fast(U):
    """Fast FFT-based score computation for larger sets."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    maxU = max(U)
    if maxU == 0:
        return -1
    
    # Determine FFT length
    max_sum = 2 * maxU
    max_diff = 2 * maxU  # differences range from -maxU to maxU
    fft_len = 1
    while fft_len < 2 * (max_sum + 1):
        fft_len *= 2
    
    # Build indicator
    indicator = np.zeros(fft_len, dtype=np.float64)
    for u in U:
        indicator[u] = 1.0
    
    freq = fft(indicator)
    
    # Sumset: |U+U|
    sum_conv = ifft(freq * freq).real
    sumset_size = int(np.sum(sum_conv[:max_sum + 1] > 1e-9))
    
    # Difference set: |U-U|
    # diff[d] for d in [-maxU, maxU]
    # Cross-correlation: correlate indicator with reversed indicator
    rev_ind = np.zeros(fft_len, dtype=np.float64)
    for u in U:
        rev_ind[fft_len - 1 - u] = 1.0
    
    freq_rev = fft(rev_ind)
    diff_conv = ifft(freq * freq_rev).real
    
    # The difference a-b = d corresponds to specific positions in diff_conv
    # From the correlation structure, differences in [-maxU, maxU] map to:
    # position fft_len-1-d for d >= 0, and position fft_len-1+|d| for d < 0
    # Actually, let me compute directly for reliability
    arr = np.array(U, dtype=np.int64)
    diffs = arr[:, None] - arr[None, :]
    diffset_size = len(np.unique(diffs))
    
    if sumset_size == 0 or diffset_size == 0:
        return -1
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return -1
    denom = math.log(2 * maxU + 1)
    if denom <= 0:
        return -1
    return 1.0 + math.log(ratio) / denom

def _score_set_fast(U):
    """Faster score computation using numpy vectorization."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    maxU = max(U)
    if maxU == 0:
        return -1
    
    arr = np.array(U, dtype=np.int64)
    
    # Sumset: all pairwise sums
    sums = arr[:, None] + arr[None, :]
    sumset_size = len(np.unique(sums))
    
    # Difference set: all pairwise differences  
    diffs = arr[:, None] - arr[None, :]
    diffset_size = len(np.unique(diffs))
    
    if sumset_size == 0:
        return -1
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return -1
    denom = math.log(2 * maxU + 1)
    if denom <= 0:
        return -1
    return 1.0 + math.log(ratio) / denom

def _dual_grid_set(m1, m2, max_val):
    """Generate set U = {x in [0, max_val] : x ≡ 0 mod m1 AND x ≡ 0 mod m2}.
    By CRT, this is multiples of lcm(m1, m2)."""
    import math
    lcm_val = m1 * m2 // math.gcd(m1, m2)
    if lcm_val == 0:
        return [0]
    U = list(range(0, max_val + 1, lcm_val))
    if 0 not in U:
        U.insert(0, 0)
    return U

def _dual_grid_search(max_m=50, max_val_limit=200, deadline=None):
    """Search over dual-grid modular lattice parameters."""
    best_U = None
    best_score = -1
    
    for m1 in range(1, max_m + 1):
        if deadline is not None and time.time() > deadline:
            break
        for m2 in range(1, max_m + 1):
            if deadline is not None and time.time() > deadline:
                break
            
            lcm_val = m1 * m2 // math.gcd(m1, m2)
            if lcm_val == 0:
                continue
            
            # Try different max_val to control density
            for max_val in [lcm_val * 2, lcm_val * 3, lcm_val * 4, lcm_val * 5, lcm_val * 7, lcm_val * 10]:
                if max_val > max_val_limit:
                    break
                U = _dual_grid_set(m1, m2, max_val)
                if len(U) < 2:
                    continue
                
                s = _score_set_fast(U)
                if s > best_score:
                    best_score = s
                    best_U = list(U)
    
    return best_U, best_score

def _golomb_ruler_greedy(n, max_attempts=10000):
    """Generate a Golomb ruler of order n using greedy with backtracking."""
    if n <= 1:
        return [0]
    known = {
        1: [0],
        2: [0, 1],
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 10, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 12, 17, 22],
        8: [0, 1, 4, 10, 12, 17, 22, 29],
        9: [0, 1, 4, 10, 12, 17, 22, 29, 36],
        10: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44],
        11: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52],
        12: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60],
        13: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 68],
    }
    if n in known:
        return known[n]
    
    ruler = [0]
    used_diffs = set()
    
    def backtrack(pos, current_ruler, current_diffs):
        if len(current_ruler) == n:
            return current_ruler[:]
        
        start = current_ruler[-1] + 1
        for p in range(start, start + 500):
            new_diffs = set()
            valid = True
            for i in range(len(current_ruler)):
                d = p - current_ruler[i]
                if d in current_diffs or d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if valid:
                result = backtrack(pos + 1, current_ruler + [p], current_diffs | new_diffs)
                if result is not None:
                    return result
        
        return None
    
    result = backtrack(1, [0], set())
    if result is not None:
        return result
    
    return [0] + list(range(1, n))

def _perturb_golomb(ruler, num_perturbations=50, time_limit=10.0, deadline=None):
    """Apply small perturbations to a Golomb ruler to find better configurations."""
    start = time.time()
    best = list(ruler)
    best_score = _score_set_fast(ruler)
    
    for iteration in range(num_perturbations):
        if deadline is not None and time.time() > deadline:
            break
        if time.time() - start > time_limit:
            break
        
        new_U = list(best)
        
        for _ in range(3):
            if len(new_U) > 2:
                idx = random.randint(1, len(new_U) - 1)
                new_U.pop(idx)
            max_val = max(new_U) if new_U else 1
            new_val = random.randint(1, max_val + 10)
            if new_val not in new_U:
                new_U.append(new_val)
            new_U = sorted(set(new_U))
            if 0 not in new_U or len(new_U) < 2:
                new_U = list(best)
                break
        
        s = _score_set_fast(new_U)
        if s > best_score + 1e-12:
            best_score = s
            best = new_U
    
    return best, best_score

def _local_search(U, max_iter=200, time_limit=10.0, deadline=None):
    """Local search: try adding, removing, or shifting elements."""
    start = time.time()
    U = sorted(set(U))
    best_score = _score_set_fast(U)
    best_U = list(U)
    
    max_val = max(U) if U else 10
    search_range = max(max_val * 2, 30)
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        if time.time() - start > time_limit:
            break
        improved = False
        
        moves = []
        
        for x in range(1, search_range + 1):
            if x not in U:
                moves.append(('add', x))
        
        for x in U:
            if x != 0:
                moves.append(('remove', x))
        
        for x in U:
            if x == 0:
                continue
            for delta in [-2, -1, 1, 2, 3, 5]:
                new_x = x + delta
                if new_x >= 0 and new_x != x and new_x not in U:
                    moves.append(('shift', x, new_x))
        
        random.shuffle(moves)
        
        for move in moves[:30]:
            if deadline is not None and time.time() > deadline:
                break
            if time.time() - start > time_limit:
                break
            
            if move[0] == 'add':
                new_U = list(U) + [move[1]]
            elif move[0] == 'remove':
                new_U = [x for x in U if x != move[1]]
            else:
                new_U = [move[2] if x == move[1] else x for x in U]
            
            if 0 not in new_U or len(new_U) < 2:
                continue
            
            s = _score_set_fast(new_U)
            if s > best_score + 1e-12:
                best_score = s
                best_U = new_U
                improved = True
                break
        
        if not improved:
            for _ in range(3):
                if deadline is not None and time.time() > deadline:
                    break
                if time.time() - start > time_limit:
                    break
                new_U = list(U)
                for _ in range(random.randint(1, 2)):
                    new_U.append(random.randint(1, search_range))
                if len(new_U) > 3:
                    rem = random.choice([x for x in new_U if x != 0])
                    new_U.remove(rem)
                
                if 0 not in new_U or len(new_U) < 2:
                    continue
                s = _score_set_fast(new_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = new_U
                    improved = True
                    break
    
    return best_U, best_score

def _modular_lattice_search(deadline):
    """Search over modular lattice structures with varied parameters."""
    best_U = None
    best_score = -1
    
    # Try sets defined by x ≡ r mod m for various m and r
    for m in range(2, 30):
        if time.time() > deadline:
            break
        for r in range(0, min(m, 10)):
            if time.time() > deadline:
                break
            for max_val in [m * 3, m * 5, m * 8, m * 12, m * 20]:
                if max_val > 500:
                    break
                U = [x for x in range(0, max_val + 1) if x % m == r]
                if 0 not in U:
                    # Add 0 and adjust
                    U = [0] + [x for x in range(0, max_val + 1) if x % m == r and x != 0]
                if len(U) < 2:
                    continue
                s = _score_set_fast(U)
                if s > best_score:
                    best_score = s
                    best_U = list(U)
    
    # Try mixed: union of two arithmetic progressions
    for m1 in range(2, 15):
        if time.time() > deadline:
            break
        for m2 in range(2, 15):
            if m1 == m2:
                continue
            if time.time() > deadline:
                break
            for max_val in [m1 * m2, m1 * m2 * 2, m1 * m2 * 3]:
                if max_val > 300:
                    break
                U = set()
                U.add(0)
                for x in range(0, max_val + 1):
                    if x % m1 == 0 or x % m2 == 0:
                        U.add(x)
                U = sorted(U)
                if len(U) < 2:
                    continue
                s = _score_set_fast(U)
                if s > best_score:
                    best_score = s
                    best_U = list(U)
    
    return best_U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_U = [0, 1, 3]
    best_score = _score_set_fast(best_U)
    
    # Try Golomb rulers of various sizes
    for n in range(3, 14):
        if time.time() > deadline - 10:
            break
        ruler = _golomb_ruler_greedy(n)
        if len(ruler) >= 2:
            s = _score_set_fast(ruler)
            if s > best_score:
                best_score = s
                best_U = list(ruler)
    
    # Try structured candidates
    initial_candidates = [
        [0, 1, 3],
        [0, 1, 2, 4],
        [0, 1, 3, 6],
        [0, 1, 2, 3, 5],
        [0, 1, 3, 4, 6],
        [0, 1, 2, 4, 5],
        [0, 1, 3, 5, 6],
        [0, 1, 2, 3, 4, 6],
        [0, 1, 3, 4, 5, 7],
        [0, 1, 2, 4, 6, 7],
        [0, 1, 4, 5, 8],
        [0, 1, 3, 7, 8],
        [0, 2, 3, 5, 7],
        [0, 1, 4, 6, 7],
        [0, 1, 2, 4, 5, 7, 8],
        [0, 1, 3, 4, 6, 7, 9],
        [0, 1, 2, 5],
        [0, 1, 4, 10, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 22],
        [0, 1, 2, 3, 4, 5, 7, 8, 9, 11],
        [0, 1, 2, 4, 5, 7, 8, 10, 11, 13],
        [0, 1, 3, 4, 6, 7, 9, 10, 12, 13],
    ]
    
    for cand in initial_candidates:
        if time.time() > deadline - 10:
            break
        s = _score_set_fast(cand)
        if s > best_score:
            best_score = s
            best_U = list(cand)
    
    # DUAL GRID MODULAR LATTICE SEARCH (new strategy)
    if time.time() < deadline - 30:
        dg_U, dg_score = _dual_grid_search(max_m=30, max_val_limit=300, deadline=deadline - 20)
        if dg_score > best_score:
            best_score = dg_score
            best_U = dg_U
    
    # MODULAR LATTICE SEARCH (union of APs)
    if time.time() < deadline - 20:
        ml_U, ml_score = _modular_lattice_search(deadline - 10)
        if ml_score > best_score:
            best_score = ml_score
            best_U = ml_U
    
    # Local search from best
    if time.time() < deadline - 15:
        best_U, best_score = _local_search(best_U, max_iter=300, time_limit=min(20, deadline - time.time() - 15), deadline=deadline)
    
    # Try perturbing Golomb rulers
    random.seed(42)
    for n in range(4, 12):
        if time.time() > deadline - 15:
            break
        ruler = _golomb_ruler_greedy(n)
        if len(ruler) >= 3:
            U2, s2 = _perturb_golomb(ruler, num_perturbations=30, time_limit=min(5, deadline - time.time() - 15), deadline=deadline)
            if s2 > best_score:
                best_score = s2
                best_U = U2
    
    # Try more random starts with local search
    attempts = 0
    while time.time() < deadline - 10 and attempts < 30:
        attempts += 1
        size = random.randint(4, 15)
        max_val = random.randint(15, 80)
        U = {0}
        while len(U) < size:
            U.add(random.randint(1, max_val))
        U = sorted(U)
        
        s = _score_set_fast(U)
        if s > best_score:
            best_score = s
            best_U = U
        
        if time.time() < deadline - 10:
            U2, s2 = _local_search(U, max_iter=100, time_limit=min(5, deadline - time.time() - 10), deadline=deadline)
            if s2 > best_score:
                best_score = s2
                best_U = U2
    
    # Final polish
    if time.time() < deadline - 5:
        best_U, best_score = _local_search(best_U, max_iter=200, time_limit=min(10, deadline - time.time() - 5), deadline=deadline)
    
    return best_U
