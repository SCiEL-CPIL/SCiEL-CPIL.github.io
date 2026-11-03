"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import itertools
import numpy as np
from numpy.fft import fft, ifft

# EVOLVE-BLOCK-START

def _score_set(U):
    """Compute the score for a finite set U."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    maxU = max(U)
    if maxU == 0:
        return -1
    # Compute |U+U| and |U-U|
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    ss = len(sumset)
    ds = len(diffset)
    if ss == 0:
        return -1
    ratio = ds / ss
    if ratio <= 0:
        return -1
    denom = math.log(2 * maxU + 1)
    if denom <= 0:
        return -1
    return 1.0 + math.log(ratio) / denom

def _score_set_fft(U):
    """Compute the score for a finite set U using FFT for speed."""
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    # Compute sumset using FFT
    sum_len = 2 * max_u + 1
    fft_len = 1
    while fft_len < 2 * sum_len:
        fft_len *= 2
    
    indicator = np.zeros(fft_len, dtype=np.float64)
    for u in U:
        indicator[u] = 1.0
    
    freq = fft(indicator)
    conv = ifft(freq * freq).real
    sumset_size = int(np.sum(conv[:sum_len] > 1e-9))
    
    # Compute difference set
    # U - U: values range from -max_u to max_u
    # We compute the autocorrelation
    # diff_conv[k] = sum_i indicator[i] * indicator[i + k] (with appropriate shifts)
    # Actually, let's use the property that |U-U| can be computed via the
    # autocorrelation of the indicator function.
    # The difference set U-U = {a-b : a,b in U}
    # We can compute this by convolving indicator with reversed indicator
    rev_indicator = np.zeros(fft_len, dtype=np.float64)
    for u in U:
        rev_indicator[fft_len - 1 - u] = 1.0
    
    freq_rev = fft(rev_indicator)
    diff_conv = ifft(freq * freq_rev).real
    
    # The difference set values are at positions max_u to 2*max_u in the convolution
    # Actually, let me think about this more carefully.
    # If we have indicator[i] = 1 if i in U, 0 otherwise
    # And rev_indicator[j] = 1 if (fft_len-1-j) in U, 0 otherwise
    # Then conv[k] = sum_i indicator[i] * rev_indicator[k-i]
    # = sum_i indicator[i] * indicator[fft_len-1-(k-i)]
    # = sum_i indicator[i] * indicator[fft_len-1-k+i]
    # Let j = fft_len-1-k+i, so i = j - (fft_len-1-k)
    # This is getting complicated. Let me use a different approach.
    
    # Simpler: compute U-U directly for small sets, use FFT for large sets
    if n <= 100:
        diffset = set()
        for a in U:
            for b in U:
                diffset.add(a - b)
        diffset_size = len(diffset)
    else:
        # For larger sets, use FFT-based approach
        # The number of distinct differences can be estimated
        # But for now, let's just compute it directly if feasible
        diffset = set()
        for a in U:
            for b in U:
                diffset.add(a - b)
        diffset_size = len(diffset)
    
    if sumset_size == 0:
        return -1
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return -1
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return -1
    return 1.0 + math.log(ratio) / denom

def _golomb_ruler(n):
    """Generate a Golomb ruler of order n using greedy construction."""
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
    }
    if n in known:
        return known[n]
    ruler = [0]
    for i in range(1, n):
        pos = 1
        while True:
            candidate = ruler + [pos]
            diffs = set()
            valid = True
            for j in range(len(candidate)):
                for k in range(j + 1, len(candidate)):
                    d = candidate[k] - candidate[j]
                    if d in diffs:
                        valid = False
                        break
                    diffs.add(d)
                if not valid:
                    break
            if valid:
                ruler.append(pos)
                break
            pos += 1
    return ruler

def _sidon_set(n, max_val=None):
    """Generate a Sidon (B_2) set of size n using greedy method."""
    if n <= 1:
        return [0]
    if max_val is None:
        max_val = n * n + n + 1
    S = [0]
    for _ in range(n - 1):
        found = False
        for x in range(1, max_val + 1):
            # Check if x can be added: all pairwise sums must be unique
            valid = True
            sums = set()
            for a in S:
                for b in S:
                    s = a + b
                    if s in sums:
                        valid = False
                        break
                    sums.add(s)
                if not valid:
                    break
            if valid:
                new_S = S + [x]
                sums = set()
                for a in new_S:
                    for b in new_S:
                        s = a + b
                        if s in sums:
                            valid = False
                            break
                        sums.add(s)
                if valid:
                    S.append(x)
                    found = True
                    break
        if not found:
            break
    return S

def _singer_construction(q):
    """Singer construction for Sidon sets from finite fields."""
    if q < 2:
        return [0]
    # Build finite field GF(q^2) if q is prime power
    # For simplicity, use the standard Singer construction
    # The set {0} union {g^i : i in some range} in the field
    # Simplified: use known small cases
    known = {
        2: [0, 1, 3],
        3: [0, 1, 4, 6],
        4: [0, 1, 4, 10, 11],
        5: [0, 1, 4, 10, 12, 17],
    }
    if q in known:
        return known[q]
    # Fallback to greedy Sidon
    return _sidon_set(q + 1, max_val=q * q + q)

def _local_search(U, max_iter=500, time_limit=10.0, deadline=None):
    """Enhanced local search with larger moves and better exploration."""
    start = time.time()
    U = sorted(set(U))
    best_score = _score_set(U)
    best_U = list(U)
    
    max_val = max(U) if U else 10
    search_range = max(max_val * 3, 80)
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        if time.time() - start > time_limit:
            break
        improved = False
        
        # Try adding a new element
        candidates_add = []
        if len(U) < 25:
            candidates_add = list(range(1, search_range + 1))
        
        # Try removing an element (keep 0)
        candidates_remove = [x for x in U if x != 0]
        
        # Try shifting a non-zero element with larger deltas
        candidates_shift = []
        for x in U:
            if x == 0:
                continue
            for delta in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 7, 10, 15, 20]:
                new_x = x + delta
                if new_x >= 0 and new_x != x and new_x <= search_range:
                    candidates_shift.append((x, new_x))
        
        all_moves = []
        for x in candidates_add:
            if x not in U:
                all_moves.append(('add', x))
        for x in candidates_remove:
            all_moves.append(('remove', x))
        for old, new in candidates_shift:
            if new not in U or new == old:
                all_moves.append(('shift', old, new))
        
        random.shuffle(all_moves)
        
        for move in all_moves[:30]:
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
            
            s = _score_set(new_U)
            if s > best_score + 1e-12:
                best_score = s
                best_U = new_U
                improved = True
                break
        
        if not improved:
            # Try larger random perturbations
            for _ in range(8):
                if deadline is not None and time.time() > deadline:
                    break
                if time.time() - start > time_limit:
                    break
                new_U = list(U)
                # Add 1-3 random elements
                for _ in range(random.randint(1, 3)):
                    new_U.append(random.randint(1, search_range))
                # Optionally remove 0-2 elements
                if len(new_U) > 4:
                    for _ in range(random.randint(0, 2)):
                        rem = random.choice([x for x in new_U if x != 0])
                        new_U.remove(rem)
                
                if 0 not in new_U or len(new_U) < 2:
                    continue
                s = _score_set(new_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = new_U
                    improved = True
                    break
        
        if not improved:
            # Try Sidon-inspired restart
            if random.random() < 0.1:
                sidon = _sidon_set(random.randint(3, 8), max_val=search_range)
                if len(sidon) >= 3:
                    s = _score_set(sidon)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_U = sidon
                        improved = True
    
    return best_U, best_score

def _construct_candidate(seed_params):
    """Construct a candidate set from parameters."""
    U = {0}
    for p in seed_params:
        if p >= 0:
            U.add(p)
    return sorted(U)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_U = [0, 1, 3]
    best_score = _score_set(best_U)
    
    # Try several structured initial candidates
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
        # Sparse structures
        [0, 1, 4, 5, 8],
        [0, 1, 3, 7, 8],
        [0, 2, 3, 5, 7],
        [0, 1, 4, 6, 7],
        # Larger
        [0, 1, 2, 4, 5, 7, 8],
        [0, 1, 3, 4, 6, 7, 9],
        # Golomb rulers
        _golomb_ruler(3),
        _golomb_ruler(4),
        _golomb_ruler(5),
        _golomb_ruler(6),
        _golomb_ruler(7),
        _golomb_ruler(8),
        _golomb_ruler(9),
        _golomb_ruler(10),
        # Known good configurations
        [0, 1, 2, 5],
        [0, 1, 4, 10, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 22],
        # Sidon sets
        _sidon_set(3, max_val=30),
        _sidon_set(4, max_val=40),
        _sidon_set(5, max_val=50),
        _sidon_set(6, max_val=60),
        _sidon_set(7, max_val=70),
        # Singer constructions
        _singer_construction(2),
        _singer_construction(3),
        _singer_construction(4),
        _singer_construction(5),
        # More sparse patterns
        [0, 1, 5, 6, 10],
        [0, 1, 3, 7, 9, 12],
        [0, 2, 4, 7, 11, 13],
        [0, 1, 4, 8, 9, 13],
    ]
    
    for cand in initial_candidates:
        if time.time() > deadline:
            break
        s = _score_set(cand)
        if s > best_score:
            best_score = s
            best_U = list(cand)
    
    # Local search from best
    if time.time() < deadline - 5:
        best_U, best_score = _local_search(best_U, max_iter=300, time_limit=min(15, deadline - time.time() - 5), deadline=deadline)
    
    # Try more random starts with varied strategies
    random.seed(42)
    attempts = 0
    while time.time() < deadline - 5 and attempts < 80:
        attempts += 1
        strategy = attempts % 4
        if strategy == 0:
            # Random sparse
            size = random.randint(3, 10)
            max_val = random.randint(15, 80)
            U = {0}
            while len(U) < size:
                U.add(random.randint(1, max_val))
        elif strategy == 1:
            # Sidon-based
            U = set(_sidon_set(random.randint(3, 8), max_val=80))
        elif strategy == 2:
            # Golomb-based
            U = set(_golomb_ruler(random.randint(4, 10)))
        else:
            # Structured: arithmetic-like with gaps
            base = random.randint(2, 5)
            size = random.randint(3, 8)
            U = {0}
            for i in range(1, size):
                U.add(i * base + random.randint(-1, 1) * random.randint(0, 2))
        U = sorted([x for x in U if x >= 0])
        
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U
        
        if time.time() < deadline - 5:
            U2, s2 = _local_search(U, max_iter=150, time_limit=min(6, deadline - time.time() - 5), deadline=deadline)
            if s2 > best_score:
                best_score = s2
                best_U = U2
    
    # Final polish with more iterations
    if time.time() < deadline - 2:
        best_U, best_score = _local_search(best_U, max_iter=400, time_limit=min(12, deadline - time.time() - 2), deadline=deadline)
    
    return best_U

# EVOLVE-BLOCK-END
