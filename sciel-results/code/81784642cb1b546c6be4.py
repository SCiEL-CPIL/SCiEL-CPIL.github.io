"""Model-and-optimize approach for sumset/difference-set exponent lower bound."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

def _score_set(U):
    """Compute the score for a finite set U containing 0."""
    U_set = set(U)
    n = len(U_set)
    if n == 0:
        return 0.0
    max_u = max(U_set)
    if max_u == 0:
        return 1.0
    
    # Compute sumset and difference set efficiently using sets
    sumset = set()
    for a in U_set:
        for b in U_set:
            sumset.add(a + b)
    
    diffset = set()
    for a in U_set:
        for b in U_set:
            diffset.add(a - b)
    
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + log_ratio / log_denom

def _score_set_fast(U):
    """Faster scoring using numpy where beneficial for moderate sizes."""
    U_set = set(U)
    n = len(U_set)
    if n == 0:
        return 0.0
    max_u = max(U_set)
    if max_u == 0:
        return 1.0
    
    arr = np.array(sorted(U_set), dtype=np.int64)
    
    # Sumset via broadcasting for small n
    if n <= 200:
        sums = arr[:, None] + arr[None, :]
        sumset_size = len(np.unique(sums))
        diffs = arr[:, None] - arr[None, :]
        diffset_size = len(np.unique(diffs))
    else:
        # Fall back to set-based for large n
        sumset = set()
        for a in U_set:
            for b in U_set:
                sumset.add(a + b)
        sumset_size = len(sumset)
        
        diffset = set()
        for a in U_set:
            for b in U_set:
                diffset.add(a - b)
        diffset_size = len(diffset)
    
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + log_ratio / log_denom

def _score_arithmetic_progression(n, d):
    """
    Analytically compute the score for U = {0, d, 2d, ..., (n-1)d}.
    For this set:
      - Sumset U+U = {0, d, 2d, ..., 2(n-1)d}, so |U+U| = 2n-1
      - Diffset U-U = {-(n-1)d, ..., (n-1)d}, so |U-U| = 2n-1
      - max(U) = (n-1)*d
      - ratio = (2n-1)/(2n-1) = 1, so score = 1.0
    
    This is not useful. We need to perturb the AP to break the symmetry.
    """
    # Pure AP gives ratio=1, score=1.0. Not helpful directly.
    # But it's a good starting point for perturbation.
    return 1.0

def _construct_golomb(n):
    """Construct a Golomb ruler of order n."""
    if n <= 1:
        return [0]
    
    # Known optimal Golomb rulers
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
        13: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72],
        14: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 84],
    }
    if n in known:
        return known[n]
    
    # Greedy construction
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

def _construct_perturbed_ap(n, d, seed=None):
    """
    Construct a perturbed arithmetic progression: U = {0, d+e1, 2d+e2, ..., (n-1)d+e_{n-1}}
    where e_i are small perturbations chosen to maximize the score.
    
    The idea: a pure AP has |U+U| = |U-U| = 2n-1, giving score 1.0.
    By perturbing, we can make |U-U| grow faster than |U+U| relative to max(U).
    
    Specifically, if we perturb so that all differences are distinct (Golomb-like),
    then |U-U| = n(n-1) + 1, while |U+U| may grow more slowly.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random
    
    U = [0]
    for i in range(1, n):
        base = i * d
        # Try several perturbations and pick the best
        best_val = base
        best_score = -1
        for e in range(-d//2, d//2 + 1):
            if e == 0:
                continue
            candidate = base + e
            if candidate < 0:
                continue
            # Quick check: does this maintain distinct differences?
            new_U = U + [candidate]
            new_U_sorted = sorted(new_U)
            # Check all pairwise differences are distinct
            diffs = set()
            valid = True
            for j in range(len(new_U_sorted)):
                for k in range(j + 1, len(new_U_sorted)):
                    diff = new_U_sorted[k] - new_U_sorted[j]
                    if diff in diffs:
                        valid = False
                        break
                    diffs.add(diff)
                if not valid:
                    break
            if valid:
                # This is a Golomb ruler extension - compute score
                s = _score_set_fast(new_U_sorted)
                if s > best_score:
                    best_score = s
                    best_val = candidate
        # If no valid perturbation found, use base
        if best_val == base:
            # Try to find any valid position
            for e in range(1, d + 1):
                candidate = base + e
                if candidate < 0:
                    continue
                new_U = U + [candidate]
                new_U_sorted = sorted(new_U)
                diffs = set()
                valid = True
                for j in range(len(new_U_sorted)):
                    for k in range(j + 1, len(new_U_sorted)):
                        diff = new_U_sorted[k] - new_U_sorted[j]
                        if diff in diffs:
                            valid = False
                            break
                        diffs.add(diff)
                    if not valid:
                        break
                if valid:
                    best_val = candidate
                    break
        U.append(best_val)
    
    return sorted(U)

def _construct_sidon_sequence(n, max_val=None):
    """Construct a Sidon (B_2) sequence of length n."""
    if max_val is None:
        max_val = n * n * 4
    
    seq = [0]
    used_sums = {0}
    
    for i in range(1, n):
        pos = 1
        while pos <= max_val:
            valid = True
            for j, a in enumerate(seq):
                s = a + pos
                if s in used_sums:
                    valid = False
                    break
            if valid:
                # Check all pairs including new element
                all_vals = seq + [pos]
                sums = set()
                for x in range(len(all_vals)):
                    for y in range(x, len(all_vals)):
                        s = all_vals[x] + all_vals[y]
                        if s in sums:
                            valid = False
                            break
                        sums.add(s)
                    if not valid:
                        break
                if valid:
                    seq.append(pos)
                    break
            pos += 1
        if len(seq) < i + 1:
            break
    
    return seq

def _optimize_ap_gap(n, d_range):
    """
    For a given n, find the optimal gap d for a perturbed AP.
    We try to construct a Golomb ruler of order n with elements roughly
    spaced by d, which maximizes |U-U|/|U+U| for a given max(U).
    """
    best_score = -1
    best_U = [0]
    
    for d in d_range:
        U = _construct_perturbed_ap(n, d)
        if len(U) < n:
            continue
        s = _score_set_fast(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    return best_U, best_score

def _local_search(U, max_iters=3000, deadline=None, max_size=50):
    """Improve set U via local search with add/remove/modify operations."""
    U = sorted(set(U))
    best_score = _score_set_fast(U)
    best_U = U[:]
    max_u = max(U) if U else 0
    upper_bound = max_u + 150
    
    # Use fast scorer for sets up to 200 elements
    use_fast = len(U) <= 200
    scorer = _score_set_fast if use_fast else _score_set
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        improved = False
        U_work = U[:]
        random.shuffle(U_work)
        
        # Try modifying one element
        for i in range(len(U_work)):
            old_val = U_work[i]
            # Try a few candidate replacements
            candidates = []
            # Try values near current
            for delta in [-10, -7, -5, -3, -2, -1, 1, 2, 3, 5, 7, 10, 15, 20, 30]:
                nv = old_val + delta
                if 0 <= nv <= upper_bound and nv != old_val:
                    candidates.append(nv)
            # Try random values
            for _ in range(5):
                nv = random.randint(0, upper_bound)
                if nv != old_val:
                    candidates.append(nv)
            
            for new_val in candidates:
                new_U = U_work[:]
                new_U[i] = new_val
                new_U_set = set(new_U)
                if len(new_U_set) != len(new_U):
                    continue
                new_U_sorted = sorted(new_U_set)
                if new_U_sorted[0] != 0:
                    continue
                s = scorer(new_U_sorted)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = new_U_sorted
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Try adding an element
            if len(U) < max_size:
                # Try positions that might create distinct differences
                tried = set(U)
                for new_val in range(0, upper_bound + 1):
                    if new_val in tried:
                        continue
                    new_U = U + [new_val]
                    s = scorer(new_U)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = sorted(new_U)
                        improved = True
                        break
        
        if not improved:
            # Try removing an element (if size > 2)
            if len(U) > 2:
                for i in range(len(U)):
                    if U[i] == 0:
                        continue
                    new_U = U[:i] + U[i+1:]
                    s = scorer(new_U)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = new_U
                        improved = True
                        break
        
        if not improved:
            # Random restart: mutate the whole set
            if random.random() < 0.15:
                mutated = U[:]
                # Add 1-3 random elements
                for _ in range(random.randint(1, 3)):
                    nv = random.randint(1, upper_bound)
                    if nv not in mutated:
                        mutated.append(nv)
                # Remove 0-2 random elements (not 0)
                removable = [x for x in mutated if x != 0]
                for _ in range(random.randint(0, min(2, len(removable)))):
                    if removable:
                        idx = random.randint(0, len(removable) - 1)
                        val = removable.pop(idx)
                        mutated.remove(val)
                if 0 in mutated:
                    mutated = sorted(set(mutated))
                    s = scorer(mutated)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = mutated
                        improved = True
        
        if not improved and it > 100:
            # Try expanding the search space
            upper_bound += 75
            U = best_U[:]
    
    return sorted(set(best_U)), best_score

def _optimize_parameters(start_time, deadline):
    """Optimize construction parameters directly against the objective."""
    best_score = -1.0
    best_U = [0]
    
    # Strategy 1: Golomb rulers of various sizes
    for n in range(3, 25):
        if time.time() > deadline:
            break
        gr = _construct_golomb(n)
        s = _score_set(gr)
        if s > best_score:
            best_score = s
            best_U = gr[:]
    
    # Strategy 2: Sidon sequences
    for n in range(3, 20):
        if time.time() > deadline:
            break
        ss = _construct_sidon_sequence(n, max_val=n * n * 5)
        if len(ss) >= n:
            s = _score_set(ss)
            if s > best_score:
                best_score = s
                best_U = ss[:]
    
    # Strategy 3: Perturbed arithmetic progressions (difference-sparse construction)
    # This is the key strategy: optimize gap size d for each n
    for n in range(4, 20):
        if time.time() > deadline:
            break
        # Try various gap sizes
        for d in range(2, 30):
            if time.time() > deadline:
                break
            U, s = _optimize_ap_gap(n, [d])
            if s > best_score:
                best_score = s
                best_U = U[:]
    
    # Strategy 4: Power-of-prime constructions
    for p in [2, 3, 5, 7]:
        if time.time() > deadline:
            break
        for n in range(3, 8):
            candidate = [0]
            val = 1
            for i in range(n - 1):
                val *= p
                candidate.append(val)
            s = _score_set(candidate)
            if s > best_score:
                best_score = s
                best_U = candidate[:]
    
    # Strategy 5: Quadratic-like constructions
    for n in range(3, 12):
        if time.time() > deadline:
            break
        for c in range(1, 6):
            candidate = sorted(set(i * i * c for i in range(n)))
            if 0 not in candidate:
                candidate = [0] + [x for x in candidate if x > 0]
            if len(candidate) >= 2:
                s = _score_set(candidate)
                if s > best_score:
                    best_score = s
                    best_U = candidate[:]
    
    return best_U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    best_U = [0, 1, 3]
    best_score = _score_set_fast(best_U)
    
    # Phase 1: Parameter optimization including perturbed AP
    opt_U, opt_score = _optimize_parameters(start_time, deadline)
    if opt_score > best_score:
        best_score = opt_score
        best_U = opt_U[:]
    
    # Phase 2: Local search refinement
    if time.time() < deadline:
        improved_U, improved_score = _local_search(best_U, max_iters=8000, deadline=deadline, max_size=60)
        if improved_score > best_score:
            best_score = improved_score
            best_U = improved_U
    
    # Phase 3: Try local search on other good candidates
    if time.time() < deadline:
        # Generate some diverse candidates
        candidates = []
        
        for n in [5, 8, 12, 15, 18, 20]:
            if time.time() > deadline:
                break
            gr = _construct_golomb(n)
            candidates.append(gr)
        
        # Add perturbed AP candidates
        for n in [6, 8, 10, 12]:
            if time.time() > deadline:
                break
            for d in [3, 5, 7, 10]:
                if time.time() > deadline:
                    break
                U = _construct_perturbed_ap(n, d)
                if len(U) >= n:
                    candidates.append(U)
        
        # Add quadratic and cubic sequences
        for c in [1, 2, 3]:
            if time.time() > deadline:
                break
            for n in [8, 12, 16]:
                candidate = sorted(set(i * i * c for i in range(n)))
                if len(candidate) >= 2 and 0 in candidate:
                    candidates.append(candidate)
        
        random.seed(123)
        for _ in range(15):
            if time.time() > deadline:
                break
            n = random.randint(4, 20)
            max_val = random.randint(20, 150)
            U = [0] + random.sample(range(1, max_val + 1), n - 1)
            U = sorted(U)
            candidates.append(U)
        
        for U in candidates:
            if time.time() > deadline:
                break
            s = _score_set_fast(U)
            if s > best_score - 0.05:  # Only if close to best
                improved_U, improved_score = _local_search(U, max_iters=3000, deadline=deadline, max_size=50)
                if improved_score > best_score:
                    best_score = improved_score
                    best_U = improved_U
    
    return sorted(set(best_U))
# EVOLVE-BLOCK-END
