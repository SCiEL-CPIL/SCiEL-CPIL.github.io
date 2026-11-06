import time
import random
import numpy as np

N = 128
TARGET_LO = 128
TARGET_HI = 256
NUM_TARGETS = 129  # targets 128..256 inclusive

_START_TIME = time.time()
DEADLINE = 55.0  # Safety margin before 60s kill

def _elapsed():
    return time.time() - _START_TIME

def _pair_counts(A):
    """Compute representation counts for each target in [128,256] using vectorized ops."""
    A = np.asarray(A, dtype=np.int32)
    n = len(A)
    if n == 0:
        return np.zeros(NUM_TARGETS, dtype=np.int32)
    # Use outer sum approach: for each i, compute A[i] + A[i:]
    counts = np.zeros(NUM_TARGETS, dtype=np.int32)
    for i in range(n):
        sums = A[i] + A[i:]
        mask = (sums >= TARGET_LO) & (sums <= TARGET_HI)
        if np.any(mask):
            idx = sums[mask] - TARGET_LO
            counts += np.bincount(idx, minlength=NUM_TARGETS).astype(np.int32)[:NUM_TARGETS]
    return counts

def _pair_counts_fast(A):
    """Faster pair counts using full outer product for small sets."""
    A = np.asarray(A, dtype=np.int32)
    n = len(A)
    if n == 0:
        return np.zeros(NUM_TARGETS, dtype=np.int32)
    # Full outer sum
    outer = A[:, None] + A[None, :]
    mask = (outer >= TARGET_LO) & (outer <= TARGET_HI)
    # Only count upper triangle (i <= j) for unordered pairs
    triu_mask = np.triu(np.ones((n, n), dtype=bool), k=0)
    mask = mask & triu_mask
    if np.any(mask):
        idx = outer[mask] - TARGET_LO
        counts = np.bincount(idx, minlength=NUM_TARGETS).astype(np.int32)[:NUM_TARGETS]
    else:
        counts = np.zeros(NUM_TARGETS, dtype=np.int32)
    return counts

def _is_valid(counts):
    return int(np.min(counts)) > 0

def _get_peak(counts):
    return int(np.max(counts))

def _delta_swap(A, i, x_new):
    """
    Compute the delta in counts when replacing A[i] with x_new.
    Returns a numpy array of length NUM_TARGETS with the net change.
    """
    A = np.asarray(A, dtype=np.int32)
    n = len(A)
    ai = A[i]
    delta = np.zeros(NUM_TARGETS, dtype=np.int32)
    
    # Remove contributions of ai
    # Pair (ai, ai)
    s = ai + ai
    if TARGET_LO <= s <= TARGET_HI:
        delta[s - TARGET_LO] -= 1
    # Pairs (ai, Aj) for j != i
    for j in range(n):
        if j == i:
            continue
        s = ai + A[j]
        if TARGET_LO <= s <= TARGET_HI:
            delta[s - TARGET_LO] -= 1
    # Pairs (Aj, ai) for j < i (already counted as (ai, Aj) above since we do all j != i)
    # Actually we need to be careful: unordered pairs (i,j) with i<j.
    # When we remove element at index i, we remove:
    #   - pair (i,i) if i exists
    #   - pair (j,i) for all j < i
    #   - pair (i,j) for all j > i
    # But since pairs are unordered, (j,i) and (i,j) are the same.
    # Let me redo: the pairs involving index i are (i,j) for all j != i, plus (i,i).
    # Each unordered pair {i,j} with j != i appears once.
    
    # Actually my original approach counts each pair once by iterating j from i to n-1.
    # So when removing A[i], I remove:
    #   - (i,i): s = 2*ai
    #   - (i,j) for j > i: s = ai + A[j]
    #   - (j,i) for j < i: s = A[j] + ai
    # But (j,i) and (i,j) are the same unordered pair. In my _pair_counts, I do:
    #   for i in range(n): for j in range(i, n): sum = A[i]+A[j]
    # So pair {i,j} is counted when min(i,j) is the outer loop index.
    # When removing A[i], the pairs that involved A[i] are:
    #   - (i,i) if i==i: always
    #   - (i,j) for j > i (where i is outer)
    #   - (j,i) for j < i (where j is outer)
    # All these have sum ai + A[j] (or 2*ai for the self-pair).
    
    # Add contributions of x_new
    s = x_new + x_new
    if TARGET_LO <= s <= TARGET_HI:
        delta[s - TARGET_LO] += 1
    for j in range(n):
        if j == i:
            continue
        s = x_new + A[j]
        if TARGET_LO <= s <= TARGET_HI:
            delta[s - TARGET_LO] += 1
    
    return delta

def _local_search_swap(A, max_time=50.0, rng=None):
    """
    Swap-based local search to minimize peak multiplicity while maintaining validity.
    Returns (best_A, best_peak).
    """
    if rng is None:
        rng = random.Random()
    
    A = sorted(list(A))
    A_set = set(A)
    counts = _pair_counts_fast(A)
    
    if not _is_valid(counts):
        return (A, float('inf'))
    
    current_peak = _get_peak(counts)
    best_A = list(A)
    best_peak = current_peak
    
    n = len(A)
    no_improve = 0
    max_no_improve = 200
    
    while _elapsed() < max_time and no_improve < max_no_improve:
        no_improve += 1
        improved = False
        
        # Shuffle indices
        indices = list(range(n))
        rng.shuffle(indices)
        
        for i in indices:
            if _elapsed() > max_time:
                break
            ai = A[i]
            
            # Generate candidates: values in [0,128] not in A_set
            # To speed up, try a random subset
            all_candidates = [x for x in range(0, N + 1) if x not in A_set and x != ai]
            if not all_candidates:
                break
            rng.shuffle(all_candidates)
            
            # Try a limited number of candidates per position
            for x_new in all_candidates[:50]:
                delta = _delta_swap(A, i, x_new)
                new_counts = counts + delta
                
                if np.min(new_counts) <= 0:
                    continue
                
                new_peak = int(np.max(new_counts))
                
                if new_peak < current_peak:
                    # Accept
                    A[i] = x_new
                    A_set.discard(ai)
                    A_set.add(x_new)
                    A.sort()
                    counts = new_counts
                    current_peak = new_peak
                    improved = True
                    no_improve = 0
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A = list(A)
                    break  # Restart after a move
        
        if not improved:
            no_improve += 10  # Accelerate giving up if no improvement
    
    return (best_A, best_peak)

def _try_remove(A, counts, peak):
    """
    Try to remove elements from A while maintaining validity and not increasing peak.
    Returns (new_A, new_counts, new_peak) or None.
    """
    A = sorted(list(A))
    A_set = set(A)
    n = len(A)
    
    for i in range(n):
        if _elapsed() > DEADLINE - 5:
            break
        ai = A[i]
        
        # Compute what happens if we remove A[i]
        new_A = A[:i] + A[i+1:]
        if len(new_A) == 0:
            continue
        
        new_counts = _pair_counts_fast(new_A)
        
        if not _is_valid(new_counts):
            continue
        
        new_peak = _get_peak(new_counts)
        
        if new_peak <= peak:
            return (new_A, new_counts, new_peak)
    
    return None

def _greedy_build(rng, max_time=50.0):
    """
    Build a set greedily to cover all targets, then optimize.
    """
    A = []
    A_set = set()
    counts = np.zeros(NUM_TARGETS, dtype=np.int32)
    
    # Start with anchors
    initial = [0, 128, 64]
    for x in initial:
        if x not in A_set:
            A.append(x)
            A_set.add(x)
    A.sort()
    counts = _pair_counts_fast(A)
    
    while _elapsed() < max_time:
        if _is_valid(counts):
            break
        
        # Find uncovered targets
        uncovered = np.where(counts == 0)[0]
        if len(uncovered) == 0:
            break
        
        # Pick a random uncovered target
        target_idx = rng.choice(uncovered)
        target_val = TARGET_LO + target_idx
        
        best_x = None
        best_gain = -1
        
        for x in range(0, N + 1):
            if x in A_set:
                continue
            
            gain = 0
            for y in A:
                s = x + y
                if TARGET_LO <= s <= TARGET_HI and counts[s - TARGET_LO] == 0:
                    gain += 1
            s = x + x
            if TARGET_LO <= s <= TARGET_HI and counts[s - TARGET_LO] == 0:
                gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_x = x
        
        if best_x is None or best_gain <= 0:
            break
        
        A.append(best_x)
        A_set.add(best_x)
        A.sort()
        counts = _pair_counts_fast(A)
    
    return A, counts

def _simulated_annealing_polish(A, counts, peak, max_time=50.0, rng=None):
    """
    Simulated annealing style polish: accept moves that don't increase peak,
    occasionally accept worse moves to escape local optima.
    """
    if rng is None:
        rng = random.Random()
    
    A = sorted(list(A))
    A_set = set(A)
    n = len(A)
    
    temperature = 2.0
    cooling = 0.995
    
    while _elapsed() < max_time and temperature > 0.1:
        temperature *= cooling
        
        # Random swap
        i = rng.randint(0, n - 1)
        ai = A[i]
        candidates = [x for x in range(0, N + 1) if x not in A_set and x != ai]
        if not candidates:
            continue
        x_new = rng.choice(candidates)
        
        delta = _delta_swap(A, i, x_new)
        new_counts = counts + delta
        
        if np.min(new_counts) <= 0:
            continue
        
        new_peak = int(np.max(new_counts))
        
        if new_peak <= peak:
            # Accept (improvement or neutral)
            A[i] = x_new
            A_set.discard(ai)
            A_set.add(x_new)
            A.sort()
            counts = new_counts
            peak = new_peak
        else:
            # Accept with probability exp(-(new_peak - peak) / temperature)
            prob = np.exp(-(new_peak - peak) / max(temperature, 0.01))
            if rng.random() < prob:
                A[i] = x_new
                A_set.discard(ai)
                A_set.add(x_new)
                A.sort()
                counts = new_counts
                # Don't update peak for worse moves in SA
    
    return (A, counts, peak)

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    rng = random.Random(12345)
    
    best_A = None
    best_peak = float('inf')
    
    # Strategy 1: Greedy build + local search + removal
    # Strategy 2: Random init + repair + local search
    # Strategy 3: Structured init (arithmetic progressions, etc.)
    
    # First, try a structured approach: use a base that's known to work
    # A = {0, 1, 2, ..., k} covers [0, 2k]. For [128, 256], need 2k >= 256, k >= 128.
    # That's the full set [0,128], which has high peak.
    # Better: use a sparse set.
    
    # Let's try multiple initializations
    
    initializations = []
    
    # 1. Greedy
    A_greedy, _ = _greedy_build(rng, max_time=DEADLINE - 20)
    if len(A_greedy) > 0:
        initializations.append(A_greedy)
    
    # 2. Random sets of various sizes
    for size in [16, 20, 24, 28, 32, 36, 40, 48, 56]:
        if _elapsed() > DEADLINE - 10:
            break
        A_rand = rng.sample(range(0, N + 1), size)
        initializations.append(sorted(A_rand))
    
    # 3. Structured: arithmetic progression
    for step in [3, 4, 5, 6, 8]:
        A_ap = list(range(0, N + 1, step))
        initializations.append(A_ap)
    
    # 4. Two interleaved progressions
    for step in [4, 6, 8]:
        A_two = list(range(0, N + 1, step)) + list(range(step//2, N + 1, step))
        A_two = sorted(set(A_two))
        initializations.append(A_two)
    
    # Process each initialization
    for A_init in initializations:
        if _elapsed() > DEADLINE - 5:
            break
        
        A_init = sorted(set(A_init))
        counts = _pair_counts_fast(A_init)
        
        if not _is_valid(counts):
            # Try to repair by adding elements
            A_repaired = list(A_init)
            A_set = set(A_repaired)
            counts_repaired = counts.copy()
            
            for _ in range(20):
                if _is_valid(counts_repaired):
                    break
                uncovered = np.where(counts_repaired == 0)[0]
                if len(uncovered) == 0:
                    break
                target_idx = rng.choice(uncovered)
                best_x = None
                best_gain = -1
                for x in range(0, N + 1):
                    if x in A_set:
                        continue
                    gain = 0
                    for y in A_repaired:
                        s = x + y
                        if TARGET_LO <= s <= TARGET_HI and counts_repaired[s - TARGET_LO] == 0:
                            gain += 1
                    s = x + x
                    if TARGET_LO <= s <= TARGET_HI and counts_repaired[s - TARGET_LO] == 0:
                        gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None or best_gain <= 0:
                    break
                A_repaired.append(best_x)
                A_set.add(best_x)
                A_repaired.sort()
                counts_repaired = _pair_counts_fast(A_repaired)
            
            if not _is_valid(counts_repaired):
                continue
            A_init = A_repaired
            counts = counts_repaired
        
        # Local search
        A_opt, peak = _local_search_swap(A_init, max_time=min(30.0, DEADLINE - 5), rng=rng)
        
        if peak < best_peak:
            best_peak = peak
            best_A = A_opt
        
        # Try removal
        if best_A is not None:
            counts_best = _pair_counts_fast(best_A)
            for _ in range(5):
                result = _try_remove(best_A, counts_best, best_peak)
                if result is None:
                    break
                best_A, counts_best, best_peak = result
        
        # Simulated annealing polish
        if best_A is not None and _elapsed() < DEADLINE - 3:
            counts_best = _pair_counts_fast(best_A)
            peak_best = _get_peak(counts_best)
            A_sa, counts_sa, peak_sa = _simulated_annealing_polish(
                best_A, counts_best, peak_best, 
                max_time=min(10.0, DEADLINE - 3), rng=rng
            )
            if peak_sa < best_peak:
                best_peak = peak_sa
                best_A = A_sa
        
        # Early termination
        if best_peak <= 4:
            break
    
    # Fallback
    if best_A is None or not _is_valid(_pair_counts_fast(best_A)):
        best_A = list(range(0, 129))
    
    return best_A
