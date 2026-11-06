"""Additive bases with few representations - simulated annealing + local search."""
import time
import random
import numpy as np

N = 128
TARGET_LO = 128
TARGET_HI = 256
NUM_TARGETS = 129

_START_TIME = time.time()
DEADLINE = 85.0

def _elapsed():
    return time.time() - _START_TIME

def _pair_counts(A):
    """Compute representation counts for each target in [128,256]."""
    counts = [0] * NUM_TARGETS
    n = len(A)
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            s = ai + A[j]
            if TARGET_LO <= s <= TARGET_HI:
                counts[s - TARGET_LO] += 1
    return counts

def _peak(counts):
    return max(counts)

def _is_valid(counts):
    return all(c > 0 for c in counts)

def _objective(counts):
    """Lower is better. Penalize uncovered targets heavily."""
    if not _is_valid(counts):
        uncovered = sum(1 for c in counts if c == 0)
        return 1000 + uncovered
    return _peak(counts)

def _try_swap(A_sorted, A_set, counts, i, x_new):
    """Try replacing A_sorted[i] with x_new. Returns (new_counts, new_peak) or None."""
    ai = A_sorted[i]
    if x_new == ai:
        return None
    if x_new in A_set:
        return None
    
    n = len(A_sorted)
    new_counts = list(counts)
    
    # Remove contributions of ai
    for j in range(n):
        if j == i:
            continue
        aj = A_sorted[j]
        s = ai + aj
        if TARGET_LO <= s <= TARGET_HI:
            new_counts[s - TARGET_LO] -= 1
    s = ai + ai
    if TARGET_LO <= s <= TARGET_HI:
        new_counts[s - TARGET_LO] -= 1
    
    # Add contributions of x_new
    for j in range(n):
        if j == i:
            continue
        aj = A_sorted[j]
        s = x_new + aj
        if TARGET_LO <= s <= TARGET_HI:
            new_counts[s - TARGET_LO] += 1
    s = x_new + x_new
    if TARGET_LO <= s <= TARGET_HI:
        new_counts[s - TARGET_LO] += 1
    
    if any(c < 0 for c in new_counts):
        return None
    
    peak = max(new_counts)
    return (new_counts, peak)

def _local_search(A_sorted, max_iters=500, time_limit=None):
    """Perform swap-based local search to minimize peak multiplicity."""
    A_set = set(A_sorted)
    counts = _pair_counts(A_sorted)
    current_peak = _peak(counts)
    current_obj = _objective(counts)
    
    if not _is_valid(counts):
        return (A_sorted, current_peak, counts)
    
    best_A = list(A_sorted)
    best_peak = current_peak
    best_counts = list(counts)
    
    n = len(A_sorted)
    iters = 0
    while iters < max_iters:
        if time_limit is not None and _elapsed() > time_limit:
            break
        iters += 1
        improved = False
        # Try all possible swaps
        for i in range(n):
            ai = A_sorted[i]
            for x_new in range(0, N + 1):
                if x_new == ai or x_new in A_set:
                    continue
                result = _try_swap(A_sorted, A_set, counts, i, x_new)
                if result is None:
                    continue
                new_counts, new_peak = result
                new_obj = _objective(new_counts)
                if new_obj < current_obj:
                    # Accept
                    A_sorted[i] = x_new
                    A_set.discard(ai)
                    A_set.add(x_new)
                    A_sorted.sort()
                    counts = new_counts
                    current_peak = new_peak
                    current_obj = new_obj
                    improved = True
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A = list(A_sorted)
                        best_counts = list(counts)
                    break
            if improved:
                break
        if not improved:
            break
    
    return (best_A, best_peak, best_counts)

def _simulated_annealing(A_sorted, start_temp=10.0, end_temp=0.1, max_iters=2000, time_limit=None):
    """Simulated annealing with swap moves."""
    A_set = set(A_sorted)
    counts = _pair_counts(A_sorted)
    current_obj = _objective(counts)
    
    best_A = list(A_sorted)
    best_obj = current_obj
    best_counts = list(counts)
    best_peak = _peak(counts)
    
    n = len(A_sorted)
    rng = random.Random(42)
    
    for it in range(max_iters):
        if time_limit is not None and _elapsed() > time_limit:
            break
        
        temp = start_temp * (end_temp / start_temp) ** (it / max_iters)
        
        # Random swap
        i = rng.randint(0, n - 1)
        ai = A_sorted[i]
        x_new = rng.randint(0, N)
        
        if x_new == ai or x_new in A_set:
            continue
        
        result = _try_swap(A_sorted, A_set, counts, i, x_new)
        if result is None:
            continue
        
        new_counts, new_peak = result
        new_obj = _objective(new_counts)
        
        delta = new_obj - current_obj
        
        if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-10)):
            # Accept
            A_sorted[i] = x_new
            A_set.discard(ai)
            A_set.add(x_new)
            A_sorted.sort()
            counts = new_counts
            current_obj = new_obj
            
            if current_obj < best_obj:
                best_obj = current_obj
                best_A = list(A_sorted)
                best_counts = list(counts)
                best_peak = _peak(counts)
    
    return (best_A, best_peak, best_counts)

def _generate_initial_A(size, rng):
    """Generate an initial random set of given size from [0, 128]."""
    return sorted(rng.sample(range(0, N + 1), size))

def _fix_coverage(A, max_add=20):
    """Add elements to cover all targets if needed."""
    A = list(A)
    A_set = set(A)
    counts = _pair_counts(A)
    
    for _ in range(max_add):
        if _is_valid(counts):
            break
        uncovered = [i for i in range(NUM_TARGETS) if counts[i] == 0]
        if not uncovered:
            break
        
        # Find best x to add
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
            candidates = [x for x in range(0, N + 1) if x not in A_set]
            if not candidates:
                break
            best_x = random.choice(candidates)
        
        A.append(best_x)
        A_set.add(best_x)
        A.sort()
        counts = _pair_counts(A)
    
    return A, counts

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    rng = random.Random(123)
    
    best_A = None
    best_peak = float('inf')
    best_counts = None
    
    # Try various sizes
    sizes_to_try = [35, 38, 40, 42, 45, 48, 50, 55]
    
    for size in sizes_to_try:
        if _elapsed() > DEADLINE - 5:
            break
        
        for restart in range(3):
            if _elapsed() > DEADLINE - 3:
                break
            
            # Generate initial set
            A = _generate_initial_A(size, rng)
            A, counts = _fix_coverage(A)
            
            if not _is_valid(counts):
                continue
            
            peak = _peak(counts)
            
            # Simulated annealing
            A_sorted = sorted(A)
            A_sorted, peak, counts = _simulated_annealing(
                A_sorted, start_temp=5.0, end_temp=0.01, 
                max_iters=1000, time_limit=DEADLINE - 3
            )
            
            # Local search polish
            A_sorted, peak, counts = _local_search(
                A_sorted, max_iters=200, time_limit=DEADLINE - 2
            )
            
            if peak < best_peak:
                best_peak = peak
                best_A = list(A_sorted)
                best_counts = list(counts)
            
            if best_peak <= 2:
                break
        
        if best_peak <= 2:
            break
    
    # If we still don't have a valid set, fall back
    if best_A is None or not _is_valid(best_counts):
        best_A = list(range(64, 129))
        best_counts = _pair_counts(best_A)
        best_peak = _peak(best_counts)
    
    return best_A
