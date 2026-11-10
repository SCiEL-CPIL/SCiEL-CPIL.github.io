"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _compute_peak(A_set):
    """Compute the peak representation multiplicity for sums in [128, 256].
    Returns (peak, coverage_ok) where coverage_ok is True if all targets are covered."""
    A = sorted(A_set)
    n = len(A)
    # Count representations for each sum in [128, 256]
    counts = np.zeros(129, dtype=np.int32)  # indices 0..128 correspond to sums 128..256
    for i in range(n):
        a = A[i]
        for j in range(i, n):
            s = a + A[j]
            if 128 <= s <= 256:
                counts[s - 128] += 1
    peak = int(counts.max())
    coverage_ok = bool((counts > 0).all())
    return peak, coverage_ok

def _local_search(A_set, max_iters=50000, deadline=None):
    """Local search: try adding/removing elements to reduce peak while maintaining coverage."""
    A_set = set(A_set)
    peak, cov = _compute_peak(A_set)
    if not cov:
        return A_set, peak, False
    
    best_peak = peak
    best_set = set(A_set)
    
    rng = random.Random(42)
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        # Try a random modification
        choice = rng.random()
        
        if choice < 0.5:
            # Try adding a random element
            candidates = [x for x in range(129) if x not in A_set]
            if not candidates:
                continue
            new_elem = rng.choice(candidates)
            new_set = A_set | {new_elem}
        else:
            # Try removing a random element
            if len(A_set) <= 10:
                continue
            candidates = list(A_set)
            old_elem = rng.choice(candidates)
            new_set = A_set - {old_elem}
        
        new_peak, new_cov = _compute_peak(new_set)
        
        if new_cov and new_peak < best_peak:
            best_peak = new_peak
            best_set = new_set
            A_set = new_set
            peak = new_peak
        elif new_cov and new_peak == peak and len(new_set) < len(A_set):
            # Prefer smaller sets with same peak
            A_set = new_set
            best_set = new_set
        elif new_cov and new_peak == peak:
            # Accept with some probability for diversity
            if rng.random() < 0.3:
                A_set = new_set
                best_set = new_set
    
    return best_set, best_peak, True

def _greedy_construction():
    """Greedy construction: start with a base set and add elements to cover gaps."""
    # Start with a reasonable initial set
    # Use a pattern that tends to have low multiplicity
    A_set = set()
    
    # Add elements from both ends to help cover the range
    # Sums range from 0+0=0 to 128+128=256
    # We need to cover [128, 256]
    
    # Initial seed: try a structured set
    # Elements near 64 help with middle sums, elements at extremes help with edge sums
    for x in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
              64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
              118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128]:
        A_set.add(x)
    
    return A_set

def _ensure_coverage(A_set, deadline=None):
    """Ensure all targets in [128, 256] are covered by adding elements if needed."""
    A_set = set(A_set)
    
    # Check which targets are not covered
    while True:
        _, cov = _compute_peak(A_set)
        if cov:
            return A_set, True
        
        if deadline and time.time() > deadline:
            return A_set, False
        
        # Find uncovered targets
        A = sorted(A_set)
        counts = np.zeros(129, dtype=np.int32)
        for i in range(len(A)):
            a = A[i]
            for j in range(i, len(A)):
                s = a + A[j]
                if 128 <= s <= 256:
                    counts[s - 128] += 1
        
        uncovered = [s for s in range(129) if counts[s] == 0]
        if not uncovered:
            return A_set, True
        
        # Try to add an element that covers the most uncovered targets
        best_elem = None
        best_count = 0
        for candidate in range(129):
            if candidate in A_set:
                continue
            # Count how many uncovered targets this would cover
            cover_count = 0
            for t in uncovered:
                target = t + 128
                # Check if candidate + a in A_set == target for some a
                needed = target - candidate
                if 0 <= needed <= 128 and needed in A_set:
                    cover_count += 1
            if cover_count > best_count:
                best_count = cover_count
                best_elem = candidate
        
        if best_elem is None or best_count == 0:
            # Can't fix with single addition, try random
            candidate = random.randint(0, 128)
            A_set.add(candidate)
        else:
            A_set.add(best_elem)
    
    return A_set, False

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75 second deadline with 15s safety margin
    
    # Phase 1: Initial construction
    A_set = _greedy_construction()
    A_set, cov = _ensure_coverage(A_set, deadline)
    
    if not cov:
        # Fallback to the seed solution
        return list(range(64, 129))
    
    peak, _ = _compute_peak(A_set)
    
    # Phase 2: Local search to minimize peak
    if time.time() < deadline - 5:
        A_set, peak, cov = _local_search(A_set, max_iters=200000, deadline=deadline - 2)
    
    if not cov:
        return list(range(64, 129))
    
    return sorted(A_set)
# EVOLVE-BLOCK-END
