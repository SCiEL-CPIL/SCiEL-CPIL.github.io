"""Rewritten program: search over structured integer sets to minimize max(|A+A|, |A*A|)."""
import time
import itertools
import random
from functools import lru_cache

# EVOLVE-BLOCK-START

def _sumset_size(A):
    """Compute |A+A| with equal summands allowed."""
    s = set()
    n = len(A)
    for i in range(n):
        for j in range(i, n):
            s.add(A[i] + A[j])
    return len(s)

def _prodset_size(A):
    """Compute |A*A| with equal factors allowed."""
    s = set()
    n = len(A)
    for i in range(n):
        for j in range(i, n):
            s.add(A[i] * A[j])
    return len(s)

def _objective(A):
    """Return max(|A+A|, |A*A|)."""
    return max(_sumset_size(A), _prodset_size(A))

def _is_valid(A, n=20, maximum=10000):
    """Check that A has n distinct positive integers <= maximum."""
    if len(A) != n:
        return False
    if len(set(A)) != n:
        return False
    if any(x <= 0 or x > maximum for x in A):
        return False
    return True

def _hamming_numbers_5smooth(limit=10000):
    """Generate 5-smooth numbers (of form 2^a * 3^b * 5^c) up to limit."""
    nums = set()
    a = 0
    while (2 ** a) <= limit:
        b = 0
        while (2 ** a) * (3 ** b) <= limit:
            c = 0
            while (2 ** a) * (3 ** b) * (5 ** c) <= limit:
                nums.add((2 ** a) * (3 ** b) * (5 ** c))
                c += 1
            b += 1
        a += 1
    return sorted(nums)

def _hamming_numbers_3smooth(limit=10000):
    """Generate 3-smooth numbers (of form 2^a * 3^b) up to limit."""
    nums = set()
    a = 0
    while (2 ** a) <= limit:
        b = 0
        while (2 ** a) * (3 ** b) <= limit:
            nums.add((2 ** a) * (3 ** b))
            b += 1
        a += 1
    return sorted(nums)

def _geometric_seq(ratio_num, ratio_den, start, count, max_val=10000):
    """Generate a geometric sequence."""
    seq = []
    val = start
    for _ in range(count):
        if val > max_val:
            break
        seq.append(val)
        # Multiply by ratio
        val = val * ratio_num
        if val % ratio_den == 0:
            val //= ratio_den
        else:
            val = int(val * ratio_num / ratio_den)
    return seq

def _local_search(A, deadline, rng):
    """Local search: try swapping elements to reduce objective."""
    A = sorted(A)
    current_obj = _objective(A)
    n = len(A)
    
    # Precompute sumset and prodset for faster updates
    while time.time() < deadline:
        improved = False
        # Try all single-element swaps
        for i in range(n):
            for j in range(i + 1, n):
                # Try replacing A[i] with A[j] won't work since they're distinct
                # Instead, try perturbing
                pass
        
        # Try replacing one element with a nearby value
        for i in range(n):
            old_val = A[i]
            # Try a few nearby values
            candidates = []
            for delta in range(-5, 6):
                new_val = old_val + delta
                if new_val > 0 and new_val <= 10000 and new_val != old_val:
                    if new_val not in A:
                        candidates.append(new_val)
            
            for new_val in candidates:
                A[i] = new_val
                if not _is_valid(A):
                    A[i] = old_val
                    continue
                obj = _objective(A)
                if obj < current_obj:
                    current_obj = obj
                    improved = True
                    break
                A[i] = old_val
            if improved:
                A = sorted(A)
                break
        
        if not improved:
            # Try random restarts within a neighborhood
            found = False
            for _ in range(50):
                i = rng.randint(0, n - 1)
                old_val = A[i]
                new_val = old_val + rng.randint(-10, 10)
                if new_val > 0 and new_val <= 10000 and new_val != old_val:
                    if new_val not in A:
                        A[i] = new_val
                        obj = _objective(A)
                        if obj < current_obj:
                            current_obj = obj
                            found = True
                            A = sorted(A)
                            break
                        A[i] = old_val
            if not found:
                break
    
    return A, current_obj

def _sliding_window_5smooth(window_size=20, limit=10000):
    """Sliding window over 5-smooth numbers."""
    smooth = _hamming_numbers_5smooth(limit)
    best_A = None
    best_obj = float('inf')
    
    for start in range(len(smooth) - window_size + 1):
        A = smooth[start:start + window_size]
        obj = _objective(A)
        if obj < best_obj:
            best_obj = obj
            best_A = list(A)
    
    return best_A, best_obj

def _sliding_window_3smooth(window_size=20, limit=10000):
    """Sliding window over 3-smooth numbers."""
    smooth = _hamming_numbers_3smooth(limit)
    best_A = None
    best_obj = float('inf')
    
    for start in range(len(smooth) - window_size + 1):
        A = smooth[start:start + window_size]
        obj = _objective(A)
        if obj < best_obj:
            best_obj = obj
            best_A = list(A)
    
    return best_A, best_obj

def _geometric_search(deadline, rng):
    """Search over geometric sequences."""
    best_A = None
    best_obj = float('inf')
    
    ratios = [(3, 2), (4, 3), (5, 4), (6, 5), (7, 6), (8, 7), (9, 8), (10, 9),
              (5, 3), (7, 4), (9, 5), (11, 6), (13, 7), (15, 8), (17, 9), (19, 10)]
    starts = range(1, 50)
    
    for ratio_num, ratio_den in ratios:
        for start in starts:
            if time.time() > deadline:
                return best_A, best_obj
            A = _geometric_seq(ratio_num, ratio_den, start, 20)
            if len(A) < 20:
                continue
            if not _is_valid(A):
                continue
            obj = _objective(A)
            if obj < best_obj:
                best_obj = obj
                best_A = list(A)
    
    return best_A, best_obj

def _random_structured_search(deadline, rng, n=20, maximum=10000):
    """Random search with structured perturbations."""
    best_A = None
    best_obj = float('inf')
    
    # Start from various seeds
    seed_sets = []
    
    # 5-smooth based
    smooth5 = _hamming_numbers_5smooth(maximum)
    if len(smooth5) >= n:
        for _ in range(100):
            idx = sorted(rng.sample(range(len(smooth5)), n))
            A = [smooth5[i] for i in idx]
            seed_sets.append(A)
    
    # 3-smooth based
    smooth3 = _hamming_numbers_3smooth(maximum)
    if len(smooth3) >= n:
        for _ in range(100):
            idx = sorted(rng.sample(range(len(smooth3)), n))
            A = [smooth3[i] for i in idx]
            seed_sets.append(A)
    
    # Geometric based
    for ratio_num, ratio_den in [(3, 2), (4, 3), (5, 4), (6, 5), (7, 6)]:
        for start in range(1, 30):
            A = _geometric_seq(ratio_num, ratio_den, start, n)
            if len(A) == n and _is_valid(A):
                seed_sets.append(A)
    
    # Small integers
    for start in range(1, 10):
        A = list(range(start, start + n))
        if _is_valid(A):
            seed_sets.append(A)
    
    for A in seed_sets:
        if time.time() > deadline:
            break
        obj = _objective(A)
        if obj < best_obj:
            best_obj = obj
            best_A = list(A)
    
    # Local search on best
    if best_A is not None:
        best_A, best_obj = _local_search(best_A, deadline, rng)
    
    return best_A, best_obj

def construct():
    """Find 20 distinct positive integers <= 10000 minimizing max(|A+A|, |A*A|)."""
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    rng = random.Random(42)
    
    best_A = None
    best_obj = float('inf')
    
    # 1. Sliding window over 5-smooth numbers
    A, obj = _sliding_window_5smooth()
    if A is not None and obj < best_obj:
        best_obj = obj
        best_A = A
    
    # 2. Sliding window over 3-smooth numbers
    A, obj = _sliding_window_3smooth()
    if A is not None and obj < best_obj:
        best_obj = obj
        best_A = A
    
    # 3. Geometric sequence search
    if time.time() < deadline:
        A, obj = _geometric_search(deadline, rng)
        if A is not None and obj < best_obj:
            best_obj = obj
            best_A = A
    
    # 4. Random structured search with local improvement
    if time.time() < deadline:
        A, obj = _random_structured_search(deadline, rng)
        if A is not None and obj < best_obj:
            best_obj = obj
            best_A = A
    
    # 5. Final local search on the best candidate
    if best_A is not None and time.time() < deadline:
        best_A, best_obj = _local_search(best_A, deadline, rng)
    
    if best_A is None:
        # Fallback: first 20 integers
        best_A = list(range(1, 21))
    
    return best_A

# EVOLVE-BLOCK-END
