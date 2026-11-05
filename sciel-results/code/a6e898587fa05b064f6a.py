"""Optimized additive basis construction for [0,128] covering [128,256]."""
import time
import random
import numpy as np

N = 128
TARGET_LO = 128
TARGET_HI = 256
TARGETS = list(range(TARGET_LO, TARGET_HI + 1))
NUM_TARGETS = len(TARGETS)

def _compute_peak(A_set):
    """Compute the maximum representation multiplicity for sums in [128,256]."""
    if not A_set:
        return float('inf')
    
    # Create indicator vector
    v = np.zeros(N + 1, dtype=np.int32)
    for a in A_set:
        v[a] = 1
        
    # Convolution gives ordered pair counts
    # w[k] = number of pairs (a,b) in A x A such that a+b=k
    w = np.convolve(v, v, mode='full')
    
    # We are interested in k in [128, 256]
    # The index in w corresponds to the sum k
    # w has length 2*(N+1)-1 = 257. Indices 0..256.
    
    peak = 0
    for k in range(TARGET_LO, TARGET_HI + 1):
        ordered_count = w[k]
        # Check if k/2 is in A
        if k % 2 == 0:
            half = k // 2
            if 0 <= half <= N and v[half] == 1:
                unordered_count = (ordered_count + 1) // 2
            else:
                unordered_count = ordered_count // 2
        else:
            unordered_count = ordered_count // 2
            
        if unordered_count > peak:
            peak = unordered_count
            
    return peak

def _is_valid(A_set):
    """Check if every target in [128,256] has at least one representation."""
    if not A_set:
        return False
    
    v = np.zeros(N + 1, dtype=np.int32)
    for a in A_set:
        v[a] = 1
        
    w = np.convolve(v, v, mode='full')
    
    for k in range(TARGET_LO, TARGET_HI + 1):
        ordered_count = w[k]
        if k % 2 == 0:
            half = k // 2
            if 0 <= half <= N and v[half] == 1:
                unordered_count = (ordered_count + 1) // 2
            else:
                unordered_count = ordered_count // 2
        else:
            unordered_count = ordered_count // 2
            
        if unordered_count == 0:
            return False
            
    return True

def _local_search(A_set, max_iters=500, deadline=None):
    """Local search to improve the set."""
    if deadline is not None and time.time() > deadline:
        return A_set, _compute_peak(A_set)
    
    A = sorted(A_set)
    current_peak = _compute_peak(A)
    
    for _ in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        
        # Try removing elements
        # We prioritize removing elements that are "expensive" or redundant
        # For simplicity, iterate through sorted A
        for a in A:
            new_set = A_set - {a}
            if not new_set:
                continue
            if _is_valid(new_set):
                new_peak = _compute_peak(new_set)
                if new_peak < current_peak:
                    A_set = new_set
                    A = sorted(A_set)
                    current_peak = new_peak
                    improved = True
                    break
        
        if improved:
            continue
        
        # Try swapping elements
        # Only try swaps if the set is not too small
        if len(A) >= 2:
            # Sample a few candidates to avoid O(N^2 * N) blowup
            # We'll try to replace each element with a random candidate
            # Or better: try to replace elements that contribute to high peaks
            # For now, random swaps with a limited budget
            
            candidates = [x for x in range(N + 1) if x not in A_set]
            if candidates:
                random.shuffle(candidates)
                # Limit the number of swap attempts per iteration to keep it fast
                max_swaps_to_try = 20
                for _swap_iter in range(max_swaps_to_try):
                    a_remove = random.choice(A)
                    a_add = random.choice(candidates)
                    new_set = A_set - {a_remove} | {a_add}
                    if _is_valid(new_set):
                        new_peak = _compute_peak(new_set)
                        if new_peak < current_peak:
                            A_set = new_set
                            A = sorted(A_set)
                            current_peak = new_peak
                            improved = True
                            # Update candidates list if needed? 
                            # Since we only do a few swaps, we can re-shuffle or just continue
                            # To be safe, break and restart the inner loop logic if improved
                            break
                            
                if improved:
                    continue

        if not improved:
            break
    
    return A_set, current_peak

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin
    
    # Start with a heuristic initial solution
    # A = {0, 1, ..., 63} U {64, ..., 127} ? No, that's all of them.
    # We need a subset.
    # A good starting point is often a dense set around the middle.
    # Let's try A = {0, 2, 4, ..., 128} (even numbers).
    # Sums of two evens are even. We need to cover odd sums too.
    # So we need both parities.
    
    # Try a random valid set first
    # We can construct a valid set by starting with all and removing, or adding to a small set.
    # Starting with all [0, 128] is valid. Peak is high.
    # Let's start with a medium-sized set.
    
    random.seed(42)
    
    # Initial guess: Random subset of size ~60
    best_set = set(random.sample(range(N + 1), 60))
    
    # Ensure validity by adding elements if necessary
    # This might be slow, so let's just use the local search which handles invalidity by not accepting invalid moves?
    # Wait, _local_search only accepts moves that maintain validity.
    # So the initial set MUST be valid.
    
    # Let's generate a valid set by taking a union of arithmetic progressions or similar.
    # Or simply start with the full set and remove elements?
    # Full set [0, 128] is valid.
    # Let's start with full set and run local search to remove elements.
    # This is safe because validity is maintained (removing from a valid set might break validity, 
    # but _local_search checks validity before accepting a removal).
    
    best_set = set(range(N + 1))
    best_peak = _compute_peak(best_set)
    
    # Run local search to reduce peak
    best_set, best_peak = _local_search(best_set, max_iters=1000, deadline=deadline)
    
    # Try a few random restarts if time permits
    for trial in range(10):
        if time.time() > deadline:
            break
            
        # Generate a random set that is likely to be dense enough
        # Size 50-70
        size = random.randint(50, 70)
        A_set = set(random.sample(range(N + 1), size))
        
        # Check if valid. If not, try to fix by adding random elements?
        # Or just skip if not valid?
        # If we skip, we might miss good solutions.
        # Let's try to make it valid by adding elements from the complement.
        # This is a heuristic.
        
        if not _is_valid(A_set):
            # Try to add random elements until valid or max attempts
            candidates = [x for x in range(N + 1) if x not in A_set]
            random.shuffle(candidates)
            for x in candidates:
                A_set.add(x)
                if _is_valid(A_set):
                    break
            
            if not _is_valid(A_set):
                continue
        
        # Local search
        A_set, peak = _local_search(A_set, max_iters=500, deadline=deadline)
        
        if peak < best_peak:
            best_peak = peak
            best_set = A_set
            
    return list(best_set)
