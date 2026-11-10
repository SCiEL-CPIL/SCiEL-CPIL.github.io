"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _compute_peak(A_set, target_lo=128, target_hi=256):
    """Compute the maximum number of unordered pair representations for any target in [target_lo, target_hi]."""
    A = sorted(A_set)
    n = len(A)
    # For each target t in [target_lo, target_hi], count pairs (i,j) with i<=j, A[i]+A[j]==t
    # Use a dict or array for efficiency
    # A is a subset of [0,128], so we can use an indicator array
    present = np.zeros(129, dtype=np.int8)
    for a in A:
        present[a] = 1
    
    max_mult = 0
    for t in range(target_lo, target_hi + 1):
        # Count unordered pairs (a, b) with a <= b, a + b = t, a in A, b in A
        count = 0
        for a in A:
            b = t - a
            if b < a:
                break
            if b > 128:
                continue
            if present[b]:
                count += 1
        if count > max_mult:
            max_mult = count
    return max_mult

def _compute_peak_fast(A_set, target_lo=128, target_hi=256):
    """Faster peak computation using numpy."""
    present = np.zeros(129, dtype=np.int8)
    for a in A_set:
        present[a] = 1
    
    A_arr = np.array(sorted(A_set), dtype=np.int32)
    max_mult = 0
    for t in range(target_lo, target_hi + 1):
        # a ranges from max(0, t-128) to min(128, t//2)
        a_min = max(0, t - 128)
        a_max = min(128, t // 2)
        if a_min > a_max:
            continue
        # For each a in [a_min, a_max], check if a and t-a are both in A
        a_range = np.arange(a_min, a_max + 1)
        b_range = t - a_range
        # Both must be in A
        valid = present[a_range] & present[b_range]
        count = np.sum(valid)
        if count > max_mult:
            max_mult = count
    return max_mult

def _local_search(A_set, max_time=55.0, target_lo=128, target_hi=256):
    """Local search to minimize peak multiplicity while maintaining coverage."""
    start_time = time.time()
    A_set = set(A_set)
    
    # Precompute coverage: which targets are covered by which pairs
    # For each element a in A, removing it affects targets t where there exists b in A with a+b=t
    # We need to track multiplicity for each target
    
    def _get_multiplicities(A):
        """Return dict: target -> count of unordered pairs summing to target."""
        mult = {}
        A_sorted = sorted(A)
        for t in range(target_lo, target_hi + 1):
            count = 0
            for a in A_sorted:
                b = t - a
                if b < a:
                    break
                if b > 128:
                    continue
                if b in A:
                    count += 1
            mult[t] = count
        return mult
    
    mult = _get_multiplicities(A_set)
    current_peak = max(mult.values())
    
    best_peak = current_peak
    best_A = set(A_set)
    
    # Local search: try adding/removing/swapping elements
    while time.time() - start_time < max_time:
        improved = False
        
        # Try removing elements that don't break coverage
        for a in list(A_set):
            if time.time() - start_time > max_time - 2:
                break
            # Check if removing a still covers all targets
            # A target t is still covered if there exists another pair (a', b') != (a, b) summing to t
            # For each target t, check if mult[t] > 1 or if there's another representation
            # Actually, we need: for every t in [128,256], after removing a, there's still at least one pair
            # t is covered by pairs (a,b) with a<=b, a+b=t. After removing a, t is still covered iff
            # there exists a' != a (or a' == a but with a different partner) such that a' + b' = t.
            
            # Simpler: check if any target becomes uncovered
            ok = True
            for t in range(target_lo, target_hi + 1):
                # Count representations not involving a
                count_without_a = 0
                for a2 in A_set:
                    if a2 == a:
                        continue
                    b2 = t - a2
                    if b2 < a2:
                        break
                    if b2 > 128 or b2 not in A_set:
                        continue
                    count_without_a += 1
                # Also check pair (a, a) if a+a == t: this is lost
                # And pairs (a, b) where b != a: these are lost
                # So we need count_without_a >= 1 for all t
                if count_without_a == 0:
                    ok = False
                    break
            
            if ok:
                new_A = A_set - {a}
                new_mult = _get_multiplicities(new_A)
                new_peak = max(new_mult.values())
                if new_peak < current_peak:
                    A_set = new_A
                    mult = new_mult
                    current_peak = new_peak
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A = set(A_set)
                    improved = True
        
        # Try adding elements
        if not improved:
            for a in range(129):
                if time.time() - start_time > max_time - 2:
                    break
                if a in A_set:
                    continue
                new_A = A_set | {a}
                new_mult = _get_multiplicities(new_A)
                new_peak = max(new_mult.values())
                if new_peak < current_peak:
                    A_set = new_A
                    mult = new_mult
                    current_peak = new_peak
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A = set(A_set)
                    improved = True
                    break
        
        if not improved:
            # Try swaps: remove one, add another
            found_swap = False
            for a_remove in list(A_set):
                if time.time() - start_time > max_time - 2:
                    break
                for a_add in range(129):
                    if time.time() - start_time > max_time - 2:
                        break
                    if a_add == a_remove or a_add in A_set:
                        continue
                    new_A = (A_set - {a_remove}) | {a_add}
                    # Quick check: does new_A cover all targets?
                    covers = True
                    for t in range(target_lo, target_hi + 1):
                        found = False
                        for a2 in new_A:
                            b2 = t - a2
                            if b2 < a2:
                                break
                            if b2 > 128 or b2 not in new_A:
                                continue
                            found = True
                            break
                        if not found:
                            covers = False
                            break
                    if not covers:
                        continue
                    new_mult = _get_multiplicities(new_A)
                    new_peak = max(new_mult.values())
                    if new_peak < current_peak:
                        A_set = new_A
                        mult = new_mult
                        current_peak = new_peak
                        if current_peak < best_peak:
                            best_peak = current_peak
                            best_A = set(A_set)
                        found_swap = True
                        break
                if found_swap:
                    break
        
        if not improved and not found_swap:
            # No improvement possible with single moves; try 2-opt or break
            # Try random restarts within local search
            # For now, break out
            break
    
    return best_A, best_peak

def _greedy_construct():
    """Greedy construction: start with a base set and iteratively add elements to minimize peak."""
    # Start with a structured set: every other number
    A_set = set(range(0, 129, 2))  # Even numbers: 65 elements
    
    # Check coverage
    def _covers(A):
        for t in range(128, 257):
            found = False
            for a in A:
                b = t - a
                if b < a:
                    break
                if b > 128 or b not in A:
                    continue
                found = True
                break
            if not found:
                return False
        return True
    
    if not _covers(A_set):
        # Try odd numbers
        A_set = set(range(1, 129, 2))
        if not _covers(A_set):
            # Try a mix
            A_set = set(range(0, 129, 1))  # All numbers (fallback)
    
    return A_set

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # 58s budget, 2s safety margin
    
    # Step 1: Greedy construction
    A_set = _greedy_construct()
    
    # Step 2: Local search to minimize peak
    if time.time() < deadline - 5:
        A_set, peak = _local_search(A_set, max_time=deadline - time.time() - 2)
    
    # Verify coverage
    A_set = set(A_set)
    for t in range(128, 257):
        found = False
        for a in A_set:
            b = t - a
            if b < a:
                break
            if b > 128 or b not in A_set:
                continue
            found = True
            break
        if not found:
            # Fallback: add elements to cover
            for a in range(129):
                if a not in A_set:
                    A_set.add(a)
                    # Re-check
                    for t2 in range(128, 257):
                        found2 = False
                        for a2 in A_set:
                            b2 = t2 - a2
                            if b2 < a2:
                                break
                            if b2 > 128 or b2 not in A_set:
                                continue
                            found2 = True
                            break
                        if not found2:
                            break
                    else:
                        break
            break
    
    return sorted(A_set)
# EVOLVE-BLOCK-END
