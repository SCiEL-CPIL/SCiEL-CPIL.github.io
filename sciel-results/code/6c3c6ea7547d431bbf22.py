import math
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = 24
    L = 1
    for i in range(1, n + 1):
        L = math.lcm(L, i)
    
    V = [0] * (n + 1)
    for i in range(1, n + 1):
        V[i] = L // i

    S = set(range(1, n + 1))
    
    # Cache for subset sum queries
    subset_cache = {}
    
    def subset_sum_exists(target, values):
        """Check if any subset of values sums to target.
        Returns (bool, list_of_indices) or (False, None)."""
        if not values:
            return (False, None)
        if sum(values) < target:
            return (False, None)
        if target == 0:
            return (True, [])
        
        key = (target, tuple(sorted(values)))
        if key in subset_cache:
            return subset_cache[key]
        
        half = len(values) // 2
        left = values[:half]
        right = values[half:]
        
        left_sums = {}
        for mask in range(1 << half):
            s = 0
            for i in range(half):
                if mask & (1 << i):
                    s += left[i]
            if s not in left_sums:
                left_sums[s] = mask
        
        right_sums = {}
        for mask in range(1 << (len(values) - half)):
            s = 0
            for i in range(len(values) - half):
                if mask & (1 << i):
                    s += right[i]
            if s not in right_sums:
                right_sums[s] = mask
        
        for s_l, mask_l in left_sums.items():
            need = target - s_l
            if need in right_sums:
                mask_r = right_sums[need]
                subset = []
                for i in range(half):
                    if mask_l & (1 << i):
                        subset.append(i)
                for i in range(len(values) - half):
                    if mask_r & (1 << i):
                        subset.append(half + i)
                result = (True, subset)
                subset_cache[key] = result
                return result
        
        result = (False, None)
        subset_cache[key] = result
        return result

    def find_conflict(x, S):
        """Find if any subset of S\{x} has reciprocal sum equal to 1/x.
        Returns list of elements in the conflicting subset, or None."""
        target = V[x]
        other_vals = [(V[y], y) for y in S if y != x]
        if not other_vals:
            return None
        
        vals = [v for v, _ in other_vals]
        if sum(vals) < target:
            return None
        
        result, indices = subset_sum_exists(target, vals)
        if result:
            return [other_vals[i][1] for i in indices]
        return None

    def is_valid(S):
        """Check if S is reciprocal-subsum-free."""
        for x in S:
            conflict = find_conflict(x, S)
            if conflict is not None:
                return False
        return True

    # Phase 1: Greedy removal with better conflict resolution
    while time.time() < deadline - 15.0:
        conflicts = {}
        for x in S:
            conflict = find_conflict(x, S)
            if conflict is not None:
                conflicts[x] = conflict
        
        if not conflicts:
            break
        
        # Count how often each element appears in conflicts
        counts = {y: 0 for y in S}
        for x, witness in conflicts.items():
            counts[x] += 1
            for y in witness:
                counts[y] += 1
        
        # Remove the element with highest count
        best_elem = max(counts, key=lambda y: (counts[y], y))
        S.remove(best_elem)
    
    # Phase 2: Try to add elements back
    removed = set(range(1, n + 1)) - S
    for y in sorted(removed, reverse=True):
        if time.time() > deadline - 10.0:
            break
        trial = S | {y}
        if is_valid(trial):
            S = trial
    
    # Phase 3: Local search - try remove-1-add-1 swaps
    improved = True
    while improved and time.time() < deadline - 8.0:
        improved = False
        for x in list(S):
            for y in range(1, n + 1):
                if y in S:
                    continue
                trial = (S - {x}) | {y}
                if is_valid(trial):
                    S = trial
                    improved = True
                    break
            if improved:
                break
    
    # Phase 4: Try single additions again
    removed = set(range(1, n + 1)) - S
    for y in sorted(removed, reverse=True):
        if time.time() > deadline - 3.0:
            break
        trial = S | {y}
        if is_valid(trial):
            S = trial
    
    # Phase 5: Final local search with 1-1 swaps
    improved = True
    while improved and time.time() < deadline - 1.0:
        improved = False
        for x in list(S):
            for y in range(1, n + 1):
                if y in S:
                    continue
                trial = (S - {x}) | {y}
                if is_valid(trial):
                    S = trial
                    improved = True
                    break
            if improved:
                break
    
    return sorted(S)
