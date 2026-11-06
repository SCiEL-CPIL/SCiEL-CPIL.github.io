import time
from itertools import combinations
from fractions import Fraction

def _check_valid(S):
    """Check if S is reciprocal-subsum-free using exact arithmetic and pruning."""
    S_set = set(S)
    for x in S:
        target = Fraction(1, x)
        # Only elements y > x can contribute (1/y < 1/x for y > x)
        # Elements y < x have 1/y > 1/x, so any subset containing them exceeds target
        candidates = [y for y in S if y > x]
        if not candidates:
            continue
        # Sort candidates by value (ascending = descending reciprocal)
        candidates.sort(reverse=True)  # largest first = smallest reciprocal first
        # DFS with pruning: build subset, prune if partial sum >= target
        n = len(candidates)
        if n > 20:
            # For very large candidate sets, use iterative approach
            # But in practice n is at most 24-x which is small for x>=4
            pass
        
        def dfs(idx, current_sum, current_count):
            if current_sum == target:
                return True  # violation found
            if current_sum > target:
                return False
            if idx >= n:
                return False
            for i in range(idx, n):
                new_sum = current_sum + Fraction(1, candidates[i])
                if new_sum > target:
                    break  # further candidates are even smaller reciprocals, so this won't help... wait
                    # Actually candidates sorted descending, so candidates[i] >= candidates[i+1]
                    # So 1/candidates[i] <= 1/candidates[i+1]
                    # If new_sum > target, then adding more (larger reciprocals) will only increase
                    # But we're going in order of decreasing candidate value = increasing reciprocal
                    # So if 1/candidates[i] makes sum too big, 1/candidates[i+1] is even bigger
                    # So we can break
                    return False
                if dfs(i + 1, new_sum, current_count + 1):
                    return True
            return False
        
        if dfs(0, Fraction(0), 0):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    def time_left():
        return deadline - time.time()
    
    # Seed: the top half
    best = list(range(13, 25))
    if not _check_valid(best):
        # Find a valid subset of the top half
        best = [24]
        for x in range(23, 12, -1):
            if time_left() < 1:
                break
            candidate = best + [x]
            if _check_valid(candidate):
                best = candidate
    
    # Strategy 1: Greedy from large to small (add elements in descending order)
    ordering = list(range(24, 0, -1))
    current = []
    for x in ordering:
        if time_left() < 2:
            break
        candidate = current + [x]
        if _check_valid(candidate):
            current = candidate
    if len(current) > len(best):
        best = current
    
    # Strategy 2: Greedy from small to large
    if time_left() > 5:
        current2 = []
        for x in range(1, 25):
            if time_left() < 2:
                break
            candidate = current2 + [x]
            if _check_valid(candidate):
                current2 = candidate
        if len(current2) > len(best):
            best = current2
    
    # Strategy 3: Try to add elements to best
    if time_left() > 5:
        improved = list(best)
        for x in range(1, 25):
            if time_left() < 2:
                break
            if x not in improved:
                candidate = improved + [x]
                if _check_valid(candidate):
                    improved = candidate
        if len(improved) > len(best):
            best = improved
    
    # Strategy 4: Local search - try removing one element and adding one or two others
    if time_left() > 10:
        found = True
        while found and time_left() > 5:
            found = False
            best_set = set(best)
            for i in range(len(best)):
                if time_left() < 3:
                    break
                removed = best[i]
                remaining = [x for idx, x in enumerate(best) if idx != i]
                remaining_set = set(remaining)
                # Try adding one element
                for j in range(1, 25):
                    if time_left() < 3:
                        break
                    if j in remaining_set:
                        continue
                    candidate = remaining + [j]
                    if len(candidate) > len(best) and _check_valid(candidate):
                        best = candidate
                        found = True
                        break
                if found:
                    break
                # Try adding two elements
                for j in range(1, 25):
                    if time_left() < 3:
                        break
                    if j in remaining_set:
                        continue
                    for k in range(j + 1, 25):
                        if time_left() < 3:
                            break
                        if k in remaining_set:
                            continue
                        candidate = remaining + [j, k]
                        if len(candidate) > len(best) and _check_valid(candidate):
                            best = candidate
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
    
    # Strategy 5: Branch and bound for exact search if time permits
    if time_left() > 15:
        # Try to find a set larger than best
        target_size = len(best) + 1
        elements = list(range(1, 25))
        
        def branch_and_bound(idx, current, best_size):
            if time_left() < 2:
                return current
            if len(current) + (24 - idx) <= best_size:
                return current
            if idx >= 24:
                return current
            
            # Option 1: skip elements[idx]
            result = branch_and_bound(idx + 1, current, best_size)
            if len(result) > best_size:
                best_size = len(result)
            
            # Option 2: include elements[idx]
            candidate = current + [elements[idx]]
            if _check_valid(candidate):
                result2 = branch_and_bound(idx + 1, candidate, best_size)
                if len(result2) > len(result):
                    result = result2
            
            return result
        
        # Only do this if we have enough time and haven't found size 14+
        if len(best) < 14:
            result = branch_and_bound(0, [], len(best))
            if len(result) > len(best):
                best = result
    
    return best
