# EVOLVE-BLOCK-START
import time
import random
import numpy as np

# Precompute conflict structure: for each value v in [1,512],
# conflicts[v] is a set of values that conflict with v (i.e., some element
# could be the mean of a subset containing both v and that other value,
# or v is the mean of a subset containing that other value).
#
# Actually, the constraint is: no element a in the set is the arithmetic mean
# of any subset of >=2 OTHER elements in the set.
#
# Two elements x, y conflict if there exists a third element a (possibly equal
# to x or y? No, a must be distinct from both) such that:
#   - a is the mean of {x, y}: 2a = x + y, so a = (x+y)/2
#   - x is the mean of {a, y}: 2x = a + y, so a = 2x - y
#   - y is the mean of {a, x}: 2y = a + x, so a = 2y - x
#
# But we also need to consider larger subsets. For now, let's focus on the
# pairwise conflicts (subset size 2) which are the most restrictive, and
# handle larger subsets in the validation step.
#
# For the hypergraph approach: two vertices x, y are adjacent (conflict) if
# adding both to any independent set would create a violation. The simplest
# necessary condition is that no third vertex a exists such that a is the
# mean of {x,y}, or x is the mean of {a,y}, or y is the mean of {a,x}.
#
# But actually for the independent set, we need: for any subset S of the
# chosen set, and any element a in S, a is NOT the mean of any subset of
# S\{a} with >=2 elements.
#
# Let's precompute for each pair (x,y) whether they "conflict" in the sense
# that there exists some a in [1,512] such that {x,y,a} would be invalid.
# This is a necessary condition for the pair to be in a valid set.

def _precompute_conflicts():
    """Precompute pairwise conflicts: conflicts[x] is a set of y where
    {x, y} cannot both be in a valid set (because some a would make one
    the mean of a subset containing the other)."""
    N = 512
    conflicts = [set() for _ in range(N + 1)]
    
    for x in range(1, N + 1):
        for y in range(x + 1, N + 1):
            # Check if there exists a in [1, N] such that:
            # a is the mean of {x, y}: a = (x+y)/2 must be integer, a != x, a != y
            # x is the mean of {a, y}: a = 2x - y must be in [1,N], a != x, a != y
            # y is the mean of {a, x}: a = 2y - x must be in [1,N], a != x, a != y
            
            # Case 1: a = (x+y)/2
            if (x + y) % 2 == 0:
                a = (x + y) // 2
                if 1 <= a <= N and a != x and a != y:
                    conflicts[x].add(y)
                    conflicts[y].add(x)
                    continue
            
            # Case 2: a = 2x - y (x is mean of {a, y})
            a = 2 * x - y
            if 1 <= a <= N and a != x and a != y:
                conflicts[x].add(y)
                conflicts[y].add(x)
                continue
            
            # Case 3: a = 2y - x (y is mean of {a, x})
            a = 2 * y - x
            if 1 <= a <= N and a != x and a != y:
                conflicts[x].add(y)
                conflicts[y].add(x)
                continue
    
    return conflicts

def _validate_set(full_set):
    """Full validation: check that no element is the mean of any subset of
    >=2 other elements in full_set."""
    n = len(full_set)
    if n < 3:
        return True
    
    set_vals = set(full_set)
    
    for a in full_set:
        others = [x for x in full_set if x != a]
        m = len(others)
        if m < 2:
            continue
        
        # DP: dp[k] is a bitmask where bit s is set if sum s is achievable
        # with exactly k elements from others
        max_sum = sum(others)
        # We only care about sums up to m*a (since k*a <= m*a)
        max_target = m * a
        if max_target > max_sum:
            max_target = max_sum
        
        dp = [0] * (m + 1)
        dp[0] = 1  # sum 0 with 0 elements
        
        for x in others:
            for k in range(m, 0, -1):
                dp[k] |= (dp[k-1] << x)
            # Mask to keep integers manageable
            if max_target < 200000:
                mask = (1 << (max_target + 1)) - 1
                for k in range(m + 1):
                    dp[k] &= mask
        
        # Check: for k >= 2, is k*a achievable?
        for k in range(2, m + 1):
            target = k * a
            if target <= max_target:
                if (dp[k] >> target) & 1:
                    return False
    
    return True

def _check_addition_hyper(current_set, x, conflicts):
    """Check if adding x to current_set is safe using precomputed conflicts
    plus a quick subset-sum check for larger subsets."""
    # First, check pairwise conflicts
    for y in current_set:
        if y in conflicts[x]:
            return False
    
    # Quick 3-AP check (subset size 2): is x the mean of any pair?
    set_curr = set(current_set)
    for a in current_set:
        b = 2 * x - a
        if b in set_curr and b != a:
            return False
    
    # Is any a in current_set the mean of {x, b} for some b in current_set?
    for a in current_set:
        b = 2 * a - x
        if b in set_curr and b != a and b != x:
            return False
    
    # For larger subsets (size >= 3), do a DP check
    n = len(current_set)
    if n >= 2:
        # Check if x is the mean of any subset of >=2 elements from current_set
        m = n
        max_target = m * x
        if max_target <= 200000:
            dp = [0] * (m + 1)
            dp[0] = 1
            for elem in current_set:
                for k in range(m, 0, -1):
                    dp[k] |= (dp[k-1] << elem)
                if max_target < 200000:
                    mask = (1 << (max_target + 1)) - 1
                    for k2 in range(m + 1):
                        dp[k2] &= mask
            for k in range(2, m + 1):
                target = k * x
                if target <= max_target and (dp[k] >> target) & 1:
                    return False
        
        # Check if any a in current_set is the mean of a subset including x
        for a in current_set:
            others_no_a = [y for y in current_set if y != a]
            m2 = len(others_no_a)
            if m2 < 1:
                continue
            # Need subset of {x} U others_no_a with >=2 elements summing to k*a
            # This means: x + sum(subset of others_no_a) = k*a for k >= 2
            # Or: sum(subset of others_no_a) = k*a - x for k >= 2, and subset has >= 1 elem
            # Or: sum(subset of others_no_a) = k*a for k >= 2 (subset doesn't include x), 
            #   but that's already checked since a was in current_set and others_no_a are
            #   all in current_set, so this was already validated when current_set was built.
            # Actually we only need to check subsets that INCLUDE x.
            
            max_t = a * (m2 + 1) - x
            if max_t < 0 or max_t > 200000:
                continue
            
            dp2 = [0] * (m2 + 1)
            dp2[0] = 1
            for elem in others_no_a:
                for k in range(m2, 0, -1):
                    dp2[k] |= (dp2[k-1] << elem)
                if max_t < 200000:
                    mask = (1 << (max_t + 1)) - 1
                    for k2 in range(m2 + 1):
                        dp2[k2] &= mask
            
            for t in range(1, m2 + 1):
                target = a * (t + 1) - x
                if 0 <= target <= max_t:
                    if (dp2[t] >> target) & 1:
                        return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 87.0  # 3s safety margin
    
    # Precompute conflicts
    conflicts = _precompute_conflicts()
    
    best_set = []
    
    # Strategy 1: Greedy from low to high using hypergraph conflicts
    selected = []
    for x in range(1, 513):
        if time.time() > deadline:
            break
        if len(selected) >= 48:
            break
        if _check_addition_hyper(selected, x, conflicts):
            selected.append(x)
    
    if len(selected) > len(best_set):
        best_set = selected[:]
    
    # Strategy 2: Greedy from high to low
    selected = []
    for x in range(512, 0, -1):
        if time.time() > deadline:
            break
        if len(selected) >= 48:
            break
        if _check_addition_hyper(selected, x, conflicts):
            selected.append(x)
    
    if len(selected) > len(best_set):
        best_set = selected[:]
    
    # Strategy 3: Randomized greedy with multiple seeds
    for seed in range(200):
        if time.time() > deadline:
            break
        rng = random.Random(seed)
        order = list(range(1, 513))
        rng.shuffle(order)
        selected = []
        for x in order:
            if time.time() > deadline:
                break
            if len(selected) >= 48:
                break
            if _check_addition_hyper(selected, x, conflicts):
                selected.append(x)
        if len(selected) > len(best_set):
            best_set = selected[:]
    
    # Strategy 4: Local search - try to add more elements
    if time.time() < deadline - 5:
        current = best_set[:]
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for x in range(1, 513):
                if time.time() > deadline:
                    break
                if x in current:
                    continue
                if len(current) >= 48:
                    break
                if _check_addition_hyper(current, x, conflicts):
                    current.append(x)
                    improved = True
            if len(current) > len(best_set):
                best_set = current[:]
    
    # Strategy 5: Swap-based local search
    if time.time() < deadline - 3:
        current = best_set[:]
        for _ in range(50):
            if time.time() > deadline:
                break
            improved = False
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                for x in range(1, 513):
                    if x in current:
                        continue
                    candidate = current[:i] + [x] + current[i+1:]
                    # Quick conflict check
                    ok = True
                    for j, y in enumerate(candidate):
                        if j != i and y in conflicts[x]:
                            ok = False
                            break
                    if not ok:
                        continue
                    # Try to extend
                    new_size = len(candidate)
                    for y in range(1, 513):
                        if y in candidate:
                            continue
                        if new_size >= 48:
                            break
                        if _check_addition_hyper(candidate, y, conflicts):
                            candidate.append(y)
                            new_size += 1
                            improved = True
                    if len(candidate) > len(current):
                        current = candidate
                        if len(current) > len(best_set):
                            best_set = current[:]
                        break
                if improved:
                    break
            if not improved:
                break
    
    # Final validation
    if best_set and not _validate_set(best_set):
        # Fallback: re-validate and remove problematic elements
        validated = []
        for x in best_set:
            if _check_addition_hyper(validated, x, conflicts):
                validated.append(x)
        if len(validated) > len(best_set):
            best_set = validated
    
    return best_set[:48]
# EVOLVE-BLOCK-END
