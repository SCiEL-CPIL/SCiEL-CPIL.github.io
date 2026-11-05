"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def _precompute_conflicts():
    """
    Precompute all conflict triples (i, S) where 1/i = sum(1/j for j in S).
    S is a frozenset of distinct elements from [1,24], i not in S.
    Returns a list of (i, frozenset(S)).
    """
    conflicts = []
    universe = list(range(1, 25))
    
    for i in universe:
        # 1/i must equal sum of reciprocals of distinct elements from [1,24]\{i}
        # The sum must be exactly 1/i.
        # We can iterate over subsets of the other elements.
        # However, 2^23 is too large. We need a smarter way or just check relevant subsets.
        # Note that if 1/i = sum 1/j_k, then min(j_k) > i is not required, but typically j_k are larger?
        # Actually, if any j < i, 1/j > 1/i, so the sum would exceed 1/i if only one term.
        # If multiple terms, it's possible.
        # But wait, if j < i, then 1/j > 1/i.
        # If S has one element j < i, sum = 1/j > 1/i. No match.
        # If S has elements all > i, then 1/j < 1/i.
        # It is possible to have a mix.
        
        # Let's use meet-in-the-middle or just iterate subsets of size up to a limit?
        # The maximum size of S: 1/i = 1/a + 1/b + ...
        # Smallest reciprocal is 1/24.
        # Max terms: i / 24. For i=1, max terms ~ 24.
        # But distinct elements.
        
        # Efficient check:
        # Iterate over subsets of [1,24]\{i}.
        # To avoid 2^23, we can prune.
        # If we build a subset S, and sum(1/j) > 1/i, we can prune?
        # No, because we are adding positive terms.
        
        # Let's just iterate all subsets of [1,24]\{i} that could potentially sum to 1/i.
        # Since 1/i is small for large i, the subsets must consist of large numbers.
        # For i=1, 1 = sum 1/j. Egyptian fractions.
        
        # Given the small universe (24), let's just do a DFS with pruning.
        others = [x for x in universe if x != i]
        target = 1.0 / i
        
        def dfs(start_idx, current_sum, current_set):
            if abs(current_sum - target) < 1e-12:
                if current_set:
                    conflicts.append((i, frozenset(current_set)))
                return
            if current_sum > target:
                return
            if start_idx >= len(others):
                return
            
            for k in range(start_idx, len(others)):
                # Pruning: if current_sum + 1/others[k] > target, and since others are sorted ascending?
                # If others are sorted ascending, 1/others[k] is descending.
                # Let's sort others descending to prune faster?
                # If we pick a large number, sum increases a lot.
                # Let's sort others ascending.
                # If current_sum + 1/others[k] > target, then for any j > k, 1/others[j] < 1/others[k], so it might still fit?
                # No, if we are adding, and we exceeded, we should stop?
                # If we sort ascending, 1/others[0] is largest.
                # If current_sum + 1/others[k] > target, then current_sum + 1/others[k+1] might be <= target.
                # So we can't break immediately.
                
                # Better pruning:
                # If current_sum + sum(1/others[j] for j in range(k, len(others))) < target, then we can't reach target.
                # Precompute suffix sums of reciprocals.
                
                val = 1.0 / others[k]
                new_sum = current_sum + val
                if new_sum > target + 1e-12:
                    continue
                dfs(k + 1, new_sum, current_set + [others[k]])
                
        # To make pruning effective, sort others in descending order of value (ascending index)?
        # Actually, standard subset sum DFS works best if we can bound the remaining sum.
        # Let's just run it. 24 elements is small enough for a simple DFS with basic pruning.
        # But 2^23 is 8 million. For each i, that's 24 * 8M = 200M operations. Might be too slow in Python.
        
        # Alternative: Meet in the middle for each i?
        # Or just hardcode the known conflicts?
        # The task is [1,24].
        # Known small conflicts:
        # 1/1 = 1/2 + 1/3 + 1/6
        # 1/1 = 1/2 + 1/4 + 1/6 + 1/12
        # 1/1 = 1/2 + 1/3 + 1/7 + 1/42 (42 not in range)
        # 1/2 = 1/3 + 1/7 + 1/42 (no)
        # 1/2 = 1/4 + 1/5 + 1/20 (20 in range) -> 1/2 = 1/4+1/5+1/20. Yes.
        # 1/2 = 1/4 + 1/6 + 1/12. Yes.
        # 1/2 = 1/3 + 1/8 + 1/24. Yes.
        # 1/3 = 1/4 + 1/13 + 1/156 (no)
        # 1/3 = 1/4 + 1/5 + 1/20 + 1/60 (no)
        # 1/3 = 1/4 + 1/12 + 1/6? No 1/4+1/12+1/6 = 3/12+1/12+2/12 = 6/12 = 1/2.
        # 1/3 = 1/4 + 1/18 + 1/6? 1/4+1/6 = 5/12. 1/3 = 4/12. No.
        # 1/3 = 1/7 + 1/8 + 1/56 (no)
        
        # Let's optimize the DFS.
        # Sort others in descending order (largest numbers first, smallest reciprocals).
        # This allows us to prune if the sum of the *remaining* smallest reciprocals isn't enough?
        # Actually, if we sort ascending (1, 2, ...), the first few are large reciprocals.
        # If we include 1, sum >= 1.
        # If target < 1, we can't include 1.
        
        # Let's use integer arithmetic to avoid float errors and speed up.
        # L = lcm(i, others).
        # Target = L/i.
        # Values = [L/j for j in others].
        # Find subset of Values summing to Target.
        
        pass # Placeholder for actual implementation below

    # Implementation of _precompute_conflicts with integer arithmetic and MITM
    conflicts = []
    universe = list(range(1, 25))
    
    for i in universe:
        others = [x for x in universe if x != i]
        n = len(others)
        
        # Calculate LCM of i and all others
        L = i
        for x in others:
            L = _lcm(L, x)
            
        target_int = L // i
        vals = [L // x for x in others]
        
        # Meet in the middle
        mid = n // 2
        left_vals = vals[:mid]
        right_vals = vals[mid:]
        
        # Generate subset sums for left
        left_sums = {0}
        for v in left_vals:
            new_sums = set()
            for s in left_sums:
                ns = s + v
                if ns < target_int:
                    new_sums.add(ns)
                elif ns == target_int:
                    # Found a subset in left that sums to target
                    # We need to map back to original indices
                    # This is tricky with just sums. We need to store the subset.
                    # For conflict detection, we just need to know IF a conflict exists.
                    # But to build the hypergraph, we need the specific subset S.
                    # However, for the purpose of finding a maximal independent set,
                    # we can just check validity on the fly or precompute a boolean matrix?
                    # The conflict is: if S is selected, and i is selected, and S is the specific subset, it's invalid.
                    # Actually, the condition is: "no reciprocal of a selected element may equal the sum of reciprocals of a subset of the OTHER selected elements".
                    # This means if i is in the set, and S is a subset of the OTHER elements, and 1/i = sum(1/j for j in S), then the set is invalid.
                    # So for each i, we need to know ALL subsets S of [1,24]\{i} such that 1/i = sum(1/j).
                    # Then, if we select i, we cannot select ALL elements of any such S.
                    # This is a constraint: NOT (i in Set AND S subset of Set\{i}).
                    # Equivalently: If i in Set, then for every conflict S_i, S_i is NOT a subset of Set\{i}.
                    
                    # Storing all S might be expensive if there are many.
                    # But for small N, it's manageable.
                    pass
            left_sums.update(new_sums)
            
        # This approach is getting complicated to retrieve the actual subsets S.
        # Given the time limit and the small N (24), maybe we can just use a faster validity check
        # and a smarter search?
        
        # Let's revert to a simpler but faster validity check and a better search strategy.
        # The current program scores 1.5833.
        # The inspiration suggests that greedy increasing fails because of retroactive conflicts.
        # Greedy decreasing might be better?
        # Or, we can use a local search that specifically targets resolving conflicts.
        
        pass

    return conflicts

def _is_valid_fast(subset):
    """
    Fast validity check.
    For each x in subset, check if 1/x is sum of reciprocals of any subset of subset\{x}.
    We can precompute the subset sums of reciprocals for the whole set?
    No, because the subset must exclude x.
    
    However, we can use the property that if 1/x = sum(1/j), then the sum of reciprocals of the whole set
    minus 1/x must be representable? No.
    
    Let's stick to the integer LCM method but optimize it.
    """
    s = list(subset)
    n = len(s)
    if n == 0:
        return True
    
    # Sort to ensure determinism
    s.sort()
    
    # Precompute LCM of all elements in s
    L = s[0]
    for x in s[1:]:
        L = _lcm(L, x)
        
    # Convert to integers
    # target for x is L // x
    # values for y are L // y
    
    # We need to check for each x if L//x is in the subset sums of {L//y for y in s, y != x}.
    
    # We can generate all subset sums of the integer values once?
    # There are 2^n sums. n=24 -> 16 million. Too many.
    
    # Meet in the middle for each x?
    # Or, just check the specific conflicts we know about?
    # Since the universe is small, let's precompute the list of all valid conflicts (i, S)
    # using a brute force search that is run once at import time or first call.
    
    return True # Placeholder

# Let's implement the precomputation properly.
_CONFLICTS = None

def _get_conflicts():
    global _CONFLICTS
    if _CONFLICTS is not None:
        return _CONFLICTS
    
    _CONFLICTS = []
    universe = list(range(1, 25))
    
    for i in universe:
        others = [x for x in universe if x != i]
        n = len(others)
        
        L = i
        for x in others:
            L = _lcm(L, x)
            
        target_int = L // i
        vals = [L // x for x in others]
        
        # Find all subsets of vals that sum to target_int
        # Use MITM to find existence, but we need the subsets themselves to store as conflicts.
        # Actually, we can just store the conflict as a bitmask of the 'others' indices?
        # Or better, store the actual elements.
        
        # Since we only need to check if a candidate set is valid,
        # we can check: for each i in candidate, does there exist a conflict S_i such that S_i is subset of candidate\{i}?
        
        # Let's find all such S_i.
        # n=23. 2^23 is too big for full enumeration.
        # But we can prune.
        
        # Sort vals and others together
        pairs = sorted(zip(vals, others))
        
        def find_subsets(idx, current_sum, current_set):
            if current_sum == target_int:
                if current_set:
                    _CONFLICTS.append((i, frozenset(current_set)))
                return
            if current_sum > target_int:
                return
            if idx == len(pairs):
                return
            
            # Pruning: if current_sum + sum(vals[idx:]) < target_int, return
            # Precompute suffix sums
            # For now, simple recursion
            
            # Option 1: Skip pairs[idx]
            find_subsets(idx + 1, current_sum, current_set)
            
            # Option 2: Take pairs[idx]
            v, x = pairs[idx]
            find_subsets(idx + 1, current_sum + v, current_set | {x})
            
        # This will be slow.
        # Let's use a different approach:
        # Since the values are integers, and target is fixed,
        # we can use a dictionary to store achievable sums and the subset?
        # No, too many states.
        
        # Given the constraints, maybe we can just hardcode the few conflicts that exist in [1,24].
        # Let's identify them manually or with a quick script logic here.
        
        pass
        
    # Fallback: just return a dummy list to avoid errors, and rely on a slower but correct checker in construct.
    _CONFLICTS = []
    return _CONFLICTS

def _is_safe_int(candidate, others):
    """
    Check if 1/candidate is sum of reciprocals of any subset of others.
    Uses integer arithmetic.
    """
    if not others:
        return True
    
    L = candidate
    for y in others:
        L = _lcm(L, y)
        
    target_int = L // candidate
    vals = [L // y for y in others]
    
    n = len(vals)
    if n == 0:
        return True
        
    # MITM
    mid = n // 2
    left_vals = vals[:mid]
    right_vals = vals[mid:]
    
    left_sums = {0}
    for v in left_vals:
        new_sums = set()
        for s in left_sums:
            ns = s + v
            if ns < target_int:
                new_sums.add(ns)
            elif ns == target_int:
                return False
        left_sums.update(new_sums)
        
    right_sums = {0}
    for v in right_vals:
        new_sums = set()
        for s in right_sums:
            ns = s + v
            if ns < target_int:
                new_sums.add(ns)
            elif ns == target_int:
                return False
        right_sums.update(new_sums)
        
    for rs in right_sums:
        if rs > target_int:
            continue
        remainder = target_int - rs
        if remainder in left_sums:
            if rs == 0 and remainder == 0:
                continue
            return False
            
    return True

def _is_valid_full(subset):
    s = set(subset)
    for x in s:
        others = list(s - {x})
        if not _is_safe_int(x, others):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_set = set()
    best_size = 0
    
    # Strategy 1: Greedy Decreasing
    # Hypothesis: Adding large numbers first is safer because their reciprocals are small,
    # making it harder for them to be formed by sums of other reciprocals?
    # Actually, the risk is that 1/large = sum(1/other).
    # If 'other' are also large, their reciprocals are small.
    # If we add 24, then 23, etc.
    # When adding 24, others is empty. Safe.
    # When adding 23, others={24}. 1/23 != 1/24. Safe.
    # When adding 22, others={23,24}. 1/22 != 1/23, 1/24, 1/23+1/24.
    # 1/22 ~ 0.045. 1/23+1/24 ~ 0.085.
    # It seems decreasing greedy might work well.
    
    selected = set()
    for i in range(24, 0, -1):
        if time.time() > deadline:
            break
        candidate_set = selected | {i}
        if _is_valid_full(candidate_set):
            selected = candidate_set
            
    if len(selected) > best_size:
        best_set = set(selected)
        best_size = len(selected)
        
    # Strategy 2: Greedy Increasing
    selected = set()
    for i in range(1, 25):
        if time.time() > deadline:
            break
        candidate_set = selected | {i}
        if _is_valid_full(candidate_set):
            selected = candidate_set
            
    if len(selected) > best_size:
        best_set = set(selected)
        best_size = len(selected)
        
    # Strategy 3: Local Search / Hill Climbing
    # Start with best found so far.
    # Try to add any excluded element.
    # If adding fails, try to remove one element from the conflict and add the new one.
    
    current = set(best_set)
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        excluded = sorted(set(range(1, 25)) - current, reverse=True)
        
        for x in excluded:
            if time.time() > deadline - 10:
                break
            
            # Try to add x
            new_set = current | {x}
            if _is_valid_full(new_set):
                current = new_set
                improved = True
                if len(current) > best_size:
                    best_set = set(current)
                    best_size = len(current)
                # Restart loop with new excluded
                break
        
        if not improved:
            # Try swaps: remove y, add x
            current_list = sorted(current)
            for y in current_list:
                if time.time() > deadline - 5:
                    break
                for x in excluded:
                    if x in current:
                        continue
                    new_set = current - {y} | {x}
                    if len(new_set) <= best_size:
                        continue
                    if _is_valid_full(new_set):
                        # Accept swap? Only if it leads to further improvement?
                        # For now, just accept if it's valid and same size, hoping to unblock?
                        # No, we want to maximize size.
                        # If we swap, size is same.
                        # We should only swap if it allows adding more later.
                        # This is complex. Let's stick to additions for now.
                        pass
                        
    # Strategy 4: Randomized Restart
    for _ in range(10):
        if time.time() > deadline - 5:
            break
        
        # Random subset of size 12-15
        import random
        random.seed(int(time.time() * 1000) % 10000)
        init_size = random.randint(12, 15)
        current = set(random.sample(range(1, 25), init_size))
        
        # Refine
        # Remove invalid elements
        changed = True
        while changed:
            changed = False
            for x in list(current):
                if not _is_safe_int(x, list(current - {x})):
                    current.remove(x)
                    changed = True
        
        # Add
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            excluded = sorted(set(range(1, 25)) - current, reverse=True)
            for x in excluded:
                new_set = current | {x}
                if _is_valid_full(new_set):
                    current = new_set
                    improved = True
                    break
        
        if len(current) > best_size:
            best_set = set(current)
            best_size = len(current)
            
    if not best_set:
        best_set = {24}
        
    return sorted(best_set)
# EVOLVE-BLOCK-END
