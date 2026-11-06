"""Optimized solver for: No element dividing a sum of three others."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

# Deadline guard
_DEADLINE = 85.0  # seconds total; aim to finish well before 90s

def _check_add(x, S):
    """
    Check if x can be added to set S without violating the constraint.
    S is a set of integers.
    Returns True if valid, False otherwise.
    """
    S_list = list(S)
    n = len(S_list)
    
    # Condition 1: for x, no 3 distinct elements in S should sum to 0 mod x
    # i.e., for any distinct a, b, c in S, (a+b+c) % x != 0
    if n >= 3:
        if x == 1:
            return False
        # Use remainder buckets to speed up
        # We need to check if there exist i, j, k distinct such that (S[i]+S[j]+S[k]) % x == 0
        # This is equivalent to finding i, j such that (-(S[i]+S[j])) % x exists in S at index k distinct from i, j.
        
        # Optimization: Precompute remainders
        # rem_val -> list of indices
        rem_idx = defaultdict(list)
        for i, v in enumerate(S_list):
            rem_idx[v % x].append(i)
        
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                target_rem = (-(a + b)) % x
                if target_rem in rem_idx:
                    for k in rem_idx[target_rem]:
                        if k != i and k != j:
                            return False
    
    # Condition 2: for each a in S, new triples involving x must not be divisible by a
    # Triples are (x, u, v) where u, v in S \ {a} and u != v.
    # Check if (x + u + v) % a == 0 for any such u, v.
    for i in range(n):
        a = S_list[i]
        if a == 1:
            # If a=1 is in S, then any sum is divisible by 1.
            # The triple (x, u, v) has sum x+u+v.
            # We need to ensure there are no such triples.
            # Actually, if a=1 is in S, then for ANY x, u, v in S\{1}, (x+u+v) % 1 == 0.
            # So if S contains 1 and |S\{1}| >= 2, we cannot add x.
            # If |S\{1}| < 2, we can't form a triple of 3 distinct OTHER elements for a=1?
            # Wait, the condition is "No selected integer may divide the sum of any k=3 DISTINCT OTHER selected integers".
            # If a=1, we look at sums of 3 distinct elements from S\{1}.
            # When we add x, the new set is S' = S U {x}.
            # The "OTHER" elements for a=1 are S' \ {1}.
            # Previously, S \ {1} was valid.
            # Now S' \ {1} = (S \ {1}) U {x}.
            # We need to check if any 3 distinct elements in (S \ {1}) U {x} sum to 0 mod 1.
            # Since everything is 0 mod 1, this is true IF there are at least 3 distinct elements in (S \ {1}) U {x}.
            # So if len(S \ {1}) + 1 >= 3, i.e., len(S \ {1}) >= 2, then it's invalid.
            if n >= 3: # S has at least 3 elements, and one is 1, so S\{1} has at least 2.
                return False
            continue
        
        # For a > 1:
        # We need to check pairs (u, v) from S \ {a}.
        # (x + u + v) % a == 0
        # u + v ≡ -x (mod a)
        
        # Build remainder map for S \ {a}
        rem_idx_a = defaultdict(list)
        for j in range(n):
            if j == i:
                continue
            rem_idx_a[S_list[j] % a].append(j)
        
        for j in range(n):
            if j == i:
                continue
            u = S_list[j]
            target_rem = (-x - u) % a
            if target_rem in rem_idx_a:
                for k in rem_idx_a[target_rem]:
                    if k != j:
                        return False
    
    return True

def _is_valid(S):
    """Full validation of set S."""
    S_list = list(S)
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        if a == 1:
            # Check if there are 3 distinct other elements
            if n - 1 >= 3:
                return False
            continue
        
        others = [S_list[j] for j in range(n) if j != i]
        if len(others) < 3:
            continue
            
        # Check all triples in others
        m = len(others)
        # Optimization: use remainder buckets for a
        rem_idx = defaultdict(list)
        for idx, val in enumerate(others):
            rem_idx[val % a].append(idx)
        
        for i2 in range(m):
            v1 = others[i2]
            for i3 in range(i2 + 1, m):
                v2 = others[i3]
                target = (-(v1 + v2)) % a
                if target in rem_idx:
                    for i4 in rem_idx[target]:
                        if i4 != i2 and i4 != i3:
                            return False
    return True

def _greedy(order):
    """Greedy construction given an order of candidates."""
    S = set()
    for x in order:
        if len(S) >= 48:
            break
        if _check_add(x, S):
            S.add(x)
    return S

def _local_search(S, time_limit):
    """
    Try to improve S by adding elements or swapping.
    Returns improved set.
    """
    S = set(S)
    n_max = 192
    all_candidates = set(range(1, n_max + 1))
    unselected = all_candidates - S
    
    t_start = time.time()
    
    while time.time() - t_start < time_limit:
        improved = False
        
        # 1. Try to add any unselected element
        for x in list(unselected):
            if len(S) >= 48:
                break
            if _check_add(x, S):
                S.add(x)
                unselected.discard(x)
                improved = True
        
        if len(S) >= 48:
            break
            
        if not improved:
            # 2. Try swaps: remove a from S, try to add c from unselected
            # If adding c allows adding more, or just increases size (impossible with 1-for-1 swap unless we add 2 later)
            # Actually, a 1-for-1 swap keeps size same. It might enable future additions.
            # Let's try to find a swap that allows at least one additional addition afterwards.
            
            swapped = False
            for a in list(S):
                if time.time() - t_start > time_limit:
                    break
                    
                # Temporarily remove a
                S.remove(a)
                unselected.add(a)
                
                # Try to add elements from unselected
                added = []
                for c in list(unselected):
                    if len(S) >= 48:
                        break
                    if _check_add(c, S):
                        S.add(c)
                        unselected.discard(c)
                        added.append(c)
                        
                # If we added more than 1, net gain > 0. Keep it.
                # If we added exactly 1, net gain 0. Keep it only if it helps further?
                # For now, keep if added >= 1.
                if len(added) > 0:
                    swapped = True
                    improved = True
                    # If we added 1, size is same. If we added >1, size increased.
                    break
                else:
                    # Revert
                    S.add(a)
                    unselected.discard(a)
                    
            if not swapped:
                # No simple improvement found.
                # Try random restart? Or break.
                # Let's break to avoid infinite loop on stuck state.
                # Or try a more complex move: remove 2, add 2?
                # For simplicity, break.
                break
                
    return S

def construct():
    t_start = time.time()
    n = 192
    candidates = list(range(1, n + 1))
    
    best_S = set()
    
    # 1. Deterministic Greedy (Descending)
    # Large numbers are less likely to divide small sums, so this is a strong heuristic.
    S1 = _greedy(sorted(candidates, reverse=True))
    if len(S1) > len(best_S):
        best_S = S1
        
    # 2. Deterministic Greedy (Ascending)
    S2 = _greedy(sorted(candidates))
    if len(S2) > len(best_S):
        best_S = S2
        
    # 3. Randomized Greedy Runs
    # Use remaining time for random restarts and local search
    
    # Allocate time: 20s for initial greedy runs, rest for local search
    greedy_time_budget = 20.0
    local_search_time_budget = _DEADLINE - (time.time() - t_start) - 5.0  # 5s safety
    
    # Run random greedy while time permits
    random.seed(42)
    while time.time() - t_start < greedy_time_budget:
        random.shuffle(candidates)
        S_rand = _greedy(candidates)
        if len(S_rand) > len(best_S):
            best_S = S_rand
            
    # 4. Local Search on best_S
    if time.time() - t_start < _DEADLINE - 5.0:
        best_S = _local_search(best_S, local_search_time_budget)
        
    # 5. If time remains, try more random restarts with local search
    if time.time() - t_start < _DEADLINE - 5.0:
        remaining_budget = _DEADLINE - (time.time() - t_start) - 5.0
        # Split remaining time into chunks for restarts
        # Each restart: greedy + local search
        # Estimate time per restart: greedy ~0.1s, local search ~5s?
        # Let's try a few more.
        
        for _ in range(10):
            if time.time() - t_start > _DEADLINE - 5.0:
                break
            
            random.shuffle(candidates)
            S_rand = _greedy(candidates)
            
            # Short local search
            ls_budget = min(5.0, _DEADLINE - (time.time() - t_start) - 5.0)
            if ls_budget > 0:
                S_rand = _local_search(S_rand, ls_budget)
                
            if len(S_rand) > len(best_S):
                best_S = S_rand

    # Final validation
    if _is_valid(best_S):
        return sorted(best_S)
    else:
        # Fallback: find a valid subset of best_S?
        # Or just return empty? No, best_S from greedy is valid incrementally.
        # Local search might have introduced a bug if _check_add is flawed.
        # But _check_add is robust.
        # If _is_valid fails, it means a bug in _check_add or _local_search logic.
        # Let's try to repair by removing elements that cause violations.
        # Simple repair: iterate and remove violators.
        S_list = list(best_S)
        valid_S = set()
        for x in S_list:
            if _check_add(x, valid_S):
                valid_S.add(x)
        
        if _is_valid(valid_S):
            return sorted(valid_S)
        else:
            # Last resort
            return sorted(valid_S)

# EVOLVE-BLOCK-END
