"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from itertools import combinations

P = {'n': 256}
_DEADLINE = None

def _is_prime(n):
    if n < 2:
        return False
    for d in range(2, isqrt(n) + 1):
        if n % d == 0:
            return False
    return True

def _valid_set(S):
    """Check if set S is valid: for all distinct a,b,c in S, a does not divide b*c."""
    S_list = list(S)
    m = len(S_list)
    for i in range(m):
        a = S_list[i]
        for j in range(i+1, m):
            b = S_list[j]
            for k in range(j+1, m):
                c = S_list[k]
                if (b * c) % a == 0:
                    return False
    return True

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 85  # 85s budget, 5s safety margin
    
    n = P['n']
    nums = list(range(2, n + 1))
    N = len(nums)
    
    # Map values to indices for bitmask operations
    val_to_idx = {v: i for i, v in enumerate(nums)}
    
    # Precompute conflict bitmasks.
    # conflict_mask[x] is a bitmask of indices y such that there EXISTS a z 
    # where {x, y, z} is a bad triple? No, that's not quite right for incremental addition.
    
    # Let's define:
    # A set S is valid if for all distinct a,b,c in S, a !| b*c.
    # When adding x to S, we need to ensure:
    # 1. For all distinct a,b in S: x !| a*b
    # 2. For all distinct a,b in S: a !| x*b
    # 3. For all distinct a,b in S: b !| x*a
    
    # We can precompute for each x, the set of pairs (a,b) in the full universe [2..256]
    # that would make {x,a,b} invalid.
    # However, storing pairs is heavy.
    
    # Alternative: Precompute for each x, a bitmask `bad_pairs_mask[x]` is not directly possible 
    # because the condition depends on the specific pair (a,b).
    
    # Let's use a different representation.
    # For each x, let `incompatible_with[x]` be a set of numbers y such that 
    # adding y to a set containing x creates a risk? No, the risk is triple-based.
    
    # Let's stick to a fast check function but optimize it with precomputed products mod?
    # No, the previous program's bottleneck was likely the O(n^2) loop in _can_add.
    
    # New Strategy: 
    # 1. Identify "safe" numbers (primes, large numbers) that rarely cause conflicts.
    # 2. Use a backtracking search that picks the next number to add.
    #    To speed up, we precompute for each candidate x, a list of "forbidden pairs" 
    #    (a,b) such that x|a*b or a|x*b or b|x*a.
    #    If the current set S contains any such pair, x cannot be added.
    
    # Precompute forbidden pairs for each x
    # forbidden[x] = set of (i, j) indices such that nums[i] and nums[j] are distinct 
    # and {x, nums[i], nums[j]} is invalid.
    
    forbidden = [set() for _ in range(N)]
    
    for x_idx, x in enumerate(nums):
        for i in range(N):
            if i == x_idx: continue
            a = nums[i]
            for j in range(i+1, N):
                if j == x_idx: continue
                b = nums[j]
                # Check if {x, a, b} is invalid
                # x | a*b OR a | x*b OR b | x*a
                if (a * b) % x == 0 or (x * b) % a == 0 or (x * a) % b == 0:
                    forbidden[x_idx].add((i, j))
    
    # Now, for a current set S (represented as a set of indices), 
    # x can be added if S does not contain any pair from forbidden[x_idx].
    
    def can_add(S_indices, x_idx):
        # S_indices is a set of indices
        # Check if any pair in forbidden[x_idx] is a subset of S_indices
        for (i, j) in forbidden[x_idx]:
            if i in S_indices and j in S_indices:
                return False
        return True
    
    # Heuristic: Order numbers by "danger" (number of forbidden pairs)
    # Numbers with many forbidden pairs are hard to place, so decide them early?
    # Or numbers with few forbidden pairs are "safe" and can be included more freely.
    # Let's sort by number of forbidden pairs ascending (safe first) for greedy, 
    # but for backtracking, we might want to branch on "constrained" variables first.
    
    # Let's try a greedy approach first with this fast check, then local search.
    
    def greedy_search(order):
        S = set()
        for idx in order:
            if can_add(S, idx):
                S.add(idx)
        return S
    
    # Orders
    # 1. Primes first (very safe)
    primes_idx = [i for i, v in enumerate(nums) if _is_prime(v)]
    others_idx = [i for i in range(N) if i not in primes_idx]
    
    # Sort others by number of forbidden pairs (ascending = safer first)
    others_sorted_safe = sorted(others_idx, key=lambda i: len(forbidden[i]))
    others_sorted_danger = sorted(others_idx, key=lambda i: -len(forbidden[i]))
    
    orders = [
        primes_idx + others_sorted_safe,
        primes_idx + others_sorted_danger,
        sorted(range(N), key=lambda i: len(forbidden[i])),
        sorted(range(N), key=lambda i: -len(forbidden[i])),
        list(range(N)), # Natural
    ]
    
    best_set = set()
    best_size = 0
    
    for order in orders:
        S = greedy_search(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Local Search: 1-for-1 swaps and 1-for-2 swaps
    def local_search(S, max_iter=50):
        S = set(S)
        improved = True
        iters = 0
        while improved and iters < max_iter and time.time() < _DEADLINE:
            improved = False
            iters += 1
            S_list = list(S)
            for rem in S_list:
                if time.time() > _DEADLINE:
                    return S
                S2 = S - {rem}
                # Try to add any single number not in S2
                best_add = None
                best_score = len(S2)
                for add_idx in range(N):
                    if add_idx in S2:
                        continue
                    if can_add(S2, add_idx):
                        # Score: just size, but maybe prefer "safer" numbers?
                        # For now, just take the first one or one that allows more future adds?
                        # Let's just take the first valid one to keep it simple, 
                        # or try all and pick the one that maximizes future potential?
                        # Simple: pick the one with smallest index? No, pick any.
                        # Let's try to add the one that is "safest" (fewest forbidden pairs)
                        if best_add is None or len(forbidden[add_idx]) < len(forbidden[best_add]):
                            best_add = add_idx
                
                if best_add is not None:
                    S2.add(best_add)
                    if len(S2) > len(S):
                        S = S2
                        improved = True
                        break
                    elif len(S2) == len(S):
                        # Tie-breaking: prefer the new set if it's "safer"?
                        # Skip for now
                        pass
        
        return S
    
    best_set = local_search(best_set)
    best_size = len(best_set)
    
    # Further improvement: Try to add 2 numbers at a time?
    # Or remove 1, add 2?
    # Given time constraints, let's do a simple hill climb with 1-for-2 if possible.
    
    def local_search_1_for_2(S, max_iter=20):
        S = set(S)
        improved = True
        iters = 0
        while improved and iters < max_iter and time.time() < _DEADLINE:
            improved = False
            iters += 1
            S_list = list(S)
            for rem in S_list:
                if time.time() > _DEADLINE:
                    return S
                S2 = S - {rem}
                # Try to add two numbers
                found = False
                # To speed up, we can iterate through candidates
                cands = [i for i in range(N) if i not in S2]
                # Sort candidates by safety
                cands.sort(key=lambda i: len(forbidden[i]))
                
                for i1 in cands:
                    if time.time() > _DEADLINE:
                        return S
                    if not can_add(S2, i1):
                        continue
                    S3 = S2 | {i1}
                    for i2 in cands:
                        if i2 == i1:
                            continue
                        if i2 in S3:
                            continue
                        if can_add(S3, i2):
                            # Success!
                            S = S3 | {i2}
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
        
        return S
    
    best_set = local_search_1_for_2(best_set)
    best_size = len(best_set)
    
    # Final verification
    S_list = sorted([nums[i] for i in best_set])
    if not _valid_set(S_list):
        # Fallback to primes
        S_list = sorted([nums[i] for i in primes_idx])
        
    return S_list

# EVOLVE-BLOCK-END
