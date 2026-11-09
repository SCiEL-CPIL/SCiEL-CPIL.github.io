"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

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

def _can_add(S, x):
    """Check if x can be added to set S without violating the constraint."""
    S_list = list(S)
    m = len(S_list)
    # x does not divide a*b for all distinct a,b in S
    for i in range(m):
        a = S_list[i]
        for j in range(i+1, m):
            b = S_list[j]
            if (a * b) % x == 0:
                return False
    # a does not divide x*b for all distinct a,b in S
    for i in range(m):
        a = S_list[i]
        for j in range(i+1, m):
            b = S_list[j]
            if (x * b) % a == 0:
                return False
    return True

def _branch_and_bound(candidates, current_set, current_size, best_size, best_set, deadline):
    """
    Branch and bound search for maximum valid set.
    candidates: list of remaining candidates to consider
    current_set: set of elements chosen so far
    current_size: len(current_set)
    best_size: current best size found
    best_set: current best set
    deadline: time limit
    """
    global _DEADLINE
    if time.time() > deadline:
        return best_size, best_set
    
    if not candidates:
        if current_size > best_size:
            best_size = current_size
            best_set = set(current_set)
        return best_size, best_set
    
    # Upper bound: current_size + len(candidates)
    if current_size + len(candidates) <= best_size:
        return best_size, best_set
    
    # Try adding the first candidate
    x = candidates[0]
    remaining = candidates[1:]
    
    # Option 1: Add x
    if _can_add(current_set, x):
        new_set = current_set | {x}
        best_size, best_set = _branch_and_bound(
            remaining, new_set, current_size + 1, best_size, best_set, deadline
        )
    
    # Option 2: Skip x
    best_size, best_set = _branch_and_bound(
        remaining, current_set, current_size, best_size, best_set, deadline
    )
    
    return best_size, best_set

def _local_search(S, candidates, deadline):
    """Try to improve S by removing one element and adding others."""
    best = set(S)
    best_size = len(best)
    
    # Try all removals
    for rem in list(best):
        if time.time() > deadline:
            break
        S2 = best - {rem}
        # Try to add candidates in various orders
        for x in candidates:
            if x in S2:
                continue
            if _can_add(S2, x):
                S2.add(x)
                # Try to add more
                for y in candidates:
                    if y in S2:
                        continue
                    if _can_add(S2, y):
                        S2.add(y)
                if len(S2) > best_size:
                    best = set(S2)
                    best_size = len(best)
    
    return best

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 80  # 80 second budget (10s safety margin)
    
    n = P['n']
    nums = list(range(2, n + 1))
    
    # Start with primes (always a valid set)
    primes = [x for x in nums if _is_prime(x)]
    S = set(primes)
    
    # Try greedy with different orderings
    def _greedy_order(order_func):
        S_greedy = set()
        for x in order_func(nums):
            if _can_add(S_greedy, x):
                S_greedy.add(x)
        return S_greedy
    
    orderings = [
        lambda nums: sorted(nums, key=lambda x: -x),  # descending
        lambda nums: sorted(nums, key=lambda x: x),    # ascending
        lambda nums: sorted(nums, key=lambda x: (x, -x)),  # natural
    ]
    
    best_set = set(S)
    best_size = len(S)
    
    for order_func in orderings:
        S_greedy = _greedy_order(order_func)
        if len(S_greedy) > best_size:
            best_set = S_greedy
            best_size = len(S_greedy)
    
    # Local search from best
    candidates = [x for x in nums if x not in best_set]
    if time.time() < _DEADLINE:
        best_set = _local_search(best_set, candidates, _DEADLINE)
        best_size = len(best_set)
    
    # Branch and bound with remaining time
    if time.time() < _DEADLINE - 10:  # Leave 10s for verification and return
        # Use a smaller candidate set for B&B to keep it tractable
        # Focus on numbers that are likely to be in the optimal set
        # Heuristic: prefer numbers with fewer conflicts
        def _conflict_count(x, S_ref):
            count = 0
            for a in S_ref:
                for b in S_ref:
                    if a == b:
                        continue
                    if (a * b) % x == 0 or (x * b) % a == 0:
                        count += 1
            return count
        
        # Sort candidates by conflict count (ascending) - try less conflicting ones first
        candidate_list = [x for x in nums if x not in best_set]
        candidate_list.sort(key=lambda x: _conflict_count(x, best_set))
        
        # Limit B&B to a manageable subset
        bb_candidates = candidate_list[:50]
        
        # Reset best for B&B
        bb_best_size = best_size
        bb_best_set = set(best_set)
        
        # Try B&B starting from empty set with these candidates
        # This is expensive, so we only do it if we have enough time
        if time.time() < _DEADLINE - 20:
            # Use a simpler B&B that builds up from scratch
            def _bb_search(cands, current, size, b_size, b_set, dl):
                if time.time() > dl:
                    return b_size, b_set
                if not cands:
                    if size > b_size:
                        b_size = size
                        b_set = set(current)
                    return b_size, b_set
                if size + len(cands) <= b_size:
                    return b_size, b_set
                
                x = cands[0]
                rest = cands[1:]
                
                # Try adding x
                if _can_add(current, x):
                    b_size, b_set = _bb_search(
                        rest, current | {x}, size + 1, b_size, b_set, dl
                    )
                
                # Try skipping x
                b_size, b_set = _bb_search(
                    rest, current, size, b_size, b_set, dl
                )
                
                return b_size, b_set
            
            bb_best_size, bb_best_set = _bb_search(
                bb_candidates, set(), 0, bb_best_size, bb_best_set, _DEADLINE
            )
            
            if bb_best_size > best_size:
                best_set = bb_best_set
                best_size = bb_best_size
    
    # Verify
    S_list = sorted(best_set)
    if not _valid_set(S_list):
        # Fallback to primes
        S_list = sorted(primes)
    
    return list(S_list)
# EVOLVE-BLOCK-END
