"""Rewritten program for the reciprocal-subsum-free set task."""
import time
import random
from fractions import Fraction
from itertools import combinations

_START = None
DEADLINE = 85.0  # Safety margin for 90s limit

def _deadline_hit():
    global _START
    if _START is None:
        return False
    return (time.time() - _START) > DEADLINE

def _check_violation(S, a):
    """
    Check if 1/a equals sum of reciprocals of some non-empty subset of S \ {a}.
    Returns True if a violation exists.
    """
    others = [x for x in S if x != a]
    if not others:
        return False
    
    target = Fraction(1, a)
    n = len(others)
    
    # If n is small, brute force is fine.
    # If n is large, meet-in-the-middle is better.
    # For n <= 20, brute force 2^n is ~1M, which might be slow in Python if called many times.
    # Let's use a smarter check.
    
    # Optimization: 
    # 1/a = 1/b1 + ... + 1/bk
    # Since all terms are positive, the sum is strictly less than 1/min(others) * k? No.
    # The sum of reciprocals of a subset of distinct integers >= 2 is bounded.
    # Max sum for k terms from {2..24} is 1/2+1/3+1/4... which is small.
    # Actually, for a=1, 1/1 = 1. Max sum of others is < 1 (since 1/2+1/3+... diverges slowly but for distinct integers up to 24, sum is ~3). 
    # Wait, 1/1 = 1. Can we make 1 from 1/2+1/3+...? 
    # 1/2 + 1/3 + 1/6 = 1. So {1, 2, 3, 6} is invalid.
    
    # Let's stick to the meet-in-the-middle for n > 10 to be safe and fast.
    
    if n <= 10:
        for mask in range(1, 1 << n):
            s = Fraction(0, 1)
            for i in range(n):
                if mask & (1 << i):
                    s += Fraction(1, others[i])
            if s == target:
                return True
        return False
    else:
        half = n // 2
        left = others[:half]
        right = others[half:]
        
        left_sums = set()
        # Add 0 for empty subset
        left_sums.add(Fraction(0, 1))
        for mask in range(1, 1 << len(left)):
            s = Fraction(0, 1)
            for i in range(len(left)):
                if mask & (1 << i):
                    s += Fraction(1, left[i])
            left_sums.add(s)
        
        # Check if target - sum_right is in left_sums
        # Note: The subset T must be non-empty.
        # If sum_right == target and left_sum == 0, then T is non-empty (subset of right).
        # If sum_right != target and left_sum == 0, then T is subset of left.
        # We just need to ensure the combined subset is not empty.
        # Since we iterate all right subsets, and check if (target - s_right) is in left_sums:
        # If s_right == target, we need 0 in left_sums (which is true). T = subset of right (non-empty if mask != 0).
        # If s_right != target, we need non-zero left sum? No, if left sum is 0, T is subset of right.
        # The only invalid case is if T is empty, which means s_right = 0 and left_sum = 0.
        # So we must exclude the case where mask_right == 0 AND the required left_sum is 0.
        
        for mask in range(1 << len(right)):
            s = Fraction(0, 1)
            for i in range(len(right)):
                if mask & (1 << i):
                    s += Fraction(1, right[i])
            
            diff = target - s
            if diff in left_sums:
                # Check if the combined subset is non-empty
                # If mask == 0 and diff == 0, then T is empty.
                if not (mask == 0 and diff == 0):
                    return True
        return False

def _is_valid(S):
    """Check if S is a valid reciprocal-subsum-free set."""
    S_list = list(S)
    for a in S_list:
        if _check_violation(S_list, a):
            return False
    return True

def _greedy_construct():
    """Greedy construction: Start with all, remove conflicts."""
    S = list(range(1, 25))
    changed = True
    while changed and not _deadline_hit():
        changed = False
        # Try removing elements that cause the most conflicts or just smallest first
        # Smallest elements are most likely to be represented as sums of larger ones?
        # 1/1 = 1/2+1/3+1/6. 1/2 = 1/3+1/6.
        # Usually small numbers are the 'targets' that are easy to represent.
        for a in sorted(S):
            if _check_violation(S, a):
                S.remove(a)
                changed = True
                break
    return S

def _local_search(initial_set):
    """
    Local search to maximize cardinality.
    """
    S = list(initial_set)
    best_S = list(S)
    best_len = len(S)
    
    # Ensure best is valid
    if not _is_valid(best_S):
        # Fallback to a known valid set
        best_S = [24]
        best_len = 1
    
    # If initial set is not valid, we need to fix it first
    # The greedy constructor should produce a valid set.
    
    while not _deadline_hit():
        improved = False
        
        # 1. Try to add any element
        for x in range(1, 25):
            if x not in S:
                new_S = S + [x]
                if _is_valid(new_S):
                    S = new_S
                    improved = True
                    if len(S) > best_len:
                        best_len = len(S)
                        best_S = list(S)
                    break # Restart adding phase with new S
        
        # 2. If no addition possible, try swaps
        if not improved:
            swapped = False
            # Sort S to make deterministic? Or random?
            # Let's try all swaps
            for x in S:
                for y in range(1, 25):
                    if y not in S:
                        new_S = [z for z in S if z != x] + [y]
                        if _is_valid(new_S):
                            S = new_S
                            swapped = True
                            # Size is same, so best_len doesn't change unless we find a path to larger size later
                            break
                if swapped:
                    break
            
            if not swapped:
                # Stuck. Try a random perturbation to escape local optimum
                # Remove a random element, then try to add back or add new
                if len(S) > 1:
                    # Pick a random element to remove
                    idx = random.randint(0, len(S) - 1)
                    removed = S.pop(idx)
                    
                    # Try to add it back or something else
                    # Actually, just restart the addition phase with the smaller set
                    # This effectively allows "removal" if it leads to a better configuration later
                    # But we must ensure we don't degrade best_S
                    pass
                else:
                    # If stuck at size 1, we are in trouble, but [24] is always valid
                    break

def construct():
    global _START
    _START = time.time()
    
    # 1. Get a good starting point via greedy
    S_greedy = _greedy_construct()
    
    # 2. Run local search
    S_best = _local_search(S_greedy)
    
    # 3. Try a few random restarts if time permits
    # Generate random valid sets? Hard.
    # Instead, try greedy with different removal orders?
    # The greedy removal order "smallest first" is standard.
    # Let's try "largest first" removal.
    
    if not _deadline_hit():
        S_alt = list(range(1, 25))
        changed = True
        while changed and not _deadline_hit():
            changed = False
            for a in sorted(S_alt, reverse=True):
                if _check_violation(S_alt, a):
                    S_alt.remove(a)
                    changed = True
                    break
        S_best_alt = _local_search(S_alt)
        if len(S_best_alt) > len(S_best):
            S_best = S_best_alt
            
    if not _deadline_hit():
        # Try random subset initialization
        # Start with a random large subset, make it valid, then local search
        random.seed(42) # Reproducibility
        for _ in range(5):
            if _deadline_hit():
                break
            # Pick a random subset of size ~15
            S_rand = random.sample(range(1, 25), 15)
            # Make it valid by removing conflicts
            changed = True
            while changed and not _deadline_hit():
                changed = False
                for a in S_rand:
                    if _check_violation(S_rand, a):
                        S_rand.remove(a)
                        changed = True
                        break
            S_best_rand = _local_search(S_rand)
            if len(S_best_rand) > len(S_best):
                S_best = S_best_rand

    # Final safety check
    if not _is_valid(S_best):
        S_best = [24]
        
    if not S_best:
        S_best = [24]
        
    return S_best
