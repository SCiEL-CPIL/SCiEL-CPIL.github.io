"""Optimized program: search for maximum-cardinality reciprocal-dissociated subset of [1..22]."""
import time

# LCM of 1..22
_LCM = 232792560

def _scaled(x):
    return _LCM // x

def _check_dissociated(subset):
    """Return True if all 2^len(subset) reciprocal subset sums are distinct."""
    k = len(subset)
    if k == 0:
        return True
    scaled = [_scaled(x) for x in subset]
    n = 1 << k
    # Use a set to track seen sums
    seen = set()
    seen.add(0)
    # Iterate through all masks
    for mask in range(1, n):
        # Compute sum using DP approach
        lb = mask & (-mask)
        idx = lb.bit_length() - 1
        prev = mask ^ lb
        # We need to compute sum efficiently
        # Instead of storing all sums, let's use a different approach
        pass
    
    # Better approach: generate all subset sums
    sums = [0] * n
    for mask in range(1, n):
        lb = mask & (-mask)
        idx = lb.bit_length() - 1
        prev = mask ^ lb
        sums[mask] = sums[prev] + scaled[idx]
    
    return len(set(sums)) == n

def _find_collision(subset):
    """Find two different subsets with the same sum, or None if dissociated."""
    k = len(subset)
    if k == 0:
        return None
    scaled = [_scaled(x) for x in subset]
    n = 1 << k
    sums = [0] * n
    for mask in range(1, n):
        lb = mask & (-mask)
        idx = lb.bit_length() - 1
        prev = mask ^ lb
        sums[mask] = sums[prev] + scaled[idx]
    
    # Find collision
    seen = {}
    for mask in range(n):
        s = sums[mask]
        if s in seen:
            return mask, seen[s]
        seen[s] = mask
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s safety margin
    
    # We want the maximum cardinality subset of [1..22] that is reciprocal-dissociated.
    # Strategy: Use a more systematic approach
    # 1. Start with a good initial set
    # 2. Try to add elements
    # 3. If stuck, try swaps
    
    # Known good starting point: primes and some other numbers
    # Let's start with a set that's likely to be large
    # The key insight: larger numbers have smaller reciprocals, making collisions less likely
    # But we need to be careful about subset sums
    
    # Let's try a greedy approach but with better heuristics
    # Start with all numbers, remove the "most problematic" ones
    
    # Actually, let's try a different approach:
    # 1. Try to build up from a small set
    # 2. Use backtracking with pruning
    
    # First, let's try to find a large set by greedy addition
    # Start with the largest numbers (smallest reciprocals)
    
    best = []
    in_best = set()
    
    # Try adding numbers from largest to smallest (smallest reciprocals first)
    # This should be more likely to avoid collisions
    for x in range(22, 0, -1):
        if time.time() > deadline - 5:
            break
        candidate = best + [x]
        candidate.sort()
        if _check_dissociated(candidate):
            best = candidate
            in_best.add(x)
    
    # Now try to add back numbers we couldn't add
    for x in range(1, 23):
        if x in in_best:
            continue
        if time.time() > deadline - 5:
            break
        candidate = best + [x]
        candidate.sort()
        if _check_dissociated(candidate):
            best = candidate
            in_best.add(x)
    
    # Try swaps: replace one element with another
    if time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            in_set = set(best)
            not_in_set = [x for x in range(1, 23) if x not in in_set]
            for i, a in enumerate(best):
                if time.time() > deadline - 5:
                    break
                for b in not_in_set:
                    candidate = best[:]
                    candidate[i] = b
                    candidate.sort()
                    if _check_dissociated(candidate):
                        best = candidate
                        in_set = set(best)
                        not_in_set = [x for x in range(1, 23) if x not in in_set]
                        improved = True
                        break
                if improved:
                    break
    
    return best
