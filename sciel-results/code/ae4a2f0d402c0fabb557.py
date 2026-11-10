"""Rewritten program with faster verification and multiple random restarts."""
import random
from collections import defaultdict

def _is_valid_set(selected):
    """Check if no element is the mean of any subset of >=2 other elements."""
    n = len(selected)
    if n <= 2:
        return True
    sel_set = set(selected)
    for a in selected:
        others = [x for x in selected if x != a]
        # Transform: we need subset T of others with |T|>=2 and sum(T) = |T|*a
        # Equivalently: sum(x - a for x in T) = 0 with |T|>=2
        diffs = [x - a for x in others]
        # DP: dp[k] = set of possible sums using exactly k elements
        # Use dict: sum_value -> min_size or just track which sizes are achievable
        # Better: dp[size] = set of achievable sums
        m = len(diffs)
        # dp[s] = set of sizes for which sum s is achievable
        # Or: achievable[s] = set of sizes
        # Since we only care if size>=2 is achievable for sum=0, let's track:
        # can[s] = bitmask of achievable sizes (bit i set means size i is achievable)
        # But sizes up to 47, so 48-bit mask
        # Initialize: can[0] = 1 (size 0, sum 0)
        can = {0: 1}  # sum -> bitmask of achievable sizes
        for d in diffs:
            new_can = dict(can)
            for s, mask in can.items():
                new_s = s + d
                # Shift mask left by 1 (add one element to each subset)
                new_mask = mask << 1
                if new_s in new_can:
                    new_can[new_s] |= new_mask
                else:
                    new_can[new_s] = new_mask
            can = new_can
        # Check if sum=0 is achievable with size >= 2
        if 0 in can:
            mask = can[0]
            # Check if any bit >= 2 is set
            if mask & ~3:  # bits 0 and 1 are sizes 0 and 1; we need size >= 2
                return False
    return True

def _greedy_with_order(order, n=512, max_size=48):
    """Greedy construction following a specific order of candidates."""
    selected = []
    for x in order:
        if x > n:
            continue
        if x in selected:
            continue
        candidate = selected + [x]
        if len(candidate) > max_size:
            continue
        if _is_valid_set(candidate):
            selected = candidate
    return selected

def construct():
    """Construct a set of up to 48 distinct positive integers <= 512
    where no element is the mean of any subset of >=2 other elements."""
    
    best = []
    
    # Try multiple random orderings
    for trial in range(200):
        # Create a random ordering of candidates 1..512
        candidates = list(range(1, 513))
        random.shuffle(candidates)
        result = _greedy_with_order(candidates)
        if len(result) > len(best):
            best = result
        if len(best) >= 48:
            break
    
    # Also try sorted orderings and other heuristics
    for order in [
        list(range(1, 513)),
        list(range(512, 0, -1)),
        # Try starting with large numbers
        list(range(256, 513)) + list(range(1, 257)),
        # Try primes-ish (just odd numbers)
        [x for x in range(1, 513) if x % 2 == 1] + [x for x in range(1, 513) if x % 2 == 0],
    ]:
        result = _greedy_with_order(order)
        if len(result) > len(best):
            best = result
        if len(best) >= 48:
            break
    
    return best
