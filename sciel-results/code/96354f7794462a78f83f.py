"""Stronger algorithm for the reciprocal-weight under lcm-separation task."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction
from functools import lru_cache

_DEADLINE = 80  # seconds; leave 10s safety margin

def _lcm(a, b):
    return a * b // math.gcd(a, b)

def _build_compatibility():
    """
    Returns:
      numbers: list of candidate numbers in [2,256]
      comp:    comp[i] is a bitmask of j != i such that lcm(numbers[i], numbers[j]) > 256
    """
    numbers = list(range(2, 257))
    n = len(numbers)
    comp = [0] * n
    for i in range(n):
        a = numbers[i]
        mask = 0
        for j in range(n):
            if i == j:
                continue
            b = numbers[j]
            if _lcm(a, b) > 256:
                mask |= (1 << j)
        comp[i] = mask
    return numbers, comp

def construct():
    t0 = time.time()
    numbers, comp = _build_compatibility()
    n = len(numbers)
    
    # weights: 1/x as Fraction
    weights = [Fraction(1, numbers[i]) for i in range(n)]
    
    # Precompute: for each i, the set of j incompatible with i (lcm <= 256)
    # incompatible[i] is bitmask of j such that lcm(numbers[i],numbers[j]) <= 256
    incomp = [(~comp[i]) & ((1 << n) - 1) for i in range(n)]
    # Remove self
    for i in range(n):
        incomp[i] &= ~(1 << i)
    
    # Greedy seed: sort by weight desc, greedily add if compatible with all selected
    order = sorted(range(n), key=lambda i: weights[i], reverse=True)
    selected = []
    for i in order:
        if all(i in comp[j] for j in selected):
            selected.append(i)
    
    seed_set = set(selected)
    seed_score = sum(weights[i] for i in selected)
    
    # Local search: try swapping one element for another
    # We'll do several rounds of improvement
    current = set(selected)
    current_score = seed_score
    
    # Precompute adjacency for fast compatibility checks
    # comp_mask[i] is bitmask of compatible nodes
    comp_mask = comp  # already bitmasks
    
    for iteration in range(20):
        if time.time() - t0 > _DEADLINE:
            break
        improved = False
        
        # Try removing one element and adding another
        for i in list(current):
            # Try adding each j not in current
            for j in range(n):
                if j in current:
                    continue
                # Check if j is compatible with all in current - {i}
                ok = True
                for k in current:
                    if k != i and not (comp_mask[k] & (1 << j)):
                        ok = False
                        break
                if ok:
                    new_score = current_score - weights[i] + weights[j]
                    if new_score > current_score:
                        current.discard(i)
                        current.add(j)
                        current_score = new_score
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break
    
    # Also try adding elements that are compatible with all current
    for j in range(n):
        if j in current:
            continue
        if all(comp_mask[k] & (1 << j) for k in current):
            current.add(j)
            current_score += weights[j]
    
    # Convert back to values
    result = [numbers[i] for i in current]
    result.sort()
    
    # Verify validity
    for idx_a, a in enumerate(result):
        for idx_b in range(idx_a+1, len(result)):
            b = result[idx_b]
            if _lcm(a, b) <= 256:
                # Should not happen, but fallback
                result = [2]
                break
        if len(result) == 1:
            break
    
    return result
# EVOLVE-BLOCK-END
