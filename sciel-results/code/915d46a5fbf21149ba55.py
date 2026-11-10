"""Few-term Egyptian fractions with denominators greater than four."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    def verify(denoms):
        total = sum(Fraction(1, d) for d in denoms)
        return total == 1

    # Known 16-term solution
    known_solution = [5, 6, 7, 10, 14, 21, 30, 35, 42, 70, 105, 140, 210, 280, 420, 840]
    if verify(known_solution):
        best = known_solution
    else:
        best = None

    # Structural change: subset-sum over divisor weights of a common denominator.
    # For a chosen L, each valid denominator d (d|L, d>=5) has weight w = L//d.
    # We need a subset of weights summing to L.
    # Use meet-in-the-middle for efficiency.

    def solve_subset_sum(L):
        """Find minimum-size subset of {L//d : d|L, d>=5} that sums to L."""
        # Get all valid weights
        weights = []
        denom_to_weight = {}
        for d in range(5, L + 1):
            if L % d == 0:
                w = L // d
                if w not in denom_to_weight:
                    denom_to_weight[w] = d
                    weights.append(w)
        
        if not weights:
            return None
        
        # Remove duplicates (each weight corresponds to exactly one d)
        # Sort weights
        weights_sorted = sorted(set(weights))
        
        if time.time() > deadline:
            return None
        
        # Meet in the middle
        n = len(weights_sorted)
        if n > 30:
            # Too many for MITM, try a different approach
            return None
        
        half = n // 2
        left_weights = weights_sorted[:half]
        right_weights = weights_sorted[half:]
        
        # Generate all subset sums for left half
        left_sums = {0: []}
        for w in left_weights:
            new_entries = {}
            for s, subset in left_sums.items():
                ns = s + w
                if ns <= L and ns not in new_entries:
                    new_entries[ns] = subset + [w]
            for ns, subset in new_entries.items():
                if ns not in left_sums or len(subset) < len(left_sums[ns]):
                    left_sums[ns] = subset
        
        # Generate all subset sums for right half
        right_sums = {0: []}
        for w in right_weights:
            new_entries = {}
            for s, subset in right_sums.items():
                ns = s + w
                if ns <= L and ns not in new_entries:
                    new_entries[ns] = subset + [w]
            for ns, subset in new_entries.items():
                if ns not in right_sums or len(subset) < len(right_sums[ns]):
                    right_sums[ns] = subset
        
        # Find minimum total size subset summing to L
        best_subset = None
        best_size = L + 1
        for ls, lsubset in left_sums.items():
            rs_needed = L - ls
            if rs_needed in right_sums:
                rsubset = right_sums[rs_needed]
                total_size = len(lsubset) + len(rsubset)
                if total_size < best_size:
                    best_size = total_size
                    best_subset = lsubset + rsubset
        
        if best_subset is None:
            return None
        
        # Convert weights back to denominators
        denoms = [denom_to_weight[w] for w in best_subset]
        return sorted(set(denoms))
    
    # Try several common denominators
    candidate_Ls = [840, 2520, 4200, 5040, 7560, 10080, 15120, 20160, 25200, 50400]
    
    for L in candidate_Ls:
        if time.time() > deadline:
            break
        result = solve_subset_sum(L)
        if result is not None and verify(result):
            if best is None or len(result) < len(best):
                best = result
                if len(result) <= 7:
                    return result
    
    # Fallback: try a bounded DFS for small term counts
    def dfs(remaining, min_denom, used, depth, max_depth):
        if time.time() > deadline:
            return None
        if remaining == 0:
            return sorted(used)
        if depth == max_depth:
            return None
        if remaining < 0:
            return None
        
        # Pruning: remaining must be <= (max_depth - depth) / min_denom
        if remaining > (max_depth - depth) / min_denom:
            return None
        
        # The next denominator must be at least ceil(1/remaining)
        if remaining <= 0:
            return None
        min_d = max(min_denom, math.ceil(1 / float(remaining)))
        
        # Upper bound on d: 1/d <= remaining, so d >= ceil(1/remaining)
        # Also, we need remaining - 1/d to be achievable with (max_depth-depth-1) terms
        # Each at most 1/(d+1), so remaining - 1/d <= (max_depth-depth-1)/(d+1)
        # This gives a tighter upper bound
        
        max_d = min(10000, int(1 / float(remaining)) + 50) if remaining > 0 else 10000
        
        for d in range(min_d, max_d + 1):
            if d < 5 or d in used:
                continue
            new_remaining = remaining - Fraction(1, d)
            if new_remaining < 0:
                break
            result = dfs(new_remaining, d + 1, used | {d}, depth + 1, max_depth)
            if result is not None:
                return result
        return None
    
    # Try to find solutions with fewer terms than current best
    target_terms = len(best) if best else 16
    for max_depth in range(5, min(target_terms, 16)):
        if time.time() > deadline:
            break
        result = dfs(Fraction(1, 1), 5, set(), 0, max_depth)
        if result is not None and verify(result):
            if best is None or len(result) < len(best):
                best = result
                if len(result) <= 7:
                    return result
    
    if best is not None:
        return sorted(best)
    return known_solution

# EVOLVE-BLOCK-END
