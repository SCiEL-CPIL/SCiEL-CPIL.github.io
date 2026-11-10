"""Rewritten program: structural change to bounded DFS with pruning for Egyptian fraction of 5/31."""
from fractions import Fraction
import time

def construct():
    target = Fraction(5, 31)
    deadline = time.time() + 80.0  # safety margin before 90s kill
    best_solution = None
    best_max_den = float('inf')

    def greedy_completion(remaining, current_denoms, max_den_allowed):
        """Try to complete using greedy; return list if valid else None."""
        nonlocal best_solution, best_max_den
        denoms = list(current_denoms)
        r = remaining
        while r > 0:
            # Greedy step
            d = (r.denominator + r.numerator - 1) // r.numerator
            if d <= max_den_allowed and d > (denoms[-1] if denoms else 0) and d <= 10000000:
                denoms.append(d)
                r -= Fraction(1, d)
            else:
                return None
        if r == 0:
            max_d = max(denoms)
            if max_d < best_max_den:
                best_max_den = max_d
                best_solution = denoms[:]
                return denoms[:]
        return None

    def dfs(remaining, min_den, max_den_allowed, denoms, depth):
        nonlocal best_solution, best_max_den
        if time.time() > deadline:
            return
        if remaining == 0:
            if denoms:
                max_d = max(denoms)
                if max_d < best_max_den:
                    best_max_den = max_d
                    best_solution = denoms[:]
            return
        if depth >= 256:
            return
        # Lower bound: even with infinite terms, the sum of 1/d for d >= min_den
        # But more practically, we need enough terms to reach remaining
        # If remaining is positive, we need at least 1 term
        # Upper bound check: if even using the smallest possible denominators we can't reach
        # Actually, let's use a simpler bound: if 1/min_den > remaining, then we can't use min_den
        # and all larger denominators are even smaller, so no solution
        if Fraction(1, min_den) > remaining:
            return
        # Try denominators from min_den to max_den_allowed
        # But we need to be smart: d must satisfy 1/d <= remaining, i.e., d >= 1/remaining
        d_min = min_den
        d_max = min(max_den_allowed, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
        # Actually d_max: we want 1/d <= remaining, so d >= ceil(1/remaining)
        # But we also need d to be such that remaining - 1/d >= 0, so 1/d <= remaining
        # d >= 1/remaining, so d_min_effective = max(min_den, ceil(1/remaining))
        if remaining.numerator > 0:
            d_min_effective = max(d_min, (1 + remaining.numerator - 1) // remaining.numerator)
            # ceil(1/remaining) = ceil(remaining.denominator / remaining.numerator)
            d_min_effective = max(d_min, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
            if d_min_effective > d_max:
                return
        else:
            d_min_effective = d_min
        
        # Iterate candidate denominators
        for d in range(d_min_effective, d_max + 1):
            if time.time() > deadline:
                return
            if d in denoms:
                continue
            if d > max_den_allowed:
                break
            new_remaining = remaining - Fraction(1, d)
            if new_remaining < 0:
                continue
            # Pruning: if new_remaining is positive, the next denominator must be > d
            # Also, we can try greedy completion as a quick check
            # But first, recurse
            denoms.append(d)
            dfs(new_remaining, d + 1, max_den_allowed, denoms, depth + 1)
            denoms.pop()
            # Update best if we found something
            if best_solution is not None and max_den_allowed >= best_max_den:
                # We can potentially prune more
                pass

    # Start with a reasonable max_den_allowed. We can start high and decrease.
    # First, try greedy from start to get an initial upper bound
    initial = greedy_completion(target, [], 10000000)
    if initial is not None:
        print(f"Initial greedy solution: max_den={best_max_den}, terms={len(initial)}")

    # Now do DFS to find better solutions
    # Start with a moderate max_den_allowed and increase if needed
    # Actually, let's search from a lower bound upward
    # Lower bound: the largest denominator must be at least ceil(1/target) = ceil(31/5) = 7
    # But that's a weak bound. Let's just do DFS with decreasing max_den_allowed
    # Or better: do DFS with max_den_allowed = best_max_den - 1 and see if we can find a solution
    # Then decrease further
    
    # Let's try max_den_allowed values from a reasonable range
    # Start with the greedy solution's max, then try to improve
    current_best = best_max_den
    if current_best == float('inf'):
        current_best = 10000000
    
    # Try to find solutions with max_den < current_best
    # Use DFS with max_den_allowed = current_best - 1, then decrease
    search_max = current_best - 1
    min_search = 7  # ceil(31/5)
    
    # Do a binary-search-like or linear search
    # For each candidate max_den, check if a solution exists
    # But that's expensive. Instead, do DFS once with the current best as upper bound
    # and let it find the optimal
    
    # Reset and do full DFS with pruning
    dfs(target, 1, current_best - 1, [], 0)
    
    if best_solution is None:
        # Fallback to greedy
        best_solution = [8]
        r = target - Fraction(1, 8)
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            best_solution.append(d)
            r -= Fraction(1, d)
    
    return best_solution
