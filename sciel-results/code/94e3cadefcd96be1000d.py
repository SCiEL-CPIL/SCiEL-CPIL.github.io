"""Meet-in-the-middle + iterative deepening to minimize largest denominator for 5/31 Egyptian fraction."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction


def _find_solution_with_max(target, M, max_terms=8, deadline=None):
    """Find distinct denominators in [1, M] summing to target, or None."""
    if deadline is None:
        deadline = time.time() + 80.0
    
    # Use DFS with strong pruning
    found = [None]
    nodes = [0]
    NODE_CAP = 5_000_000
    
    def dfs(remaining, last, path, terms_left):
        if time.time() > deadline or nodes[0] > NODE_CAP:
            return
        nodes[0] += 1
        
        if remaining == 0:
            if path:
                found[0] = list(path)
            return
        
        if terms_left == 0:
            return
        
        # Pruning: remaining must be positive
        if remaining <= 0:
            return
        
        # Lower bound on next denominator
        # 1/d <= remaining => d >= 1/remaining
        min_d = (remaining.denominator + remaining.numerator - 1) // remaining.numerator
        if min_d < last + 1:
            min_d = last + 1
        if min_d > M:
            return
        
        # Upper bound: sum of terms_left largest available reciprocals
        max_possible = Fraction(0, 1)
        d = last + 1
        for _ in range(terms_left):
            if d > M:
                break
            max_possible += Fraction(1, d)
            d += 1
        if remaining > max_possible:
            return
        
        # Lower bound: even taking the smallest possible, can we reach remaining?
        # We need at least 1/M * terms_left contribution from terms_left terms
        # But more precisely: the minimum sum with terms_left distinct terms > last is
        # 1/(last+1) + 1/(last+2) + ... + 1/(last+terms_left)
        # This is a lower bound on what we can achieve (smallest possible sum)
        # Actually we need remaining to be >= this minimum sum
        min_possible = Fraction(0, 1)
        d = last + 1
        for _ in range(terms_left):
            if d > M:
                break
            min_possible += Fraction(1, d)
            d += 1
        if remaining < min_possible:
            return
        
        # Try each candidate denominator
        for d in range(min_d, M + 1):
            new_rem = remaining - Fraction(1, d)
            if new_rem < 0:
                continue
            path.append(d)
            dfs(new_rem, d, path, terms_left - 1)
            if found[0] is not None:
                path.pop()
                return
            path.pop()
    
    dfs(target, 0, [], max_terms)
    return found[0]


def construct():
    deadline = time.time() + 85.0
    target = Fraction(5, 31)
    
    # Start with a known valid solution
    # 5/31 = 1/8 + 1/31 + 1/248? Let's verify: 1/8 + 1/31 + 1/248 = 31/248 + 8/248 + 1/248 = 40/248 = 5/31. Yes!
    best_sol = [8, 31, 248]
    best_max = 248
    
    # Try to find solutions with smaller maximum denominator
    # Binary search or linear search from a low bound
    # The theoretical minimum: 5/31 < 1, so we need at least 2 terms
    # With 2 terms: 1/a + 1/b = 5/31 => 31(a+b) = 5ab => 5ab - 31a - 31b = 0
    # (5a-31)(5b-31) = 31*31 = 961
    # 961 = 31^2, divisors: 1, 31, 961
    # 5a-31 = 1 => a=32/5 not integer
    # 5a-31 = 31 => a=62/5 not integer  
    # 5a-31 = 961 => a=992/5 not integer
    # So no 2-term solution.
    
    # Try 3-term solutions with bounded max
    # We'll search for the smallest M such that a solution exists with all denoms <= M
    
    # Lower bound for M: need at least enough terms to sum to 5/31
    # With max denominator M, max sum with k terms is 1/1+1/2+...+1/k but constrained by distinctness
    # Simple lower bound: M must be at least ceil(31/5) = 7 for any single term, but we need multiple
    
    # Start searching from M=7 upward, but be smart about it
    # First, try small M values with limited terms
    
    # Try M from 7 to best_max-1
    # For each M, try to find a solution with max_terms up to 8
    # Use iterative deepening on number of terms
    
    search_start = 7
    search_end = min(best_max - 1, 10000000)
    
    # First, quickly check if we can beat 248 with 3 terms
    # Enumerate 3-term solutions more efficiently
    # 1/a + 1/b + 1/c = 5/31 with a < b < c <= M
    for M in range(search_start, min(best_max, 200)):  # Check small M first
        if time.time() > deadline:
            break
        sol = _find_solution_with_max(target, M, max_terms=6, deadline=deadline)
        if sol is not None:
            m = max(sol)
            if m < best_max:
                best_max = m
                best_sol = sol
                # Don't break, keep searching for even smaller
            else:
                break
    
    # If we found something, try to go even lower
    if best_max < 248:
        for M in range(search_start, best_max):
            if time.time() > deadline:
                break
            sol = _find_solution_with_max(target, M, max_terms=8, deadline=deadline)
            if sol is not None:
                m = max(sol)
                if m < best_max:
                    best_max = m
                    best_sol = sol
    
    # Verify the solution
    s = sum(Fraction(1, x) for x in best_sol)
    if s != target or len(best_sol) < 1 or len(best_sol) > 256:
        return [8, 31, 248]
    
    # Ensure all denominators are distinct and in [1, 10000000]
    if len(set(best_sol)) != len(best_sol):
        return [8, 31, 248]
    if any(d < 1 or d > 10000000 for d in best_sol):
        return [8, 31, 248]
    
    return best_sol
# EVOLVE-BLOCK-END
