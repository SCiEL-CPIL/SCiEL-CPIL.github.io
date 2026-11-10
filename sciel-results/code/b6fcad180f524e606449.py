"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import gcd

def _try_complete_greedy(rem, max_terms=256, deadline=None):
    """Complete the remainder with greedy Egyptian fractions, enforcing distinctness."""
    if rem == 0:
        return []
    values = []
    used = set()
    r = rem
    while r != 0:
        if deadline is not None and time.time() > deadline:
            return None
        if len(values) >= max_terms:
            return None
        # Greedy step
        d = (r.denominator + r.numerator - 1) // r.numerator
        # Ensure distinct
        while d in used:
            d += 1
        if d > 10_000_000:
            return None
        values.append(d)
        used.add(d)
        r -= Fraction(1, d)
    return values

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    target = Fraction(5, 31)
    best_max = 10_000_000  # Initialize with a large value
    best_solution = None
    
    # Strategy: Systematic search over small numbers of terms with small denominators.
    # We know from the prompt that a solution with max << 1736 exists, likely in 100-500.
    # We will search for solutions with 2, 3, 4, 5 terms, increasing the max denominator limit.
    
    # To be efficient, we can iterate through possible first denominators and try to complete the rest.
    # However, a more robust approach for small term counts is to use a recursive search with pruning.
    
    def search(rem, min_den, max_terms, current_terms):
        """
        Recursively find Egyptian fraction representation for rem.
        min_den: minimum allowed denominator (to ensure increasing order and distinctness)
        max_terms: remaining number of terms allowed
        current_terms: list of denominators chosen so far
        Returns a list of denominators if found, else None.
        """
        if rem == 0:
            if current_terms:
                return list(current_terms)
            else:
                return None  # Need at least 1 term? Task says 1..256. If rem=0 initially, invalid.
        
        if max_terms == 0:
            return None
        
        if rem < 0:
            return None
            
        # Greedy bound: the smallest denominator d must satisfy 1/d >= rem/max_terms
        # So d <= max_terms / rem
        # Also d >= min_den
        # And 1/d < rem (since we need more than 1 term if max_terms > 1, or exactly 1 if max_terms==1)
        # Actually, if max_terms == 1, rem must be 1/d exactly.
        
        if max_terms == 1:
            if rem.denominator % rem.numerator == 0:
                d = rem.denominator // rem.numerator
                if d >= min_den and d <= 10_000_000:
                    current_terms.append(d)
                    return list(current_terms)
            return None
        
        # Upper bound for d:
        # We need rem = 1/d + rest, where rest is sum of (max_terms-1) fractions each < 1/d (strictly, due to distinctness/order)
        # Actually, rest < (max_terms-1)/d.
        # So rem < 1/d + (max_terms-1)/d = max_terms/d
        # => d < max_terms / rem
        # Also, since denominators are strictly increasing, the largest denominator in the rest is > d.
        # A tighter bound: The greedy algorithm gives the smallest possible next denominator if we want to minimize the max.
        # But for existence, we just need any solution.
        
        # Let's iterate d from min_den upwards.
        # To minimize the LARGEST denominator, we should prefer smaller d?
        # Not necessarily. Sometimes a slightly larger d allows a much smaller subsequent denominator.
        # But generally, smaller d leads to larger remainders and potentially larger subsequent denominators.
        # However, the task is to MINIMIZE the LARGEST denominator.
        # So we want the solution with the smallest max(current_terms).
        
        # We will search for any valid solution within the current constraint, and keep track of the best (min max).
        # But doing a full exhaustive search is expensive.
        # Let's use the greedy completion as a heuristic to find *a* solution quickly for a given starting point,
        # and then refine if time permits.
        
        # Actually, the prompt suggests "Systematic search over 2-5 term Egyptian fractions with small denominators".
        # Let's try to find solutions with exactly k terms for k=2,3,4,5.
        
        # For a given k, we can iterate d1, d2, ...
        # This is complex. Let's stick to the "try first denominator, then greedy complete" but with a smarter first denominator selection.
        
        # Better approach:
        # 1. Iterate max_limit from 10 to 1000 (step 10, then 100).
        # 2. For each max_limit, try to find a solution where all denominators <= max_limit.
        # 3. If found, update best_max and best_solution.
        # 4. To find such a solution, we can use a backtracking search that enforces d <= max_limit.
        
        pass # Placeholder for recursive search logic, which we will implement in the main loop below.

    # Implementation of the main search strategy:
    # We'll try to find solutions with a maximum denominator limit L.
    # We increase L until we find a solution.
    # To speed this up, we use a recursive search that prunes based on L.
    
    def find_solution_with_max_limit(L, max_terms=5):
        """
        Find a solution with all denominators <= L.
        Returns list of denominators or None.
        """
        # Recursive helper
        def rec(rem, min_d, terms_left, current):
            if rem == 0:
                return list(current)
            if terms_left == 0:
                return None
            if rem < 0:
                return None
                
            # If terms_left == 1
            if terms_left == 1:
                if rem.denominator % rem.numerator == 0:
                    d = rem.denominator // rem.numerator
                    if min_d <= d <= L:
                        current.append(d)
                        res = list(current)
                        current.pop()
                        return res
                return None
            
            # Determine range for next denominator d
            # 1/d must be <= rem (obviously)
            # And we need to represent rem - 1/d with terms_left-1 terms, all >= d+1 and <= L.
            # The maximum sum of terms_left-1 fractions with denominators >= d+1 is (terms_left-1)/(d+1).
            # So we need rem - 1/d <= (terms_left-1)/(d+1)
            # Also, the minimum sum is 0 (not helpful).
            # A simpler bound: d must be such that 1/d < rem (if terms_left > 1, we need positive remainder)
            # Actually, if 1/d >= rem, then rem - 1/d <= 0. Since denominators are positive, remainder must be > 0.
            # So 1/d < rem => d > 1/rem.
            
            # Also, to fit in L, we need a valid completion.
            
            # Let's iterate d from min_d to L.
            # Optimization: We can break early if 1/d is too small to allow the remainder to be formed by terms_left-1 terms maxing out at L?
            # The maximum possible sum for the remaining terms_left-1 terms is (terms_left-1)/L (if all were 1/L, but they must be distinct and >= d+1).
            # Actually, the largest possible sum for the remainder is achieved by the smallest possible denominators.
            # But we are constrained by d >= min_d.
            
            # Let's just iterate d.
            # To make it faster, we can start from the greedy choice if it's >= min_d.
            
            # Lower bound for d:
            # We need rem - 1/d > 0 => d > 1/rem
            # Also d >= min_d
            
            # Upper bound for d:
            # The remaining remainder is rem - 1/d.
            # This must be representable by terms_left-1 fractions with denominators in [d+1, L].
            # The maximum value of such a sum is (terms_left-1)/(d+1) (if we could use d+1 repeatedly, but we can't, but it's an upper bound).
            # So rem - 1/d <= (terms_left-1)/(d+1)
            # This gives a constraint on d.
            
            # For simplicity and speed, let's iterate d from min_d to L.
            # We will add a check to break if d is too large such that even the smallest possible remainder (0) is not reachable? No.
            # We break if 1/d is so small that rem - 1/d is still too large to be formed by terms_left-1 terms maxing at L?
            # Max sum of terms_left-1 terms with denominators >= d+1 is roughly (terms_left-1)/(d+1).
            # If rem - 1/d > (terms_left-1)/(d+1), then it's impossible.
            
            d_start = max(min_d, (1 / rem).numerator if rem.denominator == 1 else int(1/rem) + 1)
            # Careful with 1/rem calculation.
            # d > 1/rem => d >= floor(1/rem) + 1
            
            if rem.numerator == 0: return None # Should not happen
            
            d_min = max(min_d, (rem.denominator + rem.numerator - 1) // rem.numerator) # This is the greedy bound for d where 1/d <= rem
            # Wait, 1/d <= rem => d >= 1/rem.
            # So d_min = max(min_d, ceil(1/rem))
            
            d = d_min
            while d <= L:
                if time.time() > deadline:
                    return None
                
                # Check if remainder is formable
                rem2 = rem - Fraction(1, d)
                if rem2 < 0:
                    break # d is too small? No, d is increasing, 1/d is decreasing, so rem2 is increasing.
                    # Wait. If d increases, 1/d decreases.
                    # If rem2 < 0, it means 1/d > rem.
                    # As d increases, 1/d gets smaller, so rem2 becomes less negative, then positive.
                    # So if rem2 < 0, we should continue to larger d.
                    # If rem2 == 0, we need 0 terms left. If terms_left > 1, invalid (distinctness? No, just extra terms of 0? No, denominators must be positive).
                    # So rem2 must be > 0.
                    
                if rem2 == 0:
                    # We used 1 term, but we have terms_left - 1 left.
                    # We cannot use 0 denominators.
                    # So this is only valid if terms_left == 1. But we handled that above.
                    # So skip.
                    pass
                elif rem2 > 0:
                    # Check upper bound feasibility
                    # Max sum of terms_left-1 terms with denominators > d is (terms_left-1)/(d+1)
                    # If rem2 > (terms_left-1)/(d+1), then it's impossible to form rem2 with such terms.
                    max_possible = Fraction(terms_left - 1, d + 1)
                    if rem2 > max_possible:
                        break # For larger d, 1/d is smaller, rem2 is larger, max_possible is smaller. So it will never work.
                    
                    current.append(d)
                    res = rec(rem2, d + 1, terms_left - 1, current)
                    current.pop()
                    if res is not None:
                        return res
                d += 1
            return None

        return rec(target, 1, max_terms, [])

    # Try increasing max limits
    # We know a solution exists with max < 1736.
    # Let's try 50, 100, 200, 500, 1000, 2000.
    
    limits = [50, 100, 150, 200, 300, 500, 800, 1200, 1736, 2000, 5000, 10000]
    
    for L in limits:
        if time.time() > deadline:
            break
            
        # Try with different max_terms
        for terms in range(2, 6):
            if time.time() > deadline:
                break
            
            sol = find_solution_with_max_limit(L, terms)
            if sol:
                # Validate
                s = sum(Fraction(1, d) for d in sol)
                if s == target and len(sol) == len(set(sol)) and all(1 <= d <= 10_000_000 for d in sol):
                    if max(sol) < best_max:
                        best_max = max(sol)
                        best_solution = sol
                        # If we found a very small one, we might want to keep searching for even smaller?
                        # But the loop increases L. If we found a solution with L, we can break this inner loop for this L.
                        # But maybe a smaller L was skipped? No, we iterate L in increasing order.
                        # So the first solution found for a given L is the best for that L?
                        # Not necessarily, but we are minimizing max denominator.
                        # If we find a solution with max <= L, we record it.
                        # We should continue searching smaller L? No, L is increasing.
                        # So if we found a solution at L=100, best_max <= 100.
                        # We should then try L=101? No, the next L is 150.
                        # We might miss a solution with max=105 if we jump to 150.
                        # But we are looking for the GLOBAL minimum.
                        # So we should find the smallest L for which a solution exists.
                        # If we find a solution at L, we can stop increasing L?
                        # No, because we might find a solution with max < L in a later iteration? No, L is increasing.
                        # If a solution exists with max=105, it would have been found when L=105 (if we tested it).
                        # Since we jump 100->150, we might miss 105.
                        # To be safe, if we find a solution at L, we should try to refine L downwards?
                        # Or just accept the first found solution as a good candidate and maybe do a local search?
                        
                        # Given the time constraint, accepting the first found solution in the increasing L sequence is a good heuristic.
                        # We can break the L loop if we find a solution with a reasonably small max.
                        if best_max <= 200:
                            break
                break # Break terms loop, try next L? Or break all?
            # If no solution found for this L and terms, try next terms
        if best_max <= 200:
            break
            
    if best_solution is None:
        # Fallback to greedy
        r = target
        values = []
        used = set()
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            while d in used:
                d += 1
            values.append(d)
            used.add(d)
            r -= Fraction(1, d)
        best_solution = values
        
    return best_solution

# EVOLVE-BLOCK-END
