"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    deadline = time.time() + 55  # safety margin before 60s limit
    
    target = Fraction(5, 31)
    
    # Known solution: 5/31 = 1/8 + 1/31 + 1/248, max = 248
    best = [8, 31, 248]
    best_max = 248
    
    def check_and_update(denoms):
        nonlocal best, best_max
        m = max(denoms)
        if m < best_max:
            total = sum(Fraction(1, d) for d in denoms)
            if total == target and len(set(denoms)) == len(denoms) and all(1 <= d <= 10000000 for d in denoms):
                best_max = m
                best = list(denoms)
    
    # Strategy: iterate C from lower bound upward, check if decomposition exists with all denoms <= C
    # Lower bound: need sum of reciprocals = 5/31 with distinct denoms in [1,C]
    # Minimum C is at least ceil(31/5) = 7, but need multiple terms
    
    def can_decompose(rem, min_denom, max_denom, remaining_terms, used):
        """Check if rem can be decomposed into sum of remaining_terms unit fractions
        with denominators in [min_denom, max_denom], not in used."""
        if remaining_terms == 0:
            return rem == 0
        if rem <= 0:
            return False
        # Lower bound on next denominator: 1/d < rem → d > 1/rem
        # Upper bound: need remaining_terms/d >= rem → d <= remaining_terms/rem
        # Also d must be >= min_denom and <= max_denom
        
        # For the smallest possible denominator, we need 1/d <= rem, so d >= 1/rem
        # For the largest possible, we need remaining_terms/d >= rem, so d <= remaining_terms/rem
        
        # d_min = max(min_denom, floor(1/rem) + 1) if rem < 1, else min_denom
        # Actually: 1/d <= rem means d >= 1/rem. So d_min = max(min_denom, ceil(1/rem))
        # But we also need remaining_terms * (1/d_min) >= rem for feasibility
        
        # d must satisfy: 1/d <= rem → d >= 1/rem
        # And: remaining_terms/d >= rem → d <= remaining_terms/rem
        
        if rem.numerator > 0:
            d_min = max(min_denom, (1 + rem.denominator - 1) // rem.numerator)  # ceil(1/rem)
            # Actually ceil(1/rem) = ceil(rem.denominator / rem.numerator)
            d_min = max(min_denom, (rem.denominator + rem.numerator - 1) // rem.numerator)
            d_max = min(max_denom, (remaining_terms * rem.denominator) // rem.numerator)
        else:
            return False
        
        if d_min > d_max:
            return False
        
        # Try denominators from d_min to d_max
        for d in range(d_min, d_max + 1):
            if d in used:
                continue
            new_rem = rem - Fraction(1, d)
            if new_rem < 0:
                continue
            if can_decompose(new_rem, d + 1, max_denom, remaining_terms - 1, used | {d}):
                return True
        return False
    
    # Try increasing C from lower bound
    # Lower bound: for 3 terms, we showed min max is 248. For more terms, could be less.
    # Let's try C from 20 to best_max-1
    
    for C in range(20, best_max):
        if time.time() > deadline:
            break
        
        # Try different numbers of terms
        # Maximum number of terms with denominators in [1,C] distinct: C terms
        # But sum of 1/1 + 1/2 + ... + 1/C = H(C) ~ ln(C)
        # We need sum = 5/31 ≈ 0.161
        # H(7) ≈ 2.59, H(10) ≈ 2.93, so we need many small terms or few large terms
        
        # For a given C, try term counts from 2 to min(C, 256)
        # Start with smaller term counts for speed
        
        # Actually, let's be smarter: for a given C, the minimum sum with k distinct denoms from [1,C]
        # is 1/C + 1/(C-1) + ... + 1/(C-k+1)
        # The maximum sum is 1/1 + 1/2 + ... + 1/k
        
        # We need 5/31 to be achievable.
        
        # Try term counts from 3 to min(C, 20)
        max_terms = min(C, 20)
        for n_terms in range(3, max_terms + 1):
            if time.time() > deadline:
                break
            
            # Check feasibility: can we get sum = 5/31 with n_terms distinct denoms in [1,C]?
            # Min possible sum: 1/C + 1/(C-1) + ... + 1/(C-n_terms+1)
            min_sum = sum(Fraction(1, C - i) for i in range(n_terms))
            # Max possible sum: 1/1 + 1/2 + ... + 1/n_terms
            max_sum = sum(Fraction(1, i) for i in range(1, n_terms + 1))
            
            if min_sum > target or max_sum < target:
                continue
            
            # Try to find a solution
            # Use backtracking with pruning
            def backtrack(rem, start, terms_left, current_denoms):
                nonlocal best, best_max
                if time.time() > deadline:
                    return False
                if terms_left == 0:
                    return rem == 0
                if rem <= 0:
                    return False
                
                # d must satisfy: 1/d <= rem → d >= 1/rem
                d_min = max(start, (rem.denominator + rem.numerator - 1) // rem.numerator)
                # remaining_terms/d >= rem → d <= remaining_terms/rem
                d_max = min(C, (terms_left * rem.denominator) // rem.numerator)
                
                if d_min > d_max:
                    return False
                
                # Additional pruning: even with the largest possible remaining terms, can we reach rem?
                # Sum of 1/(d_max) + 1/(d_max+1) + ... for remaining terms is an upper bound on what we can add
                # Actually, the maximum we can add with terms_left terms starting from d_min is
                # 1/d_min + 1/(d_min+1) + ... but that's not quite right since they must be >= d_min
                
                # Simpler: if terms_left * (1/d_min) < rem, we can't reach it
                if Fraction(terms_left, 1) * Fraction(1, d_min) < rem:
                    return False
                
                # If 1/d_max + 1/(d_max+1) + ... + 1/(d_max + terms_left - 1) < rem, can't reach it
                # This is a tighter lower bound check
                max_possible = sum(Fraction(1, d_max + i) for i in range(terms_left))
                if max_possible < rem:
                    return False
                
                for d in range(d_min, d_max + 1):
                    if len(current_denoms) + 1 > 256:
                        return False
                    new_rem = rem - Fraction(1, d)
                    if new_rem < 0:
                        continue
                    current_denoms.append(d)
                    if backtrack(new_rem, d + 1, terms_left - 1, current_denoms):
                        return True
                    current_denoms.pop()
                return False
            
            found = backtrack(target, 1, n_terms, [])
            if found:
                # Extract the solution from the backtracking
                # We need to modify to capture the solution
                pass
        
    # The above backtracking doesn't capture the solution. Let me rewrite.
    
    def find_solution(C, n_terms):
        """Find a decomposition of target into n_terms distinct unit fractions with denoms in [1,C]."""
        result = [None]
        
        def backtrack(rem, start, terms_left, current):
            nonlocal best, best_max
            if time.time() > deadline:
                return False
            if terms_left == 0:
                return rem == 0
            if rem <= 0:
                return False
            
            d_min = max(start, (rem.denominator + rem.numerator - 1) // rem.numerator)
            d_max = min(C, (terms_left * rem.denominator) // rem.numerator)
            
            if d_min > d_max:
                return False
            
            if Fraction(terms_left, 1) * Fraction(1, d_min) < rem:
                return False
            
            max_possible = sum(Fraction(1, d_max + i) for i in range(terms_left))
            if max_possible < rem:
                return False
            
            for d in range(d_min, d_max + 1):
                new_rem = rem - Fraction(1, d)
                if new_rem < 0:
                    continue
                current.append(d)
                if backtrack(new_rem, d + 1, terms_left - 1, current):
                    result[0] = list(current)
                    return True
                current.pop()
            return False
        
        if backtrack(target, 1, n_terms, []):
            return result[0]
        return None
    
    # Try increasing C
    for C in range(20, best_max):
        if time.time() > deadline:
            break
        
        max_terms = min(C, 15)
        for n_terms in range(3, max_terms + 1):
            if time.time() > deadline:
                break
            
            # Feasibility check
            min_sum = sum(Fraction(1, C - i) for i in range(n_terms))
            if min_sum > target:
                continue
            max_sum = sum(Fraction(1, i) for i in range(1, n_terms + 1))
            if max_sum < target:
                continue
            
            sol = find_solution(C, n_terms)
            if sol is not None:
                check_and_update(sol)
                break  # Found solution for this C, move to next C
    
    return best
# EVOLVE-BLOCK-END
