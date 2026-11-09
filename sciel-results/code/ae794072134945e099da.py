"""Rewritten program using DFS with branch and bound to minimize the largest denominator."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import sys
    sys.setrecursionlimit(10000)

    target = Fraction(5, 31)
    max_denom_limit = 10**7
    max_terms = 256
    
    # Find an initial solution using greedy to set a good upper bound
    def greedy_solution(rem):
        terms = []
        while rem > 0:
            if rem == 0:
                break
            d = (rem.denominator + rem.numerator - 1) // rem.numerator
            if d > max_denom_limit:
                break
            terms.append(d)
            rem -= Fraction(1, d)
        return terms

    init_terms = greedy_solution(target)
    best_max = max(init_terms) if init_terms else max_denom_limit + 1
    best_solution = init_terms[:]
    
    def dfs(rem, min_d, current_terms, current_max):
        nonlocal best_solution, best_max
        
        if rem == 0:
            if current_max < best_max:
                best_max = current_max
                best_solution = current_terms[:]
            return True

        if len(current_terms) >= max_terms:
            return False
            
        if current_max >= best_max:
            return False

        # Lower bound for next denominator
        # 1/d <= rem  => d >= 1/rem
        L = max(min_d, (rem.denominator + rem.numerator - 1) // rem.numerator)
        
        # Upper bound for next denominator
        # We need rem - 1/d to be representable by k = max_terms - len(current_terms) terms
        # The maximum sum of k distinct denominators starting from d+1 is approx k/(d+1)
        # rem - 1/d <= k/(d+1)
        # rem(d+1) - (1 + 1/d) <= k
        # rem*d + rem - 1 - 1/d <= k
        # rem*d <= k - rem + 1 + 1/d
        # d <= (k - rem + 1 + 1/d) / rem
        # Approximation: d <= (k - rem + 1) / rem
        k = max_terms - len(current_terms)
        
        # Calculate U
        # (k - rem + 1) / rem = k/rem - 1 + 1/rem
        # Let's just use a slightly looser bound to be safe:
        # d * rem <= k - rem + 1 + 1
        # d <= (k - rem + 2) / rem
        # Since rem is a fraction, (k - rem + 2) / rem = (k+2)/rem - 1
        U_num = (k + 2) * rem.denominator - rem.numerator
        U_den = rem.numerator
        U = U_num // U_den
        
        # Also d must be <= best_max - 1 to improve
        if U >= best_max:
            U = best_max - 1
            
        if U > max_denom_limit:
            U = max_denom_limit
            
        if L > U:
            return False

        # Iterate d from L to U
        # We want to minimize the largest denominator.
        # Trying smaller d first might lead to solutions with smaller max?
        # Not necessarily. But smaller d keeps the "budget" of small numbers available.
        for d in range(L, U + 1):
            next_rem = rem - Fraction(1, d)
            
            if next_rem < 0:
                break
                
            if next_rem == 0:
                if d < best_max:
                    best_max = d
                    best_solution = current_terms + [d]
                    return True
                
            new_max = max(current_max, d)
            if new_max >= best_max:
                continue
                
            if dfs(next_rem, d + 1, current_terms + [d], new_max):
                return True

        return False

    dfs(target, 1, [], 0)
    
    if best_solution:
        return best_solution
    else:
        # Fallback
        return init_terms
# EVOLVE-BLOCK-END
