"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    """
    Returns the set of denominators for the Egyptian fraction expansion of 5/31
    that minimizes the largest denominator.
    Uses number-theoretic divisor enumeration for 3-term solutions,
    and systematic search for 4+ terms.
    """
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s limit
    
    target = Fraction(5, 31)
    best_max_den = 248  # current known solution
    best_solution = [8, 31, 248]
    
    # For 3-term Egyptian fractions: 1/a + 1/b + 1/c = 5/31
    # This means 1/b + 1/c = 5/31 - 1/a = (5a - 31)/(31a)
    # So bc = 31a(5a-31) / (b+c) ... not quite right
    # Better: 1/b + 1/c = (5a-31)/(31a)
    # => (b+c)/(bc) = (5a-31)/(31a)
    # => 31a(b+c) = bc(5a-31)
    # Let k = 5a-31, so 31a(b+c) = k*bc
    # => k*bc - 31a*b - 31a*c = 0
    # Multiply by k: k^2*bc - 31a*k*b - 31a*k*c = 0
    # Add (31a)^2: (kb - 31a)(kc - 31a) = (31a)^2
    # So kb - 31a and kc - 31a are divisors of (31a)^2
    
    def find_3_term_solutions(max_den_limit):
        """Find all 3-term solutions with max denominator <= max_den_limit."""
        solutions = []
        # a must be at least 3 (since 1/a <= 5/31, a >= 31/5 = 6.2, so a >= 7)
        # Actually 1/a < 5/31 => a > 31/5 = 6.2, so a >= 7
        # And a <= max_den_limit
        for a in range(7, max_den_limit + 1):
            k = 5 * a - 31  # must be positive, so a >= 7
            if k <= 0:
                continue
            
            # (kb - 31a)(kc - 31a) = (31a)^2
            n = (31 * a) ** 2
            # Find all divisor pairs (d1, d2) of n such that d1 * d2 = n
            # and b = (d1 + 31a) / k, c = (d2 + 31a) / k are positive integers
            # with b, c > a (for distinctness, we'll sort later)
            
            # Iterate over divisors of n
            import math
            sqrt_n = int(math.isqrt(n))
            for d1 in range(1, sqrt_n + 1):
                if n % d1 == 0:
                    d2 = n // d1
                    # Check if b and c are integers
                    num_b = d1 + 31 * a
                    num_c = d2 + 31 * a
                    if num_b % k == 0 and num_c % k == 0:
                        b = num_b // k
                        c = num_c // k
                        if b > 0 and c > 0 and b <= max_den_limit and c <= max_den_limit:
                            # Ensure distinctness
                            if len(set([a, b, c])) == 3:
                                sol = sorted([a, b, c])
                                if max(sol) <= max_den_limit:
                                    solutions.append(sol)
                    # Also check the swapped pair (d2, d1) gives same result
                    # but we need to check if d1 != d2 for the other ordering
                    if d1 != d2:
                        num_b2 = d2 + 31 * a
                        num_c2 = d1 + 31 * a
                        if num_b2 % k == 0 and num_c2 % k == 0:
                            b2 = num_b2 // k
                            c2 = num_c2 // k
                            if b2 > 0 and c2 > 0 and b2 <= max_den_limit and c2 <= max_den_limit:
                                if len(set([a, b2, c2])) == 3:
                                    sol = sorted([a, b2, c2])
                                    if max(sol) <= max_den_limit:
                                        solutions.append(sol)
        return solutions
    
    # First, try to find 3-term solutions with smaller max denominator
    # We know [8, 31, 248] has max 248. Let's search for smaller ones.
    # Start from max_den = 247 down to a reasonable minimum
    for max_d in range(247, 50, -1):
        if time.time() > deadline:
            break
        sols = find_3_term_solutions(max_d)
        if sols:
            # Found a solution with max denominator <= max_d
            best_sol = min(sols, key=lambda s: max(s))
            if max(best_sol) < best_max_den:
                best_max_den = max(best_sol)
                best_solution = best_sol
            break
    
    # If no better 3-term solution found, try 4-term solutions
    # For 4 terms: 1/a + 1/b + 1/c + 1/d = 5/31
    # This is more complex. Let's use a systematic search.
    # We can fix a and b, then solve for c and d:
    # 1/c + 1/d = 5/31 - 1/a - 1/b = (5ab - 31b - 31a)/(31ab)
    # Let R = 5ab - 31b - 31a, S = 31ab
    # Then (c+d)/(cd) = R/S
    # => S(c+d) = R*cd
    # => R*cd - S*c - S*d = 0
    # => (Rc - S)(Rd - S) = S^2
    
    def find_4_term_solutions(max_den_limit):
        """Find 4-term solutions with max denominator <= max_den_limit."""
        solutions = []
        # a >= 7 (same as before)
        # b > a
        # c > b, d > c
        for a in range(7, max_den_limit + 1):
            if time.time() > deadline:
                return solutions
            for b in range(a + 1, max_den_limit + 1):
                if time.time() > deadline:
                    return solutions
                # R = 5ab - 31b - 31a
                R = 5 * a * b - 31 * b - 31 * a
                S = 31 * a * b
                if R <= 0:
                    continue
                # (Rc - S)(Rd - S) = S^2
                n = S * S
                import math
                sqrt_n = int(math.isqrt(n))
                for d1 in range(1, sqrt_n + 1):
                    if n % d1 == 0:
                        d2 = n // d1
                        num_c = d1 + S
                        num_d = d2 + S
                        if num_c % R == 0 and num_d % R == 0:
                            c = num_c // R
                            d = num_d // R
                            if c > b and d > c and c <= max_den_limit and d <= max_den_limit:
                                if len(set([a, b, c, d])) == 4:
                                    sol = sorted([a, b, c, d])
                                    if max(sol) <= max_den_limit:
                                        solutions.append(sol)
                        if d1 != d2:
                            num_c2 = d2 + S
                            num_d2 = d1 + S
                            if num_c2 % R == 0 and num_d2 % R == 0:
                                c2 = num_c2 // R
                                d2_val = num_d2 // R
                                if c2 > b and d2_val > c2 and c2 <= max_den_limit and d2_val <= max_den_limit:
                                    if len(set([a, b, c2, d2_val])) == 4:
                                        sol = sorted([a, b, c2, d2_val])
                                        if max(sol) <= max_den_limit:
                                            solutions.append(sol)
        return solutions
    
    # Try 4-term solutions if 3-term didn't improve
    if best_max_den == 248:
        for max_d in range(247, 50, -1):
            if time.time() > deadline:
                break
            sols = find_4_term_solutions(max_d)
            if sols:
                best_sol = min(sols, key=lambda s: max(s))
                if max(best_sol) < best_max_den:
                    best_max_den = max(best_sol)
                    best_solution = best_sol
                break
    
    return best_solution
# EVOLVE-BLOCK-END
