import time
from fractions import Fraction
import random

def construct():
    target = Fraction(5, 31)
    deadline = time.time() + 85
    best_solution = None
    best_max_den = float('inf')
    
    def solve_two_terms(r):
        """
        Solve r = 1/a + 1/b for distinct integers a, b with a > min_a.
        Returns (a, b) with a < b, or None.
        """
        if r <= 0:
            return None
        p = r.numerator
        q = r.denominator
        N = q * q
        limit = int(N ** 0.5) + 1
        
        for x in range(1, limit + 1):
            if N % x != 0:
                continue
            y = N // x
            if (x + q) % p != 0:
                continue
            if (y + q) % p != 0:
                continue
            a = (x + q) // p
            b = (y + q) // p
            if a == b:
                continue
            if a < 1 or b < 1:
                continue
            if a > b:
                a, b = b, a
            return (a, b)
        return None

    def update_best(sol):
        nonlocal best_solution, best_max_den
        if not sol:
            return
        if len(sol) > 256:
            return
        if len(set(sol)) != len(sol):
            return
        for d in sol:
            if d < 1 or d > 10000000:
                return
        s = sum(Fraction(1, d) for d in sol)
        if s != target:
            return
        m = max(sol)
        if m < best_max_den:
            best_max_den = m
            best_solution = sol[:]

    # Phase 1: Systematic search for 3-term solutions
    # 1/d1 + 1/d2 + 1/d3 = 5/31
    # d1 must be in [7, 10] since 3/7 > 5/31 and 3/11 < 5/31? No.
    # 3/d1 >= 5/31 => d1 <= 3*31/5 = 18.6. So d1 in [7, 18].
    for d1 in range(7, 19):
        if time.time() > deadline:
            break
        r1 = target - Fraction(1, d1)
        if r1 <= 0:
            continue
        pair = solve_two_terms(r1)
        if pair:
            d2, d3 = pair
            if d1 < d2 < d3:
                update_best([d1, d2, d3])

    # Phase 2: Systematic search for 4-term solutions
    # 1/d1 + 1/d2 + 1/d3 + 1/d4 = 5/31
    # d1 in [7, 15] (since 4/7 > 5/31 and 4/16 < 5/31? 4/16=1/4=0.25, 5/31~0.16. So 4/d1 >= 5/31 => d1 <= 4*31/5=24.8)
    # But we want small max denominator, so d1 should be small.
    # Let's try d1 in [7, 15]
    for d1 in range(7, 16):
        if time.time() > deadline:
            break
        r1 = target - Fraction(1, d1)
        if r1 <= 0:
            continue
        # d2 > 1/r1
        min_d2 = int(1 / r1) + 1
        if min_d2 <= d1:
            min_d2 = d1 + 1
        max_d2 = 50 if best_max_den == float('inf') else min(50, best_max_den)
        for d2 in range(min_d2, max_d2 + 1):
            if time.time() > deadline:
                break
            r2 = r1 - Fraction(1, d2)
            if r2 <= 0:
                continue
            pair = solve_two_terms(r2)
            if pair:
                d3, d4 = pair
                if d2 < d3 < d4 and d1 not in (d3, d4) and d2 not in (d3, d4):
                    update_best([d1, d2, d3, d4])

    # Phase 3: Randomized local search to find better 4 or 5 term solutions
    # We try to improve the current best by fixing some denominators and re-solving the rest
    random.seed(42)
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        # Try random initial denominators
        num_terms = random.choice([3, 4, 5])
        # Generate random increasing sequence of denominators
        d1 = random.randint(7, 12)
        denoms = [d1]
        rem = target - Fraction(1, d1)
        success = True
        for i in range(1, num_terms - 1):
            if rem <= 0:
                success = False
                break
            min_d = denoms[-1] + 1
            # Upper bound for next denominator
            # We need (num_terms - i - 1) terms to sum to rem - 1/d_next
            # So rem - 1/d_next > 0 => 1/d_next < rem => d_next > 1/rem
            # Also rem - 1/d_next >= (num_terms - i - 1) / max_possible
            # For random search, just pick randomly in a reasonable range
            max_d = 100 if best_max_den == float('inf') else best_max_den
            if max_d < min_d + 10:
                max_d = min_d + 50
            d_next = random.randint(min_d, max_d)
            denoms.append(d_next)
            rem -= Fraction(1, d_next)
        
        if not success:
            continue
        
        if rem > 0:
            # Try to complete with 2 terms
            pair = solve_two_terms(rem)
            if pair:
                d_last, d_last2 = pair
                # Ensure distinctness and ordering
                all_denoms = denoms + [d_last, d_last2]
                if len(set(all_denoms)) == len(all_denoms):
                    if all(d > 0 and d <= 10000000 for d in all_denoms):
                        # Check sum
                        s = sum(Fraction(1, d) for d in all_denoms)
                        if s == target:
                            update_best(all_denoms)
        
        # Also try greedy completion from random start
        if time.time() > deadline:
            break

    # Phase 4: Try to improve by removing terms or swapping
    if best_solution:
        # Try greedy completion with different starting points
        for start in range(7, 15):
            if time.time() > deadline:
                break
            r = target - Fraction(1, start)
            if r <= 0:
                continue
            sol = [start]
            min_d = start + 1
            while r > 0 and time.time() < deadline:
                d = (r.denominator + r.numerator - 1) // r.numerator
                if d < min_d:
                    d = min_d
                sol.append(d)
                r -= Fraction(1, d)
                min_d = d + 1
            if r == 0:
                update_best(sol)

    # Fallback
    if best_solution is None:
        # Use a known solution: 5/31 = 1/7 + 1/55 + 1/11935 + ...? No, let's use a simpler one
        # 5/31 = 1/7 + 4/217. 4/217 = 1/55 + 3/11935. 3/11935 = 1/4000 + ...?
        # Let's just use greedy
        r = target
        sol = []
        used = set()
        min_d = 1
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            if d < min_d:
                d = min_d
            while d in used:
                d += 1
            sol.append(d)
            used.add(d)
            r -= Fraction(1, d)
            min_d = d + 1
        best_solution = sol
        best_max_den = max(sol)

    return best_solution
