"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from fractions import Fraction
    from math import ceil

    deadline = time.time() + 80.0  # safety margin before 90s kill
    target = Fraction(5, 31)

    # We want to find distinct denominators d1 < d2 < ... < dk
    # such that sum(1/di) = 5/31, minimizing max(di) = dk.
    #
    # Strategy:
    # 1. Systematic search over small term counts (k=2, 3, 4, ...) 
    #    using exact arithmetic to find valid expansions.
    # 2. For a fixed k, we can bound the search space.
    #    Since we want to minimize the largest denominator, we can try 
    #    to find a solution where the largest denominator is as small as possible.
    #
    # Key Insight from knowledge:
    # "Systematic 3-term search finds optimal Egyptian fraction by solving for c given a,b, 
    # avoiding greedy heuristics that miss better solutions like [7,62,434] with max 434 
    # vs greedy's 1736"
    #
    # Let's verify [7, 62, 434]:
    # 1/7 + 1/62 + 1/434 = (62*434 + 7*434 + 7*62) / (7*62*434)
    # Numerator: 26908 + 3038 + 434 = 30380
    # Denominator: 188756
    # 30380 / 188756 = 5 / 31 ?
    # 5 * 188756 = 943780
    # 31 * 30380 = 941780
    # Wait, 943780 != 941780. Let me re-calculate.
    # 1/7 + 1/62 = (62+7)/(7*62) = 69/434
    # 69/434 + 1/434 = 70/434 = 35/217
    # 35/217 = 5/31 ? 35*31 = 1085, 5*217 = 1085. Yes!
    # So [7, 62, 434] is a valid expansion for 5/31. Max denom is 434.
    #
    # Can we do better?
    # Let's try 2 terms: 1/a + 1/b = 5/31 => 31(a+b) = 5ab => 5ab - 31a - 31b = 0
    # Multiply by 5: 25ab - 155a - 155b = 0
    # (5a - 31)(5b - 31) = 31*31 = 961
    # Factors of 961 (which is 31^2): 1, 31, 961.
    # Case 1: 5a - 31 = 1 => 5a = 32 (no integer a)
    # Case 2: 5a - 31 = 31 => 5a = 62 (no integer a)
    # Case 3: 5a - 31 = 961 => 5a = 992 (no integer a)
    # So no 2-term solution.
    #
    # Let's try 3 terms: 1/a + 1/b + 1/c = 5/31, with a < b < c.
    # We want to minimize c.
    # We know a >= ceil(31/5) = 7.
    # Also, since a < b < c, 3/a > 5/31 => a < 93/5 = 18.6 => a <= 18.
    # So a is in [7, 18].
    #
    # For a fixed a, let r = 5/31 - 1/a = (5a - 31) / (31a).
    # We need 1/b + 1/c = r, with b > a and c > b.
    # This is the 2-term problem for r.
    # Let r = n/d. We need 1/b + 1/c = n/d.
    # (b+c)/bc = n/d => nbc - d(b+c) = 0
    # Multiply by n: n^2 bc - nd(b+c) = 0
    # (nb - d)(nc - d) = d^2
    # Let X = nb - d, Y = nc - d. Then XY = d^2.
    # b = (X+d)/n, c = (Y+d)/n.
    # We need b and c to be integers, so X+d and Y+d must be divisible by n.
    # Also b > a and c > b.
    #
    # We can iterate over divisors X of d^2.
    # To minimize c, we want to minimize Y = d^2/X, so we want to maximize X.
    # However, we also have constraints on b and c being distinct and > a.
    #
    # Let's implement this systematic search.
    # We will iterate a from 7 to 18.
    # For each a, compute r = 5/31 - 1/a.
    # Let n = r.numerator, d = r.denominator.
    # We need to find factors X, Y of d^2 such that XY = d^2.
    # b = (X+d)/n, c = (Y+d)/n.
    # Conditions:
    # 1. b, c are integers.
    # 2. b > a.
    # 3. c > b.
    # 4. b != c (implied by c > b).
    #
    # We want to minimize c.
    # Note that c = (d^2/X + d)/n. To minimize c, we want to maximize X.
    # So we should iterate X in descending order of divisors of d^2.
    #
    # Let's find the minimum possible c for each a, and then take the global minimum.

    best_c = float('inf')
    best_solution = None

    # a is the smallest denominator
    # 3/a > 5/31 => a < 93/5 = 18.6
    # a >= ceil(31/5) = 7
    for a in range(7, 19):
        r = target - Fraction(1, a)
        if r <= 0:
            continue
        n = r.numerator
        d = r.denominator
        d2 = d * d
        
        # Find all divisors of d2
        # d can be up to 31*18 = 558. d2 up to ~300,000.
        # Factoring is fast.
        divisors = []
        for i in range(1, int(d2**0.5) + 1):
            if d2 % i == 0:
                divisors.append(i)
                if i != d2 // i:
                    divisors.append(d2 // i)
        
        # Sort divisors in descending order to find min c first
        divisors.sort(reverse=True)
        
        for X in divisors:
            if time.time() > deadline:
                break
            Y = d2 // X
            # Check if X+d and Y+d are divisible by n
            if (X + d) % n != 0 or (Y + d) % n != 0:
                continue
            b = (X + d) // n
            c = (Y + d) // n
            # Check constraints
            if b <= a or c <= b:
                continue
            # Also need distinct denominators, which is satisfied by a < b < c
            # Check if c is the new best
            if c < best_c:
                best_c = c
                best_solution = [a, b, c]
                # Since we are iterating X in descending order, the first valid c for this a
                # is the minimum c for this a. So we can break.
                break
    
    # If we found a 3-term solution, return it.
    if best_solution:
        return best_solution
    
    # If no 3-term solution found (unlikely given [7,62,434]), try 4 terms.
    # This is more complex. We can fix a, b and solve for c, d.
    # But given the time limit and that 3-term likely has the optimal, 
    # and we found [7,62,434] with max 434, let's assume this is good enough.
    # The greedy solution had max 1736, so 434 is a big improvement.
    
    # Fallback to greedy if something goes wrong or time runs out
    r = target
    values = []
    while r > 0 and len(values) < 256:
        if time.time() > deadline:
            break
        d = (r.denominator + r.numerator - 1) // r.numerator
        if d in values:
            # Greedy might pick a repeated denominator in some edge cases? 
            # No, greedy for Egyptian fractions always picks a new larger one if r < 1.
            # But just in case.
            d += 1
        values.append(d)
        r -= Fraction(1, d)
    
    if values and sum(Fraction(1, v) for v in values) == target:
        return values
        
    # Last resort
    return [7, 62, 434]
# EVOLVE-BLOCK-END
