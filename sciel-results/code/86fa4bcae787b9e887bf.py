"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def _solve_two(a_num, a_den, b_num, b_den, max_d):
    """
    Solve 1/x + 1/y = a_num/a_den + b_num/b_den for distinct positive integers x, y <= max_d.
    Returns (x, y) or None.
    """
    target = Fraction(a_num, a_den) + Fraction(b_num, b_den)
    # We want 1/x + 1/y = target
    # (x+y)/(xy) = target
    # xy * target = x + y
    # target * xy - x - y = 0
    # Multiply by 4:
    # 4 * target * xy - 4x - 4y = 0
    # (2*target*x - 2) * (2*target*y - 2) = 4 + 4*target
    # Let A = 2*target. Then (Ax-2)(Ay-2) = 4 + 2A
    # Note: This formula assumes target is a Fraction.
    
    # Alternative: iterate x.
    # 1/y = target - 1/x = (target*x - 1)/x
    # y = x / (target*x - 1)
    # For y to be integer, target*x - 1 must divide x.
    # Let t = target. t = N/D.
    # y = x / ((N*x - D)/D) = x * D / (N*x - D)
    # So N*x - D must divide x * D.
    # Let k = N*x - D. Then x = (k+D)/N.
    # k must divide ((k+D)/N) * D.
    # N*k | k*D + D^2.
    # Since N*k | k*D, we need N*k | D^2.
    # So k is a divisor of D^2.
    # And x = (k+D)/N must be an integer.
    # And y = x * D / k must be an integer.
    # And x != y, x <= max_d, y <= max_d.
    
    N = target.numerator
    D = target.denominator
    
    # Find divisors of D^2
    # D^2 can be large, but target is small (sum of two reciprocals of small ints).
    # Let's just iterate x up to max_d? No, max_d can be 10000.
    # Iterating divisors of D^2 is faster if D is small.
    # D is denominator of target.
    
    # Let's stick to iterating x for simplicity and safety, with a tight bound.
    # 1/x < target => x > 1/target.
    # 1/x > target/2 => x < 2/target.
    # So x is in (1/target, 2/target).
    
    x_min = (D // N) + 1
    if N * x_min <= D: # 1/x_min >= target
        x_min += 1
        
    x_max = (2 * D // N)
    if N * x_max >= 2 * D: # 1/x_max <= target/2
        x_max -= 1
        
    # Also x <= max_d
    x_max = min(x_max, max_d)
    
    if x_min > x_max:
        return None
        
    for x in range(x_min, x_max + 1):
        # Check if x is valid
        num_y = x * D
        den_y = N * x - D
        if den_y <= 0:
            continue
        if num_y % den_y != 0:
            continue
        y = num_y // den_y
        if y < 1:
            continue
        if y > max_d:
            continue
        if x == y:
            continue
        # We need distinct denominators. The caller ensures x is not in 'used'.
        # We should also ensure y is not in 'used'. The caller will check this.
        # But here we just return a candidate pair.
        return (x, y)
        
    return None

def _try_complete(r, used, max_d, start_d, depth=0, max_depth=5):
    """Try to complete r with at most max_depth more terms, all <= max_d, >= start_d, not in used."""
    if time.time() > _DEADLINE:
        return None
    if r == 0:
        return []
    if depth >= max_depth:
        return None
    if r < 0:
        return None
    
    k = max_depth - depth
    
    # If k == 1:
    if k == 1:
        # r must be 1/d for some integer d.
        if r.denominator % r.numerator != 0:
            return None
        d = r.denominator // r.numerator
        if d < start_d or d > max_d or d in used:
            return None
        return [d]
        
    # If k == 2:
    if k == 2:
        # Solve 1/x + 1/y = r
        # Use _solve_two logic directly for r
        # We need distinct x, y >= start_d, not in used.
        N = r.numerator
        D = r.denominator
        x_min = (D // N) + 1
        if N * x_min <= D:
            x_min += 1
        x_min = max(x_min, start_d)
        
        x_max = (2 * D // N)
        if N * x_max >= 2 * D:
            x_max -= 1
        x_max = min(x_max, max_d)
        
        if x_min > x_max:
            return None
            
        for x in range(x_min, x_max + 1):
            if x in used:
                continue
            num_y = x * D
            den_y = N * x - D
            if den_y <= 0:
                continue
            if num_y % den_y != 0:
                continue
            y = num_y // den_y
            if y < start_d or y > max_d:
                continue
            if y in used:
                continue
            if x == y:
                continue
            return [x, y]
        return None
        
    # General case k >= 3
    # Pruning: upper bound on sum
    upper_sum = 0
    d_check = start_d
    count = 0
    while count < k:
        if d_check not in used:
            upper_sum += Fraction(1, d_check)
            count += 1
        d_check += 1
        if d_check > max_d:
            break
    if r > upper_sum:
        return None
        
    d_min = max(start_d, (r.denominator // r.numerator) + 1)
    if r.numerator * d_min <= r.denominator:
        d_min += 1
        
    for d in range(d_min, max_d + 1):
        if time.time() > _DEADLINE:
            return None
        if d in used:
            continue
        if Fraction(1, d) >= r:
            continue
        new_r = r - Fraction(1, d)
        if new_r < 0:
            continue
        used.add(d)
        res = _try_complete(new_r, used, max_d, d + 1, depth + 1, max_depth)
        used.discard(d)
        if res is not None:
            return [d] + res
    return None

_DEADLINE = 0.0

def construct():
    global _DEADLINE
    start_time = time.time()
    _DEADLINE = start_time + 85.0  # Leave 5s buffer for safety (90s limit)
    
    target = Fraction(5, 31)
    
    # Known good solution: [8, 28, 1736]
    best_solution = [8, 28, 1736]
    best_max = 1736
    
    # Systematic search for better solutions
    # Iterate d1 from 7 to 50
    # For each d1, iterate d2 from d2_min to d2_min + 200
    # Use _try_complete for the rest with max_depth = 4 (total 6 terms)
    # Limit max_d to best_max - 1
    
    for d1 in range(7, 51):
        if time.time() > _DEADLINE - 2:
            break
        r1 = target - Fraction(1, d1)
        if r1 <= 0:
            continue
        
        d2_min = (r1.denominator // r1.numerator) + 1
        if r1.numerator * d2_min <= r1.denominator:
            d2_min += 1
            
        # Limit d2 search range
        # If d2 is too large, r2 is close to r1, which might be hard to solve with few terms
        # But let's try a reasonable range
        # d2_max can be bounded by 1/r1 * 2 roughly? No, 1/d2 < r1 => d2 > 1/r1.
        # Also 1/d2 > r1/2 is not required, but helps.
        # Let's just go up to d2_min + 300 to be safe but fast.
        for d2 in range(d2_min, d2_min + 300):
            if time.time() > _DEADLINE - 2:
                break
            if d2 == d1:
                continue
            r2 = r1 - Fraction(1, d2)
            if r2 <= 0:
                continue
            
            used = {d1, d2}
            # Try to complete with up to 4 more terms (total 6)
            # We can also try smaller max_depth if we want to find shorter solutions,
            # but the task allows up to 256 terms. However, minimizing max denominator
            # often favors more terms with smaller denominators?
            # Actually, fewer terms usually means larger denominators.
            # Let's allow up to 6 terms total (4 more).
            res = _try_complete(r2, used, best_max - 1, d2 + 1, 0, 4)
            if res is not None:
                sol = [d1, d2] + res
                if len(set(sol)) == len(sol) and all(1 <= d <= 10000000 for d in sol):
                    s = sum(Fraction(1, d) for d in sol)
                    if s == target:
                        max_d = max(sol)
                        if max_d < best_max:
                            best_max = max_d
                            best_solution = sol
                            # Update deadline? No, keep searching.
                            # The next call will use the smaller best_max.

    # If we have time, try random restarts or different d1 ranges
    if time.time() < _DEADLINE - 5:
        # Try d1 > 50
        for d1 in range(51, 100):
            if time.time() > _DEADLINE - 2:
                break
            r1 = target - Fraction(1, d1)
            if r1 <= 0:
                continue
            d2_min = (r1.denominator // r1.numerator) + 1
            if r1.numerator * d2_min <= r1.denominator:
                d2_min += 1
            for d2 in range(d2_min, d2_min + 100):
                if time.time() > _DEADLINE - 2:
                    break
                if d2 == d1:
                    continue
                r2 = r1 - Fraction(1, d2)
                if r2 <= 0:
                    continue
                used = {d1, d2}
                res = _try_complete(r2, used, best_max - 1, d2 + 1, 0, 4)
                if res is not None:
                    sol = [d1, d2] + res
                    if len(set(sol)) == len(sol):
                        s = sum(Fraction(1, d) for d in sol)
                        if s == target:
                            max_d = max(sol)
                            if max_d < best_max:
                                best_max = max_d
                                best_solution = sol

    return best_solution
# EVOLVE-BLOCK-END
