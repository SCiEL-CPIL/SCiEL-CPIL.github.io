"""
Optimized Egyptian fraction search for 5/31 minimizing largest denominator.
Uses direct enumeration with integer arithmetic and tight bounds.
"""
import time
from math import gcd

def construct():
    start = time.time()
    deadline = start + 55.0  # safety margin before 60s kill

    # Target: 5/31
    # We want distinct denominators d1 < d2 < ... < dk with sum 1/di = 5/31
    # Minimize max(di) = dk

    # Known solution: 1/7 + 1/56 + 1/1736 = 5/31, max = 1736
    best_max = 1736
    best_result = [7, 56, 1736]

    def time_ok():
        return time.time() < deadline

    def try_3_terms(M):
        """Try to find 3 distinct denominators a < b < c <= M with 1/a+1/b+1/c = 5/31"""
        nonlocal best_max, best_result
        # 1/a + 1/b + 1/c = 5/31
        # a >= 7 (since 1/7 < 5/31 < 1/6... actually 1/6 > 5/31, so a >= 7)
        # For each a, b, check if c exists
        for a in range(7, M + 1):
            if not time_ok():
                return False
            # 1/b + 1/c = 5/31 - 1/a = (5a - 31)/(31a)
            rem_num = 5 * a - 31
            rem_den = 31 * a
            if rem_num <= 0:
                continue
            # Reduce
            g = gcd(rem_num, rem_den)
            rn = rem_num // g
            rd = rem_den // g
            # 1/b + 1/c = rn/rd => c = b*rd / (b*rn - rd)
            # Need b > a, b <= M, c > b, c <= M
            # b*rn - rd > 0 => b > rd/rn
            min_b = max(a + 1, rd // rn + 1)
            # c = b*rd / (b*rn - rd) <= M => b*rd <= M*(b*rn - rd) => b*rd <= M*b*rn - M*rd => b*(M*rn - rd) >= M*rd => b >= M*rd/(M*rn - rd)
            # Also c > b => b*rd/(b*rn - rd) > b => rd > b*rn - rd => 2*rd > b*rn => b < 2*rd/rn
            max_b = min(M, (2 * rd - 1) // rn)
            for b in range(min_b, max_b + 1):
                if not time_ok():
                    return False
                num_c = b * rd
                den_c = b * rn - rd
                if den_c <= 0:
                    continue
                if num_c % den_c != 0:
                    continue
                c = num_c // den_c
                if c <= b or c > M:
                    continue
                # Found valid: a, b, c all distinct, in [1, M]
                mx = c
                if mx < best_max:
                    best_max = mx
                    best_result = [a, b, c]
                    return True
        return False

    def try_4_terms(M):
        """Try to find 4 distinct denominators a < b < c < d <= M with sum = 5/31"""
        nonlocal best_max, best_result
        for a in range(7, M + 1):
            if not time_ok():
                return False
            rem_num = 5 * a - 31
            rem_den = 31 * a
            if rem_num <= 0:
                continue
            g = gcd(rem_num, rem_den)
            rn = rem_num // g
            rd = rem_den // g
            # Now find b < c < d with 1/b+1/c+1/d = rn/rd, b > a, all <= M
            # 1/b + 1/c + 1/d = rn/rd
            # For each b, find c, d with 1/c + 1/d = rn/rd - 1/b
            min_b = max(a + 1, rd // rn + 1)
            max_b = min(M, (3 * rd - 1) // rn)  # rough bound
            for b in range(min_b, max_b + 1):
                if not time_ok():
                    return False
                rn2 = rn * b - rd
                rd2 = rd * b
                if rn2 <= 0:
                    continue
                g2 = gcd(rn2, rd2)
                rn2 //= g2
                rd2 //= g2
                # 1/c + 1/d = rn2/rd2 => d = c*rd2/(c*rn2 - rd2)
                min_c = max(b + 1, rd2 // rn2 + 1)
                max_c = min(M, (2 * rd2 - 1) // rn2)
                for c in range(min_c, max_c + 1):
                    if not time_ok():
                        return False
                    num_d = c * rd2
                    den_d = c * rn2 - rd2
                    if den_d <= 0:
                        continue
                    if num_d % den_d != 0:
                        continue
                    d = num_d // den_d
                    if d <= c or d > M:
                        continue
                    mx = d
                    if mx < best_max:
                        best_max = mx
                        best_result = [a, b, c, d]
                        return True
        return False

    # Try increasing M from 7 upward
    # Start with 3 terms (fast), then 4 terms
    for M in range(7, 200):
        if not time_ok():
            break
        if M >= best_max:
            break
        if try_3_terms(M):
            if best_max <= M:
                break
        if try_4_terms(M):
            if best_max <= M:
                break

    # If we found a better solution, return it
    if best_result and best_max < 1736:
        return best_result

    # Try a bit more with 5 terms for small M
    def try_5_terms(M):
        nonlocal best_max, best_result
        # Only try if M is small enough to be feasible
        if M > 50:
            return False
        for a in range(7, M + 1):
            if not time_ok():
                return False
            rem_num = 5 * a - 31
            rem_den = 31 * a
            if rem_num <= 0:
                continue
            g = gcd(rem_num, rem_den)
            rn = rem_num // g
            rd = rem_den // g
            min_b = max(a + 1, rd // rn + 1)
            max_b = min(M, (4 * rd - 1) // rn)
            for b in range(min_b, max_b + 1):
                if not time_ok():
                    return False
                rn2 = rn * b - rd
                rd2 = rd * b
                if rn2 <= 0:
                    continue
                g2 = gcd(rn2, rd2)
                rn2 //= g2
                rd2 //= g2
                min_c = max(b + 1, rd2 // rn2 + 1)
                max_c = min(M, (3 * rd2 - 1) // rn2)
                for c in range(min_c, max_c + 1):
                    if not time_ok():
                        return False
                    rn3 = rn2 * c - rd2
                    rd3 = rd2 * c
                    if rn3 <= 0:
                        continue
                    g3 = gcd(rn3, rd3)
                    rn3 //= g3
                    rd3 //= g3
                    # 1/d = rn3/rd3 => d = rd3/rn3
                    if rn3 == 1:
                        d = rd3
                        if d > c and d <= M:
                            mx = d
                            if mx < best_max:
                                best_max = mx
                                best_result = [a, b, c, d]
                                return True
                    elif rd3 % rn3 == 0:
                        d = rd3 // rn3
                        if d > c and d <= M:
                            mx = d
                            if mx < best_max:
                                best_max = mx
                                best_result = [a, b, c, d]
                                return True
        return False

    for M in range(7, 60):
        if not time_ok():
            break
        if M >= best_max:
            break
        try_5_terms(M)

    return best_result
