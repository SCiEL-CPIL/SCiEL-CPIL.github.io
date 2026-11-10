"""
Optimized Egyptian fraction search for 5/31:
Minimize the LARGEST denominator over 1..256 distinct terms.

Key optimizations:
- Float pre-filtering to skip most candidates
- Tight bounds using the remaining fraction
- Early termination when best_max is small
- 3, 4, 5, 6, 7 term searches with increasing depth
"""
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill

    T_num, T_den = 5, 31
    target = T_num / T_den

    best_max = float('inf')
    best_result = None

    def update_best(denoms):
        nonlocal best_max, best_result
        mx = max(denoms)
        if mx < best_max:
            best_max = mx
            best_result = list(denoms)

    def time_ok():
        return time.time() < deadline

    # --- 3-term search: 1/a + 1/b + 1/c = 5/31, a < b < c ---
    # a must satisfy: 1/a < 5/31 => a > 31/5 = 6.2, so a >= 7
    # Also 3/a > 5/31 => a < 93/5 = 18.6, so a <= 18
    for a in range(7, 19):
        if not time_ok():
            break
        r1 = target - 1.0/a
        if r1 <= 0:
            continue
        # 1/b + 1/c = r1, b > a
        # b >= ceil(1/(2*r1)) roughly, but also b > a
        # 2/b >= r1 => b <= 2/r1
        max_b = int(2.0 / r1) + 1
        min_b = a + 1
        for b in range(min_b, max_b + 1):
            r2 = r1 - 1.0/b
            if r2 <= 0:
                continue
            # 1/c = r2 => c = 1/r2
            c_float = 1.0 / r2
            c = int(round(c_float))
            # Check nearby integers
            for cc in (c - 1, c, c + 1):
                if cc <= b:
                    continue
                if abs(1.0/cc - r2) < 1e-9:
                    # Exact check
                    # 1/a + 1/b + 1/cc == 5/31?
                    lhs_num = T_den * (b * cc + a * cc + a * b)
                    rhs_num = T_num * a * b * cc
                    if lhs_num == rhs_num:
                        update_best([a, b, cc])

    # --- 4-term search ---
    for a in range(7, 19):
        if not time_ok():
            break
        r1 = target - 1.0/a
        if r1 <= 0:
            continue
        max_b = int(3.0 / r1) + 1
        min_b = a + 1
        for b in range(min_b, max_b + 1):
            if not time_ok():
                break
            r2 = r1 - 1.0/b
            if r2 <= 0:
                continue
            max_c = int(2.0 / r2) + 1
            min_c = b + 1
            for c in range(min_c, max_c + 1):
                r3 = r2 - 1.0/c
                if r3 <= 0:
                    continue
                d_float = 1.0 / r3
                d = int(round(d_float))
                for dd in (d - 1, d, d + 1):
                    if dd <= c:
                        continue
                    if dd >= best_max:
                        continue
                    if abs(1.0/dd - r3) < 1e-9:
                        # Exact check
                        lhs_num = T_den * (b*c*dd + a*c*dd + a*b*dd + a*b*c)
                        rhs_num = T_num * a * b * c * dd
                        if lhs_num == rhs_num:
                            update_best([a, b, c, dd])

    # --- 5-term search ---
    for a in range(7, 19):
        if not time_ok():
            break
        r1 = target - 1.0/a
        if r1 <= 0:
            continue
        max_b = int(4.0 / r1) + 1
        min_b = a + 1
        for b in range(min_b, max_b + 1):
            if not time_ok():
                break
            r2 = r1 - 1.0/b
            if r2 <= 0:
                continue
            max_c = int(3.0 / r2) + 1
            min_c = b + 1
            for c in range(min_c, max_c + 1):
                if not time_ok():
                    break
                r3 = r2 - 1.0/c
                if r3 <= 0:
                    continue
                max_d = int(2.0 / r3) + 1
                min_d = c + 1
                for d in range(min_d, max_d + 1):
                    r4 = r3 - 1.0/d
                    if r4 <= 0:
                        continue
                    e_float = 1.0 / r4
                    e = int(round(e_float))
                    for ee in (e - 1, e, e + 1):
                        if ee <= d:
                            continue
                        if ee >= best_max:
                            continue
                        if abs(1.0/ee - r4) < 1e-9:
                            # Exact check
                            lhs_num = T_den * (b*c*d*ee + a*c*d*ee + a*b*d*ee + a*b*c*ee + a*b*c*d)
                            rhs_num = T_num * a * b * c * d * ee
                            if lhs_num == rhs_num:
                                update_best([a, b, c, d, ee])

    # --- 6-term search (only if best_max is still large) ---
    if best_max > 40:
        for a in range(7, 19):
            if not time_ok():
                break
            r1 = target - 1.0/a
            if r1 <= 0:
                continue
            max_b = int(5.0 / r1) + 1
            min_b = a + 1
            for b in range(min_b, max_b + 1):
                if not time_ok():
                    break
                r2 = r1 - 1.0/b
                if r2 <= 0:
                    continue
                max_c = int(4.0 / r2) + 1
                min_c = b + 1
                for c in range(min_c, max_c + 1):
                    if not time_ok():
                        break
                    r3 = r2 - 1.0/c
                    if r3 <= 0:
                        continue
                    max_d = int(3.0 / r3) + 1
                    min_d = c + 1
                    for d in range(min_d, max_d + 1):
                        if not time_ok():
                            break
                        r4 = r3 - 1.0/d
                        if r4 <= 0:
                            continue
                        max_e = int(2.0 / r4) + 1
                        min_e = d + 1
                        for e in range(min_e, max_e + 1):
                            r5 = r4 - 1.0/e
                            if r5 <= 0:
                                continue
                            f_float = 1.0 / r5
                            f = int(round(f_float))
                            for ff in (f - 1, f, f + 1):
                                if ff <= e:
                                    continue
                                if ff >= best_max:
                                    continue
                                if abs(1.0/ff - r5) < 1e-9:
                                    lhs_num = T_den * (b*c*d*e*ff + a*c*d*e*ff + a*b*d*e*ff + a*b*c*e*ff + a*b*c*d*ff + a*b*c*d*e)
                                    rhs_num = T_num * a * b * c * d * e * ff
                                    if lhs_num == rhs_num:
                                        update_best([a, b, c, d, e, ff])

    if best_result is not None:
        return best_result

    # Fallback: greedy Egyptian fraction
    denoms = []
    num, den = T_num, T_den
    for _ in range(256):
        if num == 0:
            break
        d = int(math.ceil(den / float(num)))
        if d <= 0:
            d = 1
        denoms.append(d)
        # num/den - 1/d = (num*d - den)/(den*d)
        num = num * d - den
        den = den * d
        if num < 0:
            num = -num
            den = -den
        g = math.gcd(num, den)
        num //= g
        den //= g
    if denoms:
        return denoms
    return [7]

