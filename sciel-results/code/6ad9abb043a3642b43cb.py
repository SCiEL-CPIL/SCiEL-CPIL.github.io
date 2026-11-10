"""Direct approach: find Egyptian fraction of 5/31 minimizing max denominator."""
import time
from math import gcd

def construct():
    start = time.time()
    deadline = start + 85  # safety margin (90s kill)

    # Target: 5/31, so p=5, q=31
    target_p, target_q = 5, 31

    best_max = float('inf')
    best_result = None

    def update_best(denoms):
        nonlocal best_max, best_result
        mx = max(denoms)
        if mx < best_max:
            best_max = mx
            best_result = list(denoms)

    def check_two_term(rem_p, rem_q, lower, upper_bound):
        """Find b > lower such that 1/b + 1/c = rem_p/rem_q with c > b and c <= upper_bound.
        Returns list of (b, c) pairs found."""
        results = []
        # c = b*(rem_p - rem_q) / rem_q
        # Need rem_p > rem_q for c > 0
        if rem_p <= rem_q:
            return results
        num = rem_p - rem_q  # c = b * num / rem_q
        # c integer iff rem_q | b*num
        # Let g = gcd(num, rem_q), then rem_q/g | b
        g = gcd(num, rem_q)
        step = rem_q // g
        # b must be a multiple of step, b > lower, b < upper_bound
        # Also c = b*num/rem_q > b means num > rem_q, which is rem_p > 2*rem_q
        # Wait: c > b means b*num/rem_q > b means num > rem_q means rem_p - rem_q > rem_q means rem_p > 2*rem_q
        # If rem_p <= 2*rem_q, then c <= b, no valid pair with c > b
        if rem_p <= 2 * rem_q:
            return results
        # b starts at smallest multiple of step > lower
        b0 = ((lower + 1 + step - 1) // step) * step
        if b0 <= lower:
            b0 += step
        for b in range(b0, upper_bound + 1, step):
            c = b * num // rem_q
            if c <= b or c > upper_bound:
                break
            results.append((b, c))
        return results

    # --- 3-term search: 1/a + 1/b + 1/c = 5/31, a < b < c ---
    for a in range(7, 19):
        if time.time() > deadline:
            break
        # rem = 5/31 - 1/a = (5a - 31) / (31a)
        rem_p = 5 * a - 31
        rem_q = 31 * a
        if rem_p <= 0:
            continue
        # Reduce
        g = gcd(rem_p, rem_q)
        rem_p //= g
        rem_q //= g
        pairs = check_two_term(rem_p, rem_q, a, best_max - 1)
        for b, c in pairs:
            if c < best_max:
                update_best([a, b, c])

    # --- 4-term search ---
    for a in range(7, 126):
        if time.time() > deadline:
            break
        if a >= best_max:
            break
        rem_p = 5 * a - 31
        rem_q = 31 * a
        if rem_p <= 0:
            continue
        g = gcd(rem_p, rem_q)
        rem_p //= g
        rem_q //= g
        # Now find 1/b + 1/c + 1/d = rem_p/rem_q with b > a, all < best_max
        min_b = a + 1
        # b must satisfy: 1/b < rem => b > rem_q/rem_p, and 3/b > rem => b < 3*rem_q/rem_p
        max_b = min(int(3 * rem_q / rem_p), best_max - 1)
        for b in range(min_b, max_b + 1):
            if time.time() > deadline:
                break
            rem2_p = rem_p * b - rem_q
            rem2_q = rem_q * b
            if rem2_p <= 0:
                continue
            g2 = gcd(rem2_p, rem2_q)
            rem2_p //= g2
            rem2_q //= g2
            pairs = check_two_term(rem2_p, rem2_q, b, best_max - 1)
            for c, d in pairs:
                if d < best_max:
                    update_best([a, b, c, d])

    # --- 5-term search ---
    for a in range(7, 32):
        if time.time() > deadline:
            break
        if a >= best_max:
            break
        rem_p = 5 * a - 31
        rem_q = 31 * a
        if rem_p <= 0:
            continue
        g = gcd(rem_p, rem_q)
        rem_p //= g
        rem_q //= g
        min_b = a + 1
        max_b = min(int(4 * rem_q / rem_p), best_max - 1)
        for b in range(min_b, max_b + 1):
            if time.time() > deadline:
                break
            rem2_p = rem_p * b - rem_q
            rem2_q = rem_q * b
            if rem2_p <= 0:
                continue
            g2 = gcd(rem2_p, rem2_q)
            rem2_p //= g2
            rem2_q //= g2
            min_c = b + 1
            max_c = min(int(3 * rem2_q / rem2_p), best_max - 1)
            for c in range(min_c, max_c + 1):
                if time.time() > deadline:
                    break
                rem3_p = rem2_p * c - rem2_q
                rem3_q = rem2_q * c
                if rem3_p <= 0:
                    continue
                g3 = gcd(rem3_p, rem3_q)
                rem3_p //= g3
                rem3_q //= g3
                pairs = check_two_term(rem3_p, rem3_q, c, best_max - 1)
                for d, e in pairs:
                    if e < best_max:
                        update_best([a, b, c, d, e])

    # --- 6-term search ---
    for a in range(7, 20):
        if time.time() > deadline:
            break
        if a >= best_max:
            break
        rem_p = 5 * a - 31
        rem_q = 31 * a
        if rem_p <= 0:
            continue
        g = gcd(rem_p, rem_q)
        rem_p //= g
        rem_q //= g
        min_b = a + 1
        max_b = min(int(5 * rem_q / rem_p), best_max - 1)
        for b in range(min_b, max_b + 1):
            if time.time() > deadline:
                break
            rem2_p = rem_p * b - rem_q
            rem2_q = rem_q * b
            if rem2_p <= 0:
                continue
            g2 = gcd(rem2_p, rem2_q)
            rem2_p //= g2
            rem2_q //= g2
            min_c = b + 1
            max_c = min(int(4 * rem2_q / rem2_p), best_max - 1)
            for c in range(min_c, max_c + 1):
                if time.time() > deadline:
                    break
                rem3_p = rem2_p * c - rem2_q
                rem3_q = rem2_q * c
                if rem3_p <= 0:
                    continue
                g3 = gcd(rem3_p, rem3_q)
                rem3_p //= g3
                rem3_q //= g3
                min_d = c + 1
                max_d = min(int(3 * rem3_q / rem3_p), best_max - 1)
                for d in range(min_d, max_d + 1):
                    if time.time() > deadline:
                        break
                    rem4_p = rem3_p * d - rem3_q
                    rem4_q = rem3_q * d
                    if rem4_p <= 0:
                        continue
                    g4 = gcd(rem4_p, rem4_q)
                    rem4_p //= g4
                    rem4_q //= g4
                    pairs = check_two_term(rem4_p, rem4_q, d, best_max - 1)
                    for e, f in pairs:
                        if f < best_max:
                            update_best([a, b, c, d, e, f])

    # --- 7-term search ---
    for a in range(7, 15):
        if time.time() > deadline:
            break
        if a >= best_max:
            break
        rem_p = 5 * a - 31
        rem_q = 31 * a
        if rem_p <= 0:
            continue
        g = gcd(rem_p, rem_q)
        rem_p //= g
        rem_q //= g
        min_b = a + 1
        max_b = min(int(6 * rem_q / rem_p), best_max - 1)
        for b in range(min_b, max_b + 1):
            if time.time() > deadline:
                break
            rem2_p = rem_p * b - rem_q
            rem2_q = rem_q * b
            if rem2_p <= 0:
                continue
            g2 = gcd(rem2_p, rem2_q)
            rem2_p //= g2
            rem2_q //= g2
            min_c = b + 1
            max_c = min(int(5 * rem2_q / rem2_p), best_max - 1)
            for c in range(min_c, max_c + 1):
                if time.time() > deadline:
                    break
                rem3_p = rem2_p * c - rem2_q
                rem3_q = rem2_q * c
                if rem3_p <= 0:
                    continue
                g3 = gcd(rem3_p, rem3_q)
                rem3_p //= g3
                rem3_q //= g3
                min_d = c + 1
                max_d = min(int(4 * rem3_q / rem3_p), best_max - 1)
                for d in range(min_d, max_d + 1):
                    if time.time() > deadline:
                        break
                    rem4_p = rem3_p * d - rem3_q
                    rem4_q = rem3_q * d
                    if rem4_p <= 0:
                        continue
                    g4 = gcd(rem4_p, rem4_q)
                    rem4_p //= g4
                    rem4_q //= g4
                    min_e = d + 1
                    max_e = min(int(3 * rem4_q / rem4_p), best_max - 1)
                    for e in range(min_e, max_e + 1):
                        if time.time() > deadline:
                            break
                        rem5_p = rem4_p * e - rem4_q
                        rem5_q = rem4_q * e
                        if rem5_p <= 0:
                            continue
                        g5 = gcd(rem5_p, rem5_q)
                        rem5_p //= g5
                        rem5_q //= g5
                        pairs = check_two_term(rem5_p, rem5_q, e, best_max - 1)
                        for f, h in pairs:
                            if h < best_max:
                                update_best([a, b, c, d, e, f, h])

    if best_result is not None:
        return best_result

    # Fallback
    return [7, 8, 9, 10000000]

# EVOLVE-BLOCK-END
