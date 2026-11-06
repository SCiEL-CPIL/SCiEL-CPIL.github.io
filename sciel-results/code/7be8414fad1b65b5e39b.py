# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    deadline = time.time() + 55  # safety margin before 90s limit

    # Known solution: 5/31 = 1/8 + 1/31 + 1/248, max = 248
    best = [8, 31, 248]
    best_max = 248

    def check_and_update(dens):
        nonlocal best, best_max
        mx = max(dens)
        if mx < best_max:
            best_max = mx
            best = list(dens)

    # ---------- 2-term solutions: 1/a + 1/b = 5/31 ----------
    # (5a-31)(5b-31) = 31^2 = 961
    for fa in [1, 31, 961]:
        if 961 % fa == 0:
            fb = 961 // fa
            if (fa + 31) % 5 == 0 and (fb + 31) % 5 == 0:
                a = (fa + 31) // 5
                b = (fb + 31) // 5
                if a > 0 and b > 0 and a < b and b < best_max:
                    check_and_update([a, b])

    # ---------- Factor-based search for N-term solutions ----------
    # Given remainder p/q, find 1/b + 1/c = p/q with b > b_min, c > c_min
    # => (pb - q)(pc - q) = q^2
    # Enumerate factor pairs of q^2, check divisibility.
    def find_two_terms(p, q, b_min, c_min, time_limit):
        """Find (b, c) with b >= b_min, c >= c_min, b <= c, 1/b+1/c = p/q."""
        q2 = q * q
        results = []
        # b must satisfy: pb - q divides q^2 and pb - q >= 0
        # pb - q >= 0 => b >= ceil(q/p)
        b_start = max(b_min, (q + p - 1) // p)
        # Also c >= c_min => pc - q >= 0 => c >= ceil(q/p)
        # And c = (q^2/(pb-q) + q)/p
        # For c >= b: (q^2/(pb-q) + q)/p >= b
        # Iterate over possible (pb - q) values that divide q^2
        # pb - q = d, so d divides q^2, d >= p*b_min - q, and b = (d+q)/p must be integer
        # and b >= b_min, and c = (q^2/d + q)/p must be integer, c >= max(c_min, b)
        
        # Enumerate divisors d of q^2
        # d = pb - q, so d ≡ -q (mod p), i.e., (d + q) % p == 0
        # and d >= p*b_min - q
        d_min = p * b_min - q
        if d_min < 0:
            d_min = 0
        # d must be positive for b to be positive (b = (d+q)/p > 0)
        if d_min < 1:
            d_min = 1
            
        # Find all divisors of q^2 that are >= d_min and satisfy (d+q) % p == 0
        # q can be up to ~31*37*... which can be large. Let's be careful.
        # For our cases, q is at most 31*37 = 1147 for the first level,
        # then q2 can be up to (1147*37)^2 which is large.
        # But we only need divisors d where (d+q) % p == 0 and d >= d_min.
        # Better approach: iterate b from b_start upward, check if (p*b - q) divides q^2.
        # But that's linear. Instead, iterate divisors of q^2.
        
        # Actually, let's just iterate b in a bounded range since the search space is small.
        # b < 2*q/p (since c > b means 1/b + 1/c < 2/b, so 2/b > p/q => b < 2q/p)
        b_end = min(b_start + 100000, (2 * q) // p)  # generous upper bound
        # But actually b < 2q/p strictly
        b_max_possible = (2 * q) // p
        if b_max_possible < b_start:
            return results
        b_end = b_max_possible  # b must be < 2q/p for c > b to be possible
        
        for b in range(b_start, b_end + 1):
            if time.time() > time_limit:
                break
            pb_minus_q = p * b - q
            if pb_minus_q <= 0:
                continue
            if q2 % pb_minus_q != 0:
                continue
            c_num = q2 // pb_minus_q + q
            if c_num % p != 0:
                continue
            c = c_num // p
            if c >= c_min and c >= b:
                results.append((b, c))
        return results

    # ---------- 3-term: 1/a + 1/b + 1/c = 5/31 ----------
    for a in range(7, min(19, best_max)):
        if time.time() > deadline:
            break
        p_rem = 5 * a - 31
        q_rem = 31 * a
        if p_rem <= 0:
            continue
        g = gcd(p_rem, q_rem)
        p_rem //= g
        q_rem //= g
        # Need 1/b + 1/c = p_rem/q_rem, b > a, c > b
        pairs = find_two_terms(p_rem, q_rem, a + 1, a + 2, deadline)
        for (b, c) in pairs:
            if c < best_max:
                check_and_update([a, b, c])

    # ---------- 4-term: 1/a + 1/b + 1/c + 1/d = 5/31 ----------
    for a in range(7, min(25, best_max)):
        if time.time() > deadline:
            break
        p_rem = 5 * a - 31
        q_rem = 31 * a
        if p_rem <= 0:
            continue
        g = gcd(p_rem, q_rem)
        p_rem //= g
        q_rem //= g
        # Need 1/b + 1/c + 1/d = p_rem/q_rem, b > a
        # b < 3*q_rem/p_rem (since 3/b > p_rem/q_rem)
        b_start = a + 1
        b_end = min((3 * q_rem) // p_rem, best_max - 1)
        for b in range(b_start, b_end + 1):
            if time.time() > deadline:
                break
            p2 = p_rem * b - q_rem
            if p2 <= 0:
                continue
            q2 = q_rem * b
            g2 = gcd(p2, q2)
            p2 //= g2
            q2 //= g2
            # Need 1/c + 1/d = p2/q2, c > b, d > c
            pairs = find_two_terms(p2, q2, b + 1, b + 2, deadline)
            for (c, d) in pairs:
                if d < best_max:
                    check_and_update([a, b, c, d])

    # ---------- 5-term: 1/a + 1/b + 1/c + 1/d + 1/e = 5/31 ----------
    for a in range(7, min(31, best_max)):
        if time.time() > deadline:
            break
        p_rem = 5 * a - 31
        q_rem = 31 * a
        if p_rem <= 0:
            continue
        g = gcd(p_rem, q_rem)
        p_rem //= g
        q_rem //= g
        b_start = a + 1
        b_end = min((4 * q_rem) // p_rem, best_max - 1)
        for b in range(b_start, b_end + 1):
            if time.time() > deadline:
                break
            p2 = p_rem * b - q_rem
            if p2 <= 0:
                continue
            q2 = q_rem * b
            g2 = gcd(p2, q2)
            p2 //= g2
            q2 //= g2
            c_start = b + 1
            c_end = min((3 * q2) // p2, best_max - 1)
            for c in range(c_start, c_end + 1):
                if time.time() > deadline:
                    break
                p3 = p2 * c - q2
                if p3 <= 0:
                    continue
                q3 = q2 * c
                g3 = gcd(p3, q3)
                p3 //= g3
                q3 //= g3
                # Need 1/d + 1/e = p3/q3, d > c, e > d
                pairs = find_two_terms(p3, q3, c + 1, c + 2, deadline)
                for (d, e) in pairs:
                    if e < best_max:
                        check_and_update([a, b, c, d, e])

    # ---------- 6-term: 1/a + 1/b + 1/c + 1/d + 1/e + 1/f = 5/31 ----------
    for a in range(7, min(38, best_max)):
        if time.time() > deadline:
            break
        p_rem = 5 * a - 31
        q_rem = 31 * a
        if p_rem <= 0:
            continue
        g = gcd(p_rem, q_rem)
        p_rem //= g
        q_rem //= g
        b_start = a + 1
        b_end = min((5 * q_rem) // p_rem, best_max - 1)
        for b in range(b_start, b_end + 1):
            if time.time() > deadline:
                break
            p2 = p_rem * b - q_rem
            if p2 <= 0:
                continue
            q2 = q_rem * b
            g2 = gcd(p2, q2)
            p2 //= g2
            q2 //= g2
            c_start = b + 1
            c_end = min((4 * q2) // p2, best_max - 1)
            for c in range(c_start, c_end + 1):
                if time.time() > deadline:
                    break
                p3 = p2 * c - q2
                if p3 <= 0:
                    continue
                q3 = q2 * c
                g3 = gcd(p3, q3)
                p3 //= g3
                q3 //= g3
                d_start = c + 1
                d_end = min((3 * q3) // p3, best_max - 1)
                for d in range(d_start, d_end + 1):
                    if time.time() > deadline:
                        break
                    p4 = p3 * d - q3
                    if p4 <= 0:
                        continue
                    q4 = q3 * d
                    g4 = gcd(p4, q4)
                    p4 //= g4
                    q4 //= g4
                    # Need 1/e + 1/f = p4/q4, e > d, f > e
                    pairs = find_two_terms(p4, q4, d + 1, d + 2, deadline)
                    for (e, f) in pairs:
                        if f < best_max:
                            check_and_update([a, b, c, d, e, f])

    return best
# EVOLVE-BLOCK-END
