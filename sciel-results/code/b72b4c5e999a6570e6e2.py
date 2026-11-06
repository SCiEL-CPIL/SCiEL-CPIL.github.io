"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    deadline = time.time() + 50
    target_num = 5
    target_den = 31
    
    best = [8, 31, 248]
    best_max = 248
    
    def verify_int(denoms):
        # Verify sum of 1/d for d in denoms equals 5/31 using integer arithmetic
        # Compute sum as fraction with common denominator
        num = 0
        den = 1
        for d in denoms:
            # num/den + 1/d = (num*d + den)/(den*d)
            new_num = num * d + den
            new_den = den * d
            # Reduce
            g = _gcd(new_num, new_den)
            num = new_num // g
            den = new_den // g
        return num == target_num and den == target_den
    
    def _gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    # 3-term: 1/a + 1/b + 1/c = 5/31
    # After a: 1/b + 1/c = (5a-31)/(31a)
    # c = 31*a*b / (5*a*b - 31*a - 31*b)
    for a in range(7, 19):
        if time.time() > deadline:
            return best
        rem_a_num = 5 * a - 31
        rem_a_den = 31 * a
        if rem_a_num <= 0:
            continue
        b_min = a + 1
        # b must be > rem_a_den / rem_a_num (since 1/b < rem_a_num/rem_a_den)
        # and b < 2 * rem_a_den / rem_a_num (since 1/b + 1/c = rem_a_num/rem_a_den < 2/b)
        b_upper = (2 * rem_a_den - 1) // rem_a_num
        b_max = min(b_upper, best_max - 2)
        for b in range(b_min, b_max + 1):
            if time.time() > deadline:
                return best
            denom_c = 5 * a * b - 31 * a - 31 * b
            if denom_c <= 0:
                continue
            num_c = 31 * a * b
            if num_c % denom_c != 0:
                continue
            c = num_c // denom_c
            if c <= b or c >= best_max:
                continue
            if verify_int([a, b, c]):
                if c < best_max:
                    best_max = c
                    best = [a, b, c]
    
    # 4-term: 1/a + 1/b + 1/c + 1/d = 5/31
    for a in range(7, min(25, best_max)):
        if time.time() > deadline:
            return best
        rem_a_num = 5 * a - 31
        rem_a_den = 31 * a
        if rem_a_num <= 0:
            continue
        b_min = a + 1
        b_upper = (3 * rem_a_den - 1) // rem_a_num
        b_max = min(b_upper, best_max - 3)
        for b in range(b_min, b_max + 1):
            if time.time() > deadline:
                return best
            rem_b_num = 5 * a * b - 31 * b - 31 * a
            rem_b_den = 31 * a * b
            if rem_b_num <= 0:
                continue
            c_min = b + 1
            c_upper = (2 * rem_b_den - 1) // rem_b_num
            c_max = min(c_upper, best_max - 2)
            for c in range(c_min, c_max + 1):
                if time.time() > deadline:
                    return best
                rem_c_num = rem_b_num * c - rem_b_den
                rem_c_den = rem_b_den * c
                if rem_c_num <= 0:
                    continue
                if rem_c_den % rem_c_num != 0:
                    continue
                d = rem_c_den // rem_c_num
                if d <= c or d >= best_max:
                    continue
                if verify_int([a, b, c, d]):
                    if d < best_max:
                        best_max = d
                        best = [a, b, c, d]
    
    # 5-term
    for a in range(7, min(31, best_max)):
        if time.time() > deadline:
            return best
        rem_a_num = 5 * a - 31
        rem_a_den = 31 * a
        if rem_a_num <= 0:
            continue
        b_min = a + 1
        b_upper = (4 * rem_a_den - 1) // rem_a_num
        b_max = min(b_upper, best_max - 4)
        for b in range(b_min, b_max + 1):
            if time.time() > deadline:
                return best
            rem_b_num = 5 * a * b - 31 * b - 31 * a
            rem_b_den = 31 * a * b
            if rem_b_num <= 0:
                continue
            c_min = b + 1
            c_upper = (3 * rem_b_den - 1) // rem_b_num
            c_max = min(c_upper, best_max - 3)
            for c in range(c_min, c_max + 1):
                if time.time() > deadline:
                    return best
                rem_c_num = rem_b_num * c - rem_b_den
                rem_c_den = rem_b_den * c
                if rem_c_num <= 0:
                    continue
                d_min = c + 1
                d_upper = (2 * rem_c_den - 1) // rem_c_num
                d_max = min(d_upper, best_max - 2)
                for d in range(d_min, d_max + 1):
                    if time.time() > deadline:
                        return best
                    rem_d_num = rem_c_num * d - rem_c_den
                    rem_d_den = rem_c_den * d
                    if rem_d_num <= 0:
                        continue
                    if rem_d_den % rem_d_num != 0:
                        continue
                    e = rem_d_den // rem_d_num
                    if e <= d or e >= best_max:
                        continue
                    if verify_int([a, b, c, d, e]):
                        if e < best_max:
                            best_max = e
                            best = [a, b, c, d, e]
    
    # 6-term: if time permits
    if time.time() < deadline - 5:
        for a in range(7, min(35, best_max)):
            if time.time() > deadline:
                return best
            rem_a_num = 5 * a - 31
            rem_a_den = 31 * a
            if rem_a_num <= 0:
                continue
            b_min = a + 1
            b_upper = (5 * rem_a_den - 1) // rem_a_num
            b_max = min(b_upper, best_max - 5)
            for b in range(b_min, b_max + 1):
                if time.time() > deadline:
                    return best
                rem_b_num = 5 * a * b - 31 * b - 31 * a
                rem_b_den = 31 * a * b
                if rem_b_num <= 0:
                    continue
                c_min = b + 1
                c_upper = (4 * rem_b_den - 1) // rem_b_num
                c_max = min(c_upper, best_max - 4)
                for c in range(c_min, c_max + 1):
                    if time.time() > deadline:
                        return best
                    rem_c_num = rem_b_num * c - rem_b_den
                    rem_c_den = rem_b_den * c
                    if rem_c_num <= 0:
                        continue
                    d_min = c + 1
                    d_upper = (3 * rem_c_den - 1) // rem_c_num
                    d_max = min(d_upper, best_max - 3)
                    for d in range(d_min, d_max + 1):
                        if time.time() > deadline:
                            return best
                        rem_d_num = rem_c_num * d - rem_c_den
                        rem_d_den = rem_c_den * d
                        if rem_d_num <= 0:
                            continue
                        e_min = d + 1
                        e_upper = (2 * rem_d_den - 1) // rem_d_num
                        e_max = min(e_upper, best_max - 2)
                        for e in range(e_min, e_max + 1):
                            if time.time() > deadline:
                                return best
                            rem_e_num = rem_d_num * e - rem_d_den
                            rem_e_den = rem_d_den * e
                            if rem_e_num <= 0:
                                continue
                            if rem_e_den % rem_e_num != 0:
                                continue
                            f = rem_e_den // rem_e_num
                            if f <= e or f >= best_max:
                                continue
                            if verify_int([a, b, c, d, e, f]):
                                if f < best_max:
                                    best_max = f
                                    best = [a, b, c, d, e, f]
    
    return best
# EVOLVE-BLOCK-END
