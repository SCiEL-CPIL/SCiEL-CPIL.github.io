"""Find Egyptian fraction of 5/31 minimizing the largest denominator."""
import time

def construct():
    start = time.time()
    deadline = start + 85  # safety margin (90s kill)

    target_num = 5
    target_den = 31

    best_max = float('inf')
    best_result = None

    def update_best(denoms):
        nonlocal best_max, best_result
        mx = max(denoms)
        if mx < best_max:
            best_max = mx
            best_result = list(denoms)

    def check_time():
        return time.time() > deadline

    # --- 3-term search: 1/a + 1/b + 1/c = 5/31, a < b < c ---
    # a: 1/a < 5/31 => a > 31/5 = 6.2, so a >= 7
    # 3/a >= 5/31 => a <= 93/5 = 18.6, so a <= 18
    for a in range(7, 19):
        if check_time():
            break
        # rem = 5/31 - 1/a = (5a - 31) / (31a)
        rem_num = 5 * a - 31
        rem_den = 31 * a
        if rem_num <= 0:
            continue
        # 1/b + 1/c = rem_num/rem_den
        # b > rem_den/rem_num, b <= 2*rem_den/rem_num
        min_b = max(a + 1, rem_den // rem_num + 1)
        max_b = (2 * rem_den) // rem_num
        for b in range(min_b, max_b + 1):
            if check_time():
                break
            # rem2 = rem_num/rem_den - 1/b = (rem_num*b - rem_den) / (rem_den*b)
            rem2_num = rem_num * b - rem_den
            rem2_den = rem_den * b
            if rem2_num <= 0:
                continue
            # 1/c = rem2_num/rem2_den => c = rem2_den/rem2_num
            if rem2_den % rem2_num == 0:
                c = rem2_den // rem2_num
                if c > b and c <= 10000000:
                    update_best([a, b, c])

    # --- 4-term search: 1/a + 1/b + 1/c + 1/d = 5/31, a < b < c < d ---
    # a: 4/a >= 5/31 => a <= 124/5 = 24.8, so a <= 24
    # But also a >= 7
    for a in range(7, 25):
        if check_time():
            break
        if a >= best_max:
            break
        rem_num = 5 * a - 31
        rem_den = 31 * a
        if rem_num <= 0:
            continue
        # 1/b + 1/c + 1/d = rem_num/rem_den
        # b > rem_den/rem_num, b <= 3*rem_den/rem_num
        min_b = max(a + 1, rem_den // rem_num + 1)
        max_b = min((3 * rem_den) // rem_num, best_max - 1)
        for b in range(min_b, max_b + 1):
            if check_time():
                break
            rem2_num = rem_num * b - rem_den
            rem2_den = rem_den * b
            if rem2_num <= 0:
                continue
            # 1/c + 1/d = rem2_num/rem2_den
            min_c = max(b + 1, rem2_den // rem2_num + 1)
            max_c = min((2 * rem2_den) // rem2_num, best_max - 1)
            for c in range(min_c, max_c + 1):
                if check_time():
                    break
                rem3_num = rem2_num * c - rem2_den
                rem3_den = rem2_den * c
                if rem3_num <= 0:
                    continue
                if rem3_den % rem3_num == 0:
                    d = rem3_den // rem3_num
                    if d > c and d <= 10000000 and d < best_max:
                        update_best([a, b, c, d])

    # --- 5-term search: 1/a + 1/b + 1/c + 1/d + 1/e = 5/31 ---
    # a: 5/a >= 5/31 => a <= 31, a >= 7
    for a in range(7, 32):
        if check_time():
            break
        if a >= best_max:
            break
        rem_num = 5 * a - 31
        rem_den = 31 * a
        if rem_num <= 0:
            continue
        min_b = max(a + 1, rem_den // rem_num + 1)
        max_b = min((4 * rem_den) // rem_num, best_max - 1)
        for b in range(min_b, max_b + 1):
            if check_time():
                break
            rem2_num = rem_num * b - rem_den
            rem2_den = rem_den * b
            if rem2_num <= 0:
                continue
            min_c = max(b + 1, rem2_den // rem2_num + 1)
            max_c = min((3 * rem2_den) // rem2_num, best_max - 1)
            for c in range(min_c, max_c + 1):
                if check_time():
                    break
                rem3_num = rem2_num * c - rem2_den
                rem3_den = rem2_den * c
                if rem3_num <= 0:
                    continue
                min_d = max(c + 1, rem3_den // rem3_num + 1)
                max_d = min((2 * rem3_den) // rem3_num, best_max - 1)
                for d in range(min_d, max_d + 1):
                    if check_time():
                        break
                    rem4_num = rem3_num * d - rem3_den
                    rem4_den = rem3_den * d
                    if rem4_num <= 0:
                        continue
                    if rem4_den % rem4_num == 0:
                        e = rem4_den // rem4_num
                        if e > d and e <= 10000000 and e < best_max:
                            update_best([a, b, c, d, e])

    # --- 6-term search (if needed) ---
    # a: 6/a >= 5/31 => a <= 186/5 = 37.2, so a <= 37
    for a in range(7, 38):
        if check_time():
            break
        if a >= best_max:
            break
        rem_num = 5 * a - 31
        rem_den = 31 * a
        if rem_num <= 0:
            continue
        min_b = max(a + 1, rem_den // rem_num + 1)
        max_b = min((5 * rem_den) // rem_num, best_max - 1)
        for b in range(min_b, max_b + 1):
            if check_time():
                break
            rem2_num = rem_num * b - rem_den
            rem2_den = rem_den * b
            if rem2_num <= 0:
                continue
            min_c = max(b + 1, rem2_den // rem2_num + 1)
            max_c = min((4 * rem2_den) // rem2_num, best_max - 1)
            for c in range(min_c, max_c + 1):
                if check_time():
                    break
                rem3_num = rem2_num * c - rem2_den
                rem3_den = rem2_den * c
                if rem3_num <= 0:
                    continue
                min_d = max(c + 1, rem3_den // rem3_num + 1)
                max_d = min((3 * rem3_den) // rem3_num, best_max - 1)
                for d in range(min_d, max_d + 1):
                    if check_time():
                        break
                    rem4_num = rem3_num * d - rem3_den
                    rem4_den = rem3_den * d
                    if rem4_num <= 0:
                        continue
                    min_e = max(d + 1, rem4_den // rem4_num + 1)
                    max_e = min((2 * rem4_den) // rem4_num, best_max - 1)
                    for e in range(min_e, max_e + 1):
                        if check_time():
                            break
                        rem5_num = rem4_num * e - rem4_den
                        rem5_den = rem4_den * e
                        if rem5_num <= 0:
                            continue
                        if rem5_den % rem5_num == 0:
                            f = rem5_den // rem5_num
                            if f > e and f <= 10000000 and f < best_max:
                                update_best([a, b, c, d, e, f])

    if best_result is not None:
        return best_result

    # Fallback
    return [7, 8, 9, 10000000]

# EVOLVE-BLOCK-END
