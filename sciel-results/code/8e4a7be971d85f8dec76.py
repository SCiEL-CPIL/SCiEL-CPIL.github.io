import time
from math import isqrt

def _is_perfect_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    max_val = 10210
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(max_val ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, max_val + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, max_val + 1) if is_prime[i]]
    prime_to_idx = {p: i for i, p in enumerate(primes)}

    def get_mask(n):
        mask = 0
        x = n
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                cnt = 0
                while x % p == 0:
                    x //= p
                    cnt ^= 1
                if cnt:
                    mask |= (1 << prime_to_idx[p])
        if x > 1:
            if x in prime_to_idx:
                mask |= (1 << prime_to_idx[x])
        return mask

    target_mask = get_mask(210)

    masks = [0] * (max_val + 1)
    for i in range(211, max_val + 1):
        masks[i] = get_mask(i)

    basis = {}
    best_result = None

    def reduce_mask(m, used):
        cur = m
        cur_used = used
        while cur != 0:
            hb = cur.bit_length() - 1
            if hb in basis:
                b_mask, b_used = basis[hb]
                cur ^= b_mask
                cur_used = cur_used.symmetric_difference(b_used)
            else:
                break
        return cur, cur_used

    for M in range(211, max_val + 1):
        if time.time() > deadline - 10.0:
            break

        m = masks[M]
        cur, cur_used = reduce_mask(m, frozenset([M]))

        if cur == 0:
            pass
        else:
            hb = cur.bit_length() - 1
            basis[hb] = (cur, cur_used)

        t_cur, t_used = reduce_mask(target_mask, frozenset())

        if t_cur == 0 and len(t_used) > 0:
            if len(t_used) <= 100:
                prod = 210
                for x in t_used:
                    prod *= x
                if _is_perfect_square(prod):
                    best_result = [int(x) for x in sorted(t_used)]
                    return best_result
            return [int(x) for x in sorted(t_used)]

    if best_result is not None:
        return best_result

    return [224, 240]
