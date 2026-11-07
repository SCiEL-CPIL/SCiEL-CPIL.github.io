import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin for 90s limit

    N = 210
    MAX_X = 10210
    MAX_K = 100

    # 1. Sieve of Eratosthenes
    limit = MAX_X
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, v in enumerate(is_prime) if v]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)

    # 2. Precompute squarefree vectors for all x in (210, 10210]
    vecs = {}
    for x in range(211, MAX_X + 1):
        v = 0
        temp = x
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                if count % 2 == 1:
                    v |= (1 << prime_to_idx[p])
        if temp > 1:
            if temp in prime_to_idx:
                v |= (1 << prime_to_idx[temp])
        vecs[x] = v

    # Compute target vector for 210 = 2 * 3 * 5 * 7
    t = 210
    target_vec = 0
    for p in primes:
        if p * p > t:
            break
        if t % p == 0:
            count = 0
            while t % p == 0:
                t //= p
                count += 1
            if count % 2 == 1:
                target_vec |= (1 << prime_to_idx[p])
    if t > 1:
        if t in prime_to_idx:
            target_vec |= (1 << prime_to_idx[t])

    # 3. Incremental Span Check with bitmasks for history
    basis_by_hsb = {}
    best_solution = None
    best_count = MAX_K + 1

    for Y in range(211, MAX_X + 1):
        if time.time() > deadline:
            break

        v_new = vecs[Y]
        hist_new = 1 << (Y - 211)

        # Reduce v_new by existing basis using dictionary lookup
        v_curr = v_new
        h_curr = hist_new

        while v_curr:
            hsb = v_curr.bit_length() - 1
            if hsb in basis_by_hsb:
                b_vec, b_hist = basis_by_hsb[hsb]
                v_curr ^= b_vec
                h_curr ^= b_hist
            else:
                break

        if v_curr != 0:
            # Independent vector, add to basis
            hsb = v_curr.bit_length() - 1
            basis_by_hsb[hsb] = (v_curr, h_curr)

        # Check if target_vec is in span
        v_t = target_vec
        h_t = 0

        while v_t:
            hsb = v_t.bit_length() - 1
            if hsb in basis_by_hsb:
                b_vec, b_hist = basis_by_hsb[hsb]
                v_t ^= b_vec
                h_t ^= b_hist
            else:
                break

        if v_t == 0:
            # Solution found! Count bits in h_t
            cnt = h_t.bit_count()
            if cnt <= MAX_K and cnt < best_count:
                best_count = cnt
                # Reconstruct the list
                result = []
                mask = h_t
                while mask:
                    low_bit = mask & (-mask)
                    pos = low_bit.bit_length() - 1
                    result.append(211 + pos)
                    mask ^= low_bit
                # Verify
                prod = 210
                for x in result:
                    prod *= x
                sqrt_prod = math.isqrt(prod)
                if sqrt_prod * sqrt_prod == prod:
                    best_solution = result
                    break  # Found minimal Y, this is optimal for this Y
            # If too many elements, continue searching for a sparser solution

    if best_solution is not None:
        return best_solution

    # Fallback: return the seed solution
    return [840]
