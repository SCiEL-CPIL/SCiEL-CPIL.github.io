"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import math
import time


def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    def valid(A, B):
        s = [a + b for a in A for b in B]
        d = list(set(s))
        for i in range(len(d)):
            for j in range(i + 1, len(d)):
                if math.gcd(d[i], d[j]) != 1:
                    return 0
        return len(d)

    # Precompute primes up to 256 (max sum 128+128).
    N = 256
    is_prime = bytearray([1]) * (N + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(N ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = 0

    odd = list(range(1, 129, 2))    # 1..127
    even = list(range(2, 129, 2))   # 2..128

    # Bitset per odd a: which even indices b make a+b prime.
    prime_mask = {}
    for a in odd:
        m = 0
        for bi, b in enumerate(even):
            if is_prime[a + b]:
                m |= (1 << bi)
        prime_mask[a] = m

    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = valid(best_A, best_B)

    def bits_to_indices(mask):
        out = []
        i = 0
        while mask:
            if mask & 1:
                out.append(i)
            mask >>= 1
            i += 1
        return out

    n_odd = len(odd)
    for i1 in range(n_odd):
        if time.time() > deadline:
            break
        a1 = odd[i1]
        m1 = prime_mask[a1]
        for i2 in range(i1 + 1, n_odd):
            if time.time() > deadline:
                break
            a2 = odd[i2]
            m12 = m1 & prime_mask[a2]
            if bin(m12).count("1") < 3:
                continue
            for i3 in range(i2 + 1, n_odd):
                a3 = odd[i3]
                m = m12 & prime_mask[a3]
                if bin(m).count("1") < 3:
                    continue
                idxs = bits_to_indices(m)
                A = (a1, a2, a3)
                for b1i, b2i, b3i in itertools.combinations(idxs, 3):
                    B = (even[b1i], even[b2i], even[b3i])
                    s = (A[0] + B[0], A[0] + B[1], A[0] + B[2],
                         A[1] + B[0], A[1] + B[1], A[1] + B[2],
                         A[2] + B[0], A[2] + B[1], A[2] + B[2])
                    if len(set(s)) < 9:
                        continue
                    if all(is_prime[x] for x in s):
                        return [list(A), list(B)]
                    sc = valid(list(A), list(B))
                    if sc > best_score:
                        best_score = sc
                        best_A = list(A)
                        best_B = list(B)
                        if best_score == 9:
                            return [best_A, best_B]

    return [best_A, best_B]

# EVOLVE-BLOCK-END
