import time
import math
import random

def construct():
    start = time.time()
    deadline = 82.0  # Safety margin before 90s kill

    max_val = 10210
    N = max_val - 210  # 10000

    # Sieve primes up to max_val
    is_prime = bytearray([1]) * (max_val + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if is_prime[i]:
            for j in range(i * i, max_val + 1, i):
                is_prime[j] = 0
    primes = [i for i in range(2, max_val + 1) if is_prime[i]]
    pidx = {p: i for i, p in enumerate(primes)}
    d = len(primes)

    # Parity vector of 210 = 2*3*5*7
    t = 0
    for p in (2, 3, 5, 7):
        t |= (1 << pidx[p])

    # Parity vectors for 211..max_val
    vectors = [0] * N
    for i in range(N):
        x = 211 + i
        v = 0
        n = x
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                cnt = 0
                while n % p == 0:
                    n //= p
                    cnt += 1
                if cnt & 1:
                    v |= (1 << pidx[p])
        if n > 1:
            v |= (1 << pidx[n])
        vectors[i] = v

    # Step 1: Find minimal K such that t is in span of vectors[0..K-1]
    # and simultaneously build the transposed system incrementally for efficiency.
    # We'll do a first pass to find min_K, then build the system.
    
    basis = [0] * (d + 1)
    rank = 0
    min_K = None
    for i in range(N):
        v = vectors[i]
        if v == 0:
            continue
        r = v
        while r:
            msb = r.bit_length() - 1
            if basis[msb]:
                r ^= basis[msb]
            else:
                basis[msb] = r
                rank += 1
                break
        # Check if t is in span
        chk = t
        while chk:
            msb = chk.bit_length() - 1
            if basis[msb]:
                chk ^= basis[msb]
            else:
                break
        if chk == 0:
            min_K = i + 1
            break

    if min_K is None:
        min_K = N

    K = min_K

    # Step 2: Build the transposed system (d equations, K unknowns + RHS)
    rows = []
    for j in range(d):
        row = 0
        for i in range(K):
            if (vectors[i] >> j) & 1:
                row |= (1 << i)
        if (t >> j) & 1:
            row |= (1 << K)
        rows.append(row)

    # Step 3: Gaussian elimination to RREF
    pivot_row = 0
    pivot_cols = []
    for c in range(K):
        found = -1
        for r in range(pivot_row, d):
            if (rows[r] >> c) & 1:
                found = r
                break
        if found == -1:
            continue
        rows[found], rows[pivot_row] = rows[pivot_row], rows[found]
        for r in range(d):
            if r != pivot_row and ((rows[r] >> c) & 1):
                rows[r] ^= rows[pivot_row]
        pivot_cols.append(c)
        pivot_row += 1
        if pivot_row >= d:
            break

    # Particular solution (free vars = 0)
    c_p = 0
    for r, c in enumerate(pivot_cols):
        if (rows[r] >> K) & 1:
            c_p |= (1 << c)

    # Null space basis
    pivot_set = set(pivot_cols)
    free_cols = [c for c in range(K) if c not in pivot_set]
    null_basis = []
    for f in free_cols:
        n = 1 << f
        for r, c in enumerate(pivot_cols):
            if (rows[r] >> f) & 1:
                n |= (1 << c)
        null_basis.append(n)

    # Helper functions
    def max_idx(c):
        if c == 0:
            return -1
        return c.bit_length() - 1

    def weight(c):
        return bin(c).count('1')

    # Initial solution
    current = c_p
    best_c = current
    best_max = max_idx(current)

    # Greedy single-flip to reduce max index
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for b in null_basis:
            nc = current ^ b
            if nc == 0:
                continue
            nw = weight(nc)
            if nw > 100:
                continue
            nmax = max_idx(nc)
            if nmax < best_max:
                current = nc
                best_c = nc
                best_max = nmax
                improved = True

    # 2-flip search
    if time.time() < deadline - 5:
        improved = True
        nb = len(null_basis)
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(nb):
                for j in range(i + 1, nb):
                    nc = current ^ null_basis[i] ^ null_basis[j]
                    if nc == 0:
                        continue
                    nw = weight(nc)
                    if nw > 100:
                        continue
                    nmax = max_idx(nc)
                    if nmax < best_max:
                        current = nc
                        best_c = nc
                        best_max = nmax
                        improved = True
                        break
                if improved:
                    break

    # 3-flip search (limited)
    if time.time() < deadline - 5:
        improved = True
        nb = len(null_basis)
        # Only do 3-flip if null space is small enough to be feasible
        if nb <= 50:
            while improved and time.time() < deadline - 5:
                improved = False
                for i in range(nb):
                    for j in range(i + 1, nb):
                        for k in range(j + 1, nb):
                            nc = current ^ null_basis[i] ^ null_basis[j] ^ null_basis[k]
                            if nc == 0:
                                continue
                            nw = weight(nc)
                            if nw > 100:
                                continue
                            nmax = max_idx(nc)
                            if nmax < best_max:
                                current = nc
                                best_c = nc
                                best_max = nmax
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break

    # Randomized search: try random combinations of null space vectors
    if time.time() < deadline - 3:
        rng = random.Random(42)
        for _ in range(50000):
            if time.time() > deadline - 3:
                break
            k = rng.randint(1, 7)
            if len(null_basis) < k:
                k = len(null_basis)
            idxs = rng.sample(range(len(null_basis)), k)
            delta = 0
            for idx in idxs:
                delta ^= null_basis[idx]
            nc = c_p ^ delta
            if nc == 0:
                continue
            nw = weight(nc)
            if nw > 100:
                continue
            nmax = max_idx(nc)
            if nmax < best_max:
                best_c = nc
                best_max = nmax
                current = nc

    # Final greedy polish
    if time.time() < deadline - 2:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for b in null_basis:
                nc = current ^ b
                if nc == 0:
                    continue
                nw = weight(nc)
                if nw > 100:
                    continue
                nmax = max_idx(nc)
                if nmax < best_max:
                    current = nc
                    best_c = nc
                    best_max = nmax
                    improved = True

    # Build result from best_c
    selected = []
    for i in range(K):
        if (best_c >> i) & 1:
            selected.append(211 + i)

    # Ensure at least 1 element
    if not selected:
        for i in range(K):
            if (c_p >> i) & 1:
                selected.append(211 + i)

    # Verify: product * 210 must be a perfect square
    total_parity = t
    for i in range(len(selected)):
        total_parity ^= vectors[selected[i] - 211]
    if total_parity != 0:
        # Fallback to c_p
        selected = []
        for i in range(K):
            if (c_p >> i) & 1:
                selected.append(211 + i)

    # Ensure types are correct
    return [int(x) for x in selected]


if __name__ == "__main__":
    sol = construct()
    print(f"Solution size: {len(sol)}")
    print(f"Max element: {max(sol) if sol else 0}")
    if sol:
        prod = 210
        for x in sol:
            prod *= x
        sqrt_prod = math.isqrt(prod)
        is_square = (sqrt_prod * sqrt_prod == prod)
        print(f"Is square: {is_square}")
        print(f"Raw value: {max(sol) - 210}")
