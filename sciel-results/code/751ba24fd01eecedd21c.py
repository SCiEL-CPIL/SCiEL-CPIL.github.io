import time
import math
import random

def construct():
    start = time.time()
    deadline = 88.0  # Safety margin before 90s kill

    max_val = 10210
    N = max_val - 210  # 10000

    # --- Sieve: smallest prime factor for fast factorization ---
    spf = list(range(max_val + 1))
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if spf[i] == i:
            for j in range(i * i, max_val + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # Collect primes up to max_val
    primes = [i for i in range(2, max_val + 1) if spf[i] == i]
    pidx = {p: i for i, p in enumerate(primes)}
    d = len(primes)

    # Parity vector of 210 = 2*3*5*7
    t = 0
    for p in (2, 3, 5, 7):
        t |= (1 << pidx[p])

    # Parity vectors for 211..max_val using spf
    vectors = [0] * N
    for i in range(N):
        x = 211 + i
        v = 0
        n = x
        while n > 1:
            p = spf[n]
            cnt = 0
            while n % p == 0:
                n //= p
                cnt += 1
            if cnt & 1:
                v |= (1 << pidx[p])
        vectors[i] = v

    # --- Build basis incrementally; track when t becomes in span ---
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
    K0 = min_K

    # --- Helper: given K, build null space and particular solution ---
    def build_system(K):
        # Rows: d equations over GF(2), K unknowns + RHS
        rows = []
        for j in range(d):
            row = 0
            for i in range(K):
                if (vectors[i] >> j) & 1:
                    row |= (1 << i)
            if (t >> j) & 1:
                row |= (1 << K)
            rows.append(row)

        # Gaussian elimination to RREF
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

        return c_p, null_basis

    def max_idx(c):
        if c == 0:
            return -1
        return c.bit_length() - 1

    def weight(c):
        return bin(c).count('1')

    # --- Search over K values from K0 upward ---
    best_c = None
    best_max = K0 - 1  # initial: just use first K0 vectors
    best_K = K0

    # First, get a baseline from K0
    c_p0, nb0 = build_system(K0)
    if weight(c_p0) <= 100:
        best_c = c_p0
        best_max = max_idx(c_p0)

    # For each K from K0 to some limit, search for better solutions
    K_limit = min(K0 + 40, N)  # don't go too far
    for K in range(K0, K_limit + 1):
        if time.time() > deadline - 10:
            break
        if max_idx(best_c if best_c else 0) <= K - 1:
            # Can't do better than K-1 if we already have max <= K-1
            # Actually we want max < current best_max, so if K-1 >= best_max, skip
            if K - 1 >= best_max:
                continue

        c_p, nb = build_system(K)
        if c_p == 0:
            # Zero solution is valid only if we have at least 1 element... 
            # Actually 0 elements means product is empty = 1, 210*1=210 not square
            # So we need at least 1 element
            pass

        nb_len = len(nb)

        if nb_len == 0:
            # Only one solution: c_p
            if c_p != 0 and weight(c_p) <= 100:
                m = max_idx(c_p)
                if m < best_max:
                    best_max = m
                    best_c = c_p
                    best_K = K
            continue

        # For small null space (<= 20), enumerate all 2^nb_len subsets
        if nb_len <= 20:
            best_local = None
            best_local_max = K - 1
            # Enumerate all combinations
            for mask in range(1, 1 << nb_len):
                if time.time() > deadline - 10:
                    break
                delta = 0
                m = mask
                idx = 0
                while m:
                    if m & 1:
                        delta ^= nb[idx]
                    m >>= 1
                    idx += 1
                nc = c_p ^ delta
                if nc == 0:
                    continue
                nw = weight(nc)
                if nw > 100:
                    continue
                nmax = max_idx(nc)
                if nmax < best_local_max:
                    best_local_max = nmax
                    best_local = nc
            if best_local is not None and best_local_max < best_max:
                best_max = best_local_max
                best_c = best_local
                best_K = K
        elif nb_len <= 28:
            # Meet-in-the-middle for null space up to 28
            half = nb_len // 2
            nb1 = nb[:half]
            nb2 = nb[half:]
            
            # Generate all subsets of nb1
            size1 = 1 << half
            # For each subset, compute (delta, max_idx, weight)
            # We want to minimize max_idx of c_p ^ delta1 ^ delta2
            # Strategy: for each delta1, find delta2 that minimizes max_idx
            
            # Precompute all delta2 values with their max_idx
            # Group by max_idx
            delta2_list = []
            for mask2 in range(1 << (nb_len - half)):
                d2 = 0
                m = mask2
                idx = 0
                while m:
                    if m & 1:
                        d2 ^= nb2[idx]
                    m >>= 1
                    idx += 1
                delta2_list.append(d2)
            
            # For each delta1, we need to find delta2 such that c_p ^ delta1 ^ delta2 has min max_idx
            # This is equivalent to: for target = c_p ^ delta1, find d2 in delta2_list
            # that minimizes max_idx(target ^ d2)
            
            # Build a dictionary: for each possible high bit, store the d2 values
            # Actually, let's just iterate and track best
            best_local = None
            best_local_max = K - 1
            
            # Sort delta2_list by max_idx to find low-max solutions first
            # But we need to check all... let's use a different approach:
            # For each delta1, iterate delta2 and track best
            # This is O(2^half * 2^(nb_len-half)) = 2^nb_len which is too slow for nb_len=28
            
            # Alternative: for each candidate max M from 0 upward, check if there exists
            # a solution with all bits <= M. This is a linear algebra problem.
            # For a given M, restrict to bits 0..M, check if t is in span of
            # vectors[0..M] (restricted to first M+1 bits)... 
            # Actually simpler: just do random sampling + greedy for larger null spaces
            
            # Random sampling
            rng = random.Random(K)
            for _ in range(5000):
                if time.time() > deadline - 10:
                    break
                mask = rng.randint(1, (1 << nb_len) - 1)
                delta = 0
                m = mask
                idx = 0
                while m:
                    if m & 1:
                        delta ^= nb[idx]
                    m >>= 1
                    idx += 1
                nc = c_p ^ delta
                if nc == 0:
                    continue
                nw = weight(nc)
                if nw > 100:
                    continue
                nmax = max_idx(nc)
                if nmax < best_local_max:
                    best_local_max = nmax
                    best_local = nc
            
            # Also try greedy: start from c_p, flip null vectors to reduce max
            current = c_p
            cur_max = max_idx(current)
            improved = True
            while improved and time.time() < deadline - 10:
                improved = False
                for b in nb:
                    nc = current ^ b
                    if nc == 0:
                        continue
                    if weight(nc) > 100:
                        continue
                    nmax = max_idx(nc)
                    if nmax < cur_max:
                        current = nc
                        cur_max = nmax
                        improved = True
            if cur_max < best_local_max:
                best_local_max = cur_max
                best_local = current
            
            if best_local is not None and best_local_max < best_max:
                best_max = best_local_max
                best_c = best_local
                best_K = K
        else:
            # Large null space: random + greedy
            rng = random.Random(K)
            best_local = None
            best_local_max = K - 1
            for _ in range(3000):
                if time.time() > deadline - 10:
                    break
                k = rng.randint(1, min(10, nb_len))
                idxs = rng.sample(range(nb_len), k)
                delta = 0
                for idx in idxs:
                    delta ^= nb[idx]
                nc = c_p ^ delta
                if nc == 0:
                    continue
                if weight(nc) > 100:
                    continue
                nmax = max_idx(nc)
                if nmax < best_local_max:
                    best_local_max = nmax
                    best_local = nc
            
            # Greedy polish
            current = best_local if best_local else c_p
            cur_max = max_idx(current)
            improved = True
            while improved and time.time() < deadline - 10:
                improved = False
                for b in nb:
                    nc = current ^ b
                    if nc == 0:
                        continue
                    if weight(nc) > 100:
                        continue
                    nmax = max_idx(nc)
                    if nmax < cur_max:
                        current = nc
                        cur_max = nmax
                        improved = True
            if cur_max < best_local_max:
                best_local_max = cur_max
                best_local = current
            
            if best_local is not None and best_local_max < best_max:
                best_max = best_local_max
                best_c = best_local
                best_K = K

    # --- Additional: direct search for small max values ---
    # Try to find a solution with max element = 210 + M for small M
    # This means using only vectors[0..M-1]
    if time.time() < deadline - 5:
        for M in range(best_max + 1, K0 + 30):
            if time.time() > deadline - 5:
                break
            if M > N:
                break
            # Check if t is in span of vectors[0..M-1]
            # Build basis for first M vectors
            b2 = [0] * (d + 1)
            r2 = 0
            for i in range(M):
                v = vectors[i]
                if v == 0:
                    continue
                r = v
                while r:
                    msb = r.bit_length() - 1
                    if b2[msb]:
                        r ^= b2[msb]
                    else:
                        b2[msb] = r
                        r2 += 1
                        break
            # Check if t is in span
            chk = t
            while chk:
                msb = chk.bit_length() - 1
                if b2[msb]:
                    chk ^= b2[msb]
                else:
                    break
            if chk == 0:
                # t is in span of first M vectors
                # Build null space for K=M
                c_p, nb = build_system(M)
                if c_p == 0:
                    continue
                if weight(c_p) > 100:
                    continue
                m = max_idx(c_p)
                if m < best_max:
                    best_max = m
                    best_c = c_p
                    best_K = M
                # Try to improve with null space
                if len(nb) <= 18:
                    for mask in range(1, 1 << len(nb)):
                        delta = 0
                        mm = mask
                        idx = 0
                        while mm:
                            if mm & 1:
                                delta ^= nb[idx]
                            mm >>= 1
                            idx += 1
                        nc = c_p ^ delta
                        if nc == 0:
                            continue
                        if weight(nc) > 100:
                            continue
                        nmax = max_idx(nc)
                        if nmax < best_max:
                            best_max = nmax
                            best_c = nc
                            best_K = M
                    break  # Found a valid M, no need to check larger M

    # --- Final polish on best solution ---
    if best_c is not None and time.time() < deadline - 3:
        # Rebuild null space for best_K and do final greedy
        c_p_final, nb_final = build_system(best_K)
        current = best_c
        cur_max = max_idx(current)
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for b in nb_final:
                nc = current ^ b
                if nc == 0:
                    continue
                if weight(nc) > 100:
                    continue
                nmax = max_idx(nc)
                if nmax < cur_max:
                    current = nc
                    cur_max = nmax
                    improved = True
        best_c = current
        best_max = cur_max

    # --- Build result ---
    if best_c is None:
        # Fallback: use c_p from K0
        best_c = c_p0
        best_K = K0

    selected = []
    for i in range(best_K):
        if (best_c >> i) & 1:
            selected.append(211 + i)

    # Ensure at least 1 element
    if not selected:
        # Find any valid solution
        for i in range(K0):
            if (c_p0 >> i) & 1:
                selected.append(211 + i)

    # Verify: product * 210 must be a perfect square
    total_parity = t
    for x in selected:
        total_parity ^= vectors[x - 211]
    if total_parity != 0:
        # Fallback
        selected = []
        for i in range(K0):
            if (c_p0 >> i) & 1:
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
