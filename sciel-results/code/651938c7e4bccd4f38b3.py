import itertools
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)
    prime_to_idx = {p: i for i, p in enumerate(primes)}

    def factorize(n):
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors

    all_nums = list(range(2, 41))
    exp_vectors = {}
    for n in all_nums:
        f = factorize(n)
        vec = [0] * num_primes
        for p, e in f.items():
            vec[prime_to_idx[p]] = e
        exp_vectors[n] = tuple(vec)

    def check_dissociated(selected):
        """Full check: all 2^k subset products distinct."""
        n = len(selected)
        if n == 0:
            return True
        vectors = [exp_vectors[num] for num in selected]
        seen = set()
        for mask in range(1 << n):
            sum_vec = [0] * num_primes
            for i in range(n):
                if mask & (1 << i):
                    for j in range(num_primes):
                        sum_vec[j] += vectors[i][j]
            key = tuple(sum_vec)
            if key in seen:
                return False
            seen.add(key)
        return True

    def is_linearly_independent(vecs):
        """Fast check: are vectors linearly independent over Q?"""
        if not vecs:
            return True
        # Gaussian elimination with fractions
        m = len(vecs)
        d = num_primes
        # Build matrix
        mat = [[Fraction(v[j]) for j in range(d)] for v in vecs]
        rank = 0
        for col in range(d):
            # Find pivot
            pivot = -1
            for row in range(rank, m):
                if mat[row][col] != 0:
                    pivot = row
                    break
            if pivot == -1:
                continue
            # Swap
            mat[rank], mat[pivot] = mat[pivot], mat[rank]
            # Eliminate
            pivot_val = mat[rank][col]
            for j in range(col, d):
                mat[rank][j] /= pivot_val
            for row in range(m):
                if row != rank and mat[row][col] != 0:
                    factor = mat[row][col]
                    for j in range(col, d):
                        mat[row][j] -= factor * mat[rank][j]
            rank += 1
            if rank == m:
                return True
        return rank == m

    # Step 1: Per-prime baseline
    selected_numbers = []
    for p in primes:
        exponents = []
        exp = 1
        while p ** exp <= 40:
            exponents.append(exp)
            exp += 1
        
        best_subset = []
        for r in range(len(exponents), -1, -1):
            if time.time() > deadline:
                break
            found = False
            for subset in itertools.combinations(exponents, r):
                subset_sums = set()
                is_diss = True
                for mask in range(1 << len(subset)):
                    s = 0
                    for i in range(len(subset)):
                        if mask & (1 << i):
                            s += subset[i]
                    if s in subset_sums:
                        is_diss = False
                        break
                    subset_sums.add(s)
                if is_diss:
                    best_subset = list(subset)
                    found = True
                    break
            if found:
                break
        
        for e in best_subset:
            selected_numbers.append(p ** e)

    best_set = selected_numbers[:]
    if not check_dissociated(best_set):
        best_set = []

    # Step 2: Greedy addition with fast linear independence filter
    if time.time() < deadline - 10:
        current = best_set[:]
        for candidate in all_nums:
            if time.time() > deadline - 10:
                break
            if candidate in current:
                continue
            test = current + [candidate]
            # Fast check first
            test_vecs = [exp_vectors[n] for n in test]
            if is_linearly_independent(test_vecs):
                if check_dissociated(test):
                    current = test
        if len(current) > len(best_set):
            best_set = current[:]

    # Step 3: Swap search with fast filter
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(best_set)):
                if time.time() > deadline - 5:
                    break
                for candidate in all_nums:
                    if candidate in best_set:
                        continue
                    new_set = best_set[:i] + best_set[i+1:] + [candidate]
                    if len(set(new_set)) != len(new_set):
                        continue
                    new_vecs = [exp_vectors[n] for n in new_set]
                    if is_linearly_independent(new_vecs):
                        if check_dissociated(new_set):
                            best_set = new_set
                            improved = True
                            break
                if improved:
                    break

    # Final verification
    if best_set and check_dissociated(best_set):
        if all(2 <= x <= 40 for x in best_set) and len(set(best_set)) == len(best_set):
            return [int(x) for x in best_set]

    return [int(x) for x in selected_numbers]
