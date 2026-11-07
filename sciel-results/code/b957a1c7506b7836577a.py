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

    def check_dissociated_fast(selected):
        """Check if all 2^k subset products are distinct using integer vectors."""
        n = len(selected)
        if n == 0:
            return True
        vectors = [exp_vectors[num] for num in selected]
        seen = set()
        seen.add((0,) * num_primes)
        # Iterative: build all subset sums
        current_sums = [(0,) * num_primes]
        for i in range(n):
            v = vectors[i]
            new_sums = []
            for s in current_sums:
                ns = tuple(s[j] + v[j] for j in range(num_primes))
                if ns in seen:
                    return False
                seen.add(ns)
                new_sums.append(ns)
            current_sums.extend(new_sums)
        return True

    def rank_of_vectors(vecs):
        """Compute rank of a set of integer vectors using Gaussian elimination over Q (but with ints)."""
        if not vecs:
            return 0
        m = len(vecs)
        d = num_primes
        # Work with a copy as list of lists of integers
        mat = [list(v) for v in vecs]
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
            mat[rank], mat[pivot] = mat[pivot], mat[rank]
            pivot_val = mat[rank][col]
            # Eliminate other rows
            for row in range(m):
                if row != rank and mat[row][col] != 0:
                    # mat[row] = mat[row] * pivot_val - mat[rank] * mat[row][col]
                    factor = mat[row][col]
                    for j in range(col, d):
                        mat[row][j] = mat[row][j] * pivot_val - mat[rank][j] * factor
            rank += 1
            if rank == m:
                break
        return rank

    def is_linearly_independent_fast(vecs):
        """Fast check: are vectors linearly independent over Q?"""
        if not vecs:
            return True
        return rank_of_vectors(vecs) == len(vecs)

    # Build initial solution: prime powers
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
    if not check_dissociated_fast(best_set):
        best_set = []

    # Branch and bound with improved upper bound
    n_total = len(all_nums)
    num_to_idx = {n: i for i, n in enumerate(all_nums)}
    idx_to_num = {i: n for n, i in num_to_idx.items()}

    # Precompute all exponent vectors as a list indexed by num_to_idx
    vec_by_idx = [exp_vectors[idx_to_num[i]] for i in range(n_total)]

    def compute_upper_bound_fast(selected_vecs, available_mask):
        """Upper bound: |selected| + rank of (selected + available) - rank(selected)."""
        sel_rank = rank_of_vectors(selected_vecs) if selected_vecs else 0
        all_vecs = list(selected_vecs)
        for i in range(n_total):
            if available_mask & (1 << i):
                all_vecs.append(vec_by_idx[i])
        if not all_vecs:
            return len(selected_vecs)
        total_rank = rank_of_vectors(all_vecs)
        additional = min(total_rank - sel_rank, bin(available_mask).count('1'))
        if additional < 0:
            additional = 0
        return len(selected_vecs) + additional

    best_found = len(best_set)
    best_solution = best_set[:]

    # Order candidates: prefer prime powers (fewer prime factors)
    candidate_order = sorted(all_nums, key=lambda x: (len(factorize(x)), x))
    order_to_idx = [num_to_idx[n] for n in candidate_order]

    import sys
    sys.setrecursionlimit(10000)

    def branch_and_bound(pos, selected_idx, selected_vecs, available_mask):
        nonlocal best_found, best_solution

        if time.time() > deadline:
            return

        if len(selected_idx) > best_found:
            best_found = len(selected_idx)
            best_solution = [idx_to_num[i] for i in selected_idx]

        if pos >= len(candidate_order):
            return

        ub = compute_upper_bound_fast(selected_vecs, available_mask)
        if ub <= best_found:
            return

        idx = order_to_idx[pos]
        num = idx_to_num[idx]
        v = vec_by_idx[idx]

        # Option 1: Include
        new_vecs = selected_vecs + [v]
        if is_linearly_independent_fast(new_vecs):
            if check_dissociated_fast([idx_to_num[i] for i in selected_idx] + [num]):
                branch_and_bound(pos + 1, selected_idx + [idx], new_vecs, available_mask & ~(1 << idx))

        # Option 2: Exclude
        if time.time() > deadline:
            return
        branch_and_bound(pos + 1, selected_idx, selected_vecs, available_mask)

    initial_available = 0
    for i in range(n_total):
        initial_available |= (1 << i)

    branch_and_bound(0, [], [], initial_available)

    # Local search: try to add one more element to best_solution
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for candidate in all_nums:
                if candidate in best_solution:
                    continue
                new_set = best_solution + [candidate]
                if len(set(new_set)) != len(new_set):
                    continue
                if check_dissociated_fast(new_set):
                    best_solution = new_set
                    best_found = len(best_solution)
                    improved = True
                    break

    # Final verification
    if best_solution and check_dissociated_fast(best_solution):
        if all(2 <= x <= 40 for x in best_solution) and len(set(best_solution)) == len(best_solution):
            return [int(x) for x in best_solution]

    if best_set and check_dissociated_fast(best_set):
        if all(2 <= x <= 40 for x in best_set) and len(set(best_set)) == len(best_set):
            return [int(x) for x in best_set]

    return [int(x) for x in selected_numbers]
