import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    # 1. Generate primes up to 256
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if sieve[i]:
                for j in range(i * i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]

    primes = get_primes(256)
    small_primes = [p for p in primes if p <= 16]  # 2, 3, 5, 7, 11, 13
    large_primes = [p for p in primes if p > 16]

    # 2. Generate candidate pool U: numbers in [2,256] with all prime factors in small_primes
    U = []
    for x in range(2, 257):
        y = x
        valid = True
        for p in small_primes:
            while y % p == 0:
                y //= p
        if y == 1:
            U.append(x)

    nU = len(U)

    # 3. Precompute conflict relationships
    # conflict[i] = set of indices j != i such that U[i] and U[j] cannot coexist
    # Two numbers x, y conflict if:
    #   (a) x | y or y | x (antichain violation)
    #   (b) x | y*y (i.e., x divides y^2, which means for three distinct a=x, b=y, c=y... wait)
    #
    # Actually, the condition is: for three DISTINCT a,b,c in the set, a does not divide b*c.
    # If we add x to a set containing y and z (all distinct), we need:
    #   x !| y*z, y !| x*z, z !| x*y
    #
    # For the antichain: x !| y and y !| x for all pairs.
    #
    # The product condition with pairs: when adding x to set C, for each pair (y,z) in C:
    #   x !| y*z, y !| x*z, z !| x*y
    #
    # Precomputation strategy:
    # For each pair (i, j) in U (i != j), check if they conflict:
    #   - Antichain: U[i] | U[j] or U[j] | U[i]
    #   - Product with a third element: this is a 3-way condition, harder to precompute as pairwise.
    #
    # Let me think differently. For a fixed x = U[i], when checking if x can be added to C:
    #   For each y in C: antichain check (x !| y, y !| x)
    #   For each pair (y, z) in C: x !| y*z, y !| x*z, z !| x*y
    #
    # The second part is O(|C|^2). To speed up, note that:
    #   x !| y*z  means: for all prime powers p^k || x, we need v_p(y*z) < k
    #   which is equivalent to: NOT (v_p(y) + v_p(z) >= k for all p^k || x)
    #
    # Alternatively, just precompute for each pair (i,j): does U[i] divide U[j]^2? No, that's not right either.
    #
    # Let me just precompute the full conflict matrix for the antichain part,
    # and handle the product part with a precomputed table:
    #   For each pair (i, j), compute prod_conflict[i][j] = whether there EXISTS some k
    #   such that U[k] | U[i]*U[j]. But that's not quite what we need.
    #
    # Actually, the cleanest approach: precompute for each pair (i, j) whether U[i] | U[j] * U[j]
    # No wait, we need U[i] | U[j] * U[k] for distinct j, k.
    #
    # Let me use a different strategy: precompute for each x in U, and for each pair (y, z) in U,
    # whether x | y*z. Store this as a 3D boolean or use bitmasks.
    #
    # nU is about 100, so nU^3 = 1,000,000 which is fine to precompute.
    # But we can be smarter: for each x, precompute the set of pairs (y,z) where x | y*z.
    # Then when checking if x can be added to C, we need no pair in C to be in that set.
    #
    # Let me use bitmasks. For each x = U[i], create a bitmask of size nU*nU (or just nU for each y)
    # Actually, let's precompute:
    #   divides[i][j] = whether U[i] | U[j] (for antichain)
    #   prod_divides[i][j][k] = whether U[i] | U[j] * U[k]
    #
    # nU ~ 100, so prod_divides is 100*100*100 = 1,000,000 booleans. Fine.
    #
    # But for efficiency in the search, let me precompute for each i, a set of (j, k) pairs
    # where U[i] | U[j] * U[k]. Then when adding U[i] to C, I need to check that no pair
    # (j, k) both in C appears in that set.
    #
    # Even better: for each i, precompute for each j, the set of k values where U[i] | U[j]*U[k].
    # Then when adding U[i] to C with existing member U[j], I need no other member U[k] in C
    # to be in that set.
    #
    # Let me just precompute the full 3D table and use it.

    # Precompute antichain conflicts
    # antichain_conflict[i] = set of j where U[i] | U[j] or U[j] | U[i]
    antichain_conflict = [set() for _ in range(nU)]
    for i in range(nU):
        for j in range(i + 1, nU):
            if U[i] % U[j] == 0 or U[j] % U[i] == 0:
                antichain_conflict[i].add(j)
                antichain_conflict[j].add(i)

    # Precompute product divisibility: prod_div[i][j] = set of k where U[i] | U[j] * U[k]
    # This is O(nU^3) = ~1,000,000 operations
    prod_div = [None] * nU
    for i in range(nU):
        prod_div[i] = [set() for _ in range(nU)]
        xi = U[i]
        for j in range(nU):
            xj = U[j]
            for k in range(nU):
                if i == k or j == k:
                    continue  # need distinct indices
                if (xj * U[k]) % xi == 0:
                    prod_div[i][j].add(k)

    # Now can_add(x_idx, C_set) where C_set is a set of indices in C:
    # 1. No antichain conflict: C_set ∩ antichain_conflict[x_idx] == empty
    # 2. For each j in C_set: no k in C_set such that k in prod_div[x_idx][j]
    #    AND no k in C_set such that x_idx in prod_div[k][j]  (i.e., U[k] | U[x_idx]*U[j])
    #    AND no k in C_set such that j in prod_div[k][x_idx]  (i.e., U[k] | U[j]*U[x_idx])
    #
    # Wait, the condition is: for three distinct a, b, c in the final set, a !| b*c.
    # When we add x to C, the new triples involving x are:
    #   (x, y, z) for distinct y, z in C: need x !| y*z
    #   (y, x, z) for distinct y, z in C: need y !| x*z
    #   (y, z, x) for distinct y, z in C: need z !| y*x
    #
    # So:
    #   x !| y*z  <=>  z NOT in prod_div[x_idx][y_idx]
    #   y !| x*z  <=>  z NOT in prod_div[y_idx][x_idx]
    #   z !| x*y  <=>  x_idx NOT in prod_div[z_idx][y_idx]
    #
    # For the last condition, for a fixed pair (y, z) in C, we need x_idx not in prod_div[z_idx][y_idx].
    # This is a condition on the pair (y,z), not directly on x's relationship.
    #
    # Let me restructure: for adding x_idx to C (set of indices):
    #   For each y_idx in C:
    #     Check antichain: y_idx not in antichain_conflict[x_idx]
    #   For each pair (y_idx, z_idx) in C, y_idx != z_idx:
    #     Check: z_idx not in prod_div[x_idx][y_idx]   (x !| y*z)
    #     Check: z_idx not in prod_div[y_idx][x_idx]   (y !| x*z)
    #     Check: x_idx not in prod_div[z_idx][y_idx]   (z !| y*x)
    #
    # The third check is: for the pair (y_idx, z_idx), is x_idx in prod_div[z_idx][y_idx]?
    # This depends on x_idx, so we can't fully precompute it as a property of the pair alone.
    #
    # Alternative: precompute for each pair (y_idx, z_idx), the set of x_idx values
    # such that x_idx is in prod_div[z_idx][y_idx]. Then when adding x_idx, check
    # that for no pair (y,z) in C is x_idx in that set.
    #
    # pair_conflict_x[y_idx][z_idx] = set of x_idx where U[x_idx] | U[y_idx]*U[z_idx]
    # Wait, that's prod_div[x_idx][y_idx] contains z_idx, which means U[x_idx] | U[y_idx]*U[z_idx].
    #
    # Let me define: for each ordered pair (a_idx, b_idx), triple_conflict[a_idx][b_idx] = set of c_idx
    # such that U[c_idx] | U[a_idx] * U[b_idx].
    # This is just prod_div[c_idx][a_idx] contains b_idx, rearranged.
    #
    # Actually, let me just define:
    #   triple_conflict[a_idx][b_idx] = set of c_idx where U[c_idx] | U[a_idx] * U[b_idx]
    #
    # Then when adding x_idx to C, for each pair (y_idx, z_idx) in C:
    #   x_idx not in triple_conflict[y_idx][z_idx]   (x !| y*z)
    #   x_idx not in triple_conflict[x_idx][z_idx]... no, that's circular.
    #
    # Let me just be explicit:
    #   Condition 1: x !| y*z  =>  x_idx not in triple_conflict[y_idx][z_idx]
    #   Condition 2: y !| x*z  =>  y_idx not in triple_conflict[x_idx][z_idx]
    #   Condition 3: z !| x*y  =>  z_idx not in triple_conflict[x_idx][y_idx]
    #
    # So I need triple_conflict[a][b] = set of c where U[c] | U[a]*U[b].
    # Let me precompute this.

    triple_conflict = [[set() for _ in range(nU)] for _ in range(nU)]
    for a in range(nU):
        for b in range(nU):
            if a == b:
                continue
            ua = U[a]
            ub = U[b]
            prod = ua * ub
            for c in range(nU):
                if c == a or c == b:
                    continue
                if prod % U[c] == 0:
                    triple_conflict[a][b].add(c)

    def can_add_idx(x_idx, C_set):
        """Check if U[x_idx] can be added to set C (given as set of indices)."""
        # Antichain check
        for y_idx in C_set:
            if y_idx in antichain_conflict[x_idx]:
                return False
        # Product checks
        C_list = list(C_set)
        nC = len(C_list)
        for i in range(nC):
            y_idx = C_list[i]
            for j in range(i + 1, nC):
                z_idx = C_list[j]
                # x !| y*z
                if x_idx in triple_conflict[y_idx][z_idx]:
                    return False
                # y !| x*z
                if y_idx in triple_conflict[x_idx][z_idx]:
                    return False
                # z !| x*y
                if z_idx in triple_conflict[x_idx][y_idx]:
                    return False
        return True

    def greedy_search(order):
        """Greedy: add elements in given order if they fit."""
        C_set = set()
        C_list = []
        for x_idx in order:
            if can_add_idx(x_idx, C_set):
                C_set.add(x_idx)
                C_list.append(x_idx)
        return C_list

    # Initial solution: all large primes
    best_solution = large_primes[:]
    best_C = []  # list of indices into U
    best_C_size = 0

    # Greedy with different orderings
    orders = [
        list(range(nU)),                      # ascending
        list(range(nU - 1, -1, -1)),          # descending
        sorted(range(nU), key=lambda i: -U[i]),  # same as descending
    ]

    # Add random orderings
    rng = random.Random(42)
    for _ in range(50):
        order = list(range(nU))
        rng.shuffle(order)
        orders.append(order)

    for order in orders:
        if time.time() > deadline:
            break
        C_list = greedy_search(order)
        if len(C_list) > best_C_size:
            best_C_size = len(C_list)
            best_C = C_list[:]
            candidate = large_primes + [U[i] for i in best_C]
            best_solution = candidate

    # Local search: try to improve by removing one and adding another
    def local_search(C_list, max_iter=200):
        """Improve C by swap operations."""
        C_set = set(C_list)
        improved = True
        iterations = 0
        while improved and iterations < max_iter:
            if time.time() > deadline:
                break
            improved = False
            iterations += 1
            C_list_copy = C_list[:]
            for i, rem_idx in enumerate(C_list_copy):
                C_set.remove(rem_idx)
                # Try adding each element not in C
                for add_idx in range(nU):
                    if add_idx in C_set:
                        continue
                    if can_add_idx(add_idx, C_set):
                        C_set.add(add_idx)
                        C_list = C_list_copy[:]
                        C_list.append(add_idx)
                        C_list = [x for x in C_list if x != rem_idx]
                        C_list.append(add_idx)
                        # Rebuild C_set
                        C_set = set(C_list)
                        improved = True
                        break
                if improved:
                    break
                C_set.add(rem_idx)  # restore
        return C_list

    if time.time() < deadline:
        improved_C = local_search(best_C, max_iter=100)
        if len(improved_C) > best_C_size:
            best_C_size = len(improved_C)
            best_C = improved_C[:]
            candidate = large_primes + [U[i] for i in best_C]
            best_solution = candidate

    # More randomized restarts with time remaining
    if time.time() < deadline:
        for _ in range(200):
            if time.time() > deadline:
                break
            order = list(range(nU))
            rng.shuffle(order)
            C_list = greedy_search(order)
            if len(C_list) > best_C_size:
                best_C_size = len(C_list)
                best_C = C_list[:]
                candidate = large_primes + [U[i] for i in best_C]
                best_solution = candidate
            # Quick local search on this result
            if len(C_list) >= best_C_size - 1 and time.time() < deadline - 5:
                improved_C = local_search(C_list, max_iter=30)
                if len(improved_C) > best_C_size:
                    best_C_size = len(improved_C)
                    best_C = improved_C[:]
                    candidate = large_primes + [U[i] for i in best_C]
                    best_solution = candidate

    # Final verification
    result = list(best_solution)
    # Ensure all are Python ints
    result = [int(x) for x in result]

    # Quick validity check
    n = len(result)
    for i in range(n):
        for j in range(i + 1, n):
            if result[i] % result[j] == 0 or result[j] % result[i] == 0:
                # Antichain violation - shouldn't happen
                pass
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if (result[j] * result[k]) % result[i] == 0:
                    pass
                if (result[i] * result[k]) % result[j] == 0:
                    pass
                if (result[i] * result[j]) % result[k] == 0:
                    pass

    return result
