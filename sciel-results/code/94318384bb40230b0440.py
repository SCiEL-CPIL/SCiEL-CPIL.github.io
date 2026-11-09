"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
import random

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = 128
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]

    def lcm_lookup(a, b):
        return lcm_table[a][b]

    def is_conflict(x, selected):
        for i in range(len(selected)):
            a = selected[i]
            lab = lcm_lookup(a, x)
            for j in range(i + 1, len(selected)):
                b = selected[j]
                lac = lcm_lookup(a, b)
                lbc = lcm_lookup(b, x)
                if lab == lac == lbc:
                    return True
        return False

    def prime_exponents(x):
        """Return dict of prime -> exponent for x."""
        factors = {}
        d = 2
        while d * d <= x:
            while x % d == 0:
                factors[d] = factors.get(d, 0) + 1
                x //= d
            d += 1
        if x > 1:
            factors[x] = factors.get(x, 0) + 1
        return factors

    def omega(x):
        count = 0
        d = 2
        while d * d <= x:
            while x % d == 0:
                count += 1
                x //= d
            d += 1
        if x > 1:
            count += 1
        return count

    def largest_prime_factor(x):
        max_p = 1
        d = 2
        while d * d <= x:
            while x % d == 0:
                max_p = d
                x //= d
            d += 1
        if x > 1:
            max_p = x
        return max_p

    # Precompute prime exponent profiles
    exp_profiles = {}
    for x in range(1, n + 1):
        exp_profiles[x] = prime_exponents(x)

    # Build weighted conflict graph: edge weight = number of bad triples containing that pair
    # A bad triple is {a,b,c} where lcm(a,b)==lcm(a,c)==lcm(b,c)
    # For each pair (a,b), count how many c make {a,b,c} a bad triple
    edge_weight = [[0] * (n + 1) for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            cnt = 0
            for c in range(1, n + 1):
                if c == a or c == b:
                    continue
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    cnt += 1
            edge_weight[a][b] = cnt
            edge_weight[b][a] = cnt

    def weighted_conflict_score(x, selected):
        """Sum of edge weights between x and selected elements."""
        score = 0
        for a in selected:
            score += edge_weight[x][a]
        return score

    def is_conflict_weighted(x, selected):
        """Check if x creates any bad triple with selected (exact)."""
        for i in range(len(selected)):
            a = selected[i]
            lab = lcm_lookup(a, x)
            for j in range(i + 1, len(selected)):
                b = selected[j]
                lac = lcm_lookup(a, b)
                lbc = lcm_lookup(b, x)
                if lab == lac == lbc:
                    return True
        return False

    def spectral_ordering(selected_pool):
        """
        Build a weighted adjacency matrix from edge_weight for the pool.
        Compute the Fiedler vector (2nd eigenvector) to get an ordering
        that separates high-conflict nodes from low-conflict ones.
        """
        m = len(selected_pool)
        if m <= 1:
            return list(selected_pool)
        
        # Build adjacency matrix
        import numpy as np
        A = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                A[i][j] = edge_weight[selected_pool[i]][selected_pool[j]]
        
        # Add self-loops with degree to make it a Laplacian-like matrix
        deg = A.sum(axis=1)
        L = np.diag(deg) - A
        
        try:
            eigvals, eigvecs = np.linalg.eigh(L)
            # Fiedler vector is the eigenvector with 2nd smallest eigenvalue
            fiedler = eigvecs[:, 1]
            # Order by projection onto Fiedler vector
            order = np.argsort(fiedler)
            return [selected_pool[i] for i in order]
        except:
            return list(selected_pool)

    def greedy_by_weighted_score(pool, ordering):
        """Greedy: add elements in order, but skip if weighted conflict score is too high."""
        selected = []
        selected_set = set()
        for x in ordering:
            if x in selected_set:
                continue
            if not is_conflict_weighted(x, selected):
                selected.append(x)
                selected_set.add(x)
        return selected

    best_result = None
    best_size = 0

    random.seed(42)

    # Generate orderings using spectral relaxation and other heuristics
    orderings = []

    # Spectral ordering on the full set
    full_pool = list(range(1, n + 1))
    orderings.append(spectral_ordering(full_pool))

    # Descending by largest prime factor
    orderings.append(sorted(range(1, n + 1), key=largest_prime_factor, reverse=True))

    # Ascending by omega
    orderings.append(sorted(range(1, n + 1), key=omega))

    # Descending by value
    orderings.append(list(range(n, 0, -1)))

    # Ascending by value
    orderings.append(list(range(1, n + 1)))

    # Sort by number of divisors
    def num_divisors(x):
        cnt = 0
        for d in range(1, int(x**0.5) + 1):
            if x % d == 0:
                cnt += 1
                if d != x // d:
                    cnt += 1
        return cnt
    orderings.append(sorted(range(1, n + 1), key=num_divisors))

    # Sort by radicial
    def radicial(x):
        r = 1
        d = 2
        while d * d <= x:
            if x % d == 0:
                r *= d
                while x % d == 0:
                    x //= d
            d += 1
        if x > 1:
            r *= x
        return r
    orderings.append(sorted(range(1, n + 1), key=radicial, reverse=True))

    # Primes and prime powers first
    def is_prime(x):
        if x < 2:
            return False
        d = 2
        while d * d <= x:
            if x % d == 0:
                return False
            d += 1
        return True
    primes = [x for x in range(1, n + 1) if is_prime(x)]
    non_primes = [x for x in range(1, n + 1) if not is_prime(x)]
    orderings.append(primes + non_primes)

    # Prime powers first
    def is_prime_power(x):
        if x < 2:
            return False
        for p in range(2, int(x**0.5) + 1):
            if x % p == 0:
                while x % p == 0:
                    x //= p
                return x == 1
        return True
    prime_powers = [x for x in range(1, n + 1) if is_prime_power(x)]
    others = [x for x in range(1, n + 1) if not is_prime_power(x)]
    orderings.append(prime_powers + others)

    # Exponent spread
    def exponent_spread(x):
        pe = exp_profiles[x]
        if not pe:
            return 0
        exponents = list(pe.values())
        return max(exponents) - min(exponents) if len(exponents) > 1 else max(exponents)
    orderings.append(sorted(range(1, n + 1), key=exponent_spread, reverse=True))

    # Sort by total edge weight (degree in conflict graph) - low degree first
    def total_edge_weight(x):
        return sum(edge_weight[x][y] for y in range(1, n + 1) if y != x)
    orderings.append(sorted(range(1, n + 1), key=total_edge_weight))

    # Random orderings
    for _ in range(40):
        perm = list(range(1, n + 1))
        random.shuffle(perm)
        orderings.append(perm)

    # Spectral orderings on random subsets (perturbed)
    for _ in range(15):
        perm = list(range(1, n + 1))
        random.shuffle(perm)
        # Take a random subset of size ~80 and do spectral on it, then add rest
        subset_size = random.randint(60, 100)
        subset = perm[:subset_size]
        rest = perm[subset_size:]
        spectral_sub = spectral_ordering(subset)
        orderings.append(spectral_sub + rest)

    # Weighted score-based ordering: sort by total edge weight with random perturbation
    for _ in range(15):
        orderings.append(sorted(range(1, n + 1), key=lambda x: total_edge_weight(x) + random.random() * 5))

    for ordering in orderings:
        if time.time() > deadline - 5.0:
            break

        selected = greedy_by_weighted_score(full_pool, ordering)

        if len(selected) > best_size:
            best_size = len(selected)
            best_result = list(selected)
            if best_size >= n:
                break

    # Local search: try to add more elements
    if best_result and time.time() < deadline - 3.0:
        current = list(best_result)
        current_set = set(current)
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            for x in range(1, n + 1):
                if x in current_set:
                    continue
                if not is_conflict_weighted(x, current):
                    current.append(x)
                    current_set.add(x)
                    improved = True
                    break
        if len(current) > best_size:
            best_size = len(current)
            best_result = current

    # Local search with swaps: try replacing one element with two
    if best_result and time.time() < deadline - 5.0:
        current = list(best_result)
        current_set = set(current)
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(len(current)):
                if time.time() > deadline - 5.0:
                    break
                removed = current[i]
                temp = current[:i] + current[i+1:]
                temp_set = set(temp)
                # Try to add two elements
                added = []
                for x in range(1, n + 1):
                    if x in temp_set or x == removed:
                        continue
                    if not is_conflict_weighted(x, temp):
                        temp.append(x)
                        temp_set.add(x)
                        added.append(x)
                        if len(added) >= 2:
                            break
                if len(added) >= 2:
                    current = temp
                    current_set = set(current)
                    improved = True
                    break
        if len(current) > best_size:
            best_size = len(current)
            best_result = current

    # Additional local search: try to improve by swapping one element for another
    if best_result and time.time() < deadline - 2.0:
        current = list(best_result)
        current_set = set(current)
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for i in range(len(current)):
                if time.time() > deadline - 2.0:
                    break
                removed = current[i]
                temp = current[:i] + current[i+1:]
                temp_set = set(temp)
                # Try to find a better replacement
                best_replacement = None
                best_score = weighted_conflict_score(removed, temp)
                for x in range(1, n + 1):
                    if x in temp_set or x == removed:
                        continue
                    if not is_conflict_weighted(x, temp):
                        score = weighted_conflict_score(x, temp)
                        if score < best_score:
                            best_score = score
                            best_replacement = x
                if best_replacement is not None:
                    current = temp + [best_replacement]
                    current_set = set(current)
                    improved = True
                    break
        if len(current) > best_size:
            best_size = len(current)
            best_result = current

    if not best_result:
        best_result = [1]

    return best_result
# EVOLVE-BLOCK-END
