import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 60

    def time_check():
        return time.time() < deadline

    N = 256

    # Sieve primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, 257):
        if i * i > N:
            break
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]

    # Max prime power for each prime
    prime_powers = {}
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        prime_powers[p] = power

    n = len(primes)

    # For each pair of primes (p,q), compute max p^a * q^b <= N with a>=1, b>=1
    def max_pair_product(p, q, N=256):
        best = 0
        pa = p
        while pa <= N:
            qb = q
            while pa * qb <= N:
                prod = pa * qb
                if prod > best:
                    best = prod
                qb *= q
            pa *= p
        return best

    # Build adjacency with positive gains
    # gain(p, q) = max_pair_product(p,q) - prime_powers[p] - prime_powers[q]
    adj = [[] for _ in range(n)]
    for i in range(n):
        p = primes[i]
        for j in range(i + 1, n):
            q = primes[j]
            prod = p * q
            if prod > N:
                break
            max_prod = max_pair_product(p, q)
            g = max_prod - prime_powers[p] - prime_powers[q]
            if g > 0:
                adj[i].append((j, g, max_prod))
                adj[j].append((i, g, max_prod))

    # Find nodes with positive-degree edges
    active_nodes = set()
    for i in range(n):
        if adj[i]:
            active_nodes.add(i)

    # Map active nodes to compact indices
    active_list = sorted(active_nodes)
    node_map = {node: idx for idx, node in enumerate(active_list)}
    m = len(active_list)

    # Build weight matrix and product matrix for compact graph
    w = [[0] * m for _ in range(m)]
    prod_matrix = [[1] * m for _ in range(m)]
    for i in range(n):
        if i not in node_map:
            continue
        ci = node_map[i]
        for j, g, max_prod in adj[i]:
            if j in node_map:
                cj = node_map[j]
                w[ci][cj] = g
                w[cj][ci] = g
                prod_matrix[ci][cj] = max_prod
                prod_matrix[cj][ci] = max_prod

    # DP over bitmask for maximum weight matching
    # dp[mask] = max total gain from matching within the set of nodes in mask
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def dp(mask):
        if mask == 0:
            return 0

        # Find lowest set bit
        i = 0
        while i < m and not (mask & (1 << i)):
            i += 1
        if i >= m:
            return 0

        # Option 1: leave i unmatched
        best = dp(mask ^ (1 << i))

        # Option 2: match i with some j in remaining
        rem = mask ^ (1 << i)
        j = i + 1
        while j < m:
            if rem & (1 << j):
                if w[i][j] > 0:
                    val = w[i][j] + dp(rem ^ (1 << j))
                    if val > best:
                        best = val
            j += 1

        return best

    full_mask = (1 << m) - 1
    total_gain = dp(full_mask)

    # Reconstruct matching by walking through dp states
    # We need to track which pairs were chosen
    choice = {}

    def dp_reconstruct(mask):
        if mask == 0:
            return ()

        i = 0
        while i < m and not (mask & (1 << i)):
            i += 1
        if i >= m:
            return ()

        rem = mask ^ (1 << i)

        # Option 1: unmatched
        best_val = dp(rem)
        best_pairs = dp_reconstruct(rem)

        # Option 2: matched
        j = i + 1
        while j < m:
            if rem & (1 << j):
                if w[i][j] > 0:
                    val = w[i][j] + dp(rem ^ (1 << j))
                    if val > best_val:
                        best_val = val
                        best_pairs = ((i, j),) + dp_reconstruct(rem ^ (1 << j))
            j += 1

        choice[mask] = best_pairs
        return best_pairs

    selected_pairs = dp_reconstruct(full_mask)

    # Map back to original prime indices
    matched_prime_indices = set()
    product_values = []
    for ci, cj in selected_pairs:
        orig_i = active_list[ci]
        orig_j = active_list[cj]
        matched_prime_indices.add(orig_i)
        matched_prime_indices.add(orig_j)
        product_values.append(prod_matrix[ci][cj])

    # Build result
    result = [1]

    # Add unmatched primes as their max prime power
    for i, p in enumerate(primes):
        if i not in matched_prime_indices:
            result.append(prime_powers[p])

    # Add matched max-products
    result.extend(product_values)

    # Validate
    if len(set(result)) != len(result):
        return [1] + [prime_powers[p] for p in primes]

    for x in result:
        if not isinstance(x, int) or x < 1 or x > N:
            return [1] + [prime_powers[p] for p in primes]

    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) != 1:
                return [1] + [prime_powers[p] for p in primes]

    return result
