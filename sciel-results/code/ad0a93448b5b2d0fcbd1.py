# EVOLVE-BLOCK-START
import time
from math import gcd, isqrt

def _sieve_primes(n):
    if n < 2:
        return []
    is_prime = bytearray(b'\x01') * (n + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, isqrt(n) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, n + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = 58.0
    n = 256

    primes = _sieve_primes(n)
    primes_set = set(primes)
    num_primes = len(primes)

    # --- Compute optimal assignment via DP over primes ---
    # For each prime p, we can either:
    #   a) use p^k (largest power <= n) as a standalone element
    #   b) pair p with another prime q to form p*q (if p*q <= n)
    # We want to maximize total sum.
    #
    # Since primes are distinct and pairwise coprime, p^a and q^b are coprime
    # for distinct primes p, q. Also p*q is coprime to r^s for r != p,q.
    # So the only constraint is: each prime can be used at most once (either
    # as its max power standalone, or in exactly one product pair).
    #
    # This is a maximum weight matching on a graph where vertices are primes
    # and edges connect primes p,q where p*q <= n, with edge weight = p*q - (p^k_p + q^k_q).
    # Plus each vertex has a "stay standalone" option with weight = p^k_p.
    #
    # We can solve this with DP over subsets of primes, but 256 primes is too many.
    # However, only primes where p*q <= 256 for some q > p can participate in pairs.
    # Small primes (2,3,5,7,11,13) can pair. Larger primes cannot pair with anything
    # except themselves (which doesn't make sense) or with smaller primes already counted.
    #
    # Let's identify which primes can participate in pairs:
    # A prime p can be in a pair with q > p if p*q <= 256, i.e., q <= 256/p.
    # For p >= 17, 256/p <= 15.06, so q must be > p >= 17 but <= 15.06: impossible.
    # So only primes p <= 13 can be the smaller member of a pair.
    # Primes that can appear in pairs: 2,3,5,7,11,13 (as smaller) and their partners.

    # Max power for each prime
    max_power = {}
    for p in primes:
        power = p
        while power * p <= n:
            power *= p
        max_power[p] = power

    # Find all candidate pairs (p, q) with p < q, p*q <= n
    candidate_pairs = []
    for i, p in enumerate(primes):
        for j in range(i + 1, num_primes):
            q = primes[j]
            prod = p * q
            if prod <= n:
                weight = prod - max_power[p] - max_power[q]
                candidate_pairs.append((p, q, prod, weight))

    # Primes that appear in any candidate pair
    pairable_primes = set()
    for (p, q, prod, w) in candidate_pairs:
        pairable_primes.add(p)
        pairable_primes.add(q)

    # All other primes just use their max power
    non_pairable_sum = 0
    for p in primes:
        if p not in pairable_primes:
            non_pairable_sum += max_power[p]

    # Now solve max weight matching on pairable_primes
    # pairable_primes is small: {2,3,5,7,11,13} plus their partners
    # Partners of 2: primes q with 2q <= 256 -> q <= 128: 3,5,7,11,13,17,19,...,127
    # Partners of 3: q <= 85: 5,7,11,...,83
    # etc. This is still a moderate graph.
    # Let's just do DP over subsets of pairable_primes.

    # Index the pairable primes
    pair_list = sorted(pairable_primes)
    pair_index = {p: i for i, p in enumerate(pair_list)}
    m = len(pair_list)

    # Build edge list with weights, only for beneficial pairs
    edges = {}
    for (p, q, prod, w) in candidate_pairs:
        if w > 0:  # only beneficial pairs
            i, j = pair_index[p], pair_index[q]
            if i > j:
                i, j = j, i
            edges[(i, j)] = w

    # DP: dp[mask] = max additional sum from matching within subset mask of pair_list
    # Each element can be unmatched (contributing max_power[p]) or matched
    # We'll compute dp[mask] = max total contribution from elements in mask
    # where elements can be either standalone or paired with each other within mask.

    full_mask = (1 << m) - 1

    # Use memoized recursion
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def dp(mask):
        if mask == 0:
            return 0
        # Find the lowest set bit
        i = (mask & -mask).bit_length() - 1
        p_i = pair_list[i]
        base = max_power[p_i]  # standalone contribution

        # Option 1: p_i is standalone
        best = base + dp(mask ^ (1 << i))

        # Option 2: p_i is paired with some j in mask
        for j in range(i + 1, m):
            if mask & (1 << j):
                if (i, j) in edges:
                    w = edges[(i, j)]
                    val = (pair_list[i] * pair_list[j]) + dp(mask ^ (1 << i) ^ (1 << j))
                    if val > best:
                        best = val
                    # Even if w <= 0, pairing might still be better? No,
                    # because standalone gives p_i^k + p_j^k, pair gives p_i*p_j.
                    # w > 0 means pair > standalone, so we only need w > 0 edges.
                    # But we should also consider pairing even if w <= 0 if it
                    # frees up... no, pairing uses both, standalone uses both.
                    # So if w <= 0, standalone is better. We only consider w > 0.
        return best

    # Check time
    if time.time() - start_time > deadline:
        # Fallback: just use all max powers
        total = non_pairable_sum + sum(max_power[p] for p in pair_list)
        result = [max_power[p] for p in primes]
        result.sort()
        # Ensure 1 is included if not present (it's coprime to everything)
        if 1 not in result:
            result.append(1)
        return [int(x) for x in result]

    best_pair_sum = dp(full_mask)
    total_sum = non_pairable_sum + best_pair_sum

    # Now reconstruct which primes are paired
    @lru_cache(maxsize=None)
    def reconstruct(mask):
        """Return set of chosen pairs (as index pairs) and whether each is standalone."""
        if mask == 0:
            return frozenset(), {}
        i = (mask & -mask).bit_length() - 1
        p_i = pair_list[i]
        base = max_power[p_i]

        # Determine which option achieves dp(mask)
        dp_val = dp(mask)
        # Option 1: standalone
        standalone_val = base + dp(mask ^ (1 << i))

        if standalone_val == dp_val:
            sub_pairs, sub_standalone = reconstruct(mask ^ (1 << i))
            # i is standalone
            sub_standalone = dict(sub_standalone)
            sub_standalone[i] = True
            return frozenset(sub_pairs), sub_standalone
        else:
            # Option 2: paired with some j
            for j in range(i + 1, m):
                if mask & (1 << j):
                    if (i, j) in edges:
                        val = (pair_list[i] * pair_list[j]) + dp(mask ^ (1 << i) ^ (1 << j))
                        if val == dp_val:
                            sub_pairs, sub_standalone = reconstruct(mask ^ (1 << i) ^ (1 << j))
                            sub_pairs = set(sub_pairs)
                            sub_pairs.add((i, j))
                            return frozenset(sub_pairs), sub_standalone
            # Should not reach here
            standalone_val2 = base + dp(mask ^ (1 << i))
            sub_pairs, sub_standalone = reconstruct(mask ^ (1 << i))
            sub_standalone = dict(sub_standalone)
            sub_standalone[i] = True
            return frozenset(sub_pairs), sub_standalone

    chosen_pairs, standalone_flags = reconstruct(full_mask)

    # Build the result set
    result = []
    for p in primes:
        if p not in pairable_primes:
            result.append(max_power[p])

    # Process pairable primes
    paired_indices = set()
    for (i, j) in chosen_pairs:
        paired_indices.add(i)
        paired_indices.add(j)
        result.append(pair_list[i] * pair_list[j])

    for i, p in enumerate(pair_list):
        if i not in paired_indices:
            result.append(max_power[p])

    # Add 1 if not already present
    if 1 not in result:
        result.append(1)

    # Verify pairwise coprime
    result_set = set(result)
    if len(result_set) != len(result):
        # Fallback
        result = [max_power[p] for p in primes]
        if 1 not in result:
            result.append(1)
    else:
        valid = True
        for a in range(len(result)):
            for b in range(a + 1, len(result)):
                if gcd(result[a], result[b]) != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            result = [max_power[p] for p in primes]
            if 1 not in result:
                result.append(1)

    # Local search: try to add any missing number coprime to all
    result_set = set(result)
    for a in range(n, 1, -1):
        if time.time() - start_time > deadline:
            break
        if a in result_set:
            continue
        can_add = True
        for b in result:
            if gcd(a, b) != 1:
                can_add = False
                break
        if can_add:
            result.append(a)
            result_set.add(a)

    # Sort and return as list of Python ints
    result.sort()
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
