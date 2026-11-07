import math
import time

P = {'n': 210, 'maximum_window': 10210, 'maximum_size': 100}


def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n


def _validate(lst, n=210, max_window=10210, max_size=100):
    """Check all constraints."""
    if not isinstance(lst, (list, tuple)):
        return False
    if len(lst) < 1 or len(lst) > max_size:
        return False
    if len(set(lst)) != len(lst):
        return False
    for x in lst:
        if not isinstance(x, int) or isinstance(x, bool):
            return False
        if x <= n or x > max_window:
            return False
    prod = n
    for x in lst:
        prod *= x
    return _is_square(prod)


def _get_prime_bitmap():
    """Assign bit indices to primes up to max_window."""
    max_n = P['maximum_window']
    is_prime = [True] * (max_n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(max_n)) + 1):
        if is_prime[i]:
            for j in range(i * i, max_n + 1, i):
                is_prime[j] = False

    prime_to_bit = {}
    bit_idx = 0
    for i in range(2, max_n + 1):
        if is_prime[i]:
            prime_to_bit[i] = bit_idx
            bit_idx += 1
    return prime_to_bit


def _compute_parity_bitmaps(prime_to_bit):
    """Compute parity bitmaps for all numbers in range."""
    max_n = P['maximum_window']
    parities = [0] * (max_n + 1)

    for n in range(2, max_n + 1):
        k = n
        bm = 0
        # Factorize k
        # Since max_n is 10210, simple trial division is fast enough
        d = 2
        while d * d <= k:
            if k % d == 0:
                exp = 0
                while k % d == 0:
                    k //= d
                    exp += 1
                if exp % 2 == 1:
                    if d in prime_to_bit:
                        bm |= (1 << prime_to_bit[d])
            d += 1 if d == 2 else 2
        if k > 1:
            if k in prime_to_bit:
                bm |= (1 << prime_to_bit[k])
        parities[n] = bm
    return parities


def _solve_linear_system(vectors, target):
    """
    Solve sum(c_i * v_i) = target in GF(2).
    vectors: list of ints
    target: int
    Returns: list of indices i such that v_i is in solution, or None if no solution.
    """
    n = len(vectors)
    if n == 0:
        return [] if target == 0 else None

    # Augmented matrix: each row is [vector, coeff_mask]
    rows = []
    for i, v in enumerate(vectors):
        rows.append([v, 1 << i])
    rows.append([target, 0])  # Target row

    # Find max bit length
    max_bit = 0
    for r in rows:
        if r[0] > 0:
            max_bit = max(max_bit, r[0].bit_length() - 1)

    # Elimination
    for bit in range(max_bit, -1, -1):
        # Find pivot row with this bit set
        pivot_idx = -1
        for i in range(len(rows)):
            if (rows[i][0] >> bit) & 1:
                pivot_idx = i
                break

        if pivot_idx == -1:
            continue

        pivot_vec, pivot_mask = rows[pivot_idx]

        for i in range(len(rows)):
            if i == pivot_idx:
                continue
            if (rows[i][0] >> bit) & 1:
                rows[i][0] ^= pivot_vec
                rows[i][1] ^= pivot_mask

    # After elimination, the last row (target) should have vector 0 if solvable.
    target_vec, target_mask = rows[-1]

    if target_vec != 0:
        return None  # No solution

    # Extract indices from target_mask
    sol_indices = []
    for i in range(n):
        if (target_mask >> i) & 1:
            sol_indices.append(i)

    return sol_indices


def construct():
    t0 = time.time()
    deadline = t0 + 85.0

    n = P['n']
    max_window = P['maximum_window']

    # Precompute prime bitmaps
    prime_to_bit = _get_prime_bitmap()
    parities = _compute_parity_bitmaps(prime_to_bit)

    target = parities[n]

    # Search for minimal M
    states = {0}
    found_M = None

    for m in range(n + 1, max_window + 1):
        if time.time() > deadline:
            break

        # Check if target is already achievable with numbers < m
        if target in states:
            found_M = m - 1
            break

        p_m = parities[m]
        
        # Update states
        # To avoid modifying set while iterating, we collect new states
        new_states = set(states)
        for s in states:
            new_states.add(s ^ p_m)
        states = new_states

        # Check if target is achievable with numbers <= m
        if target in states:
            found_M = m
            break

    if found_M is None:
        # Fallback to single element
        # x = 210 * s^2
        # Smallest s such that 210*s^2 > 210 and <= 10210
        # s=2 -> 840.
        best_list = [n * 4]
        return [int(x) for x in best_list]

    # Reconstruct solution for numbers in [211, found_M]
    # Vectors are parities[211], ..., parities[found_M]
    vectors = [parities[i] for i in range(n + 1, found_M + 1)]

    sol_indices = _solve_linear_system(vectors, target)

    if sol_indices is None or len(sol_indices) == 0:
        # Should not happen if target was in states and target != 0
        best_list = [n * 4]
        return [int(x) for x in best_list]

    selected = []
    for idx in sol_indices:
        val = n + 1 + idx
        selected.append(val)

    # Check size constraint
    if len(selected) > P['maximum_size']:
        # This is unlikely for small M, but if it happens, fallback
        best_list = [n * 4]
        return [int(x) for x in best_list]

    # Validate
    if _validate(selected):
        return [int(x) for x in selected]
    else:
        # Fallback
        best_list = [n * 4]
        return [int(x) for x in best_list]
