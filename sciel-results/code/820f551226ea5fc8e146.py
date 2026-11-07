import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin before 90s limit

    N = 256
    candidates = list(range(2, N + 1))

    # 1. Identify primes
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N+1, i):
                is_prime[j] = False

    primes = [i for i in range(2, N+1) if is_prime[i]]

    small_primes = [2, 3, 5, 7, 11, 13]
    large_primes = [p for p in primes if p > 13]

    # Generate 13-smooth numbers
    smooth_candidates = []
    for n in range(2, N+1):
        if is_prime[n]:
            if n in small_primes:
                smooth_candidates.append(n)
        else:
            m = n
            smooth = True
            for p in small_primes:
                while m % p == 0:
                    m //= p
            if m == 1:
                smooth_candidates.append(n)

    smooth_set = set(smooth_candidates)
    n_smooth = len(smooth_candidates)

    # Precompute: for each smooth number a, which pairs (i,j) in smooth_candidates have a | smooth[i]*smooth[j]?
    # This is expensive, so we'll compute on the fly but cache.

    # Helper: Check if adding y to set S (represented as set of indices into smooth_candidates) is valid
    # S is a valid set. y not in S.
    # Need to check:
    # 1. y does not divide smooth[i]*smooth[j] for any distinct i,j in S
    # 2. smooth[i] does not divide y*smooth[j] for any distinct i,j in S

    def is_valid_addition_idx(S_idx, y_idx):
        """Check if smooth_candidates[y_idx] can be added to set S_idx (set of indices)."""
        y = smooth_candidates[y_idx]
        S_list = [smooth_candidates[i] for i in S_idx]
        n = len(S_list)

        # 1. y does not divide u*v for any u,v in S, u!=v
        for i in range(n):
            u = S_list[i]
            for j in range(n):
                if i == j:
                    continue
                v = S_list[j]
                if (u * v) % y == 0:
                    return False

        # 2. u does not divide y*v for any u,v in S, u!=v
        for i in range(n):
            u = S_list[i]
            for j in range(n):
                if i == j:
                    continue
                v = S_list[j]
                if (y * v) % u == 0:
                    return False

        return True

    def is_valid_subset_idx(S_idx):
        """Check if the subset of smooth_candidates at indices in S_idx is valid."""
        items = [smooth_candidates[i] for i in S_idx]
        n = len(items)
        for i in range(n):
            a = items[i]
            for j in range(n):
                if i == j:
                    continue
                b = items[j]
                for k in range(n):
                    if i == k or j == k:
                        continue
                    c = items[k]
                    if (b * c) % a == 0:
                        return False
        return True

    # Greedy with multiple orderings
    best_small_idx = set()
    best_size = 0

    # Create various orderings
    smooth_list = smooth_candidates
    n_cand = len(smooth_list)

    orderings = [
        sorted(range(n_cand), key=lambda i: -smooth_list[i]),  # descending value
        sorted(range(n_cand), key=lambda i: smooth_list[i]),   # ascending value
        sorted(range(n_cand), key=lambda i: -len([d for d in range(2, smooth_list[i]+1) if smooth_list[i]%d==0])),  # fewest divisors first
        sorted(range(n_cand), key=lambda i: len([d for d in range(2, smooth_list[i]+1) if smooth_list[i]%d==0])),   # most divisors first
    ]

    for order in orderings:
        curr_idx = set()
        for idx in order:
            if is_valid_addition_idx(curr_idx, idx):
                curr_idx.add(idx)
        if len(curr_idx) > best_size:
            best_size = len(curr_idx)
            best_small_idx = set(curr_idx)

    # Local search: 1-for-1 swaps and 2-for-1 swaps
    S = set(best_small_idx)
    C = [i for i in range(n_cand) if i not in S]

    improved = True
    while improved and time.time() < deadline:
        improved = False
        S_list = list(S)
        found_swap = False

        # Try 1-for-1 swaps
        for x in S_list:
            if time.time() > deadline:
                break
            S_minus_x = S - {x}

            # Find valid y in C to add to S_minus_x
            for y in C:
                if is_valid_addition_idx(S_minus_x, y):
                    # Swap x for y
                    S = S_minus_x | {y}
                    C = [c for c in C if c != y]
                    improved = True
                    found_swap = True
                    break
            if found_swap:
                break

        if found_swap:
            continue

        # Try 2-for-1 swaps
        for x in S_list:
            if time.time() > deadline:
                break
            S_minus_x = S - {x}

            # Find valid y in C to add to S_minus_x
            valid_ys = []
            for y in C:
                if is_valid_addition_idx(S_minus_x, y):
                    valid_ys.append(y)

            # For each valid y, try to find valid z in C \ {y} to add to S_minus_x U {y}
            for y in valid_ys:
                S_prime = S_minus_x | {y}
                for z in C:
                    if z == y:
                        continue
                    if is_valid_addition_idx(S_prime, z):
                        # Success: Swap x for y, z
                        S = S_prime | {z}
                        C = [c for c in C if c not in {y, z}]
                        improved = True
                        found_swap = True
                        break
                if found_swap:
                    break
            if found_swap:
                break

    best_small_set = [smooth_candidates[i] for i in S]

    # Combine
    final_set = list(set(large_primes + best_small_set))

    # Verify final set
    if not is_valid_subset_idx(set(range(n_cand))):  # This check is wrong, let me fix
        pass

    # Proper verification
    def verify(final):
        s = set(final)
        items = list(s)
        n = len(items)
        for i in range(n):
            a = items[i]
            for j in range(n):
                if i == j:
                    continue
                b = items[j]
                for k in range(n):
                    if i == k or j == k:
                        continue
                    c = items[k]
                    if (b * c) % a == 0:
                        return False
        return True

    if not verify(final_set):
        # Fallback to just large primes
        final_set = large_primes

    # Ensure return type is list of ints
    return [int(x) for x in final_set]
