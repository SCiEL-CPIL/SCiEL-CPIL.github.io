"""Rewritten program: search for M with many k such that k*sigma(k) = M."""
import time
from collections import defaultdict
from math import isqrt

def construct():
    t_start = time.time()
    deadline = 150.0  # wall-clock limit; return well before 120s

    # Small primes to use in subset enumeration
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43,
              47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103,
              107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163,
              167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227,
              229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
              283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353,
              359, 367, 373, 379, 383, 389, 397]
    n = len(primes)

    # sigma_factor[i] = p_i + 1
    sf = [p + 1 for p in primes]

    # Enumerate all subsets of the first n primes
    # For each subset, compute k = product of primes, sigma = product of (p+1)
    # M = k * sigma
    # Track collisions: dict M -> list of (k, mask)
    collision_map = defaultdict(list)

    # Use iterative bit-by-bit subset enumeration
    # For n=80, 2^80 is too large. Limit to first ~24 primes (2^24 = 16M)
    max_primes = 24
    n_use = min(n, max_primes)

    # Precompute products for incremental subset building
    # Use a list of (k, sigma, mask) and grow it
    # Start with empty subset: k=1, sigma=1
    entries = [(1, 1, 0)]  # (k, sigma, mask)

    for i in range(n_use):
        p = primes[i]
        s = sf[i]
        bit = 1 << i
        current_len = len(entries)
        for j in range(current_len):
            k_old, sigma_old, mask_old = entries[j]
            k_new = k_old * p
            sigma_new = sigma_old * s
            mask_new = mask_old | bit
            entries.append((k_new, sigma_new, mask_new))

    # Now compute M = k * sigma for each entry and track collisions
    # But entries can be up to 2^24 = 16M which is a lot for memory
    # Let's be more careful: 2^24 entries each with 3 ints = ~48 bytes each = ~768MB
    # That's too much. Let's reduce to 2^22 = 4M entries.
    # Actually, let's just iterate and not store all entries.
    # Re-do: iterate subsets and compute on the fly.

    # Better approach: iterate all masks, compute k and sigma incrementally
    # But 2^24 iterations with big int arithmetic might be slow.
    # Let's use n_use = 22 (4M subsets)

    n_use = 22
    total_subsets = 1 << n_use

    # We'll iterate through all subsets and compute M = k*sigma(k)
    # For efficiency, use the fact that we can build up from previous subsets
    # Use a list to store (k, sigma) for each mask, but that's 4M * 2 big ints
    # Let's just do it iteratively with a list of tuples

    # Actually, let's use a different strategy:
    # For each subset mask, k = product of primes in mask
    # sigma = product of (p+1) for primes in mask
    # M = k * sigma

    # To avoid storing 4M entries, let's just iterate and use a dict for collisions
    # But we need to be careful about time.

    # Let's use a recursive/iterative approach that computes k and sigma for each mask
    # by using the least significant set bit.

    # Precompute for each prime index i: p_i and s_i
    # For mask m, let lb = m & (-m), i = lb.bit_length() - 1
    # prev = m ^ lb
    # k[m] = k[prev] * p[i]
    # sigma[m] = sigma[prev] * s[i]

    # But storing k and sigma for all masks is 4M * 2 big ints.
    # Let's try n_use = 20 (1M subsets) first, then increase if time allows.

    def search_with_n(n_use):
        """Search for collisions using first n_use primes."""
        total = 1 << n_use
        # Store k and sigma for each mask
        k_arr = [0] * total
        s_arr = [0] * total
        k_arr[0] = 1
        s_arr[0] = 1

        for m in range(1, total):
            lb = m & (-m)
            i = lb.bit_length() - 1
            prev = m ^ lb
            k_arr[m] = k_arr[prev] * primes[i]
            s_arr[m] = s_arr[prev] * sf[i]

        # Now find collisions
        coll_map = defaultdict(list)
        for m in range(1, total):
            M = k_arr[m] * s_arr[m]
            if M <= 10**18:
                coll_map[M].append(k_arr[m])

        # Find the M with the most solutions
        best_M = 0
        best_ks = []
        for M, ks in coll_map.items():
            if len(ks) > len(best_ks):
                best_M = M
                best_ks = ks
        return best_M, best_ks

    # Start with n_use = 20
    best_M, best_ks = search_with_n(20)
    elapsed = time.time() - t_start
    if best_ks:
        print(f"n_use=20: found M={best_M}, {len(best_ks)} solutions in {elapsed:.1f}s")

    # Try larger n_use if time permits
    if elapsed < 60:
        best_M2, best_ks2 = search_with_n(22)
        elapsed = time.time() - t_start
        if best_ks2:
            print(f"n_use=22: found M={best_M2}, {len(best_ks2)} solutions in {elapsed:.1f}s")
            if len(best_ks2) > len(best_ks):
                best_M, best_ks = best_M2, best_ks2

    if elapsed < 100:
        best_M3, best_ks3 = search_with_n(24)
        elapsed = time.time() - t_start
        if best_ks3:
            print(f"n_use=24: found M={best_M3}, {len(best_ks3)} solutions in {elapsed:.1f}s")
            if len(best_ks3) > len(best_ks):
                best_M, best_ks = best_M3, best_ks3

    # Verify and prepare result
    if not best_ks:
        # Fallback to known solution
        return {'M': 333312, 'k': [336, 372, 434]}

    # Deduplicate and verify
    seen = set()
    verified_ks = []
    for k in best_ks:
        if k in seen:
            continue
        seen.add(k)
        # Verify k * sigma(k) == best_M
        # Compute sigma(k)
        n_val = k
        sigma_val = 1
        d = 2
        temp = n_val
        while d * d <= temp:
            if temp % d == 0:
                p_power = 1
                p_sum = 1
                while temp % d == 0:
                    temp //= d
                    p_power *= d
                    p_sum += p_power
                sigma_val *= p_sum
            d += 1 if d == 2 else 2
        if temp > 1:
            sigma_val *= (1 + temp)
        if k * sigma_val == best_M:
            verified_ks.append(k)

    # Ensure k_i <= isqrt(M)
    limit = isqrt(best_M)
    verified_ks = [k for k in verified_ks if k <= limit]

    if not verified_ks:
        return {'M': 333312, 'k': [336, 372, 434]}

    # Sort and limit to 1000
    verified_ks = sorted(verified_ks)[:1000]

    return {'M': int(best_M), 'k': [int(k) for k in verified_ks]}
