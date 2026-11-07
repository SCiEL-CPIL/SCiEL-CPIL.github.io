"""Approximate a ratio inside a divisor-sum fiber."""
import time
import math
from collections import defaultdict

def construct():
    """
    Find a, b <= 10^7 with sigma(a) = sigma(b) minimizing |a/b - 3/2|.
    Uses linear sieve for SPF, computes sigma via multiplicative recurrence,
    then for each sigma group uses two-pointer to find pairs minimizing |2a-3b|/(2b).
    """
    deadline = time.time() + 80.0  # 10s safety margin before 90s kill
    N = 10_000_000

    # --- Linear sieve for SPF ---
    spf = [0] * (N + 1)
    spf[1] = 1
    primes = []
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
        for p in primes:
            if p > spf[i] or i * p > N:
                break
            spf[i * p] = p

    # --- Compute sigma(n) for all n using SPF ---
    # For n = p^k * m where gcd(p, m) = 1 and p = spf[n]:
    #   sigma(n) = sigma(m) * (1 + p + p^2 + ... + p^k)
    # We track:
    #   pk[n] = p^k (the full power of spf[n] in n)
    #   spk[n] = 1 + p + ... + p^k
    sigma = [0] * (N + 1)
    sigma[1] = 1
    pk = [1] * (N + 1)    # p^k
    spk = [1] * (N + 1)   # 1 + p + ... + p^k

    for n in range(2, N + 1):
        p = spf[n]
        m = n // p
        if m % p == 0:
            # p still divides m, so n = p^(k+1) * m'
            pk[n] = pk[m] * p
            spk[n] = spk[m] + pk[n]
            sigma[n] = sigma[m // pk[m]] * spk[n]
        else:
            # p does not divide m
            pk[n] = p
            spk[n] = 1 + p
            sigma[n] = sigma[m] * spk[n]

    # --- Group numbers by sigma value ---
    # Build dict: sigma_value -> list of n
    groups = defaultdict(list)
    for n in range(1, N + 1):
        groups[sigma[n]].append(n)

    # --- For each group, find pair (a,b) minimizing |2a-3b|/(2b) ---
    best_error = float('inf')
    best_a = 11
    best_b = 6  # fallback: sigma(11)=12, sigma(6)=12

    for s, nums in groups.items():
        if time.time() > deadline:
            break
        k = len(nums)
        if k < 2:
            continue
        # nums is already sorted (we appended in increasing n order)
        # Two-pointer: for each b, find a closest to 3b/2
        a_ptr = 0
        for b_idx in range(k):
            b = nums[b_idx]
            target = 3 * b // 2  # floor of 3b/2, we'll check around here
            # Advance a_ptr while the next element is closer to 3b/2
            while a_ptr < k - 1:
                # Compare |nums[a_ptr] - 3b/2| vs |nums[a_ptr+1] - 3b/2|
                # Using integer comparison: |2*nums[a_ptr] - 3b| vs |2*nums[a_ptr+1] - 3b|
                d1 = abs(2 * nums[a_ptr] - 3 * b)
                d2 = abs(2 * nums[a_ptr + 1] - 3 * b)
                if d2 < d1:
                    a_ptr += 1
                else:
                    break
            # Check a_ptr and a_ptr-1 (if valid)
            for idx in [a_ptr, a_ptr - 1] if a_ptr > 0 else [a_ptr]:
                a = nums[idx]
                diff = abs(2 * a - 3 * b)
                if diff == 0:
                    continue  # raw value 0 is invalid
                error = diff / (2.0 * b)
                if error < best_error:
                    best_error = error
                    best_a = a
                    best_b = b

    # Ensure types are Python int
    best_a = int(best_a)
    best_b = int(best_b)

    # Validate
    if best_a <= 0 or best_b <= 0 or best_a > 10_000_000 or best_b > 10_000_000:
        return [11, 6]

    return [best_a, best_b]
