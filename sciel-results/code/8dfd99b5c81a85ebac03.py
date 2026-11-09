import math
import time
from collections import defaultdict

P = {'n': 210, 'maximum_window': 10210, 'maximum_size': 100}

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill

    n = P['n']  # 210
    max_val = P['maximum_window']  # 10210
    max_size = P['maximum_size']  # 100

    # 210 = 2 * 3 * 5 * 7 (square-free)
    # Target: product of selected numbers * 210 must be a perfect square.
    # This means the parity vector of the product of selected numbers must equal
    # the parity vector of 210 over all primes (odd for 2,3,5,7; even for all else).

    # Sieve for primes up to max_val
    sieve = [True] * (max_val + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if sieve[i]:
            for j in range(i * i, max_val + 1, i):
                sieve[j] = False
    primes = [i for i in range(2, max_val + 1) if sieve[i]]
    prime_to_idx = {p: i for i, p in enumerate(primes)}

    # Target parity: 210 = 2*3*5*7, all exponent 1 (odd)
    target_primes = [2, 3, 5, 7]
    target_set = frozenset(prime_to_idx[p] for p in target_primes if p in prime_to_idx)

    def get_parity(x):
        """Return frozenset of prime indices with odd exponent in x."""
        odd_primes = set()
        temp = x
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                cnt = 0
                while temp % p == 0:
                    temp //= p
                    cnt += 1
                if cnt % 2 == 1:
                    odd_primes.add(prime_to_idx[p])
        if temp > 1:
            if temp in prime_to_idx:
                odd_primes.add(prime_to_idx[temp])
        return frozenset(odd_primes)

    # Precompute parities for numbers 211 to max_val
    parities = {}
    for x in range(211, max_val + 1):
        parities[x] = get_parity(x)

    best_result = [840]  # fallback: 840*210 = 176400 = 420^2
    best_raw = 630
    current_best_max = 840

    # Build a map from parity to sorted list of numbers
    parity_map = defaultdict(list)
    for x in range(211, max_val + 1):
        parity_map[parities[x]].append(x)
    # Each list is already sorted since we iterate x in increasing order

    # Search for size 1: x such that x*210 is a perfect square
    # x = 210 * k^2, need 211 <= x <= max_val
    for k in range(1, 20):
        x = 210 * k * k
        if x > 210 and x <= max_val:
            if _is_square(x * 210):
                raw = x - 210
                if raw < best_raw:
                    best_raw = raw
                    best_result = [x]
                    current_best_max = x

    # Search for size 2: x*y*210 is a perfect square
    # parity(x) XOR parity(y) = target_set
    # For each x, find y with parity = parity(x) XOR target_set, y != x, y > 210
    # To minimize max(x,y), we want both small.
    # Iterate x from 211 upward, and for each x, find the smallest valid y.

    for x in range(211, current_best_max):
        if time.time() > deadline:
            break
        px = parities[x]
        py = px ^ target_set
        if py in parity_map:
            # Find smallest y in parity_map[py] such that y != x and y > 210
            # Since list is sorted, we can find the first valid y
            for y in parity_map[py]:
                if y == x:
                    continue
                if y <= 210:
                    continue
                max_xy = max(x, y)
                if max_xy < current_best_max:
                    # Verify
                    prod = x * y * 210
                    if _is_square(prod):
                        current_best_max = max_xy
                        best_raw = max_xy - 210
                        best_result = [x, y]
                        break  # Found best y for this x, move to next x
                else:
                    break  # y is too large, no point continuing for this x

    # Search for size 3: x*y*z*210 is a perfect square
    # parity(x) XOR parity(y) XOR parity(z) = target_set
    # Fix x and y, find z with parity = target_set XOR parity(x) XOR parity(y)
    # To keep max small, use small x and y.

    if best_raw > 50 and time.time() < deadline:
        # Use small numbers for x and y
        small_numbers = list(range(211, min(400, max_val + 1)))
        for i in range(len(small_numbers)):
            if time.time() > deadline:
                break
            x = small_numbers[i]
            for j in range(i + 1, len(small_numbers)):
                if time.time() > deadline:
                    break
                y = small_numbers[j]
                needed_parity = target_set ^ parities[x] ^ parities[y]
                if needed_parity in parity_map:
                    # Find smallest z with that parity, z != x, z != y, z > 210
                    for z in parity_map[needed_parity]:
                        if z == x or z == y or z <= 210:
                            continue
                        max_xyz = max(x, y, z)
                        if max_xyz < current_best_max:
                            prod = x * y * z * 210
                            if _is_square(prod):
                                current_best_max = max_xyz
                                best_raw = max_xyz - 210
                                best_result = [x, y, z]
                                break
                    # Continue to find potentially better solutions

    # Verify the best result
    def verify(solution):
        if not isinstance(solution, (list, tuple)):
            return False
        if len(solution) < 1 or len(solution) > max_size:
            return False
        if len(set(solution)) != len(solution):
            return False
        for x in solution:
            if not isinstance(x, int) or isinstance(x, bool):
                return False
            if x <= n or x > max_val:
                return False
        prod = n
        for x in solution:
            prod *= x
        r = math.isqrt(prod)
        if r * r != prod:
            return False
        return True

    if not verify(best_result):
        # Fallback to known good solution
        best_result = [840]
        best_raw = 630

    # Return as list of Python ints
    return [int(x) for x in best_result]
