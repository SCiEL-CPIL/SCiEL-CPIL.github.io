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
    deadline = start_time + 85.0  # Safety margin before 90s kill

    n = P['n']  # 210
    max_val = P['maximum_window']  # 10210
    max_size = P['maximum_size']  # 100

    # 210 = 2 * 3 * 5 * 7 (square-free)
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

    # Search for size 1: x such that x*210 is a perfect square
    for k in range(1, 200):
        x = 210 * k * k
        if x > max_val:
            break
        if x > 210:
            if _is_square(x * 210):
                raw = x - 210
                if raw < best_raw:
                    best_raw = raw
                    best_result = [x]
                    current_best_max = x

    # Search for size 2: x*y*210 is a perfect square
    # parity(x) XOR parity(y) = target_set
    for x in range(211, current_best_max + 1):
        if time.time() > deadline:
            break
        px = parities[x]
        py = px ^ target_set
        if py in parity_map:
            for y in parity_map[py]:
                if y == x:
                    continue
                max_xy = max(x, y)
                if max_xy >= current_best_max:
                    break
                prod = x * y * 210
                if _is_square(prod):
                    current_best_max = max_xy
                    best_raw = max_xy - 210
                    best_result = [x, y]
                    break

    # Search for size 3: x*y*z*210 is a perfect square
    if best_raw > 50 and time.time() < deadline:
        search_limit = min(600, max_val + 1)
        small_numbers = list(range(211, search_limit))
        for i in range(len(small_numbers)):
            if time.time() > deadline:
                break
            x = small_numbers[i]
            if x >= current_best_max:
                break
            for j in range(i + 1, len(small_numbers)):
                if time.time() > deadline:
                    break
                y = small_numbers[j]
                if y >= current_best_max:
                    break
                needed_parity = target_set ^ parities[x] ^ parities[y]
                if needed_parity in parity_map:
                    for z in parity_map[needed_parity]:
                        if z == x or z == y:
                            continue
                        max_xyz = max(x, y, z)
                        if max_xyz >= current_best_max:
                            break
                        prod = x * y * z * 210
                        if _is_square(prod):
                            current_best_max = max_xyz
                            best_raw = max_xyz - 210
                            best_result = [x, y, z]
                            break

    # Search for size 4: x*y*z*w*210 is a perfect square
    if best_raw > 50 and time.time() < deadline:
        search_limit = min(350, max_val + 1)
        small_numbers = list(range(211, search_limit))
        for i in range(len(small_numbers)):
            if time.time() > deadline:
                break
            x = small_numbers[i]
            if x >= current_best_max:
                break
            for j in range(i + 1, len(small_numbers)):
                if time.time() > deadline:
                    break
                y = small_numbers[j]
                if y >= current_best_max:
                    break
                for k in range(j + 1, len(small_numbers)):
                    if time.time() > deadline:
                        break
                    z = small_numbers[k]
                    if z >= current_best_max:
                        break
                    needed_parity = target_set ^ parities[x] ^ parities[y] ^ parities[z]
                    if needed_parity in parity_map:
                        for w in parity_map[needed_parity]:
                            if w == x or w == y or w == z:
                                continue
                            max_xyzw = max(x, y, z, w)
                            if max_xyzw >= current_best_max:
                                break
                            prod = x * y * z * w * 210
                            if _is_square(prod):
                                current_best_max = max_xyzw
                                best_raw = max_xyzw - 210
                                best_result = [x, y, z, w]
                                break

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
        best_result = [840]
        best_raw = 630

    return [int(x) for x in best_result]
