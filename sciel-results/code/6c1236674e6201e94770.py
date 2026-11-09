import math
import time

P = {'n': 210, 'maximum_window': 10210, 'maximum_size': 100}

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    n = 210
    max_val = 10210
    max_size = 100

    # 210 = 2 * 3 * 5 * 7 (square-free)
    # Target parity: odd exponent for primes 2, 3, 5, 7; even for all others.
    # We need product of selected * 210 to be a perfect square.
    # So XOR of parities of selected numbers must equal parity of 210.

    # Sieve for primes up to max_val
    sieve = [True] * (max_val + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if sieve[i]:
            for j in range(i * i, max_val + 1, i):
                sieve[j] = False
    primes = [i for i in range(2, max_val + 1) if sieve[i]]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)

    # Target parity: 210 = 2*3*5*7
    target_primes = [2, 3, 5, 7]
    target_mask = 0
    for p in target_primes:
        if p in prime_to_idx:
            target_mask |= (1 << prime_to_idx[p])

    def get_parity_mask(x):
        """Return bitmask of prime indices with odd exponent in x."""
        mask = 0
        temp = x
        for i, p in enumerate(primes):
            if p * p > temp:
                break
            if temp % p == 0:
                cnt = 0
                while temp % p == 0:
                    temp //= p
                    cnt += 1
                if cnt % 2 == 1:
                    mask |= (1 << i)
        if temp > 1:
            if temp in prime_to_idx:
                mask |= (1 << prime_to_idx[temp])
        return mask

    # Precompute parities for numbers 211 to max_val
    parities = [0] * (max_val + 1)
    for x in range(211, max_val + 1):
        parities[x] = get_parity_mask(x)

    # Build parity_map: map from parity mask to sorted list of numbers
    from collections import defaultdict
    parity_map = defaultdict(list)
    for x in range(211, max_val + 1):
        parity_map[parities[x]].append(x)

    best_result = [840]  # fallback: 840*210 = 176400 = 420^2
    best_raw = 630
    current_best_max = 840

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
    # parity(x) XOR parity(y) = target_mask
    # To minimize max(x,y), iterate over possible max values M from 211 upward
    # For each M, check all pairs (x,y) with x < y <= M
    
    # Build a set for quick lookup of numbers by parity
    # For each parity, store numbers sorted
    # We'll search systematically: for M from 212 to current_best_max-1,
    # check if there exists x,y with x < y <= M such that parity(x)^parity(y)=target_mask
    
    found_size2 = False
    for M in range(212, current_best_max):
        if time.time() > deadline:
            break
        # For each y from 211 to M, check if there's x < y with parity(x) = parity(y) ^ target_mask
        # and x > 210
        found = False
        for y in range(M, 210, -1):  # Try largest y first to potentially find solution faster
            py = parities[y]
            px_needed = py ^ target_mask
            # Check if there's an x in (210, y) with parity px_needed
            # We can use parity_map but need x < y
            # Simpler: just check a few candidates
            # Actually, let's be smarter: precompute for each parity, the list of numbers
            # and use binary search or just scan
            if px_needed in parity_map:
                # Find if any number in parity_map[px_needed] is < y and > 210
                # parity_map[px_needed] is sorted, so we can find the largest < y
                plist = parity_map[px_needed]
                # Binary search for rightmost element < y
                lo, hi = 0, len(plist)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if plist[mid] < y:
                        lo = mid + 1
                    else:
                        hi = mid
                # lo is now the index of first element >= y, so lo-1 is last element < y
                if lo > 0:
                    x = plist[lo - 1]
                    if x > 210 and x != y:
                        # Verify
                        prod = x * y * 210
                        if _is_square(prod):
                            current_best_max = M
                            best_raw = M - 210
                            best_result = [x, y]
                            found = True
                            break
        if found:
            found_size2 = True
            break

    # If we found a size-2 solution, also try to find even better ones with more elements
    # Search for size 3: x*y*z*210 is a perfect square
    # For M from 213 to current_best_max-1, check if there exist x<y<z<=M
    # such that parity(x)^parity(y)^parity(z)=target_mask
    
    if time.time() < deadline:
        # Use a more efficient approach: for each pair (x,y) with x<y,
        # compute needed parity for z, then check if any z>y with that parity exists and z<=current_best_max-1
        # To minimize max, iterate M upward
        
        # Precompute: for each parity, store sorted list (already done in parity_map)
        # For efficiency, limit search range
        
        search_limit = min(current_best_max - 1, 500)
        for M in range(213, search_limit + 1):
            if time.time() > deadline:
                break
            found = False
            # Try pairs (x,y) with x < y < M (since z must be <= M and z > y)
            # To keep it fast, limit x and y to reasonable ranges
            max_x = min(M - 2, 300)
            for x in range(211, max_x + 1):
                if time.time() > deadline:
                    break
                px = parities[x]
                for y in range(x + 1, M):
                    if time.time() > deadline:
                        break
                    py = parities[y]
                    pz_needed = target_mask ^ px ^ py
                    if pz_needed in parity_map:
                        plist = parity_map[pz_needed]
                        # Find if any z in plist with y < z <= M
                        # Binary search for first element > y
                        lo, hi = 0, len(plist)
                        while lo < hi:
                            mid = (lo + hi) // 2
                            if plist[mid] <= y:
                                lo = mid + 1
                            else:
                                hi = mid
                        # lo is first index where plist[lo] > y
                        if lo < len(plist) and plist[lo] <= M:
                            z = plist[lo]
                            if z != x and z != y:
                                prod = x * y * z * 210
                                if _is_square(prod):
                                    current_best_max = M
                                    best_raw = M - 210
                                    best_result = [x, y, z]
                                    found = True
                                    break
                if found:
                    break
            if found:
                break

    # Search for size 4: x*y*z*w*210 is a perfect square
    if time.time() < deadline and best_raw > 20:
        search_limit = min(current_best_max - 1, 400)
        for M in range(214, search_limit + 1):
            if time.time() > deadline:
                break
            found = False
            max_x = min(M - 3, 250)
            max_y = min(M - 2, 250)
            max_z = min(M - 1, 250)
            for x in range(211, max_x + 1):
                if time.time() > deadline:
                    break
                px = parities[x]
                for y in range(x + 1, max_y + 1):
                    if time.time() > deadline:
                        break
                    py = parities[y]
                    for z in range(y + 1, max_z + 1):
                        if time.time() > deadline:
                            break
                        pz = parities[z]
                        pw_needed = target_mask ^ px ^ py ^ pz
                        if pw_needed in parity_map:
                            plist = parity_map[pw_needed]
                            # Find if any w in plist with z < w <= M
                            lo, hi = 0, len(plist)
                            while lo < hi:
                                mid = (lo + hi) // 2
                                if plist[mid] <= z:
                                    lo = mid + 1
                                else:
                                    hi = mid
                            if lo < len(plist) and plist[lo] <= M:
                                w = plist[lo]
                                if w != x and w != y and w != z:
                                    prod = x * y * z * w * 210
                                    if _is_square(prod):
                                        current_best_max = M
                                        best_raw = M - 210
                                        best_result = [x, y, z, w]
                                        found = True
                                        break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if found:
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
        # Fallback to known good solution
        best_result = [840]
        best_raw = 630

    # Return as list of Python ints
    return [int(x) for x in best_result]
