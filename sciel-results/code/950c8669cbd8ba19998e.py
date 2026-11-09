"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

P = {'n': 210, 'maximum_window': 10210, 'maximum_size': 100}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    n = P['n']  # 210
    max_val = P['maximum_window']  # 10210
    max_size = P['maximum_size']   # 100
    
    # 210 = 2 * 3 * 5 * 7
    # Target parity: odd exponents for 2, 3, 5, 7. Even for others.
    
    # Sieve primes up to max_val to factorize numbers
    sieve = [True] * (max_val + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if sieve[i]:
            for j in range(i*i, max_val + 1, i):
                sieve[j] = False
    primes = [i for i in range(2, max_val + 1) if sieve[i]]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
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
            else:
                # Should not happen
                odd_primes.add(-1)
        return frozenset(odd_primes)
    
    target_parity = get_parity(n)
    
    # Precompute parities for all candidates
    parities = {}
    for x in range(211, max_val + 1):
        parities[x] = get_parity(x)
        
    # Build parity map: parity -> list of numbers
    # We will use this to search for solutions
    parity_map = defaultdict(list)
    for x in range(211, max_val + 1):
        parity_map[parities[x]].append(x)
        
    # Sort the numbers in each parity list for efficient min/max checks?
    # They are already added in increasing order, so lists are sorted.
    
    best_solution = [840]
    best_max = 840
    
    # Check k=1
    # x = 210 * k^2
    # Smallest k=2 gives 840.
    # We already have this.
    
    # Search for k=2
    # We want to find x, y such that max(x,y) < best_max
    # and parity(x) ^ parity(y) == target_parity
    
    # We can iterate max_val_candidate from 211 to best_max - 1
    # For each M, check if there is x < M such that parity(x) == parity(M) ^ target_parity
    
    # To do this efficiently, we can maintain a map of parities seen so far.
    seen_parity_count = defaultdict(int)
    # Actually, we just need to know if a number exists.
    # We can iterate M from 211 to best_max - 1.
    # Before adding M to seen, check if target ^ P(M) is in seen.
    
    # Reset seen for this search phase
    seen_parity_nums = defaultdict(list)
    
    for M in range(211, best_max):
        if time.time() > deadline:
            break
        pM = parities[M]
        needed = pM ^ target_parity
        if needed in seen_parity_nums:
            # Found a partner
            # The partner is any number in seen_parity_nums[needed]
            # Since we want to minimize max, and M is the current max candidate,
            # any partner x < M works.
            # We found a valid pair with max M.
            # Since we are iterating M upwards, this is the smallest possible max for k=2.
            # We can break here.
            x = seen_parity_nums[needed][0] # Just pick one
            best_solution = [x, M]
            best_max = M
            break
        seen_parity_nums[pM].append(M)
        
    # If we found a better solution with k=2, best_max is updated.
    # Now search for k=3 with max < best_max
    if time.time() < deadline:
        # We need x < y < z < best_max
        # parity(x) ^ parity(y) ^ parity(z) == target_parity
        # Iterate z from 211 to best_max - 1
        # Iterate y from 211 to z - 1
        # Check if target ^ P(z) ^ P(y) exists in numbers < y
        
        # To make this fast, we can precompute for each y, the set of parities of numbers < y?
        # Or just use the parity_map and check if min number in list < y.
        
        # Since best_max is likely small (e.g., < 300), the loop is small.
        found_k3 = False
        for z in range(211, best_max):
            if time.time() > deadline:
                break
            pz = parities[z]
            for y in range(211, z):
                if time.time() > deadline:
                    break
                py = parities[y]
                needed = target_parity ^ pz ^ py
                if needed in parity_map:
                    # Check if there is a number x < y with this parity
                    # parity_map[needed] is sorted
                    # We need the first element < y
                    nums = parity_map[needed]
                    # Since y is small, we can just check the first few or use binary search
                    # But list is small.
                    for x in nums:
                        if x < y:
                            # Found x, y, z
                            best_solution = [x, y, z]
                            best_max = z
                            found_k3 = True
                            break
                        else:
                            break # x is getting larger, no point
                    if found_k3:
                        break
            if found_k3:
                break
                
    # Verify the solution
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

    if not verify(best_solution):
        # Fallback to known valid solution
        best_solution = [840]
        best_max = 840
        
    return list(best_solution)
# EVOLVE-BLOCK-END
