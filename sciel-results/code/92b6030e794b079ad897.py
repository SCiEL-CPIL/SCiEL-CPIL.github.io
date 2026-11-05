import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Primes up to enough to cover product <= 10^18
    # 2*3*5*...*47 is ~6e17. Next prime 53 makes it > 3e19.
    # So max k is around 15.
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

    def get_divisors_and_score(powers):
        """
        powers: list of (prime, exponent).
        Returns (score, n) or (-1, -1) if invalid.
        """
        n = 1
        for p, e in powers:
            if e == 0:
                continue
            n *= (p ** e)
            if n > 10**18:
                return -1, -1
        if n < 2:
            return -1, -1
        
        # Generate divisors
        divs = [1]
        for p, e in powers:
            if e == 0:
                continue
            current_len = len(divs)
            power = 1
            for _ in range(e):
                power *= p
                for i in range(current_len):
                    divs.append(divs[i] * power)
        
        if len(divs) < 2:
            return 0, n

        divs.sort()
        
        s = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i + 1]) == 1:
                s += 1
        return s, n

    # Find max k for square-free product of first k primes
    prefix_prod = [1]
    for p in primes:
        prefix_prod.append(prefix_prod[-1] * p)

    max_k = 0
    for i in range(1, len(primes) + 1):
        if prefix_prod[i] <= 10**18:
            max_k = i
        else:
            break

    best_score = 0
    best_n = 2
    best_powers = [(2, 1)]

    def try_update(powers):
        nonlocal best_score, best_n, best_powers
        if time.time() > deadline:
            return False
        s, n = get_divisors_and_score(powers)
        if s > best_score:
            best_score = s
            best_n = n
            best_powers = [p for p in powers]
            return True
        return False

    # Initialize with a few starting seeds: square-free products of first k, k-1, k-2 primes
    seeds = []
    for k in range(max(1, max_k - 2), max_k + 1):
        seed = [(primes[i], 1) for i in range(k)]
        seeds.append(seed)

    # Also try a seed with slightly fewer primes but potentially more room for exponents
    # e.g. first 10 primes
    if max_k > 10:
        seeds.append([(primes[i], 1) for i in range(10)])
    
    # Remove duplicates in seeds (by tuple of powers)
    unique_seeds = []
    seen = set()
    for s in seeds:
        key = tuple(s)
        if key not in seen:
            seen.add(key)
            unique_seeds.append(s)

    for seed in unique_seeds:
        if time.time() > deadline:
            break
        
        current_powers = [p for p in seed]
        # Ensure we evaluate the seed first
        try_update(current_powers)
        
        # Local Search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try all possible moves
            # 1. Increase exponent of existing prime
            # 2. Decrease exponent of existing prime
            # 3. Add a new prime (exponent 1)
            
            # Get list of active primes in current_powers
            active_indices = [i for i, (p, e) in enumerate(current_powers) if e > 0]
            
            # Try increasing exponents
            for idx in active_indices:
                p, e = current_powers[idx]
                new_powers = [list(x) for x in current_powers]
                new_powers[idx][1] = e + 1
                if try_update(new_powers):
                    current_powers = [tuple(x) for x in new_powers]
                    improved = True
                    break # Restart loop with new state
            
            if improved:
                continue
                
            # Try decreasing exponents
            for idx in active_indices:
                p, e = current_powers[idx]
                if e > 1:
                    new_powers = [list(x) for x in current_powers]
                    new_powers[idx][1] = e - 1
                    if try_update(new_powers):
                        current_powers = [tuple(x) for x in new_powers]
                        improved = True
                        break
            
            if improved:
                continue

            # Try adding a new prime
            # We can try adding any prime from our list that is not already present with e>0
            # To keep it efficient, we only try adding primes smaller than the largest active prime?
            # Or just try the first few unused primes.
            active_primes = {p for p, e in current_powers if e > 0}
            for p in primes:
                if p in active_primes:
                    continue
                # Check if we can add it
                n_curr = 1
                for pp, ee in current_powers:
                    n_curr *= (pp ** ee)
                    if n_curr > 10**18 // p:
                        break
                if n_curr <= 10**18 // p:
                    new_powers = [list(x) for x in current_powers]
                    # Append the new prime
                    new_powers.append([p, 1])
                    if try_update(new_powers):
                        current_powers = [tuple(x) for x in new_powers]
                        improved = True
                        break
                # Heuristic: if we can't add small primes, we probably can't add larger ones either
                # But we should check a few. Let's limit to first 20 primes for adding.
                if p > 97:
                    break

    return best_n
