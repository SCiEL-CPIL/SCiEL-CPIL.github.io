import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    # Primes <= 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # For each prime, find all numbers <= 40 that are pure powers of that prime
    # and determine the maximum dissociated subset of exponents
    prime_powers = {}
    for p in primes:
        powers = []
        e = 1
        val = p
        while val <= 40:
            powers.append((val, e))
            e += 1
            val = p ** e
        prime_powers[p] = powers
    
    # For each prime, find all dissociated subsets of exponents
    def max_dissociated(exponents):
        """Find max size subset of exponents with all distinct subset sums."""
        best = 0
        n = len(exponents)
        for size in range(n, 0, -1):
            for subset in combinations(exponents, size):
                # Check if all subset sums are distinct
                sums = set()
                valid = True
                for r in range(size + 1):
                    for combo in combinations(subset, r):
                        s = sum(combo)
                        if s in sums:
                            valid = False
                            break
                        sums.add(s)
                    if not valid:
                        break
                if valid:
                    return size
        return 0
    
    # Calculate maximum per-prime contributions
    max_per_prime = {}
    for p in primes:
        exponents = [e for (_, e) in prime_powers[p]]
        max_per_prime[p] = max_dissociated(exponents)
    
    # The theoretical maximum is the sum of max_per_prime
    theoretical_max = sum(max_per_prime.values())
    
    # Now find an actual valid set achieving this maximum
    # For each prime, find a dissociated subset of exponents of max size
    # and map back to actual numbers
    
    def find_dissociated_subset(exponents, target_size):
        """Find a dissociated subset of exactly target_size."""
        n = len(exponents)
        for subset in combinations(exponents, target_size):
            sums = set()
            valid = True
            for r in range(target_size + 1):
                for combo in combinations(subset, r):
                    s = sum(combo)
                    if s in sums:
                        valid = False
                        break
                    sums.add(s)
                if not valid:
                    break
            if valid:
                return list(subset)
        return None
    
    result = []
    for p in primes:
        exponents = [e for (_, e) in prime_powers[p]]
        target = max_per_prime[p]
        if target > 0:
            subset = find_dissociated_subset(exponents, target)
            if subset:
                for e in subset:
                    result.append(p ** e)
    
    # Verify the result
    def verify_valid(S):
        """Check all subset products are distinct using exact integer arithmetic."""
        if not S:
            return True
        products = set()
        n = len(S)
        for mask in range(1 << n):
            prod = 1
            for i in range(n):
                if mask & (1 << i):
                    prod *= S[i]
            if prod in products:
                return False
            products.add(prod)
        return True
    
    if not verify_valid(result):
        # Fallback: use the known valid construction
        result = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Ensure all values are in range and are Python ints
    result = [int(x) for x in result if isinstance(x, int) and 2 <= x <= 40]
    
    # If we haven't hit the theoretical max, try a quick search for improvement
    if len(result) < theoretical_max and time.time() < deadline - 5:
        # Try to find a better set via local search
        candidates = list(range(2, 41))
        best = result[:]
        
        # Try adding one element at a time
        for x in candidates:
            if x not in best and time.time() > deadline - 3:
                break
            trial = best + [x]
            if len(trial) <= 18 and verify_valid(trial):
                best = trial
    
    return best
