import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s
    
    # Precompute prime factor sets for numbers up to 20000
    MAX_SUM = 20000
    # Sieve smallest prime factor
    spf = list(range(MAX_SUM + 1))
    for i in range(2, int(MAX_SUM**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def prime_factors(n):
        """Return set of prime factors of n."""
        if n <= 1:
            return set()
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return factors
    
    # Precompute prime factor sets for all numbers up to 20000
    pf_cache = [set() for _ in range(MAX_SUM + 1)]
    for i in range(2, MAX_SUM + 1):
        n = i
        while n > 1:
            p = spf[n]
            pf_cache[i].add(p)
            while n % p == 0:
                n //= p
    
    # Strategy: Try small sets of primes and find 12 numbers whose pairwise sums use only those primes
    # Try candidate prime sets in order of size
    
    # First, let's try to find a solution with a very small prime set
    # Generate candidate prime sets
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    best_result = None
    best_score = float('inf')
    
    # Try prime sets of increasing size
    for num_primes in range(2, 8):
        if time.time() > deadline - 5:
            break
        for prime_set in combinations(small_primes, num_primes):
            if time.time() > deadline - 5:
                break
            ps = set(prime_set)
            
            # Find all numbers <= 10000 such that for ALL other candidates, the sum is in ps
            # First, find all "valid" numbers: a number x is valid if for all y in our set, x+y has only primes in ps
            # This is circular, so let's use a greedy/iterative approach
            
            # Find all numbers whose "potential" pairwise sums with themselves+1, +2, etc. are smooth
            # Actually, let's find numbers x such that x + y is ps-smooth for many y
            
            # Precompute: for each number x in 1..10000, check if x+y is ps-smooth for y in 1..10000
            # That's too expensive. Instead, let's find ps-smooth numbers up to 20000
            
            smooth_numbers = []
            for s in range(2, MAX_SUM + 1):
                if pf_cache[s].issubset(ps):
                    smooth_numbers.append(s)
            
            if len(smooth_numbers) < 100:
                continue
            
            # For a number x to be in our set, x + y must be smooth for all y in our set
            # Let's try: pick x from 1..10000, and for each candidate x, 
            # find how many other numbers y in 1..10000 have x+y smooth
            # Then greedily build a set
            
            # Build adjacency: x ~ y if x+y is ps-smooth
            # We need a clique of size 12 in this graph
            
            # For efficiency, precompute which sums are smooth
            smooth_set = set(smooth_numbers)
            
            # For each x in 1..10000, compute neighbors
            # neighbors[x] = set of y in 1..10000 such that x+y is in smooth_set
            # But we need distinct x, y and x != y
            
            # Let's be smarter: try small x values first (they tend to have more smooth sums)
            candidates = list(range(1, 10001))
            
            # Greedy: repeatedly pick the number with the most compatible remaining numbers
            # Start with a few seeds and expand
            
            # Try multiple random/greedy starts
            found = None
            for attempt in range(50):
                if time.time() > deadline - 5:
                    break
                
                # Pick a starting number
                if attempt == 0:
                    start = 1
                else:
                    start = 1 + (attempt * 199) % 10000
                
                current_set = [start]
                remaining = set(candidates) - {start}
                
                for _ in range(11):
                    if time.time() > deadline - 5:
                        break
                    # Find the number in remaining that is compatible with ALL in current_set
                    best_candidate = None
                    best_compatibility = -1
                    for y in remaining:
                        compatible = True
                        for x in current_set:
                            if (x + y) not in smooth_set:
                                compatible = False
                                break
                        if compatible:
                            # Score: how many other remaining numbers are also compatible with x and y
                            count = 0
                            for z in remaining:
                                if z != y:
                                    if (y + z) in smooth_set:
                                        count += 1
                            if count > best_compatibility:
                                best_compatibility = count
                                best_candidate = y
                    if best_candidate is None:
                        break
                    current_set.append(best_candidate)
                    remaining.discard(best_candidate)
                
                if len(current_set) == 12:
                    # Verify
                    valid = True
                    for i in range(12):
                        for j in range(i+1, 12):
                            if (current_set[i] + current_set[j]) not in smooth_set:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        found = sorted(current_set)
                        break
            
            if found:
                # Compute actual score
                all_primes = set()
                for i in range(12):
                    for j in range(i+1, 12):
                        all_primes |= pf_cache[found[i] + found[j]]
                score = len(all_primes)
                if score < best_score:
                    best_score = score
                    best_result = found
                if score <= num_primes:
                    return best_result
            else:
                # No clique of size 12 found for this prime set, continue
                pass
    
    if best_result is not None:
        return best_result
    
    # Fallback: use the seed
    return list(range(1, 13))

