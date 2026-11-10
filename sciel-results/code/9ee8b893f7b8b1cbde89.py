import time
from math import gcd

def construct():
        start_time = time.time()
        deadline = start_time + 110  # Safety margin
        
        # 1. Generate primes < 512
        def is_prime(n):
            if n < 2: return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0: return False
            return True
        
        primes = [p for p in range(2, 512) if is_prime(p)]
        
        # Initial set: all primes
        current_set = set(primes)
        
        def score(s):
            return sum(1.0 / (512 - a) for a in s)
        
        best_score = score(current_set)
        best_set = set(current_set)
        
        # Helper to get prime factors
        def prime_factors(n):
            factors = []
            d = 2
            while d * d <= n:
                while n % d == 0:
                    factors.append(d)
                    n //= d
                d += 1
            if n > 1:
                factors.append(n)
            return factors
            
        # Local search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try merging pairs
            # We need to check all pairs in current_set
            # To be efficient, we can just iterate
            items = list(current_set)
            for i in range(len(items)):
                if time.time() > deadline: break
                for j in range(i+1, len(items)):
                    a = items[i]
                    b = items[j]
                    prod = a * b
                    if prod >= 512:
                        continue
                    # Check if prod is coprime to rest
                    # Since a, b are in set and set is pairwise coprime, 
                    # prod is coprime to all others in set \ {a,b}
                    # We just need to check if the gain is positive
                    old_val = 1.0/(512-a) + 1.0/(512-b)
                    new_val = 1.0/(512-prod)
                    if new_val > old_val + 1e-12:
                        current_set.remove(a)
                        current_set.remove(b)
                        current_set.add(prod)
                        improved = True
                        # Update best
                        curr_score = score(current_set)
                        if curr_score > best_score:
                            best_score = curr_score
                            best_set = set(current_set)
                        # Restart loop? Or continue? 
                        # Breaking inner loop to restart outer is safer for consistency
                        break
                if improved:
                    break
            
            # Try splitting
            # Check each element in current_set
            items = list(current_set)
            for a in items:
                if time.time() > deadline: break
                if a in primes:
                    continue # Can't split prime
                factors = prime_factors(a)
                # Check if sum of weights of factors > weight of a
                # Factors are distinct? No, a might be p^k.
                # If a = p^2, factors are [p, p]. But we can only select p once.
                # Wait, the set contains distinct integers.
                # If we split a into its prime factors, we add those primes to the set.
                # We must ensure they are not already in the set.
                # Since a was in the set, and set was pairwise coprime,
                # no other element in set shares factors with a.
                # So the prime factors of a are not in the set.
                # However, if a = p^2, the factor is p (with multiplicity 2 in a, but 1 in set).
                # We add p to the set.
                # Is p coprime to others? Yes.
                # So we can always split a into its distinct prime factors.
                
                distinct_factors = set(prime_factors(a))
                old_val = 1.0/(512-a)
                new_val = sum(1.0/(512-p) for p in distinct_factors)
                
                if new_val > old_val + 1e-12:
                    current_set.remove(a)
                    current_set.update(distinct_factors)
                    improved = True
                    curr_score = score(current_set)
                    if curr_score > best_score:
                        best_score = curr_score
                        best_set = set(current_set)
                    break
            
            # Try adding a new number?
            # This is harder. We would need to find a number a < 512
            # such that its prime factors are a subset of the prime factors of elements in current_set
            # that we are willing to remove.
            # This is essentially the "Replace" move.
            # Given time, the merge/split heuristic might be sufficient.
            # However, consider the case where we have {2, 3, 5} and 2*3*5=30 is better?
            # Unlikely for small primes.
            # What if we have {2, 251} and we can add 3*251? No, 251 is used.
            
            # Let's add a "Replace" heuristic:
            # Try to replace a subset of current_set with a new number.
            # This is expensive.
            # Alternative: Randomized restarts?
            # Given the structure, the greedy merge from primes is likely very strong.
            
        return list(best_set)
