import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s limit, 10s margin
    
    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.sqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]

    primes = sieve(1000)
    
    # Define candidate sets of primes
    # Set 1: Seed-like (13..997)
    # Set 2: Include 3, exclude 7, 11 to keep sum <= 1
    # Sum(3) + Sum(13..997) - Sum(7) - Sum(11)
    # 0.33 + 0.67 - 0.14 - 0.09 = 0.77? 
    # Wait, Sum(13..997) was 0.67.
    # Sum(7, 11) = 0.23.
    # Sum(3) = 0.33.
    # If we replace 7, 11 with 3: 0.67 - 0.23 + 0.33 = 0.77.
    # We have budget 0.23 left.
    # Can we add 5? 1/5 = 0.20.
    # 0.77 + 0.20 = 0.97 <= 1.
    # So set {3, 5, 13, 17, ..., 997} is valid.
    
    candidates = [
        [p for p in primes if p >= 13], # Seed
        [3, 5] + [p for p in primes if p >= 13], # Improved
        [3, 11] + [p for p in primes if p >= 13 and p != 7], # Alternative
        [5, 7, 11] + [p for p in primes if p >= 13], # Alternative
    ]
    
    best_L = 0
    best_result = []
    
    for P in candidates:
        # Check budget
        budget = sum(1.0/p for p in P)
        if budget > 1.0:
            continue
            
        # Greedy assignment
        assigned = {}
        covered = [False] * 2000 # Assume L < 2000
        # Actually, we need to check coverage dynamically
        
        # Sort primes by size (smallest first) to prioritize high density
        P_sorted = sorted(P)
        
        current_L = 0
        n = 1
        while n < 2000:
            # Check if n is covered
            is_covered = False
            for p in assigned:
                if n % p == assigned[p]:
                    is_covered = True
                    break
            
            if is_covered:
                n += 1
                current_L = n
                continue
            
            # Not covered. Must assign a prime.
            # Find a prime in P not in assigned
            # Choose the one that covers the most future uncovered numbers?
            # Heuristic: Smallest prime available.
            chosen_p = None
            for p in P_sorted:
                if p not in assigned:
                    chosen_p = p
                    break
            
            if chosen_p is None:
                break # No more primes to assign
            
            # Assign residue
            r = n % chosen_p
            assigned[chosen_p] = r
            
            # Update covered status for this prime
            # We don't need to update the array immediately if we check on the fly,
            # but for speed, we can mark covered numbers.
            # However, we are iterating n sequentially.
            # Just set n to n+1 and let the loop handle it.
            # But we must ensure we don't skip checking.
            
            # Optimization: Jump to the next uncovered number?
            # If we assigned p, it covers n, n+p, n+2p...
            # The next uncovered number might be n+1.
            
            n += 1
            
            # Deadline check
            if time.time() > deadline:
                break
        
        if current_L > best_L:
            best_L = current_L
            best_result = [[p, assigned[p]] for p in assigned]
            
            # Early exit if we hit the target
            if best_L >= 999:
                # Try to push for higher?
                # The prompt says maximize.
                # If we have time, we can try more complex searches.
                # But for now, this is a solid improvement over seed if seed is suboptimal.
                pass

    # Final check and return
    # Ensure we return a valid list
    if not best_result:
        # Fallback to seed
        primes_seed = [p for p in primes if p >= 13]
        return [[p, (i + 1) % p] for i, p in enumerate(primes_seed)]
        
    return best_result

# EVOLVE-BLOCK-END
