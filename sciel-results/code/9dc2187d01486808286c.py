import time
from fractions import Fraction
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin

    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(1000)
    
    # Generate candidate prime sets using greedy with various skips
    def build_candidate_set(skip_primes):
        s = Fraction(0, 1)
        final_set = []
        for p in primes:
            if p in skip_primes:
                continue
            if s + Fraction(1, p) <= 1:
                final_set.append(p)
                s += Fraction(1, p)
        return final_set

    # We want sets that can cover a long prefix.
    # Including small primes is generally good for density, but we need to balance.
    # The "Erdos-Ruzsa" hint suggests we should aim for L >= 999.
    # A set of primes with sum 1/p <= 1.
    
    # Strategy:
    # 1. Generate several candidate sets of primes.
    # 2. For each set, use a greedy sequential algorithm to assign residues to cover 1..L.
    #    This avoids the "global" optimization trap and focuses on the prefix.
    # 3. Use NumPy for fast coverage checks.

    candidates = []
    # Set 1: Smallest primes {2, 3, 7}
    candidates.append(build_candidate_set([]))
    # Set 2: Skip 3 {2, 5, 7, 11, 13, ...}
    candidates.append(build_candidate_set({3}))
    # Set 3: Skip 3, 5 {2, 7, 11, 13, ...}
    candidates.append(build_candidate_set({3, 5}))
    # Set 4: Skip 2 {3, 5, 7, 11, ...}
    candidates.append(build_candidate_set({2}))
    # Set 5: Skip 2, 3 {5, 7, 11, ...}
    candidates.append(build_candidate_set({2, 3}))
    # Set 6: Skip 3, 7 {2, 5, 11, 13, ...}
    candidates.append(build_candidate_set({3, 7}))
    # Set 7: Skip 2, 5 {3, 7, 11, ...}
    candidates.append(build_candidate_set({2, 5}))
    
    best_L = 0
    best_classes = []
    
    # Evaluation limit for the prefix. 
    # The target is 999, but let's aim a bit higher to be safe, say 2000 or 5000.
    # If we can cover 1000, we score well. 
    # NumPy makes checking large ranges fast.
    eval_limit = 5000
    
    for P in candidates:
        if time.time() > deadline - 5:
            break
        if not P:
            continue
        
        # Sort primes for the greedy process
        P_sorted = sorted(P)
        
        # Greedy sequential assignment
        # We iterate n from 1 to eval_limit.
        # If n is not covered, we must assign a residue to one of the primes to cover n.
        # Which prime? 
        # Heuristic: Pick the prime p in P such that n mod p is "best" or simply the smallest p?
        # Actually, we can pick any p in P and set r = n % p.
        # But we can only set r for each p ONCE.
        # So, if n is uncovered, we MUST pick a prime p that hasn't had its residue set yet?
        # No, the residue r is fixed for the whole class. 
        # If we pick p and set r = n % p, then ALL numbers congruent to n mod p are covered.
        # This might cover future numbers, but it might also "waste" the prime if it doesn't help much.
        
        # Wait, the problem says "construct() -> classes [[p, r], ...]".
        # The residue r is chosen. It is not assigned dynamically per n.
        # However, we can CHOOSE r to optimize coverage.
        
        # Greedy Algorithm for Residue Selection:
        # 1. Start with no residues assigned.
        # 2. Iterate n = 1, 2, ...
        # 3. If n is covered by current assignments, continue.
        # 4. If n is NOT covered, we must pick a prime p from P that hasn't been "used" to cover a previous gap?
        #    No, we can pick any prime p from P. We set r_p = n % p.
        #    This fixes r_p. 
        #    We should pick the prime p that maximizes future coverage or is "cheapest" in some sense.
        #    A simple heuristic: Pick the smallest prime p in P that is not yet assigned?
        #    Or pick the prime p that covers the most NEW numbers in the range [n, eval_limit]?
        
        # Let's try:
        # Maintain a set of available primes.
        # When we hit an uncovered n:
        #   For each available prime p:
        #     Calculate how many numbers in [n, eval_limit] would be covered by setting r_p = n % p.
        #     (Note: some might already be covered by other primes, but let's just count potential hits).
        #   Pick the p that covers the most NEW numbers.
        #   Set r_p = n % p.
        #   Remove p from available primes.
        
        # This is a "Greedy Set Cover" variant.
        
        covered = np.zeros(eval_limit + 1, dtype=bool)
        residues = {}
        available_primes = list(P_sorted)
        
        # Precompute indices for each prime for speed?
        # Or just use numpy slicing.
        
        for n in range(1, eval_limit + 1):
            if time.time() > deadline - 2:
                break
                
            if covered[n-1]: # n-1 is index for n
                continue
            
            # n is uncovered. We must assign a prime.
            best_p = None
            best_count = -1
            
            for p in available_primes:
                r = n % p
                # If we set residue r for p, which numbers are covered?
                # k such that k % p == r.
                # Indices: r, r+p, r+2p, ...
                # In 1-based: r, r+p, ...
                # In 0-based array (index i corresponds to i+1):
                # We want indices i such that (i+1) % p == r.
                # i+1 = r + kp => i = r - 1 + kp.
                
                # Let's just count how many of these are NOT yet covered.
                # This might be slow if done naively.
                # Let's approximate by just counting total hits in range, or just pick smallest p?
                
                # Heuristic 1: Pick smallest p.
                # Why? Small p has higher density (1/p). It might cover more future numbers.
                # Let's try picking the smallest available prime.
                
                # Actually, let's try to be a bit smarter.
                # Count how many uncovered numbers in [n, eval_limit] are congruent to r mod p.
                # This is O(eval_limit/p) per prime. With ~100 primes and eval_limit=5000, it's manageable.
                
                # Optimization:
                # Instead of checking all, just pick the smallest p.
                # Let's test if "smallest p" works well.
                
                if best_p is None or p < best_p:
                    best_p = p
                    best_count = 0 # Placeholder
                    
            if best_p is None:
                break # No primes left, can't cover n
                
            r = n % best_p
            residues[best_p] = r
            available_primes.remove(best_p)
            
            # Update covered array
            # Indices i such that (i+1) % p == r
            # i = r - 1, r - 1 + p, ...
            start_idx = r - 1
            if start_idx < 0:
                start_idx += p # Should not happen if r in 0..p-1, r-1 in -1..p-2.
                # If r=0, start_idx = -1. Indices: -1, p-1, 2p-1...
                # 1-based numbers: 0 (invalid), p, 2p...
                # So for r=0, we cover p, 2p, ...
                # Index p-1, 2p-1...
            
            # Let's just loop or use numpy.
            # Numpy is faster.
            # Indices: start_idx, start_idx+p, ... < eval_limit
            indices = np.arange(start_idx, eval_limit, p)
            indices = indices[indices >= 0] # Safety
            covered[indices] = True
            
        # After the loop, check the prefix length
        # Find first False in covered
        if not any(covered):
            L = 0
        else:
            # Find first index where covered is False
            # We want the largest L such that 1..L are covered.
            # covered[0] corresponds to 1.
            # If covered[0..L-1] are True, then L is the length.
            # We need the first False index.
            try:
                first_false = np.argmax(~covered)
                if covered[first_false]:
                    L = eval_limit
                else:
                    L = first_false
            except:
                L = 0
                
        if L > best_L:
            best_L = L
            best_classes = [[p, residues[p]] for p in P if p in residues]
            
    return best_classes
