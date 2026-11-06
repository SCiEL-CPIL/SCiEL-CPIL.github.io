"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # 1. Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]
    
    primes = sieve(1000)
    
    # 2. State variables
    # Budget: sum of 1/p <= 1
    # We track the exact fraction sum to avoid floating point errors in budget checks
    current_sum = Fraction(0, 1)
    
    # Coverage state:
    # We want to cover 1..L.
    # We maintain a boolean array or a set of covered numbers?
    # Given L can be up to 10^5, a boolean array of size 100005 is efficient.
    MAX_L = 100000
    covered = [False] * (MAX_L + 1)
    
    # Current contiguous coverage length
    current_L = 0
    
    # List of selected classes [p, r]
    selected_classes = []
    
    # Precompute reciprocals as Fractions for budget check
    reciprocals = {p: Fraction(1, p) for p in primes}
    
    # To efficiently find the next best prime/residue:
    # For each candidate prime p, we can try all residues r in [0, p-1].
    # This is expensive: sum(p for p in primes) ~ 76k operations per iteration?
    # No, for a specific p, there are p residues.
    # Total operations per "step" of greedy: sum(p for p in remaining_primes).
    # If we have ~168 primes, sum is ~76,127.
    # For each (p, r), we need to check if adding it extends L.
    
    # Optimization:
    # Instead of checking all (p, r) from scratch every time,
    # we can maintain a list of "promising" candidates.
    # However, a simple greedy loop might be slow if we do full scan every time.
    
    # Let's estimate:
    # Number of primes <= 1000 is 168.
    # Budget 1 allows roughly sum(1/p) <= 1.
    # The sum of reciprocals of primes diverges, but slowly.
    # sum(1/p for p in primes <= 1000) is approx 2.2.
    # So we can select a subset.
    
    # Greedy Step:
    # 1. Identify the first uncovered number N = current_L + 1.
    # 2. We MUST cover N.
    # 3. To cover N, we must select a prime p and residue r such that N % p == r.
    #    This implies r = N % p.
    # 4. So for each prime p not yet used, the residue is FORCED to be r = N % p.
    # 5. We check if adding (p, N % p) is within budget.
    # 6. Among all valid (p, N % p), we pick the one that maximizes the NEW contiguous coverage.
    #    Why? Because covering N is mandatory to extend L.
    #    Once N is covered, we check N+1, N+2, etc.
    
    # This reduces the search space significantly!
    # At each step, we only consider 168 candidates (one per prime).
    # For each candidate, we simulate the coverage extension.
    
    # How to simulate extension efficiently?
    # We have a `covered` array.
    # If we add (p, r), it covers all k such that k % p == r.
    # We can iterate k from N upwards, checking if k is already covered OR k % p == r.
    # We stop when we hit a number that is neither.
    
    # Complexity per step:
    # 168 candidates.
    # For each candidate, we scan forward from N until a gap.
    # In the worst case, the gap is far. But L is capped at 10^5.
    # However, as L grows, the gaps between covered numbers by random primes shrink?
    # Actually, small primes cover densely.
    
    # Let's implement this greedy strategy.
    
    available_primes = set(primes)
    
    while True:
        if time.time() > deadline:
            break
            
        N = current_L + 1
        if N > MAX_L:
            break
            
        best_new_L = current_L
        best_p = None
        best_r = None
        best_sum = current_sum
        
        # Try each available prime
        for p in available_primes:
            # The residue is forced to cover N
            r = N % p
            
            # Check budget
            new_sum = current_sum + reciprocals[p]
            if new_sum > Fraction(1, 1):
                continue
                
            # Simulate coverage with (p, r)
            # We need to find the new contiguous L.
            # Start from N.
            # A number k is covered if:
            #   1. It was already covered in `covered` array
            #   2. OR k % p == r
            
            # Note: `covered` array reflects the state BEFORE adding this prime.
            # So we check:
            #   is_covered(k) = covered[k] or (k % p == r)
            
            # Optimization: 
            # We can break early if we exceed current best_new_L + some margin?
            # No, we just need to find the exact new L.
            
            # Since we are adding one prime, the new coverage is the union of old and new.
            # The new contiguous block starts at N.
            
            k = N
            # We know N is covered by (p, r) by definition.
            while k <= MAX_L:
                if covered[k]:
                    k += 1
                elif k % p == r:
                    k += 1
                else:
                    break
            
            new_L = k - 1
            
            if new_L > best_new_L:
                best_new_L = new_L
                best_p = p
                best_r = r
                best_sum = new_sum
        
        if best_p is None:
            # No prime can extend the coverage within budget
            break
        
        # Commit the choice
        available_primes.remove(best_p)
        selected_classes.append([best_p, best_r])
        current_sum = best_sum
        current_L = best_new_L
        
        # Update the `covered` array to reflect the new prime
        # This is O(MAX_L / p) per step.
        # Sum of MAX_L / p for selected primes.
        # Since small primes are likely selected first, this is manageable.
        for j in range(best_r, MAX_L + 1, best_p):
            if j >= 1: # covered array is 1-indexed effectively for our range
                covered[j] = True
        
        # Safety check: if we haven't improved, break to avoid infinite loop
        # (Though best_new_L > current_L is guaranteed if best_p is found)
        
    return selected_classes
# EVOLVE-BLOCK-END
