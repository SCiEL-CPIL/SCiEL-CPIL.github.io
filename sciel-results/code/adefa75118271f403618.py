"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from fractions import Fraction
import numpy as np

def _primes_upto(n):
    """Return list of primes up to n."""
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]

def construct():
    """
    Construct a set of prime residue classes covering 1..L for the largest possible L,
    subject to sum(1/p) <= 1 and distinct primes p <= 1000.
    
    Strategy: Greedy boundary extension.
    At each step, we identify the first uncovered integer U (boundary).
    For each unused prime p, the only residue r that can extend the prefix is r = U % p.
    We evaluate the extension length for each such (p, r) and pick the best one.
    """
    # We aim to cover up to a reasonable limit. 10^5 is the cap, but likely L will be much smaller.
    # Let's set a working limit to 5000 to keep numpy operations fast and memory low.
    # If we reach 5000, we can extend, but 10^5 is the hard cap.
    # Given the budget constraint, L is likely < 2000.
    CAP = 5000 
    
    primes = _primes_upto(1000)
    
    # covered[i] is True if integer i+1 is covered.
    # Index 0 corresponds to integer 1.
    covered = np.zeros(CAP + 1, dtype=bool)
    
    used_primes = set()
    budget_used = Fraction(0, 1)
    classes = []
    
    L_cur = 0 # Current prefix length (number of integers 1..L_cur covered)
    
    while True:
        U = L_cur + 1 # The next integer to cover (1-based)
        if U > CAP:
            break
            
        best_p = None
        best_r = None
        best_new_L = L_cur
        
        for p in primes:
            if p in used_primes:
                continue
            
            cost = Fraction(1, p)
            if budget_used + cost > Fraction(1, 1):
                continue
            
            # The only residue that covers U is r = U % p
            r = U % p
            
            # Simulate adding (p, r) and find the new prefix length
            # New covered set: covered[i] OR ((i+1) % p == r)
            # We check from U upwards.
            
            # Optimization: We only need to check until we hit a number not covered by either.
            # We can do this in a loop.
            new_L = L_cur
            # Start checking from U (index U-1)
            for n in range(U, CAP + 1):
                idx = n - 1
                if covered[idx]:
                    # Already covered
                    new_L = n
                    continue
                else:
                    # Check if covered by new class
                    if n % p == r:
                        new_L = n
                        continue
                    else:
                        break
            
            if new_L > best_new_L:
                best_new_L = new_L
                best_p = p
                best_r = r
            elif new_L == best_new_L and best_p is not None:
                # Tie-breaker: prefer smaller prime (lower cost, potentially more flexibility later)
                if p < best_p:
                    best_p = p
                    best_r = r
        
        if best_p is None:
            break
        
        # Commit the best choice
        used_primes.add(best_p)
        budget_used += Fraction(1, best_p)
        classes.append([best_p, best_r])
        
        # Update covered array
        # Indices i such that (i+1) % best_p == best_r
        # i % best_p == best_r - 1
        r_idx = (best_r - 1) % best_p
        covered[r_idx::best_p] = True
        
        L_cur = best_new_L
        
    # Calculate final L (capped at 10^5, but our array is CAP)
    # If L_cur == CAP, we might be able to go further, but we stop here for simplicity.
    # The problem asks for the largest L. If we hit CAP, we should probably return CAP.
    # But since CAP=5000 and target is 999, this is fine.
    
    return classes
# EVOLVE-BLOCK-END
