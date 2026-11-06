"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from fractions import Fraction

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
    
    Strategy: Greedy gap-covering.
    1. Find the smallest uncovered integer n_next.
    2. For each unused prime p that fits in budget, the residue r is fixed: r = n_next % p.
    3. Determine how far the prefix extends if we add (p, r).
    4. Pick the p that maximizes the new prefix length.
    5. Repeat until no more primes can be added or budget is exhausted.
    """
    CAP = 10**5
    primes = _primes_upto(1000)
    
    covered = [False] * (CAP + 1)
    used_primes = set()
    budget_used = Fraction(0, 1)
    classes = []
    
    n_next = 1
    
    while n_next <= CAP:
        best_p = None
        best_r = None
        best_new_prefix = n_next - 1
        
        remaining_budget = Fraction(1, 1) - budget_used
        
        # Try all unused primes
        for p in primes:
            if p in used_primes:
                continue
            
            cost = Fraction(1, p)
            if cost > remaining_budget:
                continue
            
            # Residue is forced to cover n_next
            r = n_next % p
            
            # Calculate new prefix length if we add (p, r)
            # We know 1..n_next-1 are covered.
            # Check m from n_next upwards.
            # m is covered if covered[m] is True OR m % p == r.
            new_prefix = n_next - 1
            
            m = n_next
            while m <= CAP:
                if covered[m] or (m % p == r):
                    new_prefix = m
                    m += 1
                else:
                    break
            
            if new_prefix > best_new_prefix:
                best_new_prefix = new_prefix
                best_p = p
                best_r = r
        
        if best_p is None:
            # No prime can cover n_next within budget
            break
        
        # Commit the choice
        used_primes.add(best_p)
        budget_used += Fraction(1, best_p)
        classes.append([best_p, best_r])
        
        # Update covered array
        p = best_p
        r = best_r
        
        if r == 0:
            start_val = p
        else:
            start_val = r
            
        for n in range(start_val, CAP + 1, p):
            covered[n] = True
            
        # Update n_next to the first uncovered integer
        n_next = best_new_prefix + 1
        
        if n_next > CAP:
            break
            
    # Calculate final L
    L = 0
    for i in range(1, CAP + 1):
        if covered[i]:
            L = i
        else:
            break
            
    return classes
# EVOLVE-BLOCK-END
