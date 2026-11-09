"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
import math

def construct():
    # Generate primes up to 1000
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    primes = [p for p in range(2, 1001) if is_prime(p)]
    
    # Limit for coverage checking
    LIMIT = 100000
    covered = [False] * (LIMIT + 1)
    
    used_primes = set()
    budget_remaining = Fraction(1, 1)
    
    classes = []
    
    # Helper to update covered array for a prime/residue
    def apply_class(p, r):
        # Mark x = r, r+p, ... <= LIMIT
        # x >= 1
        if r == 0:
            start = p
        else:
            start = r
        for x in range(start, LIMIT + 1, p):
            covered[x] = True
            
    # Helper to count new covers in a window
    def count_new_covers(p, r, start, end):
        count = 0
        # x = r + k*p
        # We want x in [start, end] and not covered[x]
        # Find first x >= start
        if r == 0:
            # x = k*p
            # k*p >= start => k >= ceil(start/p)
            k_start = math.ceil(start / p)
        else:
            # x = r + k*p
            # r + k*p >= start => k >= ceil((start - r) / p)
            # Note: r < p, so start - r can be negative? 
            # start >= 1, r >= 0. If start <= r, k can be 0.
            # If start > r, k >= ceil((start - r) / p)
            if start <= r:
                k_start = 0
            else:
                k_start = math.ceil((start - r) / p)
        
        # Iterate
        x = r + k_start * p
        # Ensure x >= start (due to ceil logic, it should be, but check)
        if x < start:
            x += p
        while x <= end:
            if not covered[x]:
                count += 1
            x += p
        return count

    while True:
        # Find the first gap
        gap = -1
        for i in range(1, LIMIT + 1):
            if not covered[i]:
                gap = i
                break
        
        if gap == -1:
            break # Covered up to LIMIT
            
        # We must cover 'gap'
        # Find the best prime to use
        best_p = None
        best_r = None
        best_score = -1
        
        # Lookahead window for scoring
        lookahead = 500
        
        for p in primes:
            if p in used_primes:
                continue
            cost = Fraction(1, p)
            if budget_remaining < cost:
                continue
            
            r = gap % p
            
            # Score: number of new covers in [gap, gap + lookahead]
            score = count_new_covers(p, r, gap, min(LIMIT, gap + lookahead))
            
            # Tie-breaker: Prefer smaller primes? Or larger?
            # Smaller primes cover more numbers generally, but cost more.
            # Since we are constrained by budget, maybe prefer larger primes to save budget?
            # But larger primes are less dense.
            # Let's prefer the one with higher score. If tie, prefer smaller p (more density for the cost)?
            # Actually, 1/p is the cost. Smaller p = higher cost.
            # If scores are equal, maybe we should pick the one that costs less (larger p) to save budget for later?
            # Or pick the one that costs more (smaller p) because it's "better" for coverage?
            # Let's stick to: Max score. If tie, min p (more aggressive coverage).
            if score > best_score:
                best_score = score
                best_p = p
                best_r = r
            elif score == best_score and best_p is not None:
                if p < best_p:
                    best_p = p
                    best_r = r
        
        if best_p is None:
            break # No prime can cover the gap within budget
            
        # Apply the chosen class
        classes.append([best_p, best_r])
        used_primes.add(best_p)
        budget_remaining -= Fraction(1, best_p)
        apply_class(best_p, best_r)
        
    # Final check: What is the largest L?
    L = 0
    for i in range(1, LIMIT + 1):
        if covered[i]:
            L = i
        else:
            break
            
    return classes
# EVOLVE-BLOCK-END
