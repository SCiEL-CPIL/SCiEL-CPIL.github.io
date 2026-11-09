"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import isqrt

def _primes_upto(n):
    if n < 2:
        return []
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    primes = _primes_upto(1000)
    # Precompute 1/p as Fraction for each prime
    recip = {p: Fraction(1, p) for p in primes}
    
    # We want to maximize L such that every integer in 1..L is covered.
    # Strategy: Greedy set cover with exact rational budget.
    # Start with empty set, iteratively add the prime-residue class that
    # covers the most currently-uncovered integers in 1..L_max, subject to
    # budget constraint.
    
    # First, let's figure out a reasonable L_max. The seed covers up to ~999.
    # Let's try to cover as much as possible.
    
    # Approach: 
    # 1. Start with all primes available.
    # 2. Greedily select residue classes that maximize coverage of the gap.
    #    For each prime p, the best residue r is the one that covers the most
    #    uncovered numbers in 1..L_target.
    
    # Let's set L_target = 100000 (the cap) and see what we can achieve.
    # But that's too large. Let's be smarter.
    
    # Actually, let's think about this differently. The sum of reciprocals is at most 1.
    # The sum of reciprocals of all primes up to 1000 is about 2.2, so we have budget.
    # The key insight: for a prime p, choosing residue r covers numbers congruent to r mod p.
    # In range 1..L, this covers about L/p numbers.
    
    # Greedy algorithm:
    # 1. Maintain a boolean array "covered" of size L_max+1.
    # 2. Maintain remaining budget as Fraction.
    # 3. For each step, for each unused prime p, find the residue r that maximizes
    #    the number of newly covered integers in 1..L_max.
    # 4. If the best prime adds at least 1 new coverage and fits in budget, add it.
    # 5. Stop when no prime can add coverage within budget, or time runs out.
    
    # But L_max = 100000 is large. Let's use a smarter approach.
    
    # Alternative: Work with intervals. We want to cover 1..L. 
    # Start with L=1, try to extend.
    
    # Better approach: Use the fact that we're covering 1..L.
    # For a given set of (p, r) classes, the coverage is the union of arithmetic progressions.
    
    # Let's use a greedy approach with a target L.
    # Start with L = 1000 (matching the seed), try to extend.
    
    # Actually, let's just do a proper greedy set cover.
    # The universe is {1, 2, ..., L_target}.
    # Each "set" is a (p, r) pair, covering {r mod p, r mod p + p, ...} ∩ [1, L_target].
    # But there are 168 primes × p residues each = too many sets.
    
    # Key optimization: For each prime p, only one residue r is chosen.
    # So for each p, we need to find the best r.
    
    # Let's use a different strategy:
    # 1. Sort primes by size (smaller primes cover more numbers per unit budget).
    # 2. Greedily assign residues.
    
    # Even better: Use a local search / hill climbing approach.
    # Start with the seed solution, then try to improve by swapping or adding classes.
    
    # Let's implement a greedy construction:
    
    L_target = 2000  # Start with a target, we'll see how far we get
    covered = [False] * (L_target + 1)
    used_primes = set()
    classes = []
    budget = Fraction(1, 1)
    
    # Precompute for each prime p and residue r, which numbers in 1..L_target it covers
    # This is expensive, so let's compute on the fly.
    
    # For each prime, find the residue that covers the most uncovered numbers
    def best_residue(p, covered, L_target):
        best_r = 0
        best_count = -1
        for r in range(p):
            count = 0
            # Count uncovered numbers in r mod p within 1..L_target
            # Numbers are r, r+p, r+2p, ... but we need them in [1, L_target]
            # If r == 0, numbers are p, 2p, 3p, ...
            # If r > 0, numbers are r, r+p, r+2p, ...
            start = r if r > 0 else p
            if start > L_target:
                continue
            for n in range(start, L_target + 1, p):
                if not covered[n]:
                    count += 1
            if count > best_count:
                best_count = count
                best_r = r
        return best_r, best_count
    
    # Greedy: repeatedly pick the prime that gives the most new coverage per unit cost
    # Cost is 1/p, so value is count * p (coverage per unit budget)
    
    while True:
        if time.time() > deadline - 10:
            break
        
        best_p = None
        best_r = None
        best_value = 0  # coverage per unit budget = count * p
        best_count = 0
        
        for p in primes:
            if p in used_primes:
                continue
            if budget < recip[p]:
                continue
            r, count = best_residue(p, covered, L_target)
            if count <= 0:
                continue
            value = count * p  # proportional to count / (1/p)
            if value > best_value:
                best_value = value
                best_p = p
                best_r = r
                best_count = count
        
        if best_p is None:
            break
        
        # Add the class
        used_primes.add(best_p)
        classes.append([best_p, best_r])
        budget -= recip[best_p]
        
        # Update covered
        p = best_p
        r = best_r
        start = r if r > 0 else p
        for n in range(start, L_target + 1, p):
            covered[n] = True
        
        # Check if we've covered all of 1..L_target
        if all(covered[1:L_target+1]):
            break
    
    # Now compute the actual L (largest such that 1..L are all covered)
    L = 0
    for i in range(1, L_target + 1):
        if covered[i]:
            L = i
        else:
            break
    
    # Try to extend L by adding more classes if budget allows
    # Actually, let's try a second pass: try to cover beyond L_target
    
    # Let's compute L properly and try to extend
    # First, let's see what L we have
    
    # If we have time, try to extend by considering larger L
    if L >= L_target and time.time() < deadline - 20:
        # Extend L_target
        L_target2 = min(L_target * 2, 100000)
        covered2 = [False] * (L_target2 + 1)
        for p, r in classes:
            start = r if r > 0 else p
            for n in range(start, L_target2 + 1, p):
                covered2[n] = True
        
        # Continue greedy with extended target
        used_primes2 = set(p for p, r in classes)
        classes2 = list(classes)
        budget2 = budget
        
        for _ in range(50):  # Limit iterations
            if time.time() > deadline - 10:
                break
            
            best_p = None
            best_r = None
            best_value = 0
            best_count = 0
            
            for p in primes:
                if p in used_primes2:
                    continue
                if budget2 < recip[p]:
                    continue
                r, count = best_residue(p, covered2, L_target2)
                if count <= 0:
                    continue
                value = count * p
                if value > best_value:
                    best_value = value
                    best_p = p
                    best_r = r
                    best_count = count
            
            if best_p is None:
                break
            
            used_primes2.add(best_p)
            classes2.append([best_p, best_r])
            budget2 -= recip[best_p]
            
            p = best_p
            r = best_r
            start = r if r > 0 else p
            for n in range(start, L_target2 + 1, p):
                covered2[n] = True
        
        # Compute new L
        L2 = 0
        for i in range(1, L_target2 + 1):
            if covered2[i]:
                L2 = i
            else:
                break
        
        if L2 > L:
            L = L2
            classes = classes2
    
    # Final verification: compute L exactly
    # Build a coverage array up to min(L+100, 100000)
    max_check = min(L + 100, 100000)
    final_covered = [False] * (max_check + 1)
    for p, r in classes:
        start = r if r > 0 else p
        for n in range(start, max_check + 1, p):
            final_covered[n] = True
    
    final_L = 0
    for i in range(1, max_check + 1):
        if final_covered[i]:
            final_L = i
        else:
            break
    
    return classes

# EVOLVE-BLOCK-END
