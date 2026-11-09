"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
from itertools import product

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
    
    # We want to cover 1..L for the largest possible L
    # Use greedy set cover: at each step, pick (p, r) that maximizes
    # new coverage / (1/p) while keeping total sum <= 1
    
    TARGET = 1000  # We'll try to cover up to 1000
    covered = set()
    budget = Fraction(1, 1)
    used_primes = set()
    classes = []
    
    # Precompute for each prime p and residue r, which numbers in [1, TARGET] it covers
    # But that's too expensive. Instead, for a given p, the best residue r is the one
    # that covers the most uncovered numbers.
    
    for _ in range(len(primes)):  # At most len(primes) iterations
        best_gain = Fraction(0, 1)
        best_p = None
        best_r = None
        
        for p in primes:
            if p in used_primes:
                continue
            if budget < Fraction(1, p):
                continue
            
            # For this prime, find the residue r that covers the most uncovered numbers
            # Numbers covered by (p, r) are: r, r+p, r+2p, ... in [1, TARGET]
            # But we only care about numbers in [1, TARGET] that are not yet covered
            
            best_r_for_p = 0
            best_count_for_p = 0
            
            # Try all residues 0..p-1, but p can be up to 997, so this is expensive
            # Optimization: for large p, only a few numbers in [1,TARGET] are covered
            # Actually, for p > TARGET, only one number is covered (if r in [1,TARGET])
            # For p <= TARGET, multiple numbers
            
            # Let's try residues more cleverly
            # The uncovered numbers in [1, TARGET] - let's see which residues they map to
            uncovered_in_range = [x for x in range(1, TARGET + 1) if x not in covered]
            
            if not uncovered_in_range:
                break
            
            # Count how many uncovered numbers fall into each residue class mod p
            residue_counts = {}
            for x in uncovered_in_range:
                r_mod = x % p
                residue_counts[r_mod] = residue_counts.get(r_mod, 0) + 1
            
            if not residue_counts:
                continue
            
            best_r_for_p = max(residue_counts, key=lambda r: residue_counts[r])
            best_count_for_p = residue_counts[best_r_for_p]
            
            # Gain per unit cost: best_count_for_p / (1/p) = best_count_for_p * p
            # But we want to maximize absolute gain, subject to budget
            gain = Fraction(best_count_for_p, 1)
            
            if gain > best_gain:
                best_gain = gain
                best_p = p
                best_r = best_r_for_p
        
        if best_p is None:
            break
        
        # Add this class
        used_primes.add(best_p)
        budget -= Fraction(1, best_p)
        classes.append([best_p, best_r])
        
        # Update covered
        for x in range(best_r, TARGET + 1, best_p):
            if x >= 1:
                covered.add(x)
    
    # Find the largest L such that 1..L are all covered
    L = 0
    for i in range(1, TARGET + 1):
        if i in covered:
            L = i
        else:
            break
    
    # Verify budget constraint with exact rationals
    total = sum(Fraction(1, p) for p, r in classes)
    assert total <= 1, f"Budget exceeded: {total}"
    
    return classes
# EVOLVE-BLOCK-END
