import math
import time
import random

def construct():
    """
    Maximize the number of pairwise disjoint infinite residue classes with distinct moduli in [2, 128].
    
    Strategy:
    1. Select a set of moduli S such that for every prime p, the number of multiples of p in S is at most p.
       This is a necessary condition for the existence of disjoint residues.
    2. Use a greedy heuristic to select moduli, prioritizing those with large smallest prime factors 
       to avoid saturating constraints for small primes.
    3. Assign residues greedily to satisfy the disjointness constraints.
    """
    deadline = time.time() + 85.0
    
    # Primes up to 128
    primes = []
    for i in range(2, 129):
        is_prime = True
        for j in range(2, int(math.isqrt(i)) + 1):
            if i % j == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(i)
            
    prime_set = set(primes)
    
    # Precompute prime factors for each modulus
    moduli_info = {}
    for m in range(2, 129):
        factors = set()
        temp = m
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                factors.add(p)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            factors.add(temp)
        spf = min(factors) if factors else m
        moduli_info[m] = (list(factors), spf)
        
    # Sort moduli by SPF descending, then by number of factors ascending
    # This prioritizes moduli that consume "loose" constraints (large primes)
    sorted_moduli = sorted(range(2, 129), key=lambda m: (-moduli_info[m][1], len(moduli_info[m][0])))
    
    # Greedy selection of moduli
    selected = []
    counts = {p: 0 for p in primes}
    
    for m in sorted_moduli:
        factors, _ = moduli_info[m]
        valid = True
        for p in factors:
            if counts[p] >= p:
                valid = False
                break
        if valid:
            selected.append(m)
            for p in factors:
                counts[p] += 1
        
        if time.time() > deadline:
            break
            
    # Assign residues
    # To ensure disjointness, for each prime p, residues of moduli divisible by p must be distinct mod p.
    # We can try to assign residues one by one.
    
    # Sort selected moduli by size? Or by number of prime factors?
    # Let's try sorting by the product of prime factors (roughly the modulus) or just random.
    # A good heuristic: process moduli with many constraints first?
    # Let's just try a few orders and pick the one that works.
    
    best_result = []
    
    # Try a few random orders for residue assignment
    for _ in range(50):
        random.shuffle(selected)
        assignment = {}
        success = True
        
        for m in selected:
            assigned = False
            # Try residues 0..m-1
            # We can optimize by checking only against assigned moduli
            for r in range(m):
                conflict = False
                for m2, r2 in assignment.items():
                    g = math.gcd(m, m2)
                    if r % g == r2 % g:
                        conflict = True
                        break
                if not conflict:
                    assignment[m] = r
                    assigned = True
                    break
            
            if not assigned:
                success = False
                break
                
        if success and len(assignment) > len(best_result):
            best_result = assignment
            
        if time.time() > deadline:
            break
            
    # Convert to required format
    result = []
    for m, r in best_result.items():
        result.append([r, m])
        
    return result
