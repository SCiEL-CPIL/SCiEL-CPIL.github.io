"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
from math import isqrt

def _primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]

PRIMES = _primes_up_to(1000)
PRIME_SET = set(PRIMES)

def construct():
    """
    Construct a set of prime residue classes covering 1..L with sum 1/p <= 1.
    """
    # Generate candidate prime sets
    # Strategy: Greedily add smallest primes while sum(1/p) < 1
    # Then try to optimize residues for that set
    
    candidate_sets = []
    
    # Set 1: Smallest primes
    p_list = []
    s = Fraction(0, 1)
    for p in PRIMES:
        if s + Fraction(1, p) <= 1:
            p_list.append(p)
            s += Fraction(1, p)
        else:
            break
    candidate_sets.append(p_list)
    
    # Set 2: Try swapping some large primes for smaller ones if possible?
    # Actually, the greedy smallest is usually best for density.
    # But let's try a set that includes 2.
    # If we include 2, cost 0.5. Remaining 0.5.
    # Add 3 (0.33) -> 0.83. Add 5 (0.2) -> 1.03 (Too high).
    # So {2, 3} is valid. Sum 0.83.
    # Add 7 (0.14) -> 0.97.
    # Add 11 (0.09) -> 1.06 (Too high).
    # So {2, 3, 7} is valid.
    # Add 13 (0.07) -> 1.04 (Too high).
    # So {2, 3, 7, 13} is valid? 0.5+0.33+0.14+0.07 = 1.04. No.
    # {2, 3, 7} sum 0.97.
    # {2, 3, 5} sum 1.03. No.
    # {2, 3, 11} sum 0.92.
    # {2, 3, 13} sum 0.90.
    
    # Let's just use the first candidate set (smallest primes) and a few others.
    # Set 2: {3, 5, 7, 11, 13, 17, 19, 23} (Manually verified sum ~0.99)
    set2 = [3, 5, 7, 11, 13, 17, 19, 23]
    if sum(Fraction(1, p) for p in set2) <= 1:
        candidate_sets.append(set2)
        
    # Set 3: {2, 3, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997}
    # This is too long to type. I'll generate it programmatically.
    
    best_classes = []
    best_L = 0
    
    for p_set in candidate_sets:
        # Verify budget
        total_cost = sum(Fraction(1, p) for p in p_set)
        if total_cost > 1:
            continue
            
        # Initialize residues
        # Start with 0 for all
        residues = {p: 0 for p in p_set}
        
        # Repair Algorithm
        # We want to cover 1..L_max
        L_max = 10000
        covered = [False] * (L_max + 1)
        
        # Precompute coverage for initial residues?
        # No, we update as we go.
        
        # Initialize covered array
        for p in p_set:
            r = residues[p]
            for n in range(1, L_max + 1):
                if n % p == r:
                    covered[n] = True
        
        # Find first gap
        n = 1
        while n <= L_max and covered[n]:
            n += 1
            
        if n > L_max:
            L_found = L_max
        else:
            L_found = n - 1
            
        # Try to extend
        # If n <= L_max, n is uncovered.
        # We try to change a residue to cover n.
        # We iterate n from 1 to L_max
        # If n is covered, continue.
        # If n is not covered, try to fix it.
        
        # Actually, the above loop just finds the first gap.
        # We need to actively repair.
        
        covered = [False] * (L_max + 1)
        for p in p_set:
            r = residues[p]
            for k in range(r, L_max + 1, p):
                if k >= 1:
                    covered[k] = True
        
        max_L = 0
        for n in range(1, L_max + 1):
            if not covered[n]:
                # Try to fix n
                fixed = False
                for p in p_set:
                    new_r = n % p
                    if residues[p] == new_r:
                        continue # Already covers n? No, n is not covered.
                    
                    # Check if changing p to new_r breaks any m < n
                    # We only care about m in [1, n-1]
                    # If we change p to new_r, it will cover new_r, new_r+p, ...
                    # It will UNCOVER old_r, old_r+p, ...
                    
                    old_r = residues[p]
                    
                    # Check if any m < n is currently covered ONLY by p with old_r
                    # This is expensive.
                    # Heuristic: Just try it and see if [1, n-1] is still covered.
                    
                    # To speed up, we can just check if the change is "safe"
                    # by verifying that for all m < n, if m % p == old_r, then m is covered by some other prime.
                    
                    safe = True
                    for m in range(1, n):
                        if m % p == old_r:
                            # Check if m is covered by others
                            is_covered_by_others = False
                            for q in p_set:
                                if q == p:
                                    continue
                                if m % q == residues[q]:
                                    is_covered_by_others = True
                                    break
                            if not is_covered_by_others:
                                safe = False
                                break
                    
                    if safe:
                        residues[p] = new_r
                        # Update covered array?
                        # We can just update covered[n] = True
                        # But we need to be careful about future checks.
                        # For the purpose of finding max L, we just need to know
                        # if [1, n] is covered.
                        # If we commit the change, [1, n] is covered.
                        covered[n] = True
                        fixed = True
                        break
                
                if not fixed:
                    break # Cannot cover n
            
            max_L = n
        
        if max_L > best_L:
            best_L = max_L
            best_classes = [[p, residues[p]] for p in p_set]
            
    return best_classes
# EVOLVE-BLOCK-END
