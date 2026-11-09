"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
import random
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
    
    # We will try several candidate sets of primes
    # Strategy: Find sets of primes where sum(1/p) <= 1
    # We prefer sets with more small primes to increase coverage density.
    
    candidate_sets = []
    
    # Set 1: Smallest primes without 2 (to allow more primes)
    # 3, 5, 7, 11, 13, 17, 19, 23
    sum_1 = sum(Fraction(1, p) for p in [3, 5, 7, 11, 13, 17, 19, 23])
    if sum_1 <= 1:
        candidate_sets.append([3, 5, 7, 11, 13, 17, 19, 23])
        
    # Set 2: With 2
    # 2, 3, 7, 41
    sum_2 = sum(Fraction(1, p) for p in [2, 3, 7, 41])
    if sum_2 <= 1:
        candidate_sets.append([2, 3, 7, 41])
        
    # Set 3: Another with 2
    # 2, 3, 7, 37? No, 1/2+1/3+1/7+1/37 = 1.002 > 1
    # 2, 3, 7, 43? 1/2+1/3+1/7+1/43 = 0.998
    sum_3 = sum(Fraction(1, p) for p in [2, 3, 7, 43])
    if sum_3 <= 1:
        candidate_sets.append([2, 3, 7, 43])

    # Set 4: Try to pack more odd primes
    # 3, 5, 7, 11, 13, 17, 19, 29?
    # Sum 3,5,7,11,13,17,19 = 0.954
    # Add 23: 0.997
    # Add 29: 1.031 > 1
    # So 3,5,7,11,13,17,19,23 is maximal for that chain.
    
    # Set 5: 2, 5, 7, 11, 13, 17, 19?
    # 1/2 + 1/5 + 1/7 + 1/11 + 1/13 + 1/17 + 1/19
    # 0.5 + 0.2 + 0.142 + 0.0909 + 0.0769 + 0.0588 + 0.0526 = 1.12 > 1
    # Drop 19: 1.067 > 1
    # Drop 17: 1.008 > 1
    # Drop 13: 0.931
    # Add 19: 0.983
    # So {2, 5, 7, 11, 19}
    sum_5 = sum(Fraction(1, p) for p in [2, 5, 7, 11, 19])
    if sum_5 <= 1:
        candidate_sets.append([2, 5, 7, 11, 19])
        
    # Set 6: {2, 5, 7, 13, 17, 19}?
    # 0.5 + 0.2 + 0.142 + 0.0769 + 0.0588 + 0.0526 = 1.03 > 1
    # Drop 19: 0.977
    # So {2, 5, 7, 13, 17}
    sum_6 = sum(Fraction(1, p) for p in [2, 5, 7, 13, 17])
    if sum_6 <= 1:
        candidate_sets.append([2, 5, 7, 13, 17])

    best_L = 0
    best_classes = []
    
    TARGET = 10000 # We cap evaluation at 10000 for speed, but we want L up to 10^5?
    # The task says capped at 10^5. But we want to maximize L.
    # If we find L=999, we are good.
    # Let's evaluate up to 2000 to be safe and fast, but if we find a gap, we stop.
    # Actually, if we cover 1..1000, we should check further.
    # But the prompt says "score is the largest L... capped at 10^5".
    # If we can't cover 10^5, we just report the actual L.
    
    def evaluate_L(primes, residues, limit=20000):
        """
        Returns the largest L such that 1..L are covered.
        """
        covered = [False] * (limit + 1)
        
        for p, r in zip(primes, residues):
            # Mark numbers r, r+p, r+2p... <= limit
            # Note: r is in 0..p-1.
            # Numbers are r, r+p, ...
            # If r == 0, it covers p, 2p, ... (and 0, but we start at 1)
            # If r > 0, it covers r, r+p, ...
            
            start = r
            if start == 0:
                start = p
            elif start < 1:
                # r is 0..p-1, so r>=0. If r=0, handled.
                pass
            
            # Actually, simpler:
            # x = r + k*p
            # We want x >= 1 and x <= limit
            # k = ceil((1 - r) / p)
            if r == 0:
                k_start = 1
            else:
                k_start = 0
                if r < 1: # r is int, so r=0 only case < 1
                    k_start = 1
            
            # Let's just iterate
            x = r
            if x == 0:
                x = p
            while x <= limit:
                covered[x] = True
                x += p
        
        L = 0
        for i in range(1, limit + 1):
            if covered[i]:
                L = i
            else:
                break
        return L

    def optimize_residues(primes, max_iterations=50, random_restarts=5):
        """
        Uses coordinate descent to find residues that maximize L.
        """
        best_local_L = 0
        best_local_residues = []
        
        for _ in range(random_restarts):
            # Initialize residues randomly
            residues = [random.randint(0, p - 1) for p in primes]
            
            # Initial evaluation
            current_L = evaluate_L(primes, residues, limit=5000)
            
            improved = True
            iteration = 0
            while improved and iteration < max_iterations:
                improved = False
                iteration += 1
                for i, p in enumerate(primes):
                    best_r_for_p = residues[i]
                    best_L_for_p = current_L
                    
                    # Try all residues for this prime
                    # Optimization: Only try residues that might help.
                    # But p can be up to 1000. 1000 * 5000 checks = 5M ops.
                    # In Python, this is slow.
                    # We can sample residues or use a heuristic.
                    # Heuristic: The best residue is often the one that covers the first gap.
                    
                    # Find the first gap in current coverage
                    # We need to know which numbers are covered.
                    # Re-evaluating covered array for each residue is expensive.
                    
                    # Alternative:
                    # Just try a few residues: 0, 1, p-1, and the residue of the first gap.
                    # Let's find the first gap.
                    covered = [False] * (5000 + 1)
                    for j, q in enumerate(primes):
                        r_j = residues[j]
                        x = r_j
                        if x == 0: x = q
                        while x <= 5000:
                            covered[x] = True
                            x += q
                    
                    gap = -1
                    for k in range(1, 5001):
                        if not covered[k]:
                            gap = k
                            break
                    
                    if gap == -1:
                        # Covered up to 5000, try to extend?
                        # For now, assume 5000 is good enough for local opt.
                        break
                    
                    # Candidate residues:
                    # 1. Current residue
                    # 2. Residue that covers the gap: gap % p
                    # 3. Residue that covers gap-1, gap+1 etc?
                    # 4. Random few
                    candidates = set()
                    candidates.add(residues[i])
                    candidates.add(gap % p)
                    candidates.add((gap - 1) % p)
                    candidates.add((gap + 1) % p)
                    candidates.add(0)
                    candidates.add(1)
                    candidates.add(p - 1)
                    
                    # Add some random
                    for _ in range(10):
                        candidates.add(random.randint(0, p - 1))
                        
                    for r_cand in candidates:
                        residues[i] = r_cand
                        L_cand = evaluate_L(primes, residues, limit=5000)
                        if L_cand > best_L_for_p:
                            best_L_for_p = L_cand
                            best_r_for_p = r_cand
                        else:
                            # Restore if not better? We'll restore at end of loop if no change
                            pass
                    
                    # If we found a better residue, update
                    if best_r_for_p != residues[i]:
                        residues[i] = best_r_for_p
                        current_L = best_L_for_p
                        improved = True
                    elif best_L_for_p > current_L:
                        # This shouldn't happen if we initialized best_L_for_p = current_L
                        # and only update if >
                        pass
                    else:
                        # If no improvement for this prime, keep old residue
                        # (It was already set to best_r_for_p which is same as old if no change)
                        pass
                        
            # End while
            
            if current_L > best_local_L:
                best_local_L = current_L
                best_local_residues = residues[:]
                
        # Final evaluation with larger limit for the best found
        if best_local_L > 0:
            final_L = evaluate_L(primes, best_local_residues, limit=100000)
            return final_L, best_local_residues
        else:
            return 0, []

    for p_set in candidate_sets:
        L, residues = optimize_residues(p_set)
        if L > best_L:
            best_L = L
            best_classes = [[p, r] for p, r in zip(p_set, residues)]
            
    # Verify budget
    total = sum(Fraction(1, p) for p, r in best_classes)
    assert total <= 1, f"Budget exceeded: {total}"
    
    return best_classes
# EVOLVE-BLOCK-END
