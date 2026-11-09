"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
import random
import math
import numpy as np

def construct():
    # 1. Generate Primes
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0: return False
        return True
        
    primes = [p for p in range(2, 1001) if is_prime(p)]
    
    # 2. Generate Candidate Prime Sets
    candidate_sets = []
    
    # Set 1: Small odd primes
    s1 = [3, 5, 7, 11, 13, 17, 19, 23]
    if sum(Fraction(1, p) for p in s1) <= 1: candidate_sets.append(s1)
    
    # Set 2: Includes 2
    s2 = [2, 3, 7, 43]
    if sum(Fraction(1, p) for p in s2) <= 1: candidate_sets.append(s2)
    
    # Set 3: Another with 2
    s3 = [2, 3, 7, 41]
    if sum(Fraction(1, p) for p in s3) <= 1: candidate_sets.append(s3)
    
    # Set 4: {2, 5, 7, 11, 13}
    s4 = [2, 5, 7, 11, 13]
    if sum(Fraction(1, p) for p in s4) <= 1: candidate_sets.append(s4)
    
    # Set 5: {2, 5, 7, 13, 17}
    s5 = [2, 5, 7, 13, 17]
    if sum(Fraction(1, p) for p in s5) <= 1: candidate_sets.append(s5)
    
    # Set 6: {2, 5, 7, 11, 19}
    s6 = [2, 5, 7, 11, 19]
    if sum(Fraction(1, p) for p in s6) <= 1: candidate_sets.append(s6)
    
    # Add a greedy set
    current_set = [2]
    current_sum = Fraction(1, 2)
    for p in primes:
        if p == 2: continue
        if current_sum + Fraction(1, p) <= 1:
            current_set.append(p)
            current_sum += Fraction(1, p)
            if len(current_set) >= 8: break
    if current_set not in candidate_sets:
        candidate_sets.append(current_set)

    best_L = 0
    best_classes = []
    L_MAX = 100000
    
    def evaluate_coverage(primes, residues, L_max):
        """
        Returns the largest L such that 1..L are covered.
        """
        covered = np.zeros(L_max + 1, dtype=bool)
        for p, r in zip(primes, residues):
            if r == 0:
                idx = np.arange(p, L_max + 1, p)
            else:
                idx = np.arange(r, L_max + 1, p)
            covered[idx] = True
            
        gaps = np.where(covered[1:] == False)[0]
        if len(gaps) == 0:
            return L_max
        else:
            return gaps[0]
            
    def optimize_residues(primes, L_max_local=10000):
        """
        Coordinate descent with full residue sweep.
        """
        best_local_L = 0
        best_local_res = []
        
        for _ in range(10): # 10 restarts
            residues = [random.randint(0, p-1) for p in primes]
            current_L = evaluate_coverage(primes, residues, L_max_local)
            
            improved = True
            while improved:
                improved = False
                for i, p in enumerate(primes):
                    best_r = residues[i]
                    best_L = current_L
                    
                    for r in range(p):
                        residues[i] = r
                        L_cand = evaluate_coverage(primes, residues, L_max_local)
                        if L_cand > best_L:
                            best_L = L_cand
                            best_r = r
                            
                    residues[i] = best_r
                    if best_L > current_L:
                        current_L = best_L
                        improved = True
                        
            if current_L > best_local_L:
                best_local_L = current_L
                best_local_res = residues[:]
                
        if best_local_L > 0:
            final_L = evaluate_coverage(primes, best_local_res, L_MAX)
            return final_L, best_local_res
        return 0, []
        
    for p_set in candidate_sets:
        L, residues = optimize_residues(p_set)
        if L > best_L:
            best_L = L
            best_classes = [[p, r] for p, r in zip(p_set, residues)]
            
    # Verify budget
    if best_classes:
        total = sum(Fraction(1, p) for p, r in best_classes)
        if total > 1:
            # Fallback
            best_classes = []
    else:
        # If no classes found, return a known valid empty or small set?
        # Task implies we must return valid classes.
        # If best_classes is empty, we should probably return a valid trivial one.
        # But L=0 is score 0.
        # Let's ensure we always have a valid set.
        # If the loop failed, use a hardcoded valid set.
        best_classes = [[2, 0]] # Sum 1/2 <= 1. L=0? No, 2 covers 2,4... 1 is not covered. L=0.
        
    return best_classes
# EVOLVE-BLOCK-END
