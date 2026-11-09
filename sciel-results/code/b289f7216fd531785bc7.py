"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
import random

def construct():
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
    
    best_L_global = 0
    best_solution_global = []
    
    # Run multiple times with random tie-breaking
    for _ in range(20):
        L_max_search = 5000
        covered = np.zeros(L_max_search + 1, dtype=bool)
        used_primes = set()
        budget = Fraction(1, 1)
        solution = []
        
        # Precompute indices for speed
        indices_cache = {}
        for p in primes:
            indices_cache[p] = []
            for r in range(p):
                start = r
                if start == 0:
                    start = p
                if start <= L_max_search:
                    idx = np.arange(start, L_max_search + 1, p)
                    indices_cache[p].append(idx)
                else:
                    indices_cache[p].append(np.array([], dtype=int))

        while True:
            best_gain = -1
            best_p = None
            best_r = None
            
            for p in primes:
                if p in used_primes:
                    continue
                if Fraction(1, p) > budget:
                    continue
                
                max_gain_for_p = 0
                best_r_for_p = 0
                
                for r in range(p):
                    idx = indices_cache[p][r]
                    if len(idx) == 0:
                        continue
                    gain = np.sum(~covered[idx])
                    
                    if gain > max_gain_for_p:
                        max_gain_for_p = gain
                        best_r_for_p = r
                    elif gain == max_gain_for_p and max_gain_for_p > 0:
                        # Random tie break: 50% chance to replace
                        if random.random() < 0.5:
                            best_r_for_p = r
                
                if max_gain_for_p > best_gain:
                    best_gain = max_gain_for_p
                    best_p = p
                    best_r = best_r_for_p
                elif max_gain_for_p == best_gain and max_gain_for_p > 0:
                    # Random tie break between primes
                    if random.random() < 0.5:
                        best_p = p
                        best_r = best_r_for_p
            
            if best_gain <= 0:
                break
            
            solution.append([best_p, best_r])
            used_primes.add(best_p)
            budget -= Fraction(1, best_p)
            
            p = best_p
            r = best_r
            idx = indices_cache[p][r]
            if len(idx) > 0:
                covered[idx] = True
        
        # Evaluate exact L for this solution
        if not solution:
            continue
            
        max_L_check = 100000
        L = 0
        for n in range(1, max_L_check + 1):
            is_covered = False
            for p, r in solution:
                if n % p == r:
                    is_covered = True
                    break
            if is_covered:
                L = n
            else:
                break
                
        if L > best_L_global:
            best_L_global = L
            best_solution_global = solution
            
    return best_solution_global
# EVOLVE-BLOCK-END
