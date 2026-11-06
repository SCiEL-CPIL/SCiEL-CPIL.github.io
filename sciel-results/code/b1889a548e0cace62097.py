"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction

def _distinct_prime_factors(n):
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    dpf = {}
    for i in range(1, 512):
        dpf[i] = _distinct_prime_factors(i)
    
    contrib = {a: Fraction(1, 512 - a) for a in range(1, 512)}
    
    chosen = []
    chosen_set = set()
    
    # Greedy with better heuristic: prefer high contribution, low prime factor count
    for _ in range(511):
        if time.time() > deadline:
            break
        
        best_a = None
        best_key = None
        
        for a in range(511, 0, -1):
            if a in chosen_set:
                continue
            if not all(gcd(a, b) == 1 for b in chosen):
                continue
            
            pf_count = len(dpf[a])
            key = float(contrib[a]) / (1.0 + 0.1 * pf_count)
            
            if best_key is None or key > best_key:
                best_key = key
                best_a = a
        
        if best_a is None:
            break
        chosen.append(best_a)
        chosen_set.add(best_a)
    
    # Local search: try to swap elements for better ones
    if time.time() < deadline:
        current_score = sum(contrib[a] for a in chosen)
        for _ in range(200):
            if time.time() > deadline:
                break
            found = False
            for i in range(len(chosen)):
                if time.time() > deadline:
                    break
                for a in range(511, 0, -1):
                    if time.time() > deadline:
                        break
                    if a in chosen_set:
                        continue
                    others = [b for j, b in enumerate(chosen) if j != i]
                    if all(gcd(a, b) == 1 for b in others):
                        new_score = current_score - contrib[chosen[i]] + contrib[a]
                        if new_score > current_score:
                            chosen[i] = a
                            chosen_set = set(chosen)
                            current_score = new_score
                            found = True
                            break
                if found:
                    break
            if not found:
                break
    
    return chosen
# EVOLVE-BLOCK-END
