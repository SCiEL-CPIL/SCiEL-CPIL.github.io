"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import gcd

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # 150s limit, 10s safety margin
    
    # Precompute prime factors for all numbers 1..511
    def prime_factors(n):
        factors = []
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
        if n > 1:
            factors.append(n)
        return factors
    
    pf_cache = {}
    for i in range(1, 512):
        pf_cache[i] = prime_factors(i)
    
    # Precompute the weight for each a: 1/(512-a)
    weight = {}
    for a in range(1, 512):
        weight[a] = Fraction(1, 512 - a)
    
    # Sort candidates by weight descending (greedy priority)
    candidates = list(range(511, 0, -1))
    
    # Greedy initial solution
    chosen = []
    used_primes = set()
    for a in candidates:
        if time.time() > DEADLINE:
            break
        conflict = False
        for p in pf_cache[a]:
            if p in used_primes:
                conflict = True
                break
        if not conflict:
            chosen.append(a)
            for p in pf_cache[a]:
                used_primes.add(p)
    
    def current_score():
        return sum(weight[a] for a in chosen)
    
    best_score = current_score()
    best_chosen = chosen[:]
    
    # Local search: try 1-for-1 swaps and additions
    # Build a set for fast membership
    chosen_set = set(chosen)
    
    # Try to add any unchosen a that doesn't conflict
    # Then try swaps: remove one, add one or more
    
    # More powerful: iterative improvement with 1-for-k and k-for-1
    # Use exact rational arithmetic for comparisons
    
    # First, try simple additions
    changed = True
    while changed:
        changed = False
        if time.time() > DEADLINE - 5:
            break
        for a in candidates:
            if a in chosen_set:
                continue
            conflict = False
            for p in pf_cache[a]:
                if p in used_primes:
                    conflict = True
                    break
            if not conflict:
                chosen.append(a)
                chosen_set.add(a)
                for p in pf_cache[a]:
                    used_primes.add(p)
                changed = True
                new_score = current_score()
                if new_score > best_score:
                    best_score = new_score
                    best_chosen = chosen[:]
    
    # Now try 1-for-1 swaps: remove a selected element, try to add a better one
    for iteration in range(200):
        if time.time() > DEADLINE - 10:
            break
        improved = False
        for i, a_out in enumerate(chosen):
            if time.time() > DEADLINE - 10:
                break
            # Temporarily remove a_out
            used_primes_copy = set(used_primes)
            for p in pf_cache[a_out]:
                used_primes_copy.discard(p)
            
            # Try to add each candidate a_in not in chosen
            for a_in in candidates:
                if a_in in chosen_set:
                    continue
                conflict = False
                for p in pf_cache[a_in]:
                    if p in used_primes_copy:
                        conflict = True
                        break
                if conflict:
                    continue
                # Check if adding a_in gives better score than keeping a_out
                # Gain = weight[a_in] - weight[a_out]
                if weight[a_in] > weight[a_out]:
                    # Perform swap
                    chosen[i] = a_in
                    chosen_set.discard(a_out)
                    chosen_set.add(a_in)
                    for p in pf_cache[a_out]:
                        used_primes.discard(p)
                    for p in pf_cache[a_in]:
                        used_primes.add(p)
                    improved = True
                    new_score = current_score()
                    if new_score > best_score:
                        best_score = new_score
                        best_chosen = chosen[:]
                    break  # Move to next a_out
        
        if not improved:
            break
    
    # Try 1-for-2 swaps: remove one, add two
    for iteration in range(50):
        if time.time() > DEADLINE - 15:
            break
        improved = False
        for i, a_out in enumerate(chosen):
            if time.time() > DEADLINE - 15:
                break
            used_primes_copy = set(used_primes)
            for p in pf_cache[a_out]:
                used_primes_copy.discard(p)
            
            # Try pairs of candidates
            for j, a_in1 in enumerate(candidates):
                if a_in1 in chosen_set:
                    continue
                conflict1 = False
                for p in pf_cache[a_in1]:
                    if p in used_primes_copy:
                        conflict1 = True
                        break
                if conflict1:
                    continue
                
                used_primes_copy2 = set(used_primes_copy)
                for p in pf_cache[a_in1]:
                    used_primes_copy2.add(p)
                
                for a_in2 in candidates:
                    if a_in2 in chosen_set or a_in2 == a_in1:
                        continue
                    conflict2 = False
                    for p in pf_cache[a_in2]:
                        if p in used_primes_copy2:
                            conflict2 = True
                            break
                    if conflict2:
                        continue
                    
                    gain = weight[a_in1] + weight[a_in2] - weight[a_out]
                    if gain > 0:
                        # Perform swap
                        chosen[i] = a_in1
                        chosen.append(a_in2)
                        chosen_set.discard(a_out)
                        chosen_set.add(a_in1)
                        chosen_set.add(a_in2)
                        for p in pf_cache[a_out]:
                            used_primes.discard(p)
                        for p in pf_cache[a_in1]:
                            used_primes.add(p)
                        for p in pf_cache[a_in2]:
                            used_primes.add(p)
                        improved = True
                        new_score = current_score()
                        if new_score > best_score:
                            best_score = new_score
                            best_chosen = chosen[:]
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    if time.time() > DEADLINE - 5:
        return best_chosen
    
    # Final polish: try all single additions again
    for a in candidates:
        if a in chosen_set:
            continue
        conflict = False
        for p in pf_cache[a]:
            if p in used_primes:
                conflict = True
                break
        if not conflict:
            chosen.append(a)
            chosen_set.add(a)
            for p in pf_cache[a]:
                used_primes.add(p)
            new_score = current_score()
            if new_score > best_score:
                best_score = new_score
                best_chosen = chosen[:]
    
    return best_chosen
# EVOLVE-BLOCK-END
