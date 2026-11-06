"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from math import gcd
from itertools import combinations
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    N = 512
    
    # Precompute values as both float and exact Fraction
    values_float = [0.0] * N
    values_frac = [Fraction(0, 1)] * N
    for a in range(1, N):
        values_float[a] = 1.0 / (N - a)
        values_frac[a] = Fraction(1, N - a)
        
    # Precompute prime factors
    def get_prime_factors(n):
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

    all_factors = [set() for _ in range(N)]
    for i in range(2, N):
        all_factors[i] = get_prime_factors(i)
        
    # State
    chosen = []
    chosen_set = set()
    prime_count = [0] * N  # prime_count[p] is count of chosen numbers divisible by p
    current_score_float = 0.0
    current_score_frac = Fraction(0, 1)
    
    def add_num(a):
        nonlocal current_score_float, current_score_frac
        chosen.append(a)
        chosen_set.add(a)
        current_score_float += values_float[a]
        current_score_frac += values_frac[a]
        for p in all_factors[a]:
            prime_count[p] += 1
            
    def remove_num_by_value(a):
        """Remove a from state without removing from list (for swap logic)"""
        nonlocal current_score_float, current_score_frac
        chosen_set.discard(a)
        current_score_float -= values_float[a]
        current_score_frac -= values_frac[a]
        for p in all_factors[a]:
            prime_count[p] -= 1

    def is_coprime_to_state(a):
        """Checks if a is coprime to all numbers currently in 'chosen' (reflected in prime_count)"""
        fa = all_factors[a]
        if not fa:
            return True
        for p in fa:
            if prime_count[p] > 0:
                return False
        return True

    # Phase 1: Greedy Initialization
    for a in range(511, 0, -1):
        if is_coprime_to_state(a):
            add_num(a)
            
    best_score_float = current_score_float
    best_score_frac = current_score_frac
    best_chosen = chosen[:]
    
    def get_top_candidates(L):
        """
        Returns top L candidates coprime to current state.
        """
        cands = []
        for a in range(511, 0, -1):
            if a in chosen_set:
                continue
            if is_coprime_to_state(a):
                cands.append(a)
                if len(cands) >= L:
                    break
        return cands

    def solve_k_opt(cands, k):
        """
        Finds best k numbers from cands that are pairwise coprime.
        cands is sorted by value descending.
        Returns list of selected numbers and their total value as Fraction.
        """
        if k == 1:
            if not cands: return [], Fraction(0, 1)
            return [cands[0]], values_frac[cands[0]]
        
        if k == 2:
            best_val = Fraction(0, 1)
            best_pair = []
            # Check pairs from top candidates
            for i in range(len(cands)):
                a = cands[i]
                fa = all_factors[a]
                for j in range(i + 1, len(cands)):
                    b = cands[j]
                    # Check coprime
                    if not (fa & all_factors[b]):
                        val = values_frac[a] + values_frac[b]
                        if val > best_val:
                            best_val = val
                            best_pair = [a, b]
            return best_pair, best_val
            
        if k == 3:
            best_val = Fraction(0, 1)
            best_triple = []
            # Limit search to top 18 candidates to keep O(n^3) manageable
            search_cands = cands[:18] 
            for i in range(len(search_cands)):
                a = search_cands[i]
                fa = all_factors[a]
                for j in range(i + 1, len(search_cands)):
                    b = search_cands[j]
                    fb = all_factors[b]
                    if fa & fb: continue
                    for k_idx in range(j + 1, len(search_cands)):
                        c = search_cands[k_idx]
                        fc = all_factors[c]
                        if (fa & fc) or (fb & fc): continue
                        val = values_frac[a] + values_frac[b] + values_frac[c]
                        if val > best_val:
                            best_val = val
                            best_triple = [a, b, c]
            return best_triple, best_val
        return [], Fraction(0, 1)

    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try Add 1
        cands = get_top_candidates(10)
        if cands:
            a = cands[0]
            if is_coprime_to_state(a):
                add_num(a)
                if current_score_frac > best_score_frac:
                    best_score_float = current_score_float
                    best_score_frac = current_score_frac
                    best_chosen = chosen[:]
                    improved = True

        # 2. Try 1-Opt Swap
        # We iterate over a copy of chosen to avoid modification issues
        for i in range(len(chosen)):
            if time.time() > deadline:
                return best_chosen
            
            old_a = chosen[i]
            
            # Temporarily remove old_a from state
            chosen.pop(i)
            remove_num_by_value(old_a)
                
            # Find best replacement
            cands = get_top_candidates(25)
            new_subset, new_val_frac = solve_k_opt(cands, 1)
            
            old_val_frac = values_frac[old_a]
            
            if new_val_frac > old_val_frac and new_subset:
                # Perform swap
                for a in new_subset:
                    add_num(a)
                best_score_float = current_score_float
                best_score_frac = current_score_frac
                best_chosen = chosen[:]
                improved = True
            else:
                # Restore old_a
                add_num(old_a)
                
        # 3. Try 2-Opt Swap
        if len(chosen) >= 2:
            # Try more pairs now that we have exact arithmetic
            pairs_to_try = list(combinations(range(len(chosen)), 2))[:8]
            for i, j in pairs_to_try:
                if time.time() > deadline:
                    return best_chosen
                
                old_a = chosen[i]
                old_b = chosen[j]
                
                # Remove both
                if i > j:
                    chosen.pop(i)
                    chosen.pop(j)
                else:
                    chosen.pop(j)
                    chosen.pop(i)
                
                remove_num_by_value(old_a)
                remove_num_by_value(old_b)
                
                cands = get_top_candidates(25)
                new_subset, new_val_frac = solve_k_opt(cands, 2)
                
                old_total_frac = values_frac[old_a] + values_frac[old_b]
                
                if new_val_frac > old_total_frac and len(new_subset) == 2:
                    for a in new_subset:
                        add_num(a)
                    best_score_float = current_score_float
                    best_score_frac = current_score_frac
                    best_chosen = chosen[:]
                    improved = True
                else:
                    # Restore
                    add_num(old_a)
                    add_num(old_b)
                    
        # 4. Try 3-Opt Swap (Limited)
        if len(chosen) >= 3:
             # Try a few triples
             triples_to_try = list(combinations(range(len(chosen)), 3))[:3]
             for i, j, k in triples_to_try:
                if time.time() > deadline:
                    return best_chosen
                
                old_a = chosen[i]
                old_b = chosen[j]
                old_c = chosen[k]
                
                # Sort indices to pop safely
                indices = sorted([i, j, k], reverse=True)
                
                for idx in indices:
                    chosen.pop(idx)
                
                remove_num_by_value(old_a)
                remove_num_by_value(old_b)
                remove_num_by_value(old_c)
                
                cands = get_top_candidates(20)
                new_subset, new_val_frac = solve_k_opt(cands, 3)
                
                old_total_frac = values_frac[old_a] + values_frac[old_b] + values_frac[old_c]
                
                if new_val_frac > old_total_frac and len(new_subset) == 3:
                    for a in new_subset:
                        add_num(a)
                    best_score_float = current_score_float
                    best_score_frac = current_score_frac
                    best_chosen = chosen[:]
                    improved = True
                else:
                    # Restore
                    add_num(old_a)
                    add_num(old_b)
                    add_num(old_c)

    # Final polish: try to add any remaining coprime numbers
    if time.time() < deadline - 5:
        cands = get_top_candidates(50)
        for a in cands:
            if is_coprime_to_state(a):
                add_num(a)
        if current_score_frac > best_score_frac:
            best_score_float = current_score_float
            best_score_frac = current_score_frac
            best_chosen = chosen[:]

    return best_chosen
# EVOLVE-BLOCK-END
