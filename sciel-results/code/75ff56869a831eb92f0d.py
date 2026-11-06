"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    N = 512
    
    # Precompute values
    values = [0.0] * N
    for a in range(1, N):
        values[a] = 1.0 / (N - a)
        
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
    current_score = 0.0
    
    def add_num(a):
        nonlocal current_score
        chosen.append(a)
        chosen_set.add(a)
        current_score += values[a]
        for p in all_factors[a]:
            prime_count[p] += 1
            
    def remove_num(a):
        nonlocal current_score
        # Remove from list
        # Since we iterate carefully, we can just pop if we know index, 
        # but for K-opt we remove multiple. 
        # For simplicity in the swap logic, we will rebuild or use indices.
        # Actually, it's better to pass indices for removal.
        pass # Placeholder, we will implement specific remove logic in the loop

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
            
    best_score = current_score
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
        Returns list of selected numbers and their total value.
        """
        if k == 1:
            if not cands: return [], 0.0
            return [cands[0]], values[cands[0]]
        
        if k == 2:
            best_val = 0.0
            best_pair = []
            # Check pairs from top candidates
            # We only need to check a subset if L is large, but L is small (e.g. 20-50)
            for i in range(len(cands)):
                a = cands[i]
                fa = all_factors[a]
                for j in range(i + 1, len(cands)):
                    b = cands[j]
                    # Check coprime
                    if not (fa & all_factors[b]):
                        val = values[a] + values[b]
                        if val > best_val:
                            best_val = val
                            best_pair = [a, b]
            return best_pair, best_val
            
        if k == 3:
            best_val = 0.0
            best_triple = []
            # Limit search to top 15 candidates to keep O(n^3) low
            search_cands = cands[:15] 
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
                        val = values[a] + values[b] + values[c]
                        if val > best_val:
                            best_val = val
                            best_triple = [a, b, c]
            return best_triple, best_val
        return [], 0.0

    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try Add 1
        cands = get_top_candidates(10)
        if cands:
            a = cands[0]
            if is_coprime_to_state(a):
                add_num(a)
                if current_score > best_score + 1e-12:
                    best_score = current_score
                    best_chosen = chosen[:]
                    improved = True
                else:
                    # Revert
                    # Need to implement proper remove
                    # For now, just don't add if it doesn't improve? 
                    # But adding might enable future swaps? 
                    # Usually adding a positive value number is good.
                    # But if it blocks a better combination, it's bad.
                    # The swap logic will handle replacements.
                    # So we can just add it.
                    pass

        # 2. Try 1-Opt Swap
        # We iterate over a copy of chosen to avoid modification issues
        for i in range(len(chosen)):
            if time.time() > deadline:
                return best_chosen
            
            old_a = chosen[i]
            
            # Temporarily remove old_a
            chosen.pop(i)
            chosen_set.remove(old_a)
            current_score -= values[old_a]
            for p in all_factors[old_a]:
                prime_count[p] -= 1
                
            # Find best replacement
            cands = get_top_candidates(20)
            new_subset, new_val = solve_k_opt(cands, 1)
            
            old_val = values[old_a]
            
            if new_val > old_val + 1e-12 and new_subset:
                # Perform swap
                for a in new_subset:
                    add_num(a)
                best_score = current_score
                best_chosen = chosen[:]
                improved = True
            else:
                # Restore old_a
                add_num(old_a)
                
        # 3. Try 2-Opt Swap
        # To save time, only try if 1-opt didn't improve in this pass? 
        # Or limit the number of 2-opt moves.
        # Let's try a few 2-opt moves.
        # We pick random pairs or first few pairs.
        # Since chosen is small, we can try all pairs if it's fast.
        # But solve_k_opt for k=2 is O(L^2).
        
        # We will try 2-opt only if we have time and haven't improved recently?
        # Let's just try the first 5 pairs in the list to keep it fast.
        if len(chosen) >= 2:
            # Try a subset of pairs
            pairs_to_try = list(combinations(range(len(chosen)), 2))[:5]
            for i, j in pairs_to_try:
                if time.time() > deadline:
                    return best_chosen
                
                old_a = chosen[i]
                old_b = chosen[j]
                
                # Remove both
                # Remove j first if i > j? No, pop by index.
                # If we pop i, j shifts.
                # Let's extract values first.
                vals_to_restore = [values[old_a], values[old_b]]
                
                # Remove
                if i > j:
                    chosen.pop(i)
                    chosen.pop(j)
                else:
                    chosen.pop(j)
                    chosen.pop(i)
                
                chosen_set.remove(old_a)
                chosen_set.remove(old_b)
                current_score -= vals_to_restore[0] + vals_to_restore[1]
                
                for p in all_factors[old_a]:
                    prime_count[p] -= 1
                for p in all_factors[old_b]:
                    prime_count[p] -= 1
                
                cands = get_top_candidates(20)
                new_subset, new_val = solve_k_opt(cands, 2)
                
                old_total_val = vals_to_restore[0] + vals_to_restore[1]
                
                if new_val > old_total_val + 1e-12 and len(new_subset) == 2:
                    for a in new_subset:
                        add_num(a)
                    best_score = current_score
                    best_chosen = chosen[:]
                    improved = True
                else:
                    # Restore
                    add_num(old_a)
                    add_num(old_b)
                    
        # 4. Try 3-Opt Swap (Limited)
        if len(chosen) >= 3:
             # Try only 1 or 2 triples
             triples_to_try = list(combinations(range(len(chosen)), 3))[:2]
             for i, j, k in triples_to_try:
                if time.time() > deadline:
                    return best_chosen
                
                old_a = chosen[i]
                old_b = chosen[j]
                old_c = chosen[k]
                
                # Sort indices to pop safely
                indices = sorted([i, j, k], reverse=True)
                removed_vals = [values[old_a], values[old_b], values[old_c]]
                
                for idx in indices:
                    chosen.pop(idx)
                
                for a in [old_a, old_b, old_c]:
                    chosen_set.remove(a)
                    current_score -= values[a]
                    for p in all_factors[a]:
                        prime_count[p] -= 1
                
                cands = get_top_candidates(15)
                new_subset, new_val = solve_k_opt(cands, 3)
                
                old_total_val = sum(removed_vals)
                
                if new_val > old_total_val + 1e-12 and len(new_subset) == 3:
                    for a in new_subset:
                        add_num(a)
                    best_score = current_score
                    best_chosen = chosen[:]
                    improved = True
                else:
                    # Restore
                    add_num(old_a)
                    add_num(old_b)
                    add_num(old_c)

    return best_chosen
# EVOLVE-BLOCK-END
