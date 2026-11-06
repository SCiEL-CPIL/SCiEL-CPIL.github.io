"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    N = 512
    
    # Precompute values and prime factors
    values = [0.0] * N
    for a in range(1, N):
        values[a] = 1.0 / (N - a)
        
    # Sieve for smallest prime factor to speed up factorization if needed, 
    # but gcd is fast enough for 512 numbers.
    
    def get_factors(n):
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

    # Precompute factors for all numbers
    all_factors = [set() for _ in range(N)]
    for i in range(2, N):
        all_factors[i] = get_factors(i)
        
    def is_coprime_to_all(a, chosen_set):
        fa = all_factors[a]
        if not fa: return True
        for b in chosen_set:
            if fa & all_factors[b]:
                return False
        return True

    # Phase 1: Greedy Initialization
    chosen = []
    chosen_set = set()
    for a in range(511, 0, -1):
        if is_coprime_to_all(a, chosen_set):
            chosen.append(a)
            chosen_set.add(a)
            
    def current_score():
        return sum(values[a] for a in chosen)
        
    best_score = current_score()
    best_chosen = chosen[:]
    
    def greedy_select_candidates(candidates, forbidden_set):
        """
        Greedily select a subset of candidates that are pairwise coprime 
        and coprime to forbidden_set.
        Returns the list of selected numbers and their total value.
        """
        selected = []
        selected_set = set()
        total_val = 0.0
        
        # Sort candidates by value descending
        sorted_cands = sorted(candidates, key=lambda x: values[x], reverse=True)
        
        for c in sorted_cands:
            if c in forbidden_set:
                continue
            if c in selected_set:
                continue
            if is_coprime_to_all(c, selected_set):
                selected.append(c)
                selected_set.add(c)
                total_val += values[c]
                
        return selected, total_val

    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Pass 1: Try to add any single number
        for a in range(511, 0, -1):
            if a not in chosen_set:
                if is_coprime_to_all(a, chosen_set):
                    new_score = current_score() + values[a]
                    if new_score > best_score + 1e-12:
                        chosen.append(a)
                        chosen_set.add(a)
                        best_score = new_score
                        best_chosen = chosen[:]
                        improved = True
                    # Don't break, keep trying to add more if possible
                    # But adding one might block others, so re-evaluating is complex.
                    # However, adding a high value number is usually good.
                    # Let's just do one add per pass to keep it simple and safe.
                    break
        
        # Pass 2: Try to replace one number with a set of numbers
        # We iterate through a copy of the list to avoid modification issues during iteration
        for i in range(len(chosen)):
            if time.time() > deadline:
                return best_chosen
                
            old_a = chosen[i]
            
            # Temporarily remove old_a
            chosen_set.remove(old_a)
            
            # Find all candidates coprime to the remaining set
            # CRITICAL FIX: Candidates must be coprime to the current chosen_set
            candidates = [a for a in range(1, N) if a not in chosen_set and is_coprime_to_all(a, chosen_set)]
            
            # Greedily select the best replacement set from candidates
            new_subset, new_val = greedy_select_candidates(candidates, chosen_set)
            
            old_val = values[old_a]
            
            if new_val > old_val + 1e-12:
                # Perform the swap
                # Remove old_a from list
                chosen.pop(i)
                
                # Add new subset
                for a in new_subset:
                    chosen.append(a)
                    chosen_set.add(a)
                    
                best_score = current_score()
                best_chosen = chosen[:]
                improved = True
            else:
                # Restore old_a if no improvement
                chosen_set.add(old_a)
                # Note: chosen list still has old_a at i because we didn't pop it in the 'else' branch logic above?
                # Wait, in the 'if' branch we popped it. In 'else', we didn't.
                # So if 'if' was taken, chosen is modified. If 'else', chosen is intact.
                # We just need to ensure chosen_set is correct.
                # In 'if' branch, we added new_subset to chosen_set.
                # In 'else' branch, we added old_a back to chosen_set.
                pass

    return best_chosen
# EVOLVE-BLOCK-END
