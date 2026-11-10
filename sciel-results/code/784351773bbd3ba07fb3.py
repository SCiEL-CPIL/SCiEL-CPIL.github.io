"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations

def construct():
    target = Fraction(5, 31)
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline to stay under 90s
    
    best_max_den = float('inf')
    best_solution = None
    
    # Helper to update best solution
    def update_best(dens):
        nonlocal best_max_den, best_solution
        if not dens:
            return
        m = max(dens)
        if m < best_max_den:
            best_max_den = m
            best_solution = list(dens)
    
    # Pre-compute all subset sums for small denominators
    # We will try to find a solution where all denominators are <= M.
    # We split the set of denominators into two halves: A and B.
    # Sum(A) + Sum(B) = target.
    # We pre-compute sums for subsets of a certain size from the range [1, M].
    
    # Strategy:
    # 1. Try small M values (e.g., 100, 200, 300...)
    # 2. For a given M, generate all sums of k1 denominators from [1, M] and store them.
    # 3. Generate all sums of k2 denominators from [1, M] and check for matches.
    # To keep memory and time manageable, we restrict k1 and k2 to small numbers (e.g., 3 or 4).
    # Total terms k = k1 + k2.
    
    # Let's try k1=3, k2=3 first (total 6 terms). If not found, try k1=3, k2=4, etc.
    # Actually, since we want to minimize MAX denominator, we should search M in increasing order.
    # However, the search space for M grows rapidly. 
    # A better approach: Fix the number of terms (k) and search for the minimal M.
    # Or, fix M and search for any solution.
    
    # Given the 90s limit, we can't do exhaustive search for large M.
    # Let's use a hybrid:
    # 1. Quick greedy to get a baseline.
    # 2. MITM for small M (e.g., M up to 500) with limited term counts.
    
    # Greedy baseline
    r = target
    greedy_vals = []
    while r:
        d = (r.denominator + r.numerator - 1) // r.numerator
        greedy_vals.append(d)
        r -= Fraction(1, d)
    if greedy_vals:
        update_best(greedy_vals)
    
    # MITM Search
    # We will search for solutions with up to 8 terms (4+4) or similar splits.
    # We iterate M from a small value up to the current best_max_den.
    # For each M, we generate all sums of size k1 from {1..M} and store in dict.
    # Then we generate sums of size k2 from {1..M} and check if target - sum1 is in dict.
    
    # To avoid duplicates in the final set, we must ensure the two halves are disjoint.
    # This is tricky with just sums. We need to store the actual set of denominators.
    # Storing sets for all combinations is memory heavy if M is large.
    
    # Alternative: 
    # For a fixed M, the number of subsets of size k from M is C(M, k).
    # If M=100, k=4, C(100,4) ~ 3.9M. This is too many to store in Python dict with Fraction keys quickly.
    # Let's restrict M to be smaller or k to be smaller.
    
    # Let's try k1=2, k2=2 (total 4 terms). C(100,2)=4950. Very fast.
    # Let's try k1=3, k2=3 (total 6 terms). C(100,3)=161k. Manageable.
    # Let's try k1=4, k2=4 (total 8 terms). C(100,4)=3.9M. Might be slow in Python.
    
    # We'll start with small term counts and increase M.
    # We will check M from 10 up to best_max_den - 1.
    
    max_m_to_check = int(best_max_den) - 1
    if max_m_to_check < 10:
        max_m_to_check = 100 # Just in case greedy was bad
    
    # We'll try different splits (k1, k2)
    splits = [(2, 2), (3, 2), (3, 3), (4, 2), (4, 3)]
    
    for k1, k2 in splits:
        if time.time() > deadline:
            break
            
        # We need to search M.
        # Instead of iterating M one by one, we can just generate all subsets of size k1 from [1, max_m_to_check]
        # and store them. Then generate subsets of size k2 and check.
        # But if max_m_to_check is large (e.g., 1000), C(1000, 4) is huge.
        
        # Let's limit M for MITM to a reasonable range, say 200.
        # If we don't find a solution with M <= 200, we rely on greedy or other methods.
        # The greedy solution for 5/31:
        # 5/31 = 1/7 + 1/11 + 1/15 + ... (Greedy usually gives large numbers)
        # Let's see what greedy gives.
        # 5/31 ~ 0.161
        # 1/7 ~ 0.142, rem ~ 0.019 -> 1/53
        # 1/53 ~ 0.0188, rem ~ 0.0002 -> 1/5000+
        # So greedy max den is likely > 5000.
        
        # MITM with M=200 is very likely to find a solution if one exists with small denoms.
        
        M_limit = min(max_m_to_check, 200)
        
        # Precompute sums for size k1
        sums_k1 = {}
        # Using combinations from itertools
        # To save memory, we can store the sum and the tuple of denominators.
        # But we need to check disjointness.
        
        # Optimization:
        # We can iterate M from 1 to M_limit.
        # But generating all subsets of [1..M] for each M is redundant.
        # Better: Generate all subsets of size k1 from [1..M_limit] once.
        # Sort them by max element? No, we just need to find ANY solution.
        
        # Let's generate all subsets of size k1 from [1..M_limit]
        # and store in dict: sum -> list of subsets (to handle collisions)
        # Actually, we just need one match.
        
        # Memory check: C(200, 4) ~ 64M. Too many.
        # C(100, 4) ~ 3.9M. Might be okay if we are careful, but Python dict with Fraction keys is slow.
        # Let's stick to k <= 3 for M=200. C(200, 3) ~ 1.3M. Okay.
        # So we can do (3,2), (3,3), (2,2) for M up to 200.
        # For k=4, we must restrict M to ~50. C(50,4) ~ 230k.
        
        effective_M = M_limit
        if k1 >= 4 or k2 >= 4:
            effective_M = min(effective_M, 60)
        elif k1 >= 3 or k2 >= 3:
            effective_M = min(effective_M, 150)
        else:
            effective_M = min(effective_M, 200)
            
        if effective_M < 10:
            continue
            
        # Generate and store k1 sums
        # We use a dictionary mapping Fraction sum to a list of tuples (to allow multiple choices)
        # But to save memory, we can just store one tuple per sum? No, we need disjointness.
        # If we store all, it's heavy.
        
        # Alternative approach for disjointness:
        # When we find a match sum1 + sum2 = target, we check if set1 & set2 is empty.
        # If we store all subsets, we might have many.
        
        # Let's try to be smart:
        # We iterate over subsets of size k1.
        # For each subset, we compute sum.
        # We store it in a dict: sum -> subset (just one? No, we might need another one that is disjoint).
        # Actually, if we find a match, we can just check disjointness. If it fails, we need another subset with the same sum.
        # So we should store a list.
        
        # To limit memory, we'll cap the list size per sum to, say, 10.
        
        dict_k1 = {}
        
        # Generate combinations
        # Using itertools.combinations(range(1, effective_M + 1), k1)
        
        for comb in combinations(range(1, effective_M + 1), k1):
            s = sum(Fraction(1, d) for d in comb)
            if s in dict_k1:
                if len(dict_k1[s]) < 10:
                    dict_k1[s].append(comb)
            else:
                dict_k1[s] = [comb]
                
        if time.time() > deadline:
            break
            
        # Now iterate over k2 combinations
        for comb2 in combinations(range(1, effective_M + 1), k2):
            s2 = sum(Fraction(1, d) for d in comb2)
            needed = target - s2
            
            if needed in dict_k1:
                # Found a potential match
                for comb1 in dict_k1[needed]:
                    set1 = set(comb1)
                    set2 = set(comb2)
                    if not set1.isdisjoint(set2):
                        continue
                    
                    # Valid solution
                    sol = list(set1.union(set2))
                    update_best(sol)
                    break
                if best_max_den <= effective_M:
                    break # Can't do better with this M
        
        if best_max_den <= effective_M:
            break
        
    return best_solution if best_solution else greedy_vals
# EVOLVE-BLOCK-END
