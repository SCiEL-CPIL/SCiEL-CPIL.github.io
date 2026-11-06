"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    import numpy as np
    from collections import defaultdict

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def max_sum_free_subset(S):
        """
        Find a large sum-free subset of S using greedy + local search.
        Strategy:
        1. Try multiple initial heuristics (by magnitude, by sign, random).
        2. For each, build a maximal sum-free set greedily.
        3. Perform local search: try swapping elements to increase size.
        4. Return the best found.
        """
        n = len(S)
        if n == 0:
            return []
        
        # Precompute conflict graph: for each pair (i,j), check if S[i]+S[j] is in S
        # But that's O(n^2) which is fine for n up to 240.
        # Actually, we want to check if adding an element conflicts with existing ones.
        
        # A set A is sum-free if for all a,b in A, a+b is not in A.
        # Note: a=b is allowed, so 2a must not be in A if a is in A.
        
        # Let's use a greedy approach with multiple orderings, then local search.
        
        best_subset = []
        best_size = 0
        
        # Multiple initial orderings
        orderings = []
        
        # Ordering 1: by absolute value descending
        orderings.append(sorted(range(n), key=lambda i: -abs(S[i])))
        
        # Ordering 2: by value descending
        orderings.append(sorted(range(n), key=lambda i: -S[i]))
        
        # Ordering 3: by value ascending
        orderings.append(sorted(range(n), key=lambda i: S[i]))
        
        # Ordering 4: random (with fixed seed for reproducibility)
        for seed_try in range(5):
            rng = Random(42 + seed_try)
            idxs = list(range(n))
            rng.shuffle(idxs)
            orderings.append(idxs)
        
        # For each ordering, greedily build a sum-free set
        for ordering in orderings:
            A = set()
            A_set = set()
            
            for i in ordering:
                val = S[i]
                # Check if adding val to A keeps it sum-free
                # Need to check: for all a in A, a + val not in A, and val + val not in A
                # Also for all a in A, val + a not in A (same as above)
                # And for all a in A, a + a not in A (already in A, so need 2a not in A)
                # Wait, the condition is: no a,b,c in A such that a+b=c.
                # So if we add val to A, we need to check:
                # 1. For all a in A: a + val not in A (c = a+val, a and val in A)
                # 2. For all a in A: val + a not in A (same as 1)
                # 3. val + val not in A (c = 2*val, a=b=val)
                # 4. For all a in A: a + a not in A (c = 2a, but a is already in A, so this should have been checked when a was added)
                
                # Actually, when we add val, the new triples that could be formed are:
                # (val, val, 2*val), (val, a, val+a), (a, val, a+val) for a in A
                # Since addition is commutative, (val,a) and (a,val) are the same.
                
                ok = True
                if 2 * val in A_set:
                    ok = False
                else:
                    for a in A_set:
                        if a + val in A_set:
                            ok = False
                            break
                
                if ok:
                    A.add(i)
                    A_set.add(val)
            
            if len(A) > best_size:
                best_size = len(A)
                best_subset = list(A)
        
        # Local search: try to improve by swapping elements
        # Convert best_subset to a set of indices
        best_set = set(best_subset)
        
        # Try multiple rounds of local search
        improved = True
        while improved:
            improved = False
            # For each element not in best_set, try adding it and removing one element to maintain sum-free
            for i in range(n):
                if i in best_set:
                    continue
                val = S[i]
                
                # Check if val can be added
                conflicts = set()
                for a in best_set:
                    a_val = S[a]
                    # If a_val + val in best_set's values, conflict
                    if a_val + val in {S[j] for j in best_set}:
                        conflicts.add(a)
                    if 2 * a_val == val:
                        conflicts.add(a)
                
                # Also check if 2*val is in best_set
                if 2 * val in {S[j] for j in best_set}:
                    conflicts.add(-1)  # Special marker: conflict with itself (2*val in set)
                
                if not conflicts:
                    best_set.add(i)
                    improved = True
                    continue
                
                # Try to remove one element from conflicts to add val
                # We want to remove an element that causes the fewest additional conflicts
                # For simplicity, try removing each conflicting element
                conflict_list = [c for c in conflicts if c >= 0]
                for c in conflict_list:
                    new_set = best_set - {c}
                    new_set.add(i)
                    
                    # Verify new_set is sum-free
                    new_vals = {S[j] for j in new_set}
                    valid = True
                    for a in new_set:
                        a_val = S[a]
                        if 2 * a_val in new_vals:
                            valid = False
                            break
                        for b in new_set:
                            if b != a and a_val + S[b] in new_vals:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid and len(new_set) > len(best_set):
                        best_set = new_set
                        improved = True
                        break
        
        return [S[i] for i in sorted(best_set)]

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S = instance(size, seed)
        subset = max_sum_free_subset(S)
        results.append(subset)
    
    return results
# EVOLVE-BLOCK-END
