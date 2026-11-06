"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Domain
    V_small = list(range(2, 129))
    V_large = list(range(129, 257))
    
    # Precompute reciprocal sums
    # Base sum of V_large
    base_sum_large = sum(1.0 / b for b in V_large)
    
    # Precompute conflicts for V_large
    # conf_large[a] = set of b in V_large such that lcm(a, b) <= 256
    conf_large = {}
    for a in V_small:
        confs = set()
        for b in V_large:
            # lcm(a, b) = a * b // gcd(a, b)
            if (a * b) // gcd(a, b) <= 256:
                confs.add(b)
        conf_large[a] = confs
        
    # Precompute conflicts within V_small
    # conf_small[a] = set of c in V_small such that lcm(a, c) <= 256
    conf_small = {}
    for a in V_small:
        confs = set()
        for c in V_small:
            if a == c:
                continue
            if (a * c) // gcd(a, c) <= 256:
                confs.add(c)
        conf_small[a] = confs

    # Search strategy:
    # 1. Define a core set of small numbers (2..30)
    # 2. Exhaustive search over independent subsets of core
    # 3. For each core subset, greedily extend with V_small[30..128]
    
    core_numbers = list(range(2, 31))
    other_small = list(range(31, 129))
    
    best_score = 0.0
    best_set = []
    
    # Helper to evaluate a set A
    def evaluate_set(A):
        # A is a set of numbers from V_small
        # Calculate loss in V_large
        lost_large = set()
        for a in A:
            lost_large.update(conf_large[a])
        
        # Calculate score
        sum_a = sum(1.0 / a for a in A)
        sum_b = base_sum_large - sum(1.0 / b for b in lost_large)
        return sum_a + sum_b, A.union(lost_large) # Return score and set of lost B (for debugging?)
        # Actually we just need the set A and the final S = A union (V_large \ lost_large)
        
    # Recursive search over core_numbers
    # Sort core by reciprocal desc
    core_sorted = sorted(core_numbers, key=lambda x: -1.0/x)
    
    def backtrack(idx, current_A, current_conflicts):
        nonlocal best_score, best_set
        
        if time.time() > deadline:
            return
            
        # Pruning: Upper bound
        # Current score + sum of remaining core reciprocals + sum of other_small reciprocals (optimistic)
        # This bound is loose, but helps.
        
        # Try to add core_sorted[idx]
        a = core_sorted[idx]
        
        # Option 1: Skip a
        if idx < len(core_sorted):
            backtrack(idx + 1, current_A, current_conflicts)
            
        # Option 2: Add a if no conflict
        if a not in current_conflicts:
            new_A = current_A | {a}
            new_conflicts = current_conflicts | conf_small[a] | {a}
            
            # Greedy extension with other_small
            # We can try to add any c in other_small that doesn't conflict
            # To keep it fast, we just add all valid ones? 
            # No, adding c might block other c's? 
            # Conflicts in other_small are sparse but exist.
            # Given time, let's just take the greedy best from other_small.
            
            extended_A = set(new_A)
            extended_conflicts = set(new_conflicts)
            
            # Sort other_small by reciprocal desc
            # We pre-sorted? No, let's sort once outside.
            # But we need to check conflicts dynamically.
            
            # For speed, we'll just iterate other_small and add if valid.
            # This is a greedy heuristic for the "tail".
            for c in other_small:
                if c not in extended_conflicts:
                    extended_A.add(c)
                    extended_conflicts |= conf_small[c]
                    extended_conflicts.add(c)
                    
            score, _ = evaluate_set(extended_A)
            if score > best_score:
                best_score = score
                best_set = list(extended_A)
                
            # Note: We don't backtrack into other_small.
            # This assumes the greedy extension is optimal for a fixed core.
            # This is a reasonable approximation.

    backtrack(0, set(), set())
    
    # Fallback: If best_set is empty or score is low, try a simple greedy on all V_small
    if not best_set or best_score < 0.1:
        greedy_A = set()
        greedy_conflicts = set()
        for a in sorted(V_small, key=lambda x: -1.0/x):
            if a not in greedy_conflicts:
                greedy_A.add(a)
                greedy_conflicts |= conf_small[a]
                greedy_conflicts.add(a)
        
        score, _ = evaluate_set(greedy_A)
        if score > best_score:
            best_score = score
            best_set = list(greedy_A)

    # Construct final set
    if not best_set:
        return [2] # Should not happen
        
    # Final set is best_set (from V_small) + all compatible V_large
    final_set = set(best_set)
    lost_large = set()
    for a in final_set:
        lost_large.update(conf_large[a])
        
    for b in V_large:
        if b not in lost_large:
            final_set.add(b)
            
    return list(final_set)
# EVOLVE-BLOCK-END
