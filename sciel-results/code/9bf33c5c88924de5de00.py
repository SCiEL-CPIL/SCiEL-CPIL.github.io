"""Optimized solver for: No element dividing a sum of three others."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

# Deadline guard: evaluator kills after 90s wall-clock.
# We aim to finish well before that to be safe.
_DEADLINE = 85.0  # seconds total safety margin

def construct():
    t_start = time.time()
    n = 192
    max_size = 48
    
    # Helper to check if a set is valid
    # We want to maximize the size of the subset S of {1..n}
    # such that no x in S divides the sum of any 3 distinct elements in S\{x}.
    
    # Strategy:
    # 1. Start with a good greedy heuristic to get a baseline.
    # 2. Use local search (simulated annealing or hill climbing) to improve.
    
    # --- Step 1: Greedy Initialization ---
    # We can try different orders.
    # Larger numbers are generally "safer" as divisors because they are less likely to divide sums of other numbers (which are bounded).
    # However, small numbers are "dangerous" divisors.
    # Let's try a greedy approach that adds numbers one by one.
    
    def is_valid_addition(candidate, current_set):
        # Check if adding candidate to current_set violates the condition.
        # The condition must hold for ALL elements in new_set = current_set U {candidate}.
        # For any a in new_set, and any triple t from new_set \ {a}, sum(t) % a != 0.
        
        # This is expensive to check from scratch for every addition.
        # Optimization:
        # When we add x to S, the new triples are those involving x.
        # 1. For each a in S:
        #    - Old triples from S\{a} were already valid (since S was valid).
        #    - New triples are (x, u, v) where u, v in S\{a}.
        #      We must check if a divides (x+u+v).
        # 2. For the new element x:
        #    - Triples from S\{x} = S.
        #      We must check if x divides any triple sum from S.
        
        if len(current_set) < 3:
            # If we have fewer than 3 elements, we can't form a triple of distinct others for any element?
            # Wait. If S has 1 element, S\{a} has 0 elements. No triples. Valid.
            # If S has 2 elements, S\{a} has 1 element. No triples of size 3. Valid.
            # If S has 3 elements, S\{a} has 2 elements. No triples of size 3. Valid.
            # So we only start checking when len(S) >= 3 for the "x divides triple from S" part?
            # And for "a divides triple involving x", we need at least 2 elements in S\{a} to form a pair with x.
            # So if len(S) >= 2, we have pairs.
            
            # Let's just do the full check for simplicity in the helper, but optimize in the main loop.
            pass

        # Check x dividing any triple from current_set
        if len(current_set) >= 3:
            for t in combinations(current_set, 3):
                if (t[0] + t[1] + t[2]) % candidate == 0:
                    return False
        
        # Check a in current_set dividing any new triple (x, u, v) with u,v in current_set\{a}
        for a in current_set:
            # We need pairs u, v from current_set such that u != v, u != a, v != a.
            # And (x + u + v) % a == 0.
            # Note: u and v must be distinct from each other and from a.
            # Since x is new, x != u, x != v.
            
            # Iterate pairs from current_set
            # Optimization: Precompute pair sums?
            # For a fixed a, we want to know if there exist u, v in S\{a} such that u+v == -x (mod a).
            
            # Let's just iterate.
            S_minus_a = [v for v in current_set if v != a]
            if len(S_minus_a) < 2:
                continue
                
            # Check all pairs in S_minus_a
            # We can optimize by using a set of values or residues.
            # For each u in S_minus_a, we need v such that v == (-x - u) % a.
            # And v must be in S_minus_a, v != u.
            
            # Let's use a set for O(1) lookup?
            # But we need v != u.
            # Also, we just need to find ONE such pair.
            
            # Construct a set of values in S_minus_a?
            # Since S is small (max 48), iterating pairs is O(k^2) which is ~2300 ops.
            # This is fast enough.
            
            for i in range(len(S_minus_a)):
                u = S_minus_a[i]
                for j in range(i+1, len(S_minus_a)):
                    v = S_minus_a[j]
                    if (x + u + v) % a == 0:
                        return False
        return True

    # Greedy construction
    def greedy_build(order):
        S = []
        for x in order:
            if time.time() - t_start > _DEADLINE - 5:
                break
            if is_valid_addition(x, S):
                S.append(x)
        return S

    # Try a few different orders
    # 1. Descending
    # 2. Ascending
    # 3. Random
    # 4. Sorted by "badness" (number of divisors? or something else?)
    
    best_set = []
    
    # Order 1: Descending (Large numbers first)
    S1 = greedy_build(range(n, 0, -1))
    if len(S1) > len(best_set):
        best_set = S1
    
    # Order 2: Ascending
    S2 = greedy_build(range(1, n+1))
    if len(S2) > len(best_set):
        best_set = S2
    
    # Order 3: Random (multiple attempts)
    # We can run several random restarts.
    for _ in range(50):
        if time.time() - t_start > _DEADLINE - 10:
            break
        random_order = list(range(1, n+1))
        random.shuffle(random_order)
        S3 = greedy_build(random_order)
        if len(S3) > len(best_set):
            best_set = S3

    # --- Step 2: Local Search / Improvement ---
    # Try to swap elements to increase size or fix violations if we tried to add more.
    # Actually, the greedy set is valid. We want to see if we can add more elements.
    # Or if we can swap a "bad" element for a "good" one to allow more additions?
    
    # Let's try to add elements from the remaining pool to the best_set.
    # This is effectively what the greedy did, but maybe the order mattered.
    # Let's try a "fill" phase: try adding any remaining element that fits.
    
    current_set = list(best_set)
    remaining = [x for x in range(1, n+1) if x not in current_set]
    
    changed = True
    while changed:
        changed = False
        for x in remaining:
            if time.time() - t_start > _DEADLINE - 5:
                break
            if is_valid_addition(x, current_set):
                current_set.append(x)
                remaining.remove(x)
                changed = True
        if not changed:
            break

    best_set = current_set
    
    # Try some local swaps?
    # If we are stuck at size K, maybe we can swap an element in best_set with one in remaining
    # to unlock adding more elements?
    # This is complex. Let's stick to the fill phase which is robust.
    
    # One more optimization:
    # The greedy might have rejected a number because of a specific conflict.
    # If we remove one of the conflicting numbers, we might be able to add two new ones?
    # This is a 1-for-2 swap.
    
    # Let's implement a simple local search:
    # 1. Try removing an element from best_set.
    # 2. Try adding any element from remaining.
    # 3. If size increases, keep it.
    
    # This might be slow if done naively.
    # Let's just return the current best set. The fill phase usually gets close to the maximum for this type of problem.
    
    # Final check for validity (just in case)
    # The construction ensures validity, but let's be safe.
    # Actually, is_valid_addition is strict, so the set is valid.
    
    return best_set

# EVOLVE-BLOCK-END
