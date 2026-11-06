import time
import random
from itertools import combinations_with_replacement, combinations

def construct():
    start_time = time.time()
    time_limit = 50.0  # Safety margin for 60s limit
    
    n = 256
    best_A = []
    best_B = []
    best_size = 0
    
    # Helper functions for a specific state (A, B, sums_A, diffs_A, sums_B, diffs_B)
    # To avoid passing too many args, we'll use a class or just local variables in the search loop.
    
    def get_sums(S):
        return {a + b for a, b in combinations_with_replacement(S, 2)}
    
    def get_diffs(S):
        return {b - a for a, b in combinations(S, 2)}
        
    def can_add_sidon(S, sums, x):
        # Check if x can be added to S to maintain Sidon property
        # New sums: {x + s for s in S} U {2x}
        # These must be disjoint from existing sums
        new_sums = {x + s for s in S}
        new_sums.add(2 * x)
        if new_sums & sums:
            return False
        return True

    def can_add_diff(S, diffs, D_other, x):
        # Check if x can be added to S such that new diffs don't collide with D_other
        # New diffs: {x - s for s in S} U {s - x for s in S}
        new_diffs = {x - s for s in S}
        new_diffs.update({s - x for s in S})
        if new_diffs & D_other:
            return False
        return True

    def is_valid_state(A, B):
        # Full validation for final check
        if not A or not B:
            return False
        if len(A) > 64 or len(B) > 64:
            return False
        sums_A = get_sums(A)
        if len(sums_A) != len(list(combinations_with_replacement(A, 2))):
            return False
        sums_B = get_sums(B)
        if len(sums_B) != len(list(combinations_with_replacement(B, 2))):
            return False
        diffs_A = get_diffs(A)
        diffs_B = get_diffs(B)
        if diffs_A & diffs_B:
            return False
        return True

    def local_search(A, B):
        # Returns improved A, B
        A = set(A)
        B = set(B)
        
        # Precompute sums and diffs
        sums_A = get_sums(list(A))
        diffs_A = get_diffs(list(A))
        sums_B = get_sums(list(B))
        diffs_B = get_diffs(list(B))
        
        changed = True
        while changed:
            changed = False
            
            # 1. Try to add elements
            for x in range(1, n + 1):
                if x in A or x in B:
                    continue
                
                # Try adding to A
                if len(A) < 64:
                    if can_add_sidon(A, sums_A, x) and can_add_diff(A, diffs_A, diffs_B, x):
                        A.add(x)
                        # Update sums_A
                        sums_A.update({x + s for s in A if s != x}) # Wait, A already has x
                        # Correct update:
                        # Old sums_A didn't have x.
                        # New sums involving x: {x+s for s in A\{x}} U {2x}
                        # But A now contains x.
                        # Let's just recompute or do it carefully.
                        # sums_A = sums_A | {x+s for s in A if s != x} | {2*x}
                        # But we need to remove nothing.
                        # Actually, can_add_sidon checked against OLD sums_A.
                        # So we can update:
                        new_sums_to_add = {x + s for s in A if s != x}
                        new_sums_to_add.add(2 * x)
                        sums_A.update(new_sums_to_add)
                        
                        # Update diffs_A
                        new_diffs_to_add = {x - s for s in A if s != x}
                        new_diffs_to_add.update({s - x for s in A if s != x})
                        diffs_A.update(new_diffs_to_add)
                        
                        changed = True
                        break # Restart loop with new state? Or continue?
                        # Breaking is safer to ensure consistency of sums/diffs vars.
                
                # Try adding to B
                if len(B) < 64:
                    if can_add_sidon(B, sums_B, x) and can_add_diff(B, diffs_B, diffs_A, x):
                        B.add(x)
                        new_sums_to_add = {x + s for s in B if s != x}
                        new_sums_to_add.add(2 * x)
                        sums_B.update(new_sums_to_add)
                        
                        new_diffs_to_add = {x - s for s in B if s != x}
                        new_diffs_to_add.update({s - x for s in B if s != x})
                        diffs_B.update(new_diffs_to_add)
                        
                        changed = True
                        break
            
            # 2. Try swaps (Remove from A, Add to A; or Move A->B)
            # This is more complex. Let's stick to Add and Move first.
            
            # 3. Try Move (A -> B)
            for x in list(A):
                if x in B:
                    continue
                if len(B) >= 64:
                    continue
                
                # Check if removing x from A is always valid (yes, subset of Sidon is Sidon)
                # But we need to check if adding x to B is valid.
                # However, removing x from A frees up diffs and sums.
                # But the constraint is D_A & D_B = 0.
                # If we move x from A to B:
                # New A = A \ {x}. D_A' subset of D_A.
                # New B = B U {x}. D_B' = D_B U {diffs involving x}.
                # We need D_A' & D_B' = 0.
                # Since D_A' subset D_A, and D_A & D_B = 0, D_A' & D_B = 0.
                # We just need D_A' & (New Diffs from x in B) = 0.
                
                # Let's compute Diffs of x with B
                new_diffs_x_in_B = {x - s for s in B}
                new_diffs_x_in_B.update({s - x for s in B})
                
                # Check against D_A' (which is D_A minus diffs involving x in A)
                # Diffs involving x in A: {x-s, s-x for s in A\{x}}
                # Let's just recompute D_A' and D_B' for simplicity in Move check?
                # It's O(N^2) which might be slow if done for every x.
                # Let's try a simpler Move check:
                # Can we add x to B if A was empty? No, A is not empty.
                
                # Efficient Move Check:
                # 1. Remove x from A.
                # 2. Check if x can be added to B given the NEW A.
                
                A_temp = A - {x}
                B_temp = B | {x}
                
                # Check Sidon for B_temp
                if len(B_temp) > 0:
                    # We can use can_add_sidon on B with x, but we must ensure 
                    # the sums don't collide.
                    # can_add_sidon(B, sums_B, x) checks against OLD sums_B.
                    # This is correct for Sidon.
                    if not can_add_sidon(B, sums_B, x):
                        continue
                    if not can_add_diff(B, diffs_B, diffs_A, x): # Wait, diffs_A is OLD
                        # We need to check against Diffs of A_temp
                        # Diffs of A_temp is D_A minus diffs involving x.
                        # If x was causing a conflict with B's diffs, removing it from A
                        # might allow it in B.
                        # But can_add_diff checks new_diffs_x vs D_other.
                        # Here D_other is D_A_temp.
                        # Let's compute D_A_temp.
                        pass
                
                # This is getting complicated. Let's stick to ADD only for local search 
                # and rely on multiple random restarts for the Greedy phase to find good bases.
                # Swaps/Moves are risky to implement correctly without full re-computation 
                # which is slow.
                
                # However, "Add" alone often gets stuck.
                # Let's add a simple "Swap" heuristic:
                # Pick a random x in A, remove it. Pick a random y not in A U B, try to add to A.
                # If valid, accept.
                # This is a random walk.
                
                # Given time constraints, I will implement:
                # 1. Greedy with randomization (multiple runs)
                # 2. Local Search: Add only.
                # 3. If time permits, a simple random restart of the whole process.
            
        return list(A), list(B)

    # Greedy Randomized Adaptive Search Procedure (GRASP)
    num_restarts = 20
    
    for _ in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        A = set()
        B = set()
        sums_A = set()
        diffs_A = set()
        sums_B = set()
        diffs_B = set()
        
        # Randomized Greedy Construction
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() - start_time > time_limit:
                break
                
            # Try to add to A
            if len(A) < 64:
                if can_add_sidon(A, sums_A, x) and can_add_diff(A, diffs_A, diffs_B, x):
                    A.add(x)
                    sums_A.update({x + s for s in A if s != x})
                    sums_A.add(2 * x)
                    diffs_A.update({x - s for s in A if s != x})
                    diffs_A.update({s - x for s in A if s != x})
                    continue
            
            # Try to add to B
            if len(B) < 64:
                if can_add_sidon(B, sums_B, x) and can_add_diff(B, diffs_B, diffs_A, x):
                    B.add(x)
                    sums_B.update({x + s for s in B if s != x})
                    sums_B.add(2 * x)
                    diffs_B.update({x - s for s in B if s != x})
                    diffs_B.update({s - x for s in B if s != x})
                    continue
            
            # If neither, skip x
        
        # Local Search: Add
        A, B = local_search(list(A), list(B))
        
        # Validate
        if is_valid_state(A, B):
            size = len(A) + len(B)
            if size > best_size:
                best_size = size
                best_A = A
                best_B = B

    # Final check
    if not best_A or not best_B:
        # Fallback to seed-like behavior if all fails
        return [[1], [2]]
        
    return [best_A, best_B]
