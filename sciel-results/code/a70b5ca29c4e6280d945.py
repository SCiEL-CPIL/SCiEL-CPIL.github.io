import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    MIN_SUM = 128
    MAX_SUM = 256
    NUM_SUMS = MAX_SUM - MIN_SUM + 1
    
    # Helper to calculate initial multiplicities for a given set A
    # Returns a numpy array of length NUM_SUMS
    def calc_mult(A):
        M = np.zeros(NUM_SUMS, dtype=np.int32)
        A_list = sorted(list(A))
        n = len(A_list)
        for i in range(n):
            a = A_list[i]
            # a + a
            s = 2 * a
            if MIN_SUM <= s <= MAX_SUM:
                M[s - MIN_SUM] += 1
            # a + b for b > a
            for j in range(i + 1, n):
                b = A_list[j]
                s = a + b
                if MIN_SUM <= s <= MAX_SUM:
                    M[s - MIN_SUM] += 1
        return M

    # Initial set: Use a dense interval that is known to cover the range
    # [64, 128] sums cover [128, 256]. 
    # Specifically, min sum 64+64=128, max sum 128+128=256.
    # This is a valid starting point.
    A = set(range(64, 129))
    M = calc_mult(A)
    
    # Check initial validity
    if np.min(M) < 1:
        # Fallback if the seed is somehow invalid (should not happen for this specific range)
        # Try a denser set
        A = set(range(0, 129))
        M = calc_mult(A)
        
    current_score = int(np.max(M))
    best_score = current_score
    best_A = set(A)
    
    # Simulated Annealing parameters
    T = 5.0
    cooling = 0.9995
    min_T = 0.1
    
    ALL_ELEMENTS = set(range(0, 129))
    
    # Pre-compute list of elements in A for faster random selection if needed, 
    # but set iteration is fine for random.choice if we convert to list occasionally.
    # For efficiency in the loop, we keep A as a set and M as an array.
    
    while time.time() < deadline:
        # Choose a move
        r = random.random()
        
        if r < 0.4:
            # Remove
            if len(A) <= 10:  # Keep set reasonably large to maintain coverage
                T *= cooling
                continue
            x = random.choice(list(A))
            
            # Simulate removal
            M_new = M.copy()
            
            # Subtract contributions of x
            # x + x
            s = 2 * x
            if MIN_SUM <= s <= MAX_SUM:
                M_new[s - MIN_SUM] -= 1
            
            # x + a for a in A, a != x
            for a in A:
                if a == x:
                    continue
                s = a + x
                if MIN_SUM <= s <= MAX_SUM:
                    M_new[s - MIN_SUM] -= 1
            
            # Check validity
            if np.min(M_new) >= 1:
                score_new = int(np.max(M_new))
                delta = score_new - current_score
                
                # Accept if better or with probability exp(-delta/T)
                if delta < 0 or (delta == 0 and random.random() < 0.5) or (delta > 0 and random.random() < np.exp(-delta / T)):
                    A.remove(x)
                    M = M_new
                    current_score = score_new
                    if score_new < best_score:
                        best_score = score_new
                        best_A = set(A)
                # else revert (do nothing to A or M)
            # else revert (do nothing)
            
        elif r < 0.8:
            # Swap
            if len(A) == 0:
                T *= cooling
                continue
            x = random.choice(list(A))
            # Choose y not in A
            candidates = [y for y in ALL_ELEMENTS if y not in A]
            if not candidates:
                T *= cooling
                continue
            y = random.choice(candidates)
            
            # Simulate swap: Remove x, Add y
            M_new = M.copy()
            
            # Subtract x
            s = 2 * x
            if MIN_SUM <= s <= MAX_SUM:
                M_new[s - MIN_SUM] -= 1
            for a in A:
                if a == x:
                    continue
                s = a + x
                if MIN_SUM <= s <= MAX_SUM:
                    M_new[s - MIN_SUM] -= 1
            
            # Add y
            s = 2 * y
            if MIN_SUM <= s <= MAX_SUM:
                M_new[s - MIN_SUM] += 1
            for a in A:
                if a == x:
                    continue # a is still in A during swap simulation? 
                    # Wait, in a swap, we remove x and add y. 
                    # The pairs involving x are removed. The pairs involving y are added.
                    # Pairs (a, x) where a != x are removed.
                    # Pairs (a, y) where a != y are added.
                    # Since x != y, and x is in A, y is not in A initially.
                    # So we iterate over A. For each a in A:
                    # If a == x, we already handled (x,x) and (x,a) where a!=x? 
                    # Let's be precise.
                    # Removed: (x,x) and {(a,x) | a in A, a != x}
                    # Added: (y,y) and {(a,y) | a in A, a != x} 
                    # Note: y is not in A, so a != y is always true for a in A.
                    pass 
            
            # Let's re-implement the swap update clearly.
            # Current A contains x.
            # 1. Remove x:
            #    - (x,x)
            #    - (a,x) for all a in A, a != x
            # 2. Add y:
            #    - (y,y)
            #    - (a,y) for all a in A, a != x (since x is removed, y is added, the set becomes A \ {x} U {y})
            #    - Wait, the new set is A' = (A \ {x}) U {y}.
            #    - The new pairs involving y are (y,y) and (a,y) for a in A \ {x}.
            
            # My previous loop for adding y iterated over A. 
            # If I iterate over A, I include a=x. But x is not in A'.
            # So I should iterate over A and skip x for the "add y" part?
            # Yes.
            
            # Let's redo the M_new calculation for swap properly to avoid bugs.
            M_new = M.copy()
            
            # Remove x
            s = 2 * x
            if MIN_SUM <= s <= MAX_SUM:
                M_new[s - MIN_SUM] -= 1
            for a in A:
                if a == x:
                    continue
                s = a + x
                if MIN_SUM <= s <= MAX_SUM:
                    M_new[s - MIN_SUM] -= 1
            
            # Add y
            s = 2 * y
            if MIN_SUM <= s <= MAX_SUM:
                M_new[s - MIN_SUM] += 1
            for a in A:
                if a == x:
                    continue
                s = a + y
                if MIN_SUM <= s <= MAX_SUM:
                    M_new[s - MIN_SUM] += 1
            
            if np.min(M_new) >= 1:
                score_new = int(np.max(M_new))
                delta = score_new - current_score
                
                if delta < 0 or (delta == 0 and random.random() < 0.5) or (delta > 0 and random.random() < np.exp(-delta / T)):
                    A.remove(x)
                    A.add(y)
                    M = M_new
                    current_score = score_new
                    if score_new < best_score:
                        best_score = score_new
                        best_A = set(A)
            
        else:
            # Add
            candidates = [y for y in ALL_ELEMENTS if y not in A]
            if not candidates:
                T *= cooling
                continue
            y = random.choice(candidates)
            
            # Simulate add
            M_new = M.copy()
            
            # Add y
            s = 2 * y
            if MIN_SUM <= s <= MAX_SUM:
                M_new[s - MIN_SUM] += 1
            for a in A:
                s = a + y
                if MIN_SUM <= s <= MAX_SUM:
                    M_new[s - MIN_SUM] += 1
            
            # Adding elements always increases counts, so min(M_new) >= min(M) >= 1.
            # So it is always valid in terms of coverage.
            score_new = int(np.max(M_new))
            delta = score_new - current_score
            
            # Adding usually increases max, so delta > 0.
            # Only accept if it helps escape local minima or if T is high.
            if delta < 0 or (delta == 0 and random.random() < 0.5) or (delta > 0 and random.random() < np.exp(-delta / T)):
                A.add(y)
                M = M_new
                current_score = score_new
                # We don't update best here because adding rarely improves max, 
                # but it might enable future removals that do.
            
        # Cool down
        T *= cooling
        if T < min_T:
            T = min_T
    
    return sorted(list(best_A))
