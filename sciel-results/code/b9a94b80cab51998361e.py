"""Rewritten program: greedy + vectorized local search (add/swap/drop) for B2[2] sets."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110.0  # safety margin before 120s kill (150s wall)
    N = 1000
    MAX_REP = 2
    MAX_SUM = 2 * N

    # --- Greedy initialization (fast, sequential) ---
    # We build a set where every sum a+b (a<=b, a,b in S) has at most MAX_REP representations.
    # Use a count array for sums.
    sum_counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
    chosen = []
    
    for x in range(1, N + 1):
        # Compute new sums: x + y for y in chosen (with y <= x to maintain a<=b? 
        # Actually, representations are unordered pairs {a,b} with a<=b.
        # When adding x, new pairs are (x, x) and (x, y) for y in chosen where y < x.
        # Pairs (y, x) with y > x don't exist yet since we add in increasing order? 
        # Wait, the set is not necessarily sorted in the final output, but we build it.
        # If we maintain chosen as a list, adding x creates pairs:
        # 1. (x, x) -> sum 2x
        # 2. (y, x) for all y in chosen such that y < x. Since we iterate x increasing,
        #    all y in chosen are < x. So pairs are (y, x) for all y in chosen.
        #    Sums are y + x.
        
        # Check if adding x violates the constraint.
        # New sums to check: 2x and x + y for all y in chosen.
        
        # Optimization: If chosen is large, this loop is O(|chosen|).
        # For greedy, |chosen| ~ 400-500. 1000 * 500 = 500k ops. Very fast.
        
        # Let's check validity.
        # We need to ensure that for all new sums s, count[s] + 1 <= MAX_REP.
        
        # List of new sums:
        # s_0 = 2*x
        # s_i = x + chosen[i]
        
        # We can do this in Python loop for simplicity in greedy phase, it's fast enough.
        valid = True
        new_sums = [2 * x] + [x + y for y in chosen]
        
        for s in new_sums:
            if sum_counts[s] + 1 > MAX_REP:
                valid = False
                break
        
        if valid:
            # Commit
            sum_counts[new_sums] += 1
            chosen.append(x)
    
    # Convert to numpy array for vectorized local search
    S = np.array(chosen, dtype=np.int64)
    
    # --- Local Search Phase ---
    # Goal: Maximize |S| by adding elements (if possible) or swapping to free up space?
    # Actually, adding is strictly better if possible.
    # If no element can be added, we might try to swap an existing element with a new one
    # to allow more additions later? Or just accept the greedy result if stuck.
    # A better strategy: Randomized restarts or local search that tries to add any missing element.
    
    # Let's identify which elements from 1..1000 are NOT in S.
    in_set = np.zeros(N + 1, dtype=bool)
    in_set[S] = True
    candidates = np.where(~in_set)[0]
    candidates = candidates[candidates > 0] # Exclude 0
    
    # Function to check if an element x can be added to current set S (represented by sum_counts)
    # Returns True if valid.
    def can_add(x, sum_counts):
        # New sums: 2x and x + y for y in S
        # We need sum_counts[s] < MAX_REP for all new sums s.
        # Vectorized check:
        # sums = np.concatenate(([2*x], x + S))
        # return np.all(sum_counts[sums] < MAX_REP)
        # But S changes. Let's pass S.
        pass

    # Efficient approach:
    # Maintain sum_counts.
    # Try to add candidates one by one.
    # If we can add, do it.
    # If we can't, maybe try swapping?
    # Swapping is complex. Let's stick to "add if possible" with randomization.
    
    # Better: Iterative improvement.
    # 1. Try to add any candidate.
    # 2. If none can be added, try random swaps?
    
    # Let's implement a loop that tries to add candidates in random order until stuck.
    # Then, if stuck, we can try to remove an element that is "blocking" many candidates?
    # This is getting complex.
    
    # Simpler robust strategy:
    # 1. Greedy construction (done).
    # 2. Local search:
    #    - While time remains:
    #      - Pick a random candidate x not in S.
    #      - Check if x can be added.
    #      - If yes, add it.
    #      - If no, try to find an element y in S such that removing y allows adding x?
    #        Or just skip.
    #      - To escape local optima, we can do "drop-add": remove a random y, then try to add multiple new elements.
    
    # Let's implement a simple "Add if possible" loop first, then a "Swap" loop.
    
    current_size = len(S)
    
    # Helper: add x to S and update sum_counts
    def do_add(x):
        nonlocal S, sum_counts
        sums = np.concatenate(([2 * x], x + S))
        sum_counts[sums] += 1
        S = np.append(S, x)
    
    # Helper: remove x from S and update sum_counts
    def do_remove(x):
        nonlocal S, sum_counts
        # Find index
        idx = np.where(S == x)[0][0]
        S = np.delete(S, idx)
        # Update counts
        sums = np.concatenate(([2 * x], x + S)) # Note: S is already updated, so x is not in S here.
        # Wait, the pairs involving x are (x,x) and (x, y) for y in S_new.
        # So sums are 2x and x + y for y in S_new.
        sum_counts[sums] -= 1
        # Ensure no negative counts (shouldn't happen)
        sum_counts[sums] = np.maximum(sum_counts[sums], 0)

    # Check if x can be added
    def can_add(x):
        sums = np.concatenate(([2 * x], x + S))
        return np.all(sum_counts[sums] < MAX_REP)

    # Main search loop
    # We will try to add candidates.
    # We'll keep a list of candidates.
    # To avoid O(N) scan for candidates every time, we can just iterate 1..1000 and check.
    
    # Phase 1: Exhaustive add (in increasing order, then random)
    # Actually, the greedy already did increasing order.
    # Let's try random order to see if we can squeeze in more.
    
    # Re-initialize candidates
    in_set = np.zeros(N + 1, dtype=bool)
    in_set[S] = True
    candidates = np.where(~in_set)[0]
    candidates = candidates[candidates > 0]
    
    np.random.shuffle(candidates)
    
    # Try to add them
    added_any = True
    while added_any and time.time() < deadline:
        added_any = False
        # Filter candidates that are still not in S (in case some were added)
        # Since we modify S, we need to be careful.
        # Let's just iterate through the shuffled list.
        for x in candidates:
            if in_set[x]:
                continue
            if can_add(x):
                do_add(x)
                in_set[x] = True
                added_any = True
                current_size += 1
                # Optional: break and restart shuffle? Or continue?
                # Continuing is fine.
    
    # Phase 2: Local search with swaps
    # If we are stuck, try swapping an element y in S with a candidate x not in S.
    # Condition: 
    #   1. S \ {y} U {x} is valid.
    #   2. This might allow adding more elements later?
    #   Actually, a swap keeps size constant. It's only useful if it enables future adds.
    #   Simpler: Try to find a swap that results in a state where at least one new element can be added.
    #   This is expensive.
    
    # Alternative: "Drop and Fill"
    # Remove a random y from S.
    # Try to add as many candidates as possible.
    # If net gain > 0, keep the change.
    
    # Let's implement a few iterations of Drop-and-Fill.
    
    def try_drop_and_fill(y):
        """
        Temporarily remove y, then greedily add candidates in random order.
        Returns the new set and sum_counts if size increased, else None.
        """
        # Copy state
        S_copy = S.copy()
        sc_copy = sum_counts.copy()
        in_set_copy = in_set.copy()
        
        # Remove y
        idx = np.where(S_copy == y)[0]
        if len(idx) == 0:
            return None
        idx = idx[0]
        S_copy = np.delete(S_copy, idx)
        # Update counts for removal
        sums_remove = np.concatenate(([2 * y], y + S_copy))
        sc_copy[sums_remove] -= 1
        sc_copy[sums_remove] = np.maximum(sc_copy[sums_remove], 0)
        in_set_copy[y] = False
        
        initial_size = len(S_copy)
        
        # Try to add candidates
        # Use a random order of candidates not in S_copy
        cand_mask = ~in_set_copy
        cand_mask[0] = False # 0 not allowed
        cands = np.where(cand_mask)[0]
        np.random.shuffle(cands)
        
        for x in cands:
            if in_set_copy[x]:
                continue
            # Check if x can be added to S_copy
            sums_add = np.concatenate(([2 * x], x + S_copy))
            if np.all(sc_copy[sums_add] < MAX_REP):
                # Add
                sc_copy[sums_add] += 1
                S_copy = np.append(S_copy, x)
                in_set_copy[x] = True
        
        if len(S_copy) > current_size:
            return S_copy, sc_copy, in_set_copy
        return None

    # Perform several drop-and-fill attempts
    # Pick y randomly from S
    for _ in range(20): # Max 20 attempts, or until time runs out
        if time.time() > deadline:
            break
        if len(S) == 0:
            break
        y = S[np.random.randint(len(S))]
        result = try_drop_and_fill(y)
        if result:
            S, sum_counts, in_set = result
            current_size = len(S)
            # Reset candidates? The in_set is updated.
            # The next loop iteration will use the new in_set.
            # We might want to break and restart the main add loop?
            # Let's just continue.
    
    # Final cleanup: ensure all possible adds are made (in case drop-and-fill left some space)
    in_set = np.zeros(N + 1, dtype=bool)
    in_set[S] = True
    candidates = np.where(~in_set)[0]
    candidates = candidates[candidates > 0]
    np.random.shuffle(candidates)
    
    for x in candidates:
        if time.time() > deadline:
            break
        if in_set[x]:
            continue
        if can_add(x):
            do_add(x)
            in_set[x] = True
            current_size += 1

    return S.tolist()

# EVOLVE-BLOCK-END
