import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 150s limit
    
    # 1. Generate an initial greedy set
    def greedy_initial(seed):
        random.seed(seed)
        order = list(range(1, 1001))
        random.shuffle(order)
        chosen = []
        counts = {}
        for x in order:
            ok = True
            for y in chosen:
                s = x + y
                if counts.get(s, 0) >= 2:
                    ok = False
                    break
            if ok:
                s2 = 2 * x
                if counts.get(s2, 0) >= 2:
                    ok = False
            if ok:
                chosen.append(x)
                for y in chosen[:-1]:
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
                counts[2 * x] = counts.get(2 * x, 0) + 1
        return chosen

    # Start with a deterministic greedy to get a baseline, then randomize
    best_set = greedy_initial(42)
    best_size = len(best_set)
    
    # 2. Local Search: Hill Climbing with Swap/Insert/Remove moves
    # We will try to improve the set by:
    # a) Swapping an element in the set with an element outside
    # b) Inserting a new element (if space allows)
    # c) Removing an element to allow a better insertion later (less direct, but swap covers this)
    
    # To make local search efficient, we need to quickly evaluate if a set is valid.
    # Since we are modifying the set, we can maintain the counts.
    # However, swap moves are complex to update counts for.
    # Alternative: Since N=1000 is small, we can re-verify validity of the *entire* set
    # after a move? No, O(N^2) per move is too slow if we do many moves.
    
    # Better approach for Local Search:
    # Maintain the set S and the counts of sums.
    # Move: Swap x in S with y not in S.
    # 1. Remove x from S: decrement counts for x+z (z in S\{x}) and 2x.
    # 2. Add y to S: check if counts[y+z] < 2 for all z in S (current S without x) and 2y < 2.
    #    If valid, increment counts.
    # 3. If we want to actually increase size, we just insert.
    #    But local search usually keeps size constant to find a "better" configuration
    #    that might allow insertion later?
    #    Actually, the goal is to MAXIMIZE size.
    #    So the move should be:
    #    - Try to insert a new element. If fails, try a swap that "frees up" sum capacity?
    #    - Or simpler: Randomly pick a candidate y not in S. Try to insert.
    #    - If insert fails, find a conflict. Remove one of the conflicting elements?
    
    # Let's use a simpler "Random Restart + Local Improvement" strategy.
    # 1. Greedy seed.
    # 2. Repeat:
    #    - Pick a random element y not in S.
    #    - Try to add y.
    #    - If it fails, identify which sums would exceed limit 2.
    #    - Those sums involve pairs (a, b) in S such that a+b = y+c or similar.
    #    - This is complex.
    
    # Alternative Local Search: "Swap to improve density"
    # Actually, a known effective heuristic for B-hsets is:
    # 1. Greedy construction.
    # 2. Local Search:
    #    - Randomly select x in S and y not in S.
    #    - Check if S - {x} U {y} is a valid B2[2] set.
    #    - If yes, and if we can then insert another element, or if it just maintains size?
    #    - Maintaining size doesn't help maximize size directly unless it unlocks further insertions.
    
    # Let's try a different structural change: **Simulated Annealing** or **Genetic Algorithm**?
    # Too complex for 120s.
    
    # Let's stick to **Greedy + Local Search for Insertion**.
    # Idea:
    # 1. Start with greedy set S.
    # 2. While time allows:
    #    a. Identify "blocked" elements y not in S.
    #    b. For a blocked y, the reason it's blocked is that some sum s has count 2, and y+z would create a 3rd representation.
    #    c. If we remove an element x from S that is part of one of those "critical" pairs, maybe y can be added?
    #    d. But removing x reduces size by 1. Adding y increases by 1. Net 0.
    #    e. We need to find a sequence of moves that increases size.
    
    # A more powerful move: **2-Swap** or **Chain Swap**?
    
    # Let's look at the "Local search with swap moves" inspiration.
    # "Local search with swap moves allows escaping local minima... Measured: 1 -> 1.021."
    # This implies that even with constant size swaps, we eventually find a configuration where an insertion is possible, OR the swap itself is not the goal but the *path* to a larger set.
    
    # Actually, if we just do swaps, size stays constant.
    # How does size increase?
    # Perhaps the local search is:
    # 1. Try to insert.
    # 2. If fail, perform a random swap to change the "landscape" of conflicts.
    # 3. Try to insert again.
    # 4. If insert succeeds, size increases.
    
    # Let's implement this:
    
    def is_valid_insertion(S, y, counts):
        """Check if y can be added to S given current counts."""
        for x in S:
            s = x + y
            if counts.get(s, 0) >= 2:
                return False
        if counts.get(2 * y, 0) >= 2:
            return False
        return True

    def apply_swap(S, x_remove, y_add, counts):
        """
        Swap x_remove (in S) with y_add (not in S).
        Returns True if the new set is valid, and updates S and counts in place if so.
        """
        # 1. Temporarily remove x_remove
        # We need to decrement counts for all pairs involving x_remove.
        # Pairs: (x_remove, z) for z in S, z != x_remove. And (x_remove, x_remove).
        # Note: In our counts, we store counts for a+b with a<=b.
        # If we remove x, we subtract 1 from count[x+z] for all z in S\{x} with z >= x?
        # Wait, the counts dictionary is global.
        # If we store counts for all sums a+b (a,b in S), we need to be careful about order.
        # My greedy implementation used:
        # for y in chosen[:-1]: counts[x+y] += 1
        # counts[2*x] += 1
        # This implies counts[s] is the number of pairs (a,b) in S such that a+b=s and a<=b.
        
        # To remove x:
        # For each z in S such that z != x:
        #   s = x + z
        #   counts[s] -= 1
        # counts[2*x] -= 1
        
        # To add y:
        # Check validity first.
        # For each z in S_new (which is S - {x}):
        #   s = y + z
        #   if counts[s] >= 2: invalid
        # if counts[2*y] >= 2: invalid
        
        # If valid:
        # For each z in S_new:
        #   s = y + z
        #   counts[s] += 1
        # counts[2*y] += 1
        
        # Implementation:
        S_set = set(S)
        
        # Check if y can be added to S \ {x}
        # We need to simulate the state without x.
        # Let's just calculate the "virtual" counts for S\{x}.
        # This is O(N) per check.
        
        # Actually, let's just do the swap and rollback if invalid.
        # But rollback is tricky.
        
        # Let's write a helper that checks validity of S' = S - {x} U {y} without modifying global state.
        # We can compute the "delta" counts.
        
        # Simpler:
        # 1. Remove x from counts.
        # 2. Check if y is valid.
        # 3. If valid, add y to counts and S.
        # 4. If invalid, re-add x to counts and S.
        
        # Remove x
        for z in S:
            if z == x_remove:
                continue
            s = x_remove + z
            counts[s] -= 1
        counts[2 * x_remove] -= 1
        
        # Check y
        valid = True
        for z in S:
            if z == x_remove:
                continue
            s = y_add + z
            if counts.get(s, 0) >= 2:
                valid = False
                break
        if valid:
            if counts.get(2 * y_add, 0) >= 2:
                valid = False
        
        if valid:
            # Commit swap
            S.remove(x_remove)
            S.append(y_add)
            for z in S:
                if z == y_add:
                    continue
                s = y_add + z
                counts[s] = counts.get(s, 0) + 1
            counts[2 * y_add] = counts.get(2 * y_add, 0) + 1
            return True
        else:
            # Rollback x
            for z in S:
                if z == x_remove:
                    continue
                s = x_remove + z
                counts[s] = counts.get(s, 0) + 1
            counts[2 * x_remove] = counts.get(2 * x_remove, 0) + 1
            return False

    # Main Loop:
    # We will alternate between:
    # 1. Trying to insert a random element from outside.
    # 2. If insertion fails, perform a random swap to perturb the set.
    
    current_set = list(best_set)
    current_counts = {}
    # Rebuild counts for current_set to be safe
    for i in range(len(current_set)):
        x = current_set[i]
        for j in range(i):
            y = current_set[j]
            s = x + y
            current_counts[s] = current_counts.get(s, 0) + 1
        current_counts[2 * x] = current_counts.get(2 * x, 0) + 1
        
    outside_elements = list(set(range(1, 1001)) - set(current_set))
    
    improved = True
    while time.time() < deadline and improved:
        improved = False
        # Shuffle outside elements to try different ones
        random.shuffle(outside_elements)
        
        # Try to insert
        for y in outside_elements:
            if time.time() > deadline:
                break
            if is_valid_insertion(current_set, y, current_counts):
                # Insert y
                current_set.append(y)
                for x in current_set[:-1]:
                    s = x + y
                    current_counts[s] = current_counts.get(s, 0) + 1
                current_counts[2 * y] = current_counts.get(2 * y, 0) + 1
                outside_elements.remove(y)
                improved = True
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = list(current_set)
                # After insertion, the landscape changes. We should restart the insertion loop
                # or continue? Continuing might find more immediate insertions.
                # But the "improved" flag breaks the outer while loop?
                # No, we want to keep searching.
                # Let's break out of the for loop to restart the shuffle/iteration?
                # Or just continue?
                # If we continue, we might find more.
                pass
        
        if improved:
            continue
            
        # If no insertion was possible, try swaps to perturb
        # Perform a few random swaps
        swaps_attempted = 0
        swaps_successful = 0
        max_swaps = 100
        while swaps_attempted < max_swaps and time.time() < deadline:
            # Pick random x in set, y outside
            x_idx = random.randint(0, len(current_set) - 1)
            x = current_set[x_idx]
            
            if not outside_elements:
                break
                
            y = random.choice(outside_elements)
            
            if apply_swap(current_set, x, y, current_counts):
                # Update outside_elements list
                outside_elements.remove(y)
                outside_elements.append(x)
                swaps_successful += 1
                # A successful swap changes the set. It might now allow an insertion.
                # We don't know if it's "better" in terms of size, but it breaks symmetry.
                # We should try to insert again immediately.
                # Let's break and go to the insertion phase.
                improved = True # Force restart of insertion phase
                break
            else:
                swaps_attempted += 1
                
        # If we did swaps, the 'improved' flag is True, so we loop back to insertion.
        # If no swaps were successful, we are stuck.
        # We can try to restart with a new greedy seed?
        # Or accept stagnation.

    return best_set
