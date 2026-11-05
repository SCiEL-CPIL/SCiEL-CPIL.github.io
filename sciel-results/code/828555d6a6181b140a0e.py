import time
from itertools import combinations
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # We target n=34
    target_ns = [34, 33, 32, 31, 30]
    
    for n in target_ns:
        if time.time() > deadline - 10:
            break
            
        try:
            result = solve_csp(n, time_limit=deadline - 5.0)
            if result is not None:
                return result
        except Exception as e:
            # If error, try next n
            pass
            
    # Fallback: return a small valid coloring if nothing else works
    # For n=9, a simple coloring works?
    # Let's just return a trivial valid one for small n if all else fails
    n_fallback = 9
    triples = list(combinations(range(n_fallback), 3))
    colors = [0 if (a+b+c)%3==0 else 1 for a,b,c in triples]
    return {'n': n_fallback, 'colors': colors}

def solve_csp(n, time_limit):
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    # Build K4s and K5s
    k4s = []
    k4_containing = [[] for _ in range(num_triples)]
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        t1 = triple_to_idx[(a,b,c)]
        t2 = triple_to_idx[(a,b,d)]
        t3 = triple_to_idx[(a,c,d)]
        t4 = triple_to_idx[(b,c,d)]
        idx_tuple = (t1, t2, t3, t4)
        k4s.append(idx_tuple)
        for t in idx_tuple:
            k4_containing[t].append(len(k4s) - 1)
            
    k5s = []
    k5_containing = [[] for _ in range(num_triples)]
    for quint in combinations(range(n), 5):
        idxs = []
        for triple in combinations(quint, 3):
            idxs.append(triple_to_idx[triple])
        idx_tuple = tuple(idxs)
        k5s.append(idx_tuple)
        for t in idx_tuple:
            k5_containing[t].append(len(k5s) - 1)
            
    num_k4s = len(k4s)
    num_k5s = len(k5s)
    
    # State
    colors = np.full(num_triples, -1, dtype=np.int8)
    
    # Counts
    red_count_k4 = np.zeros(num_k4s, dtype=np.int32)
    blue_count_k5 = np.zeros(num_k5s, dtype=np.int32)
    
    # Unassigned lists (we can use sets or just check colors)
    # For efficiency, we'll maintain a list of unassigned triples
    unassigned = set(range(num_triples))
    
    # Propagation queue
    # We store (triple_idx, color) to be assigned
    # If we detect a conflict, we return False
    
    def propagate(queue):
        """
        Process the queue of forced assignments.
        Returns False if a conflict is found.
        Returns True if propagation completes without conflict.
        """
        while queue:
            t, c = queue.pop()
            
            if colors[t] != -1:
                if colors[t] != c:
                    return False # Conflict
                else:
                    continue # Already assigned correctly
            
            # Assign t to c
            colors[t] = c
            unassigned.remove(t)
            
            # Update K4s
            for k4_idx in k4_containing[t]:
                if c == 0: # Red
                    red_count_k4[k4_idx] += 1
                    if red_count_k4[k4_idx] == 4:
                        return False # Red K4 found
                    if red_count_k4[k4_idx] == 3:
                        # Find the unassigned triple in this K4
                        k4_triples = k4s[k4_idx]
                        for tt in k4_triples:
                            if colors[tt] == -1:
                                queue.append((tt, 1)) # Must be Blue
                                break
            
            # Update K5s
            for k5_idx in k5_containing[t]:
                if c == 1: # Blue
                    blue_count_k5[k5_idx] += 1
                    if blue_count_k5[k5_idx] == 10:
                        return False # Blue K5 found
                    if blue_count_k5[k5_idx] == 9:
                        # Find the unassigned triple in this K5
                        k5_triples = k5s[k5_idx]
                        for tt in k5_triples:
                            if colors[tt] == -1:
                                queue.append((tt, 0)) # Must be Red
                                break
                            
        return True

    # Recursive Backtracking
    # To avoid deep recursion stack overflow and allow time checks, we use an iterative stack
    # Stack item: (unassigned_copy, colors_copy, red_count_copy, blue_count_copy) is too expensive to copy.
    # Instead, we record the changes made and undo them (trail).
    
    # Trail entry: (triple_idx, old_color, k4_idx, old_red_count, k5_idx, old_blue_count)
    # Actually, simpler: just copy the arrays. n=34, num_triples=5984.
    # Copying 5984 ints is fast. num_k4s=46376, num_k5s=237336.
    # Copying k5 counts might be slow.
    # Let's estimate: 237k ints * 4 bytes = ~1MB. Copying 1MB is ~0.1ms.
    # If we branch 1000 times, 100ms. Acceptable.
    
    def backtrack():
        if not unassigned:
            return True
            
        # Heuristic: Pick a variable
        # Prefer variables involved in "tight" constraints
        # Score a variable by sum of "urgency" of its constraints
        # Urgency for K4: (4 - red_count) if unassigned count > 0
        # Actually, just pick the first one for simplicity, or random.
        # Random is better to avoid bad ordering.
        t = list(unassigned)[0]
        # Better heuristic: pick t with highest degree in "critical" constraints
        # For now, let's use a simple random choice from unassigned to explore
        # Or, pick the one that appears in the most K4s with red_count==2 or K5s with blue_count==8
        # Let's try to compute a score quickly.
        
        # To save time, we'll just pick a random unassigned variable.
        # This is a standard DPLL approach.
        candidates = list(unassigned)
        t = candidates[np.random.randint(len(candidates))]
        
        # Try 0 then 1
        for c in [0, 1]:
            # Save state
            colors_save = colors.copy()
            red_count_k4_save = red_count_k4.copy()
            blue_count_k5_save = blue_count_k5.copy()
            unassigned_save = unassigned.copy()
            
            # Assign and propagate
            queue = [(t, c)]
            if propagate(queue):
                if backtrack():
                    return True
            
            # Restore state
            colors[:] = colors_save
            red_count_k4[:] = red_count_k4_save
            blue_count_k5[:] = blue_count_k5_save
            unassigned.clear()
            unassigned.update(unassigned_save)
            
        return False

    # Run with restarts
    # Since backtracking can be exponential, we use time-limited search with restarts.
    # We restart with a different random seed/order if stuck.
    
    # Actually, the above `backtrack` is a pure DFS. It might be too slow if the first branch is bad.
    # We need a timeout mechanism inside the recursion.
    # Python recursion limit is ~1000. Depth is 5984. We will hit recursion limit.
    # We MUST use iterative DFS or increase recursion limit.
    # sys.setrecursionlimit(10000) is risky but doable.
    # However, the time check in recursion is hard.
    
    # Let's switch to a Simulated Annealing approach but with the propagation logic used to fix local minima.
    # Or, use the backtracker but with a depth limit and random restarts.
    
    # Let's try a hybrid:
    # 1. Random initialization.
    # 2. Local Search (SA) to minimize violations.
    # 3. If violations == 0, return.
    # 4. If not, try to fix remaining violations using backtracking on the "bad" parts?
    
    # Actually, the propagation-based backtracker is powerful. Let's fix the recursion issue.
    # We can implement the backtracker iteratively.
    
    def iterative_backtrack():
        # Stack of states
        # State: (current_unassigned_set, colors, red_counts, blue_counts, next_color_to_try_for_current_var)
        # This is memory heavy.
        
        # Alternative: Use the recursive version but with sys.setrecursionlimit(10000)
        # and a global time check.
        
        global start_time
        sys.setrecursionlimit(10000)
        
        # We need to pass time_limit to the recursive function
        # We'll use a closure or global variable for the deadline.
        
        def rec():
            if time.time() > time_limit:
                return False # Timeout
            
            if not unassigned:
                return True
            
            # Heuristic: Pick variable
            # Let's pick the variable that is in the most "tight" K4s or K5s
            # Tight K4: red_count == 2 (2 unassigned, 1 red makes 3, 2 reds makes 4->conflict? No, 3 reds forces 4th blue)
            # If red_count==2, and we assign one of the unassigned to Red, red_count becomes 3, forcing the last one Blue.
            # This is good for propagation.
            
            # Calculate score for each unassigned variable
            # Score = sum of (1 if involved in tight K4/K5)
            # This is O(N * Degree). Degree is small.
            
            best_t = None
            best_score = -1
            # Sample a subset of unassigned to find a good variable to save time
            # Or just iterate all if unassigned is small.
            # At start, unassigned is 5984.
            # Let's just pick a random one to keep it simple and fast per step.
            # Random restarts will handle the ordering.
            t = list(unassigned)[np.random.randint(len(unassigned))]
            
            # Try 0 and 1
            for c in [0, 1]:
                colors_save = colors.copy()
                red_count_k4_save = red_count_k4.copy()
                blue_count_k5_save = blue_count_k5.copy()
                unassigned_save = unassigned.copy()
                
                queue = [(t, c)]
                if propagate(queue):
                    if rec():
                        return True
                
                colors[:] = colors_save
                red_count_k4[:] = red_count_k4_save
                blue_count_k5[:] = blue_count_k5_save
                unassigned.clear()
                unassigned.update(unassigned_save)
                
            return False
        
        return rec()

    # Let's try running iterative_backtrack with a time limit.
    # If it times out, we restart with a new random seed (which changes the variable order).
    
    # Actually, the copying of large arrays (k5 counts) at every step is expensive.
    # num_k5s for n=34 is 237336.
    # Copying 237k ints is ~1ms.
    # If we do 1000 steps, 1s.
    # If we do 10000 steps, 10s.
    # This might be too slow for deep searches.
    
    # Optimization: Only copy the counts that change?
    # Or, use a "trail" of changes.
    # When we assign t to c:
    #   For each K4 k containing t:
    #      old_red = red_count_k4[k]
    #      new_red = old_red + (1 if c==0 else 0)
    #      red_count_k4[k] = new_red
    #      Trail.append((k, old_red))
    #   For each K5 k containing t:
    #      old_blue = blue_count_k5[k]
    #      new_blue = old_blue + (1 if c==1 else 0)
    #      blue_count_k5[k] = new_blue
    #      Trail.append((k, old_blue))
    #   colors[t] = c
    #   Trail.append((t, old_color))
    
    # This is much faster.
    
    # Let's rewrite with Trail.
    
    def solve_with_trail():
        colors[:] = -1
        red_count_k4[:] = 0
        blue_count_k5[:] = 0
        unassigned = set(range(num_triples))
        
        trail = []
        
        def undo(trail_snapshot_len):
            while len(trail) > trail_snapshot_len:
                item = trail.pop()
                if isinstance(item, tuple) and len(item) == 3 and item[0] == 'color':
                    t, old_c, _ = item
                    colors[t] = old_c
                    unassigned.add(t)
                elif isinstance(item, tuple) and len(item) == 3 and item[0] == 'red':
                    k, old_c, _ = item
                    red_count_k4[k] = old_c
                elif isinstance(item, tuple) and len(item) == 3 and item[0] == 'blue':
                    k, old_c, _ = item
                    blue_count_k5[k] = old_c
        
        def propagate(queue):
            while queue:
                t, c = queue.pop()
                
                if colors[t] != -1:
                    if colors[t] != c:
                        return False
                    else:
                        continue
                
                # Record old state
                old_color = colors[t]
                trail.append(('color', t, old_color))
                
                colors[t] = c
                unassigned.remove(t)
                
                # Update K4s
                for k4_idx in k4_containing[t]:
                    old_red = red_count_k4[k4_idx]
                    new_red = old_red + (1 if c == 0 else 0)
                    trail.append(('red', k4_idx, old_red))
                    red_count_k4[k4_idx] = new_red
                    
                    if new_red == 4:
                        return False
                    if new_red == 3:
                        # Find unassigned in this K4
                        for tt in k4s[k4_idx]:
                            if colors[tt] == -1:
                                queue.append((tt, 1))
                                break
                
                # Update K5s
                for k5_idx in k5_containing[t]:
                    old_blue = blue_count_k5[k5_idx]
                    new_blue = old_blue + (1 if c == 1 else 0)
                    trail.append(('blue', k5_idx, old_blue))
                    blue_count_k5[k5_idx] = new_blue
                    
                    if new_blue == 10:
                        return False
                    if new_blue == 9:
                        # Find unassigned in this K5
                        for tt in k5s[k5_idx]:
                            if colors[tt] == -1:
                                queue.append((tt, 0))
                                break
                                
            return True

        def rec():
            if time.time() > time_limit:
                return False
            
            if not unassigned:
                return True
            
            # Pick variable
            # Heuristic: Pick one involved in many tight constraints
            # For speed, we pick a random one.
            t = list(unassigned)[np.random.randint(len(unassigned))]
            
            for c in [0, 1]:
                trail_len = len(trail)
                queue = [(t, c)]
                if propagate(queue):
                    if rec():
                        return True
                undo(trail_len)
            
            return False
            
        return rec()

    # Run with restarts
    # If it fails (returns False), it might be due to timeout or dead end.
    # If dead end, the recursion unwinds.
    # If timeout, it returns False immediately.
    
    # We'll run it once. If it finds a solution, great.
    # If not, we might need to try again with different random choices.
    
    # To make it robust, we run it in a loop with a small time budget per attempt.
    
    attempt_deadline = time.time() + 2.0
    while time.time() < time_limit:
        # Reset state
        colors[:] = -1
        red_count_k4[:] = 0
        blue_count_k5[:] = 0
        unassigned = set(range(num_triples))
        trail = []
        
        # We need to pass the time_limit to the recursive function.
        # The closure `rec` uses `time_limit` from the outer scope.
        
        if solve_with_trail():
            # Success
            return {'n': n, 'colors': colors.tolist()}
        
        # If we get here, it's either timeout or failure.
        # If timeout, we restart.
        # If failure (dead end), we also restart (with new random seed).
        
        # To avoid infinite loop on immediate failure, we add a small sleep or just let it run.
        # The random seed changes naturally.
        
        # Check if we hit the global time limit
        if time.time() > time_limit:
            break

    return None
