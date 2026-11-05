import numpy as np
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s kill

    def triple_index(a, b, c):
        """Compute the index of triple (a,b,c) in lexicographic order for n vertices."""
        # We need to know n for the formula. This is a helper that assumes global n.
        # Actually, let's just precompute a mapping or use a formula that depends on n.
        # The index of (a,b,c) with 0<=a<b<c<n is:
        # sum_{x=0}^{a-1} C(n-1-x, 2) + sum_{y=a+1}^{b-1} (n-1-y) + (c-b-1)
        # This is O(1) if we have precomputed factorials or just use a loop for small n.
        # For n<=40, a simple loop is fast enough, but precomputation is better.
        
        # Let's stick to a precomputed dictionary for speed in the inner loop.
        pass

    def build_indices(n):
        """Precompute K4 and K5 index tuples."""
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        k4s = []
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            k4s.append((
                triple_to_idx[(a, b, c)],
                triple_to_idx[(a, b, d)],
                triple_to_idx[(a, c, d)],
                triple_to_idx[(b, c, d)]
            ))
        
        k5s = []
        for quint in combinations(range(n), 5):
            idxs = []
            for triple in combinations(quint, 3):
                idxs.append(triple_to_idx[triple])
            k5s.append(tuple(idxs))
            
        return num_triples, k4s, k5s

    def count_violations(colors, k4s, k5s):
        """Count red K4s and blue K5s."""
        red_k4 = 0
        for idxs in k4s:
            if colors[idxs[0]] == 0 and colors[idxs[1]] == 0 and colors[idxs[2]] == 0 and colors[idxs[3]] == 0:
                red_k4 += 1
        blue_k5 = 0
        for idxs in k5s:
            # Check if all 10 are 1
            # Using numpy sum is faster
            if np.sum(colors[list(idxs)]) == 10:
                blue_k5 += 1
        return red_k4, blue_k5

    def local_search(n, time_limit):
        """
        Perform local search to find a valid coloring for n vertices.
        Returns (colors, is_valid).
        """
        num_triples, k4s, k5s = build_indices(n)
        
        # Precompute reverse indices for fast delta calculation
        # triple_idx -> list of (clique_idx, position_in_clique)
        # Actually, just list of clique indices is enough if we re-check the whole clique.
        # But re-checking 4 or 10 elements is cheap.
        
        triple_to_k4s = [[] for _ in range(num_triples)]
        triple_to_k5s = [[] for _ in range(num_triples)]
        
        for i, idxs in enumerate(k4s):
            for t in idxs:
                triple_to_k4s[t].append(i)
                
        for i, idxs in enumerate(k5s):
            for t in idxs:
                triple_to_k5s[t].append(i)

        # Initialize with a random coloring or a heuristic one
        # Heuristic: Random is often better for hypergraph Ramsey than simple modular patterns
        # which might have structure that causes violations.
        # Let's try a few restarts.
        
        best_colors = None
        best_cost = float('inf')
        
        # We will run simulated annealing or hill climbing.
        # Given the time constraint, hill climbing with random restarts is robust.
        
        max_restarts = 50
        steps_per_restart = 20000 # Adjust based on n and time
        
        # Estimate time per step to set steps_per_restart dynamically
        # For n=34, num_triples = 5984, k4s=46376, k5s=278256.
        # Delta calc involves iterating over affected cliques.
        # Avg affected K4s per triple: C(n-3, 1) = n-3. For n=34, 31.
        # Avg affected K5s per triple: C(n-3, 2) = 465.
        # So ~500 clique checks per step. Each check is ~10 ops. ~5000 ops.
        # 20000 steps ~ 100M ops. Should be < 1s in Python? Maybe 2-3s.
        # Let's be conservative.
        
        while time.time() < time_limit:
            # New restart
            colors = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
            
            # Initial cost
            red_k4, blue_k5 = count_violations(colors, k4s, k5s)
            current_cost = red_k4 + blue_k5
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors.copy()
                if best_cost == 0:
                    return best_colors, True
            
            # Hill Climbing
            # We accept moves that decrease cost.
            # We also allow some random moves to escape local minima (simulated annealing light)
            temp = 1.0
            cool_rate = 0.999
            
            # Determine max steps for this restart based on remaining time
            # Leave 10% of remaining time for next restarts
            remaining = time_limit - time.time()
            if remaining <= 0:
                break
            
            # Estimate steps per second roughly. Let's just run until time slice ends.
            # Time slice: 80% of remaining time, min 0.1s
            step_deadline = time.time() + max(0.1, remaining * 0.8)
            
            while time.time() < step_deadline:
                # Pick random triple
                t = np.random.randint(num_triples)
                
                old_color = colors[t]
                new_color = 1 - old_color
                
                delta_cost = 0
                
                # Update K4s
                for k4_idx in triple_to_k4s[t]:
                    idxs = k4s[k4_idx]
                    # Check state before
                    # Since only t changes, we can check the other 3 and infer
                    # But checking all 4 is simple and fast enough
                    is_red_before = (colors[idxs[0]] == 0 and colors[idxs[1]] == 0 and colors[idxs[2]] == 0 and colors[idxs[3]] == 0)
                    
                    # Determine if red after
                    # If t is in idxs, replace its color
                    # We know t is in idxs.
                    # Let's just build a tuple of colors for the clique with new color for t
                    # This is slightly slower but safer.
                    # Optimization: count zeros in the clique.
                    # If count_zeros == 4, it's a violation.
                    # If we flip 0->1, count_zeros decreases by 1.
                    # If we flip 1->0, count_zeros increases by 1.
                    
                    # Let's use the count approach for speed.
                    # Precompute count of 0s in each K4 and K5?
                    # That's complex to update.
                    
                    # Let's stick to the boolean check but optimize it.
                    # is_red_after:
                    # If old_color was 0, new is 1. If before was red (all 0), after is not red.
                    # If before was not red, after cannot be red (since we turned a 0 to 1).
                    # If old_color was 1, new is 0.
                    # If before was red, after is red.
                    # If before was not red, it could become red if the other 3 were 0.
                    
                    if old_color == 0:
                        # Flipping 0 -> 1
                        if is_red_before:
                            delta_cost -= 1
                        # Cannot create a new red K4
                    else:
                        # Flipping 1 -> 0
                        if is_red_before:
                            # Still red, no change
                            pass
                        else:
                            # Check if other 3 are 0
                            # Find the index of t in idxs
                            # This is slow.
                            # Alternative:
                            # Check if sum of colors in idxs (excluding t) is 0.
                            # We can precompute the sum of colors for each clique?
                            # Let's try a different approach:
                            # Just check the 4 values directly.
                            c0 = colors[idxs[0]] if idxs[0] != t else 0
                            c1 = colors[idxs[1]] if idxs[1] != t else 0
                            c2 = colors[idxs[2]] if idxs[2] != t else 0
                            c3 = colors[idxs[3]] if idxs[3] != t else 0
                            if c0 == 0 and c1 == 0 and c2 == 0 and c3 == 0:
                                delta_cost += 1
                
                # Update K5s
                for k5_idx in triple_to_k5s[t]:
                    idxs = k5s[k5_idx]
                    # Similar logic
                    # is_blue_before: all 10 are 1
                    # Check if sum is 10.
                    # This is expensive to compute every time.
                    
                    # Let's optimize:
                    # If old_color == 1 (flipping to 0):
                    #   If before was blue (all 1), after is not blue. delta -= 1
                    #   Else no change.
                    # If old_color == 0 (flipping to 1):
                    #   If before was blue, after is blue. No change.
                    #   Else, check if other 9 are 1. If so, delta += 1.
                    
                    # To check "other 9 are 1" quickly, we can maintain a count of 1s?
                    # Or just check the 10 values.
                    
                    if old_color == 1:
                        # Flipping 1 -> 0
                        # Check if it was a blue K5
                        # Sum of colors in idxs
                        # We can use np.sum on the slice? No, idxs is a list of indices.
                        # colors[list(idxs)] creates a copy.
                        # Let's do a manual loop or precomputed sum.
                        
                        # For now, manual check
                        is_blue_before = True
                        for i in idxs:
                            if colors[i] != 1:
                                is_blue_before = False
                                break
                        if is_blue_before:
                            delta_cost -= 1
                    else:
                        # Flipping 0 -> 1
                        # Check if it becomes a blue K5
                        # This requires the other 9 to be 1.
                        is_blue_after = True
                        for i in idxs:
                            if i == t:
                                continue
                            if colors[i] != 1:
                                is_blue_after = False
                                break
                        if is_blue_after:
                            delta_cost += 1
                
                new_cost = current_cost + delta_cost
                
                # Acceptance criterion
                if new_cost < current_cost:
                    colors[t] = new_color
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_colors = colors.copy()
                        if best_cost == 0:
                            return best_colors, True
                elif new_cost == current_cost and np.random.rand() < 0.5:
                    colors[t] = new_color
                elif new_cost > current_cost and np.random.rand() < np.exp(-new_cost / temp):
                    colors[t] = new_color
                    current_cost = new_cost
                
                temp *= cool_rate
                if temp < 0.1:
                    temp = 1.0

        return best_colors, (best_cost == 0)

    # Main loop: try to find the largest n
    # Start from a known good value and go up? Or start high and go down?
    # The task says maximize n.
    # Published construction is n=34.
    # Let's try n=34, 33, 32...
    # If we find a valid one, return it.
    # If not, try lower n.
    
    # However, finding n=34 is hard.
    # Maybe we should try n=20, 30, 34 in increasing order of difficulty?
    # No, higher n is harder.
    # Let's try n=34 first with a generous time budget.
    # If it fails, try n=33, etc.
    
    # Time budget allocation:
    # Try n=34 for 60s.
    # If fail, try n=33 for 30s.
    # If fail, try n=32 for 20s.
    # If fail, try n=30 for 10s.
    # If fail, try n=20 for 5s.
    # If fail, return n=9 (seed) or best found.
    
    # Actually, the seed is n=9.
    # We should ensure we always return at least n=9.
    
    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]
    
    # Try to improve
    targets = [34, 33, 32, 31, 30, 25, 20]
    
    # Allocate time
    # Total available: deadline - start_time
    # Let's use a fraction of the time for each target.
    
    time_alloc = {
        34: 50.0,
        33: 30.0,
        32: 20.0,
        31: 15.0,
        30: 10.0,
        25: 5.0,
        20: 2.0
    }
    
    for n in targets:
        if time.time() > deadline - 5:
            break
            
        t_limit = time.time() + time_alloc.get(n, 5.0)
        
        try:
            colors, is_valid = local_search(n, t_limit)
            if is_valid:
                best_n = n
                best_colors = colors.tolist()
                # If we found a valid coloring for n, we can stop?
                # The task says "Maximize n".
                # If we found n=34, that's likely the max or close to it.
                # But maybe n=35 is possible?
                # The contract says 1 <= n <= 40.
                # Published is 34.
                # Let's assume 34 is the target. If we find it, we can return it.
                # But to be safe, if we find n, we should check if we can go higher?
                # Given time constraints, finding 34 is a big win.
                # Let's break if we find n >= 34.
                if n >= 34:
                    break
        except Exception as e:
            pass

    return {'n': best_n, 'colors': best_colors}
