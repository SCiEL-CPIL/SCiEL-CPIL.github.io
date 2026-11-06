import numpy as np
import time
import itertools
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # We want to maximize n. Try from high to low.
    # Published construction is n=34.
    # We will allocate time budgets to each n.
    
    best_n = 1
    best_colors = []
    
    # Try n in descending order
    # If we find a valid coloring for n, we can return immediately if n is high enough?
    # The task says "Maximize n". So we should try to get as high as possible.
    # If we find n=34, that's likely the max (or close to it).
    # We'll try 34, 33, 32, 31, 30, 29...
    
    # Time allocation:
    # n=34: 30s
    # n=33: 20s
    # n=32: 15s
    # n=31: 10s
    # n=30: 10s
    # n=29: 5s
    # n=28: 5s
    # Remaining for lower n if needed (unlikely to need if we find high n)
    
    n_list = [34, 33, 32, 31, 30, 29, 28, 27, 26, 25]
    time_budgets = [30, 20, 15, 10, 10, 5, 5, 5, 5, 5]
    
    for i, n in enumerate(n_list):
        if time.time() > deadline - 5.0:
            break
            
        budget = time_budgets[i]
        target_deadline = time.time() + budget
        
        # Precomputation
        triples = list(itertools.combinations(range(n), 3))
        M = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Build quad and pent lists
        quads = []
        triple_to_quads = [[] for _ in range(M)]
        for quad in itertools.combinations(range(n), 4):
            q_idx = len(quads)
            tris = [triple_to_idx[t] for t in itertools.combinations(quad, 3)]
            quads.append(tris)
            for t_idx in tris:
                triple_to_quads[t_idx].append(q_idx)
                
        pents = []
        triple_to_pents = [[] for _ in range(M)]
        for pent in itertools.combinations(range(n), 5):
            p_idx = len(pents)
            tris = [triple_to_idx[t] for t in itertools.combinations(pent, 3)]
            pents.append(tris)
            for t_idx in tris:
                triple_to_pents[t_idx].append(p_idx)
                
        Q = len(quads)
        P = len(pents)
        
        # Local search function
        def solve_n(n, deadline):
            # Greedy Initialization
            colors = np.zeros(M, dtype=np.int8)
            quad_red_counts = np.zeros(Q, dtype=np.int16)
            pent_blue_counts = np.zeros(P, dtype=np.int16)
            
            # Greedy pass
            order = np.random.permutation(M)
            for t_idx in order:
                # Cost if Red (0)
                # Check quads: if other 3 are red, this makes 4
                cost_red = 0
                for q_idx in triple_to_quads[t_idx]:
                    if quad_red_counts[q_idx] == 3:
                        cost_red += 1
                        
                # Cost if Blue (1)
                # Check pents: if other 9 are blue, this makes 10
                cost_blue = 0
                for p_idx in triple_to_pents[t_idx]:
                    if pent_blue_counts[p_idx] == 9:
                        cost_blue += 1
                        
                if cost_red < cost_blue:
                    # Color Red
                    colors[t_idx] = 0
                    for q_idx in triple_to_quads[t_idx]:
                        quad_red_counts[q_idx] += 1
                else:
                    # Color Blue
                    colors[t_idx] = 1
                    for p_idx in triple_to_pents[t_idx]:
                        pent_blue_counts[p_idx] += 1
                        
            # Calculate initial violations
            violations = np.sum(quad_red_counts == 4) + np.sum(pent_blue_counts == 10)
            
            if violations == 0:
                return colors.tolist()
                
            # Simulated Annealing
            T = 1.0
            T_min = 0.01
            cooling_rate = 0.999
            step = 0
            max_steps = 1000000 # Hard cap, but time is real constraint
            
            while time.time() < deadline - 1.0:
                if violations == 0:
                    return colors.tolist()
                    
                # Pick random triple
                t_idx = np.random.randint(M)
                old_val = colors[t_idx]
                new_val = 1 - old_val
                
                delta = 0
                
                if old_val == 0: # Red -> Blue
                    # Red counts decrease
                    for q_idx in triple_to_quads[t_idx]:
                        if quad_red_counts[q_idx] == 4:
                            delta -= 1
                    # Blue counts increase
                    for p_idx in triple_to_pents[t_idx]:
                        if pent_blue_counts[p_idx] == 9:
                            delta += 1
                else: # Blue -> Red
                    # Red counts increase
                    for q_idx in triple_to_quads[t_idx]:
                        if quad_red_counts[q_idx] == 3:
                            delta += 1
                    # Blue counts decrease
                    for p_idx in triple_to_pents[t_idx]:
                        if pent_blue_counts[p_idx] == 10:
                            delta -= 1
                            
                accept = False
                if delta <= 0:
                    accept = True
                else:
                    if T > T_min:
                        prob = math.exp(-delta / T)
                        if np.random.rand() < prob:
                            accept = True
                    # else reject
                
                if accept:
                    colors[t_idx] = new_val
                    if new_val == 0: # Now Red
                        for q_idx in triple_to_quads[t_idx]:
                            quad_red_counts[q_idx] += 1
                        for p_idx in triple_to_pents[t_idx]:
                            pent_blue_counts[p_idx] -= 1
                    else: # Now Blue
                        for q_idx in triple_to_quads[t_idx]:
                            quad_red_counts[q_idx] -= 1
                        for p_idx in triple_to_pents[t_idx]:
                            pent_blue_counts[p_idx] += 1
                            
                    violations += delta
                    T *= cooling_rate
                    if T < T_min:
                        T = T_min
                        # Optional: restart with new random init if stuck?
                        # For now, just keep going or break to outer restart loop
                        # Let's add a restart mechanism inside SA if stuck for too long
                        # Simple: if T hits min, do a few random flips to escape?
                        # Or just break and rely on outer restarts.
                        # Let's break to allow a new greedy init.
                        break
                step += 1
                
                # If we hit T_min, break to allow restart
                if T <= T_min:
                    break
                    
            return None

        # Multiple restarts for this n
        found = False
        restart_deadline = target_deadline - 2.0
        restart_count = 0
        max_restarts = 10 # Limit restarts to avoid infinite loop if precomp is slow
        
        while time.time() < restart_deadline and restart_count < max_restarts:
            restart_count += 1
            res = solve_n(n, restart_deadline)
            if res is not None:
                best_n = n
                best_colors = res
                found = True
                break
                
        if found:
            # If we found a solution for n, should we continue to n+1?
            # We are iterating downwards, so if we found n=34, we are done.
            # If we found n=33, we might try to see if 34 was possible with more time?
            # But we already spent time on 34.
            # The strategy is: try 34 first. If it fails, try 33.
            # If we find 33, we return 33.
            # This is correct for "Maximize n" if we assume finding 34 is hard and we didn't find it.
            # However, if we find 33 quickly, we might have time to try 34 again?
            # No, we tried 34 first. If it failed, we move on.
            # So if we find a solution, we can return immediately?
            # Yes, because we are trying from highest to lowest.
            # If we found n=34, it's the max possible (likely).
            # If we found n=33, we failed at 34. So 33 is the best we found.
            # We should return immediately to save time?
            # The evaluator wants the max n. If we return 33, score is 33/seed.
            # If we could find 34, score would be higher.
            # But we already tried 34.
            # So returning the first found (highest n) is optimal.
            return {'n': best_n, 'colors': best_colors}
            
    # Fallback if no solution found in the list
    # Return the best we can guarantee, e.g., n=9 from seed logic or similar
    # But we should have found something for n=25 at least.
    # If not, return a trivial valid coloring for small n.
    # Let's return n=1 with empty colors? No, n>=1.
    # The seed returned n=9.
    # Let's just return n=1, colors=[] as a last resort, but hopefully we find >20.
    
    # Actually, if we didn't find any, we should return the best n we *tried* and failed?
    # No, we must return a VALID coloring.
    # So we must find one.
    # The loop above tries n down to 25.
    # If it fails for all, we need a fallback.
    # Let's add a small n fallback.
    
    for n in range(20, 5, -1):
        if time.time() > deadline - 1.0:
            break
        # Quick random search for small n
        triples = list(itertools.combinations(range(n), 3))
        M = len(triples)
        for _ in range(1000):
            colors = np.random.randint(0, 2, M).astype(np.int8)
            # Check validity
            valid = True
            # Check red K4
            for quad in itertools.combinations(range(n), 4):
                tris = [triple_to_idx[t] for t in itertools.combinations(quad, 3)]
                if all(colors[t] == 0 for t in tris):
                    valid = False
                    break
            if not valid: continue
            # Check blue K5
            for pent in itertools.combinations(range(n), 5):
                tris = [triple_to_idx[t] for t in itertools.combinations(pent, 3)]
                if all(colors[t] == 1 for t in tris):
                    valid = False
                    break
            if valid:
                return {'n': n, 'colors': colors.tolist()}
                
    # Absolute fallback
    return {'n': 1, 'colors': []}

