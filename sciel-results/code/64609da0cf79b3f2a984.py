import time
import random

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin for 120s limit
    
    best_set = None
    best_score = -1
    
    N = 32
    MAX_VAL = 512
    
    def score_set(S):
        """Calculate the number of distinct d > 0 for which a, a+d, a+2d in S."""
        diffs = set()
        # N is 32, so N^2 is 1024. Very fast.
        # We can optimize by iterating over sorted elements or just pairs.
        # Using set lookup is O(1) average.
        for a in S:
            for b in S:
                if a >= b:
                    continue
                d = b - a
                c = b + d
                if c in S:
                    diffs.add(d)
        return len(diffs)
    
    def greedy_polish(S):
        """Try to improve S by swapping elements."""
        S = set(S)
        improved = True
        while improved:
            improved = False
            current_score = score_set(S)
            # Try swapping each element in S with an element not in S
            # To avoid O(N^2 * MAX_VAL) which is 32*32*512 ~ 500k per pass, 
            # we can be smarter or just limit candidates.
            # Given N=32, even checking all 512 for each of 32 elements is 32*512=16k checks.
            # Each check involves score calculation? No, we can incremental or just re-calc.
            # Re-calc is ~1000 ops. 16k * 1000 = 16M ops. Might be slow in Python if done many times.
            # Better: Only try a few random candidates or neighbors.
            
            # Let's try a simpler greedy:
            # For each x in S, try removing it and adding y that maximizes score.
            # This is expensive.
            
            # Alternative Greedy:
            # Try adding a new element z not in S, removing x from S, if score increases.
            # To speed up, we only check a subset of candidates for z.
            
            # Actually, with N=32, a full scan of swaps might be feasible if optimized.
            # Let's try a limited greedy:
            # 1. Calculate score.
            # 2. For each x in S:
            #    Temporarily remove x.
            #    Try adding a few random y not in S (and not x).
            #    If score > current, keep change.
            
            # Given the time budget, a simple hill-climbing with random restarts inside the polish might be better.
            # But the prompt asks for a structural change. SA is the main engine.
            # Let's make the polish a simple local search:
            # Try all swaps (x in S, y not in S) but only evaluate score if potential is high?
            # No, just limit the number of swaps tried per pass.
            
            # Let's stick to a robust SA that does the heavy lifting, and a light polish.
            pass
        return S

    def run_sa(initial_set, time_limit):
        """Run Simulated Annealing on a copy of initial_set."""
        S = set(initial_set)
        current_score = score_set(S)
        best_local_set = set(S)
        best_local_score = current_score
        
        T = 50.0  # Initial temperature
        cooling = 0.999
        min_T = 0.01
        
        while T > min_T:
            if time.time() - start_time > time_limit:
                break
            
            # Generate a neighbor: swap a random element in S with a random element not in S
            x = random.choice(list(S))
            # Pick y from 1..512 not in S
            # To avoid infinite loop picking from S, we can pick from range and check
            # Or maintain a list of available elements? 
            # Simpler: pick random y, if y in S, retry (max few times) or just pick from 1..512 and if in S, skip.
            # Since |S|=32 and range=512, prob of collision is ~6%.
            y = random.randint(1, MAX_VAL)
            if y in S:
                continue
            
            # Perform swap
            S.remove(x)
            S.add(y)
            new_score = score_set(S)
            
            delta = new_score - current_score
            if delta > 0 or random.random() < pow(2.71828, delta / T):
                current_score = new_score
                if current_score > best_local_score:
                    best_local_score = current_score
                    best_local_set = set(S)
            else:
                # Reject move
                S.remove(y)
                S.add(x)
            
            T *= cooling
            
        return best_local_set, best_local_score

    # Initialization strategies
    inits = []
    
    # 1. Random uniform
    inits.append(random.sample(range(1, MAX_VAL + 1), N))
    
    # 2. Geometric spread (approx)
    # e.g., 1, 2, 4, 8... scaled to fit 512
    # Or just random with larger gaps
    geoms = []
    current = 1
    while len(geoms) < N:
        geoms.append(current)
        current *= 1.5
    # Scale to fit 1..512
    max_g = max(geoms)
    scale = 512.0 / max_g
    geoms_scaled = [int(g * scale) + 1 for g in geoms]
    # Ensure distinctness and range
    unique_geoms = []
    for g in geoms_scaled:
        if 1 <= g <= 512 and g not in unique_geoms:
            unique_geoms.append(g)
    # Fill remaining with random if needed
    while len(unique_geoms) < N:
        r = random.randint(1, MAX_VAL)
        if r not in unique_geoms:
            unique_geoms.append(r)
    inits.append(unique_geoms[:N])
    
    # 3. Two clusters
    cluster1 = random.sample(range(1, 100), 16)
    cluster2 = random.sample(range(400, 513), 16)
    inits.append(cluster1 + cluster2)
    
    # Run SA for each init, or one long SA with restarts
    # Given time budget, let's run multiple short SAs.
    
    for init in inits:
        if time.time() - start_time > timeout:
            break
        
        # Run SA for a specific duration or iterations
        # Let's allocate time proportionally
        remaining = timeout - (time.time() - start_time)
        if remaining <= 0: break
        
        # Run SA
        local_best_set, local_best_score = run_sa(init, timeout)
        
        if local_best_score > best_score:
            best_score = local_best_score
            best_set = local_best_set
            
        # If we have time, do a second pass with a different init or restart
        if time.time() - start_time < timeout - 10:
             # Generate a new random init based on best_set perturbation
             perturbed = list(best_set)
             for _ in range(4):
                 idx = random.randint(0, N-1)
                 new_val = random.randint(1, MAX_VAL)
                 while new_val in perturbed:
                     new_val = random.randint(1, MAX_VAL)
                 perturbed[idx] = new_val
             inits.append(perturbed)
             
    # Final polish on the best set found
    if best_set is not None and time.time() - start_time < timeout - 5:
        # Simple hill climbing polish
        S = set(best_set)
        current_score = score_set(S)
        improved = True
        while improved and time.time() - start_time < timeout - 2:
            improved = False
            # Try swapping each element with a random few candidates
            for x in list(S):
                if time.time() - start_time > timeout - 2:
                    break
                # Try 10 random replacements for x
                for _ in range(10):
                    y = random.randint(1, MAX_VAL)
                    if y in S:
                        continue
                    S.remove(x)
                    S.add(y)
                    new_score = score_set(S)
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                        x = y # x is now y in the set, but we are iterating over list(S) snapshot? 
                        # No, we modified S. The loop variable x is from the old list.
                        # This is getting messy. 
                        # Better: restart the loop if improved.
                        break
                    else:
                        S.remove(y)
                        S.add(x)
                if improved:
                    break # Restart the outer loop to pick up new x's
            
            if current_score > best_score:
                best_score = current_score
                best_set = S

    # Ensure valid output
    if best_set is None:
        return list(range(1, 33))
        
    return list(best_set)

# EVOLVE-BLOCK-END
