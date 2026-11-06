import numpy as np
import time
import random
import math

N = 256

def construct():
    deadline = time.time() + 55.0  # Safety margin
    rng = random.Random(42)
    
    # 1. Precompute all Arithmetic Progressions (APs)
    # An AP is defined by (start, step, length).
    # We need the list of indices for each AP to compute sums.
    # We also need for each position j, which APs contain j.
    
    ap_indices_list = []
    ap_ids = 0
    for d in range(1, N + 1):
        for a in range(1, N + 1):
            # Max length for this start and step
            max_l = (N - a) // d + 1
            for l in range(1, max_l + 1):
                indices = tuple(a + i * d for i in range(l))
                ap_indices_list.append(indices)
                ap_ids += 1
                
    num_aps = len(ap_indices_list)
    
    # Map position -> list of AP indices
    pos_to_ap = [[] for _ in range(N)]
    for ap_idx, indices in enumerate(ap_indices_list):
        for idx in indices:
            pos_to_ap[idx - 1].append(ap_idx)
            
    # Precompute flat arrays for faster access if needed, 
    # but for N=256, list of lists is likely fast enough in Python if optimized.
    # However, we will use the incremental update strategy.
    
    def compute_initial_sums(s):
        """
        s: numpy array of signs (-1, 1)
        Returns: numpy array of sums for all APs, and max abs sum.
        """
        sums = np.zeros(num_aps, dtype=np.int16)
        # Vectorized computation is hard due to variable lengths.
        # Loop over APs is O(num_aps * avg_len). 
        # num_aps ~ 180k. avg_len ~ 2. Total ops ~ 360k. Very fast.
        for ap_idx, indices in enumerate(ap_indices_list):
            # s is 0-indexed, indices are 1-indexed in generation, so subtract 1
            sums[ap_idx] = np.sum(s[list(i - 1 for i in indices)])
            
        abs_sums = np.abs(sums)
        max_sum = int(np.max(abs_sums))
        return sums, max_sum

    def build_hist(sums):
        hist = np.zeros(N + 1, dtype=np.int32)
        abs_sums = np.abs(sums)
        np.add.at(hist, abs_sums, 1)
        return hist

    def try_flip(j, s, sums, hist, current_max, temp, rng):
        """
        Attempt to flip position j.
        Returns (accepted, new_max, delta)
        """
        old_val = s[j]
        delta = -2 * old_val
        
        ap_list = pos_to_ap[j]
        
        old_max = current_max
        max_new_abs_in_affected = 0
        
        # Update sums and hist
        for ap_idx in ap_list:
            old_sum = sums[ap_idx]
            new_sum = old_sum + delta
            
            old_abs = abs(old_sum)
            new_abs = abs(new_sum)
            
            hist[old_abs] -= 1
            hist[new_abs] += 1
            
            sums[ap_idx] = new_sum
            
            if new_abs > max_new_abs_in_affected:
                max_new_abs_in_affected = new_abs
        
        # Determine new max
        if hist[old_max] == 0:
            if max_new_abs_in_affected > old_max:
                new_max = max_new_abs_in_affected
            else:
                scan_result = old_max - 1
                while scan_result > 0 and hist[scan_result] == 0:
                    scan_result -= 1
                new_max = max(max_new_abs_in_affected, scan_result)
        else:
            new_max = max(old_max, max_new_abs_in_affected)
            
        cost_diff = new_max - old_max
        
        # Acceptance criterion
        if cost_diff <= 0:
            accept = True
        else:
            if temp > 0.01:
                p = math.exp(-cost_diff / temp)
                accept = (rng.random() < p)
            else:
                accept = False
                
        if accept:
            s[j] = -old_val
            return True, new_max, delta
        else:
            # Revert
            for ap_idx in ap_list:
                new_sum = sums[ap_idx]
                old_sum = new_sum - delta
                
                old_abs = abs(old_sum)
                new_abs = abs(new_sum)
                
                hist[new_abs] -= 1
                hist[old_abs] += 1
                
                sums[ap_idx] = old_sum
                
            return False, current_max, delta

    def local_search(s, sums, hist, current_max, time_limit, rng, max_steps=1000):
        """
        Simulated Annealing local search.
        """
        temp = 2.0
        cooling_rate = 0.995
        steps = 0
        start_time = time.time()
        
        while time.time() < time_limit and steps < max_steps:
            j = rng.randint(0, N - 1)
            accepted, new_max, _ = try_flip(j, s, sums, hist, current_max, temp, rng)
            if accepted:
                current_max = new_max
            temp *= cooling_rate
            steps += 1
            
        return s, sums, hist, current_max

    def greedy_construct(rng):
        """
        Construct a sequence greedily.
        """
        s = np.zeros(N, dtype=np.int8)
        sums = np.zeros(num_aps, dtype=np.int16)
        hist = np.zeros(N + 1, dtype=np.int32)
        
        # We build the sequence position by position? 
        # No, that's hard to maintain AP sums because APs span the whole array.
        # Instead, start random and do a very aggressive greedy pass.
        s = np.array([1 if rng.random() < 0.5 else -1 for _ in range(N)], dtype=np.int8)
        sums, current_max = compute_initial_sums(s)
        hist = build_hist(sums)
        
        # Greedy pass: try flipping each position once, accept if it improves max
        improved = True
        while improved:
            improved = False
            for j in range(N):
                old_max = current_max
                old_val = s[j]
                delta = -2 * old_val
                ap_list = pos_to_ap[j]
                max_new_abs_in_affected = 0
                
                for ap_idx in ap_list:
                    new_sum = sums[ap_idx] + delta
                    new_abs = abs(new_sum)
                    if new_abs > max_new_abs_in_affected:
                        max_new_abs_in_affected = new_abs
                        
                if hist[old_max] == 0:
                    if max_new_abs_in_affected > old_max:
                        new_max = max_new_abs_in_affected
                    else:
                        scan_result = old_max - 1
                        while scan_result > 0 and hist[scan_result] == 0:
                            scan_result -= 1
                        new_max = max(max_new_abs_in_affected, scan_result)
                else:
                    new_max = max(old_max, max_new_abs_in_affected)
                    
                if new_max < current_max:
                    # Apply flip
                    for ap_idx in ap_list:
                        old_sum = sums[ap_idx]
                        new_sum = old_sum + delta
                        old_abs = abs(old_sum)
                        new_abs = abs(new_sum)
                        hist[old_abs] -= 1
                        hist[new_abs] += 1
                        sums[ap_idx] = new_sum
                    s[j] = -old_val
                    current_max = new_max
                    improved = True
                    
        return s, sums, hist, current_max

    # Main GA Loop
    pop_size = 30
    population = []
    
    # Initialize population
    # One greedy solution
    s0, sums0, hist0, score0 = greedy_construct(rng)
    population.append((s0.copy(), score0))
    
    # Random solutions
    for _ in range(pop_size - 1):
        s = np.array([1 if rng.random() < 0.5 else -1 for _ in range(N)], dtype=np.int8)
        sums, score = compute_initial_sums(s)
        population.append((s.copy(), score))
        
    best_seq = population[0][0].copy()
    best_score = population[0][1]
    
    # Sort population by score
    population.sort(key=lambda x: x[1])
    
    gen = 0
    while time.time() < deadline - 5.0:
        gen += 1
        offspring = []
        
        # Create offspring
        for i in range(pop_size):
            # Tournament selection
            p1_idx = rng.randint(0, min(5, len(population)-1))
            p2_idx = rng.randint(0, min(5, len(population)-1))
            p1 = population[p1_idx][0]
            p2 = population[p2_idx][0]
            
            # Uniform Crossover
            child = np.zeros(N, dtype=np.int8)
            for j in range(N):
                if rng.random() < 0.5:
                    child[j] = p1[j]
                else:
                    child[j] = p2[j]
                    
            # Mutation
            for j in range(N):
                if rng.random() < 0.1:
                    child[j] = -child[j]
                    
            # Local Search (SA) on child
            # We need sums and hist for the child
            sums_child, score_child = compute_initial_sums(child)
            hist_child = build_hist(sums_child)
            
            # Time budget for SA per offspring
            sa_time = 0.1 if gen < 10 else 0.05 # Less time later? Or constant?
            # Actually, fixed small time is better to allow more generations
            child, sums_child, hist_child, score_child = local_search(
                child, sums_child, hist_child, score_child, 
                time.time() + 0.1, rng, max_steps=200
            )
            
            offspring.append((child, score_child))
            
        # Combine population and offspring
        new_pop = population + offspring
        
        # Keep best pop_size
        new_pop.sort(key=lambda x: x[1])
        population = new_pop[:pop_size]
        
        # Update global best
        if population[0][1] < best_score:
            best_score = population[0][1]
            best_seq = population[0][0].copy()
            
    # Final Polish on the best sequence found
    if time.time() < deadline - 2.0:
        s_final = best_seq.copy()
        sums_final, score_final = compute_initial_sums(s_final)
        hist_final = build_hist(sums_final)
        
        # Longer SA
        s_final, sums_final, hist_final, score_final = local_search(
            s_final, sums_final, hist_final, score_final,
            deadline - 1.0, rng, max_steps=5000
        )
        
        if score_final < best_score:
            best_score = score_final
            best_seq = s_final.copy()
            
    return [int(x) for x in best_seq]
