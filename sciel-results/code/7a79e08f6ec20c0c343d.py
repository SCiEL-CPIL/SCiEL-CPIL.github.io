"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import math
import numpy as np

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters (not expected in this task)
        # Simple greedy for small n or large radius
        all_words = list(itertools.product(range(q), repeat=n))
        return [list(w) for w in all_words[:100]] # Placeholder, task is specific

    N = 3**7
    # Precompute all words and their indices
    all_words = list(itertools.product(range(3), repeat=7))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors using NumPy for speed
    # Each word is a base-3 number 0..2186
    # We can generate neighbors by modifying each digit
    neighbors = []
    neighbor_sets = []
    
    # Vectorized neighbor generation
    # Create an array of all words as digits
    # This is still a bit slow in pure python if done naively, so we stick to efficient python loops 
    # but optimize the inner loop structure. 
    # Actually, for N=2187, a python loop is ~20ms which is fine.
    # Let's try to optimize the current neighbor generation.
    
    for i, w in enumerate(all_words):
        w_list = list(w)
        nb_idxs = [i]
        for pos in range(7):
            orig = w_list[pos]
            for val in range(3):
                if val != orig:
                    w_list[pos] = val
                    nb_idxs.append(word_to_idx[tuple(w_list)])
            w_list[pos] = orig
        neighbors.append(nb_idxs)
        neighbor_sets.append(frozenset(nb_idxs))

    def greedy_init(seed):
        random.seed(seed)
        counts = [0] * N
        uncovered = N
        code = []
        code_set = set()
        
        # To speed up greedy, we can maintain a list of uncovered words and pick a random one,
        # then pick the word covering the most uncovered words.
        # But scanning all N words each time is O(N^2 * deg) = 2187^2 * 15 ~ 70M ops.
        # 70M ops in python is ~1-2 seconds. This is acceptable for 4 restarts.
        
        while uncovered > 0:
            best_idx = -1
            max_cov = 0
            # Optimization: Only check words that are not in code? 
            # Yes, but checking 'in code_set' is fast.
            for w_idx in range(N):
                if w_idx in code_set:
                    continue
                cov = 0
                for nb in neighbors[w_idx]:
                    if counts[nb] == 0:
                        cov += 1
                if cov > max_cov:
                    max_cov = cov
                    best_idx = w_idx
            
            if best_idx == -1:
                break
            
            code.append(all_words[best_idx])
            code_set.add(best_idx)
            for nb in neighbors[best_idx]:
                if counts[nb] == 0:
                    uncovered -= 1
                counts[nb] += 1
        
        return code, counts, uncovered, code_set

    def simulated_annealing(init_code, init_counts, init_covered, init_code_set, start_time, time_limit, seed):
        random.seed(seed)
        code = list(init_code)
        counts = list(init_counts)
        covered = init_covered
        size = len(code)
        code_set = set(init_code_set)
        
        P = 10.0
        def cost(sz, cov):
            return sz + P * (N - cov)
        
        current_cost = cost(size, covered)
        best_code = list(code)
        best_size = size
        best_covered = covered
        best_cost = current_cost
        
        T = 15.0
        T_min = 0.01
        cooling = 0.9999
        
        all_indices = list(range(N))
        
        last_reduction = time.time()
        reduction_interval = 0.5 # More frequent reduction
        
        # Track moves for cooling schedule
        moves_made = 0
        target_moves = 50000
        
        while time.time() - start_time < time_limit:
            # Periodic greedy reduction
            if time.time() - last_reduction > reduction_interval:
                last_reduction = time.time()
                indices = list(range(size))
                random.shuffle(indices)
                removed_any = False
                for idx in indices:
                    if idx >= size:
                        continue
                    w = code[idx]
                    w_idx = word_to_idx[w]
                    # Check if removing this word keeps coverage
                    # We need to check if any neighbor has count 1
                    # If yes, removing it might uncover that neighbor
                    # But wait, if count[nb]==1, removing w makes it 0. So it becomes uncovered.
                    # So we can only remove if all neighbors have count > 1.
                    safe = True
                    for nb in neighbors[w_idx]:
                        if counts[nb] == 1:
                            safe = False
                            break
                    if safe:
                        code.pop(idx)
                        code_set.remove(w_idx)
                        for nb in neighbors[w_idx]:
                            counts[nb] -= 1
                        size -= 1
                        current_cost = cost(size, covered)
                        if size < best_size:
                            best_size = size
                            best_code = list(code)
                            best_covered = covered
                            best_cost = current_cost
                        removed_any = True
                        # Restart scan? Or just continue? 
                        # Continuing with the old indices list is risky if indices shift.
                        # Since we pop, indices after idx shift.
                        # So we should break and restart the shuffle/scan.
                        break
                
                if not removed_any:
                    # Try a slightly more aggressive reduction: 
                    # Find a word that covers the least "unique" neighbors?
                    # Or just skip.
                    pass
            
            move_type = random.random()
            
            if move_type < 0.6:
                # Swap
                if size == 0:
                    T *= cooling
                    if T < T_min: T = T_min
                    continue
                idx_remove = random.randint(0, size - 1)
                w_remove = code[idx_remove]
                idx_w_remove = word_to_idx[w_remove]
                idx_add = random.choice(all_indices)
                
                if idx_add == idx_w_remove or idx_add in code_set:
                    T *= cooling
                    if T < T_min: T = T_min
                    continue
                
                nbs_remove = neighbors[idx_w_remove]
                nbs_add = neighbors[idx_add]
                nbs_add_set = neighbor_sets[idx_add]
                
                losing = 0
                for nb in nbs_remove:
                    if counts[nb] == 1 and nb not in nbs_add_set:
                        losing += 1
                gaining = 0
                for nb in nbs_add:
                    if counts[nb] == 0:
                        gaining += 1
                
                delta = gaining - losing
                new_covered = covered + delta
                new_cost = cost(size, new_covered)
                delta_cost = new_cost - current_cost
                
                if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                    code[idx_remove] = all_words[idx_add]
                    code_set.remove(idx_w_remove)
                    code_set.add(idx_add)
                    for nb in nbs_remove: counts[nb] -= 1
                    for nb in nbs_add: counts[nb] += 1
                    covered = new_covered
                    current_cost = new_cost
                    
                    if covered == N and size < best_size:
                        best_size = size
                        best_covered = covered
                        best_code = list(code)
                        best_cost = current_cost
                    
                    T *= cooling
                    if T < T_min: T = T_min
                    moves_made += 1
                else:
                    T *= cooling
                    if T < T_min: T = T_min
                    moves_made += 1
                    
            elif move_type < 0.85:
                # Remove
                if size == 0:
                    T *= cooling
                    if T < T_min: T = T_min
                    continue
                idx = random.randint(0, size - 1)
                w = code[idx]
                w_idx = word_to_idx[w]
                
                delta_covered = 0
                for nb in neighbors[w_idx]:
                    if counts[nb] == 1:
                        delta_covered -= 1
                
                new_size = size - 1
                new_covered = covered + delta_covered
                new_cost = cost(new_size, new_covered)
                delta_cost = new_cost - current_cost
                
                if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                    code.pop(idx)
                    code_set.remove(w_idx)
                    for nb in neighbors[w_idx]:
                        counts[nb] -= 1
                    size = new_size
                    covered = new_covered
                    current_cost = new_cost
                    
                    if covered == N and size < best_size:
                        best_size = size
                        best_covered = covered
                        best_code = list(code)
                        best_cost = current_cost
                    
                    T *= cooling
                    if T < T_min: T = T_min
                    moves_made += 1
                else:
                    T *= cooling
                    if T < T_min: T = T_min
                    moves_made += 1
                    
            else:
                # Add
                if covered == N:
                    # If fully covered, convert to swap to try to reduce size
                    if size > 0:
                        idx_remove = random.randint(0, size - 1)
                        idx_w_remove = word_to_idx[code[idx_remove]]
                        idx_add = random.choice(all_indices)
                        if idx_add == idx_w_remove or idx_add in code_set:
                            T *= cooling
                            if T < T_min: T = T_min
                            continue
                        nbs_remove = neighbors[idx_w_remove]
                        nbs_add = neighbors[idx_add]
                        nbs_add_set = neighbor_sets[idx_add]
                        
                        losing = 0
                        for nb in nbs_remove:
                            if counts[nb] == 1 and nb not in nbs_add_set:
                                losing += 1
                        gaining = 0
                        for nb in nbs_add:
                            if counts[nb] == 0:
                                gaining += 1
                        
                        delta = gaining - losing
                        new_covered = covered + delta
                        new_cost = cost(size, new_covered)
                        delta_cost = new_cost - current_cost
                        
                        if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                            code[idx_remove] = all_words[idx_add]
                            code_set.remove(idx_w_remove)
                            code_set.add(idx_add)
                            for nb in nbs_remove: counts[nb] -= 1
                            for nb in nbs_add: counts[nb] += 1
                            covered = new_covered
                            current_cost = new_cost
                            
                            if covered == N and size < best_size:
                                best_size = size
                                best_covered = covered
                                best_code = list(code)
                                best_cost = current_cost
                        T *= cooling
                        if T < T_min: T = T_min
                        moves_made += 1
                    continue
                
                # Pick word with best marginal gain
                best_idx_add = -1
                best_gain = 0
                # Sample a few candidates
                for _ in range(8):
                    idx_add = random.choice(all_indices)
                    if idx_add in code_set:
                        continue
                    gain = 0
                    for nb in neighbors[idx_add]:
                        if counts[nb] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_idx_add = idx_add
                
                if best_idx_add == -1:
                    T *= cooling
                    if T < T_min: T = T_min
                    continue
                
                nbs_add = neighbors[best_idx_add]
                delta_covered = 0
                for nb in nbs_add:
                    if counts[nb] == 0:
                        delta_covered += 1
                        
                new_size = size + 1
                new_covered = covered + delta_covered
                new_cost = cost(new_size, new_covered)
                delta_cost = new_cost - current_cost
                
                if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                    code.append(all_words[best_idx_add])
                    code_set.add(best_idx_add)
                    for nb in nbs_add:
                        counts[nb] += 1
                    size = new_size
                    covered = new_covered
                    current_cost = new_cost
                    
                    if covered == N and size < best_size:
                        best_size = size
                        best_covered = covered
                        best_code = list(code)
                        best_cost = current_cost
                    
                    T *= cooling
                    if T < T_min: T = T_min
                    moves_made += 1
                else:
                    T *= cooling
                    if T < T_min: T = T_min
                    moves_made += 1
                    
            # Adaptive cooling: if we hit T_min, reset T to a higher value to escape local minima
            if T <= T_min:
                T = 5.0
                # Also try a random restart of the temperature or a kick
                if random.random() < 0.1:
                    # Perturb: do a few random swaps
                    for _ in range(5):
                        if size > 0:
                            idx_r = random.randint(0, size-1)
                            idx_a = random.choice(all_indices)
                            if idx_a not in code_set:
                                # Just do it if it doesn't break coverage too much? 
                                # For now, just reset T.
                                pass

        return best_code, best_size

    start_time = time.time()
    time_limit = 115.0
    
    best_overall = None
    best_overall_size = float('inf')
    
    num_restarts = 3
    time_per_restart = time_limit / num_restarts
    
    for r in range(num_restarts):
        t0 = time.time()
        remaining = time_limit - (t0 - start_time)
        if remaining < 10:
            break
        init_code, init_counts, init_covered, init_code_set = greedy_init(100 + r)
        t1 = time.time()
        sa_remaining = time_limit - (t1 - start_time)
        if sa_remaining < 2:
            break
        best_code, best_size = simulated_annealing(
            init_code, init_counts, init_covered, init_code_set,
            start_time, time_limit, 200 + r
        )
        if best_size < best_overall_size:
            best_overall_size = best_size
            best_overall = best_code
    
    if best_overall is None:
        init_code, _, _, _ = greedy_init(100)
        best_overall = init_code
    
    return [list(w) for w in best_overall]
# EVOLVE-BLOCK-END
