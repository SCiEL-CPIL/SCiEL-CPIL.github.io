"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import math

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(c) for c in itertools.product(range(q), repeat=n)]

    # Precompute neighbors
    all_words = list(itertools.product(range(3), repeat=7))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    N = 2187
    
    neighbors = []
    for w in all_words:
        w_list = list(w)
        idxs = [word_to_idx[w]]
        for i in range(7):
            orig = w_list[i]
            for v in range(3):
                if v != orig:
                    w_list[i] = v
                    idxs.append(word_to_idx[tuple(w_list)])
            w_list[i] = orig
        neighbors.append(idxs)
        
    # Greedy initialization
    def greedy_init(seed):
        random.seed(seed)
        counts = [0] * N
        covered = 0
        code_indices = []
        code_set = set()
        uncovered = N
        
        while uncovered > 0:
            best_idx = -1
            max_cov = 0
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
            code_indices.append(best_idx)
            code_set.add(best_idx)
            for nb in neighbors[best_idx]:
                if counts[nb] == 0:
                    covered += 1
                    uncovered -= 1
                counts[nb] += 1
        return code_indices, counts, covered, code_set

    # Simulated Annealing
    def simulated_annealing(init_code_indices, init_counts, init_covered, init_code_set, start_time, time_limit, seed):
        random.seed(seed)
        code_indices = list(init_code_indices)
        counts = list(init_counts)
        covered = init_covered
        size = len(code_indices)
        code_set = set(init_code_set)
        
        P = 10.0
        
        def cost(sz, cov):
            return sz + P * (N - cov)
            
        current_cost = cost(size, covered)
        best_code_indices = list(code_indices)
        best_size = size
        best_covered = covered
        best_cost = current_cost
        
        T = 8.0
        T_min = 0.005
        cooling = 0.99985
        
        all_indices = list(range(N))
        
        last_reduction = time.time()
        reduction_interval = 1.0
        
        while time.time() - start_time < time_limit:
            if time.time() - last_reduction > reduction_interval:
                last_reduction = time.time()
                indices = list(range(size))
                random.shuffle(indices)
                for idx in indices:
                    if idx >= size:
                        continue
                    w_idx = code_indices[idx]
                    would_uncover = 0
                    for nb in neighbors[w_idx]:
                        if counts[nb] == 1:
                            would_uncover += 1
                    if would_uncover == 0:
                        code_indices.pop(idx)
                        code_set.remove(w_idx)
                        for nb in neighbors[w_idx]:
                            counts[nb] -= 1
                        size -= 1
                        current_cost = cost(size, covered)
                        if size < best_size and covered == N:
                            best_size = size
                            best_covered = covered
                            best_code_indices = list(code_indices)
                            best_cost = current_cost
                        break
            
            r = random.random()
            if covered < N:
                if r < 0.7:
                    move_type = 'swap'
                elif r < 0.9:
                    move_type = 'add'
                else:
                    move_type = 'remove'
            else:
                if r < 0.5:
                    move_type = 'remove'
                elif r < 0.9:
                    move_type = 'swap'
                else:
                    move_type = 'add'
            
            if move_type == 'swap':
                if size == 0:
                    T *= cooling
                    if T < T_min: T = T_min
                    continue
                idx_remove = random.randint(0, size - 1)
                idx_w_remove = code_indices[idx_remove]
                idx_add = random.choice(all_indices)
                
                if idx_add == idx_w_remove or idx_add in code_set:
                    T *= cooling
                    if T < T_min: T = T_min
                    continue
                
                nbs_remove = neighbors[idx_w_remove]
                nbs_add = neighbors[idx_add]
                nbs_add_set = set(nbs_add)
                
                losing = 0
                for nb in nbs_remove:
                    if counts[nb] == 1 and nb not in nbs_add_set:
                        losing += 1
                gaining = 0
                for nb in nbs_add:
                    if counts[nb] == 0:
                        gaining += 1
                
                delta_covered = gaining - losing
                new_covered = covered + delta_covered
                new_cost = cost(size, new_covered)
                delta_cost = new_cost - current_cost
                
                if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                    code_indices[idx_remove] = idx_add
                    code_set.remove(idx_w_remove)
                    code_set.add(idx_add)
                    for nb in nbs_remove: counts[nb] -= 1
                    for nb in nbs_add: counts[nb] += 1
                    covered = new_covered
                    current_cost = new_cost
                    
                    if covered == N and size < best_size:
                        best_size = size
                        best_covered = covered
                        best_code_indices = list(code_indices)
                        best_cost = current_cost
                        
                    T *= cooling
                    if T < T_min: T = T_min
                else:
                    T *= cooling
                    if T < T_min: T = T_min
                    
            elif move_type == 'remove':
                if size == 0:
                    T *= cooling
                    if T < T_min: T = T_min
                    continue
                idx = random.randint(0, size - 1)
                w_idx = code_indices[idx]
                
                delta_covered = 0
                for nb in neighbors[w_idx]:
                    if counts[nb] == 1:
                        delta_covered -= 1
                
                new_size = size - 1
                new_covered = covered + delta_covered
                new_cost = cost(new_size, new_covered)
                delta_cost = new_cost - current_cost
                
                if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                    code_indices.pop(idx)
                    code_set.remove(w_idx)
                    for nb in neighbors[w_idx]:
                        counts[nb] -= 1
                    size = new_size
                    covered = new_covered
                    current_cost = new_cost
                    
                    if covered == N and size < best_size:
                        best_size = size
                        best_covered = covered
                        best_code_indices = list(code_indices)
                        best_cost = current_cost
                        
                    T *= cooling
                    if T < T_min: T = T_min
                else:
                    T *= cooling
                    if T < T_min: T = T_min
                    
            else:
                best_idx_add = -1
                best_gain = 0
                for _ in range(10):
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
                    code_indices.append(best_idx_add)
                    code_set.add(best_idx_add)
                    for nb in nbs_add:
                        counts[nb] += 1
                    size = new_size
                    covered = new_covered
                    current_cost = new_cost
                    
                    if covered == N and size < best_size:
                        best_size = size
                        best_covered = covered
                        best_code_indices = list(code_indices)
                        best_cost = current_cost
                        
                    T *= cooling
                    if T < T_min: T = T_min
                else:
                    T *= cooling
                    if T < T_min: T = T_min
        
        return best_code_indices, best_size

    start_time = time.time()
    time_limit = 115.0
    
    best_overall_indices = None
    best_overall_size = float('inf')
    
    num_restarts = 4
    
    for r in range(num_restarts):
        t0 = time.time()
        remaining = time_limit - (t0 - start_time)
        if remaining < 5:
            break
        init_code_indices, init_counts, init_covered, init_code_set = greedy_init(42 + r)
        t1 = time.time()
        sa_remaining = time_limit - (t1 - start_time)
        if sa_remaining < 2:
            break
        best_code_indices, best_size = simulated_annealing(
            init_code_indices, init_counts, init_covered, init_code_set,
            start_time, time_limit, 100 + r
        )
        if best_size < best_overall_size:
            best_overall_size = best_size
            best_overall_indices = best_code_indices
    
    if best_overall_indices is None:
        init_code_indices, _, _, _ = greedy_init(42)
        best_overall_indices = init_code_indices
    
    return [list(all_words[idx]) for idx in best_overall_indices]
# EVOLVE-BLOCK-END
