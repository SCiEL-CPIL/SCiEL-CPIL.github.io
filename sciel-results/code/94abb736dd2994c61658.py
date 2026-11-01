"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import math
from array import array
import numpy as np

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(c) for c in itertools.product(range(q), repeat=n)]

    # Precompute neighbors as tuples for faster iteration
    all_words = list(itertools.product(range(3), repeat=7))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    N = 2187
    
    neighbors = []
    neighbor_sets = []
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
        neighbors.append(tuple(idxs))
        neighbor_sets.append(frozenset(idxs))
    
    # Precompute neighbor arrays as numpy for faster operations
    neighbors_np = np.array(neighbors, dtype=np.int32)
    
    # Greedy initialization with randomization for diversity
    def greedy_init(seed, randomize_prob=0.0):
        random.seed(seed)
        counts = [0] * N
        covered = 0
        code = []
        code_set = set()
        uncovered = N
        
        while uncovered > 0:
            best_idx = -1
            max_cov = 0
            # With some probability, pick a random good candidate instead of best
            if randomize_prob > 0 and random.random() < randomize_prob:
                # Pick from top 5 candidates
                candidates = []
                for w_idx in range(N):
                    if w_idx in code_set:
                        continue
                    cov = 0
                    for nb in neighbors[w_idx]:
                        if counts[nb] == 0:
                            cov += 1
                    candidates.append((cov, w_idx))
                candidates.sort(reverse=True)
                top_k = min(5, len(candidates))
                if top_k > 0:
                    best_idx = candidates[random.randint(0, top_k-1)][1]
            else:
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
                    covered += 1
                    uncovered -= 1
                counts[nb] += 1
        return code, counts, covered, code_set
    
    # Fast greedy reduction: remove all redundant words
    def greedy_reduce(code, counts, covered, code_set):
        changed = True
        while changed:
            changed = False
            # Try removing each codeword in random order
            indices = list(range(len(code)))
            random.shuffle(indices)
            for idx in indices:
                if idx >= len(code):
                    continue
                w = code[idx]
                w_idx = word_to_idx[w]
                # Check if removing this word keeps coverage
                would_uncover = 0
                for nb in neighbors[w_idx]:
                    if counts[nb] == 1:
                        would_uncover += 1
                if would_uncover == 0:
                    # Safe to remove
                    code.pop(idx)
                    code_set.remove(w_idx)
                    for nb in neighbors[w_idx]:
                        counts[nb] -= 1
                    changed = True
                    break  # Restart the scan after a removal
        return code, counts, covered, code_set
    
    # Simulated annealing with improved move strategies
    def simulated_annealing(init_code, init_counts, init_covered, init_code_set, start_time, time_limit, seed, T_init=5.0, cooling=0.9999):
        random.seed(seed)
        code = list(init_code)
        counts = list(init_counts)
        covered = init_covered
        size = len(code)
        code_set = set(init_code_set)
        
        P = 8.0
        def cost(sz, cov):
            return sz + P * (N - cov)
        
        current_cost = cost(size, covered)
        best_code = list(code)
        best_size = size
        best_covered = covered
        best_cost = current_cost
        
        T = T_init
        T_min = 0.01
        all_indices = list(range(N))
        
        # Track moves without improvement for restart
        moves_without_improvement = 0
        restart_threshold = 5000
        last_reduction_time = 0
        reduction_interval = 0.8
        
        while time.time() - start_time < time_limit:
            # Periodic greedy reduction
            if time.time() - last_reduction_time > reduction_interval:
                last_reduction_time = time.time()
                code, counts, covered, code_set = greedy_reduce(code, counts, covered, code_set)
                size = len(code)
                current_cost = cost(size, covered)
                if size < best_size:
                    best_size = size
                    best_code = list(code)
                    best_covered = covered
                    best_cost = current_cost
                    moves_without_improvement = 0
            
            # Restart if stuck
            if moves_without_improvement > restart_threshold:
                # Perturb: add a random word
                if size > 0:
                    idx_add = random.choice(all_indices)
                    if idx_add not in code_set:
                        code.append(all_words[idx_add])
                        code_set.add(idx_add)
                        for nb in neighbors[idx_add]:
                            counts[nb] += 1
                        size += 1
                        covered = min(N, covered + sum(1 for nb in neighbors[idx_add] if counts[nb] == 1))
                        current_cost = cost(size, covered)
                moves_without_improvement = 0
                T = T_init * 0.5  # Reset temperature
            
            move_type = random.random()
            
            if move_type < 0.6:
                # Swap: replace one codeword with another
                if size == 0:
                    T *= cooling
                    if T < T_min: T = T_min
                    continue
                idx_remove = random.randint(0, size - 1)
                w_remove = code[idx_remove]
                idx_w_remove = word_to_idx[w_remove]
                
                # Try a few random candidates for addition
                best_delta = float('inf')
                best_idx_add = -1
                for _ in range(3):
                    idx_add = random.choice(all_indices)
                    if idx_add == idx_w_remove or idx_add in code_set:
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
                    if delta < best_delta:
                        best_delta = delta
                        best_idx_add = idx_add
                
                if best_idx_add == -1:
                    T *= cooling
                    if T < T_min: T = T_min
                    continue
                
                nbs_remove = neighbors[idx_w_remove]
                nbs_add = neighbors[best_idx_add]
                
                new_covered = covered + best_delta
                new_cost = cost(size, new_covered)
                delta_cost = new_cost - current_cost
                
                if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                    code[idx_remove] = all_words[best_idx_add]
                    code_set.remove(idx_w_remove)
                    code_set.add(best_idx_add)
                    for nb in nbs_remove: counts[nb] -= 1
                    for nb in nbs_add: counts[nb] += 1
                    covered = new_covered
                    current_cost = new_cost
                    
                    if covered == N and size < best_size:
                        best_size = size
                        best_covered = covered
                        best_code = list(code)
                        best_cost = current_cost
                        moves_without_improvement = 0
                    else:
                        moves_without_improvement += 1
                    
                    T *= cooling
                    if T < T_min: T = T_min
                else:
                    T *= cooling
                    if T < T_min: T = T_min
                    moves_without_improvement += 1
                    
            elif move_type < 0.85:
                # Remove a codeword
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
                        moves_without_improvement = 0
                    else:
                        moves_without_improvement += 1
                    
                    T *= cooling
                    if T < T_min: T = T_min
                else:
                    T *= cooling
                    if T < T_min: T = T_min
                    moves_without_improvement += 1
                    
            else:
                # Add a codeword (only if not fully covered)
                if covered == N:
                    # Convert to swap
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
                                moves_without_improvement = 0
                            else:
                                moves_without_improvement += 1
                        T *= cooling
                        if T < T_min: T = T_min
                    continue
                
                # Pick word with best marginal gain from several candidates
                best_idx_add = -1
                best_gain = 0
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
                        moves_without_improvement = 0
                    else:
                        moves_without_improvement += 1
                    
                    T *= cooling
                    if T < T_min: T = T_min
                else:
                    T *= cooling
                    if T < T_min: T = T_min
                    moves_without_improvement += 1
        
        return best_code, best_size
    
    start_time = time.time()
    time_limit = 115.0
    
    # Multi-restart with different strategies
    best_overall = None
    best_overall_size = float('inf')
    
    # Different configurations for each restart
    configs = [
        {'seed': 42, 'randomize_prob': 0.0, 'T_init': 5.0, 'cooling': 0.9999},
        {'seed': 123, 'randomize_prob': 0.1, 'T_init': 4.0, 'cooling': 0.99995},
        {'seed': 456, 'randomize_prob': 0.2, 'T_init': 3.0, 'cooling': 0.9999},
        {'seed': 789, 'randomize_prob': 0.0, 'T_init': 6.0, 'cooling': 0.99995},
    ]
    
    num_restarts = len(configs)
    
    for r in range(num_restarts):
        t0 = time.time()
        remaining = time_limit - (t0 - start_time)
        if remaining < 5:
            break
        
        cfg = configs[r]
        init_code, init_counts, init_covered, init_code_set = greedy_init(
            cfg['seed'], cfg['randomize_prob']
        )
        t1 = time.time()
        sa_remaining = time_limit - (t1 - start_time)
        if sa_remaining < 2:
            break
        best_code, best_size = simulated_annealing(
            init_code, init_counts, init_covered, init_code_set,
            start_time, time_limit, 100 + r,
            T_init=cfg['T_init'], cooling=cfg['cooling']
        )
        if best_size < best_overall_size:
            best_overall_size = best_size
            best_overall = best_code
    
    if best_overall is None:
        # Fallback
        init_code, _, _, _ = greedy_init(42)
        best_overall = init_code
    
    return [list(w) for w in best_overall]
# EVOLVE-BLOCK-END
