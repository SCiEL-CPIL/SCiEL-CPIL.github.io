"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import math
from array import array

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
    
    # Helper to calculate unique coverage contribution of a word given current counts
    # unique_contrib[w_idx] = number of neighbors of w_idx that are covered ONLY by w_idx
    def get_unique_contrib(w_idx, counts):
        contrib = 0
        for nb in neighbors[w_idx]:
            if counts[nb] == 1:
                contrib += 1
        return contrib

    def greedy_init(seed):
        random.seed(seed)
        counts = [0] * N
        covered = 0
        code = []
        code_set = set()
        uncovered = N
        
        while uncovered > 0:
            best_idx = -1
            max_cov = 0
            # To speed up, we can sample if N is large, but N=2187 is small enough for full scan
            # We prioritize max new coverage. Tie-break: random or min redundancy?
            # Let's stick to max new coverage for simplicity, but randomize tie-breaking
            candidates = []
            for w_idx in range(N):
                if w_idx in code_set:
                    continue
                cov = 0
                for nb in neighbors[w_idx]:
                    if counts[nb] == 0:
                        cov += 1
                if cov > max_cov:
                    max_cov = cov
                    candidates = [w_idx]
                elif cov == max_cov:
                    candidates.append(w_idx)
            
            if not candidates:
                break
            
            # Randomly pick from best candidates to diversify
            best_idx = random.choice(candidates)
            code.append(all_words[best_idx])
            code_set.add(best_idx)
            for nb in neighbors[best_idx]:
                if counts[nb] == 0:
                    covered += 1
                    uncovered -= 1
                counts[nb] += 1
        return code, counts, covered, code_set

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
        
        T = 8.0
        T_min = 0.005
        cooling = 0.99985
        
        all_indices = list(range(N))
        
        # Track time for reduction
        last_reduction = time.time()
        reduction_interval = 0.5 # More frequent
        
        # Helper to perform a full greedy reduction pass
        def perform_reduction():
            nonlocal code, counts, covered, size, code_set, current_cost, best_code, best_size, best_covered, best_cost
            # Order codewords by their unique contribution (ascending)
            # Words with 0 unique contribution are candidates for removal
            indices = list(range(size))
            # Calculate unique contributions
            ucs = []
            for i in range(size):
                w_idx = word_to_idx[code[i]]
                uc = 0
                for nb in neighbors[w_idx]:
                    if counts[nb] == 1:
                        uc += 1
                ucs.append(uc)
            
            # Sort indices by UC (lowest first)
            indices.sort(key=lambda i: ucs[i])
            
            removed_any = False
            for i in indices:
                if i >= size:
                    continue
                w = code[i]
                w_idx = word_to_idx[w]
                # Check if removing this word keeps coverage
                would_uncover = 0
                for nb in neighbors[w_idx]:
                    if counts[nb] == 1:
                        would_uncover += 1
                if would_uncover == 0:
                    # Safe to remove
                    code.pop(i)
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
                    # Since we removed one, indices shift, but we are iterating over a snapshot of indices
                    # and checking `if i >= size` handles out of bounds.
                    # However, the `i` refers to the old list. 
                    # To be safe and simple, we break and restart the reduction loop if we remove anything.
                    # This is O(N^2) worst case but N is small (~180).
                    break
            
            return removed_any

        while time.time() - start_time < time_limit:
            # Periodic greedy reduction
            if time.time() - last_reduction > reduction_interval:
                last_reduction = time.time()
                # Perform reduction until no more removals are possible
                while perform_reduction():
                    pass
            
            move_type = random.random()
            
            if move_type < 0.6:
                # Swap
                if size == 0:
                    T *= cooling
                    if T < T_min: T = T_min
                    continue
                
                # Smart selection for remove: pick word with low unique contribution
                # Sample a few words and pick the one with lowest UC
                remove_candidates = []
                for _ in range(3):
                    idx_remove = random.randint(0, size - 1)
                    remove_candidates.append(idx_remove)
                
                best_remove_idx = -1
                min_uc = float('inf')
                for idx_remove in remove_candidates:
                    w = code[idx_remove]
                    w_idx = word_to_idx[w]
                    uc = 0
                    for nb in neighbors[w_idx]:
                        if counts[nb] == 1:
                            uc += 1
                    if uc < min_uc:
                        min_uc = uc
                        best_remove_idx = idx_remove
                
                idx_remove = best_remove_idx
                w_remove = code[idx_remove]
                idx_w_remove = word_to_idx[w_remove]
                
                # Smart selection for add: pick word with high uncovered coverage
                add_candidates = []
                for _ in range(5):
                    idx_add = random.choice(all_indices)
                    if idx_add != idx_w_remove and idx_add not in code_set:
                        add_candidates.append(idx_add)
                
                if not add_candidates:
                    T *= cooling
                    if T < T_min: T = T_min
                    continue
                    
                best_add_idx = -1
                max_gain = -1
                for idx_add in add_candidates:
                    gain = 0
                    for nb in neighbors[idx_add]:
                        if counts[nb] == 0:
                            gain += 1
                    if gain > max_gain:
                        max_gain = gain
                        best_add_idx = idx_add
                        
                idx_add = best_add_idx
                
                nbs_remove = neighbors[idx_w_remove]
                nbs_add = neighbors[idx_add]
                nbs_remove_set = neighbor_sets[idx_w_remove]
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
                else:
                    T *= cooling
                    if T < T_min: T = T_min
                    
            elif move_type < 0.85:
                # Remove
                if size == 0:
                    T *= cooling
                    if T < T_min: T = T_min
                    continue
                
                # Smart selection: pick word with lowest UC
                idx = random.randint(0, size - 1)
                # Maybe check a few candidates
                best_idx = -1
                min_uc = float('inf')
                for _ in range(3):
                    cand_idx = random.randint(0, size - 1)
                    w = code[cand_idx]
                    w_idx = word_to_idx[w]
                    uc = 0
                    for nb in neighbors[w_idx]:
                        if counts[nb] == 1:
                            uc += 1
                    if uc < min_uc:
                        min_uc = uc
                        best_idx = cand_idx
                        
                idx = best_idx
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
                else:
                    T *= cooling
                    if T < T_min: T = T_min
                    
            else:
                # Add
                if covered == N:
                    # Convert to swap
                    if size > 0:
                        # Reuse swap logic roughly
                        idx_remove = random.randint(0, size - 1)
                        idx_w_remove = word_to_idx[code[idx_remove]]
                        idx_add = random.choice(all_indices)
                        if idx_add == idx_w_remove or idx_add in code_set:
                            T *= cooling
                            if T < T_min: T = T_min
                            continue
                        nbs_remove = neighbors[idx_w_remove]
                        nbs_add = neighbors[idx_add]
                        nbs_remove_set = neighbor_sets[idx_w_remove]
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
                    continue
                
                # Pick word with best marginal gain from a few candidates
                best_idx_add = -1
                best_gain = 0
                for _ in range(12):
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
                else:
                    T *= cooling
                    if T < T_min: T = T_min
        
        return best_code, best_size
    
    start_time = time.time()
    time_limit = 115.0
    
    # Multi-restart: try a few different seeds for greedy init
    best_overall = None
    best_overall_size = float('inf')
    
    num_restarts = 6 # Increase restarts
    time_per_restart = time_limit / num_restarts
    
    for r in range(num_restarts):
        t0 = time.time()
        remaining = time_limit - (t0 - start_time)
        if remaining < 5:
            break
        init_code, init_counts, init_covered, init_code_set = greedy_init(42 + r)
        t1 = time.time()
        sa_remaining = time_limit - (t1 - start_time)
        if sa_remaining < 2:
            break
        best_code, best_size = simulated_annealing(
            init_code, init_counts, init_covered, init_code_set,
            start_time, time_limit, 100 + r
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
