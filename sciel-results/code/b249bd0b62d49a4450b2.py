import itertools
import random
import time
import math

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using Simulated Annealing.
    """
    start_time = time.time()
    time_limit = 115  # Leave 5s for safety
    
    # Pre-compute all t-subsets and their indices
    all_subsets = list(itertools.combinations(range(v), t))
    num_subsets = len(all_subsets)
    subset_index = {s: i for i, s in enumerate(all_subsets)}
    
    def get_block_subsets(block):
        return [subset_index[s] for s in itertools.combinations(block, t)]

    # 1. Initialization: Randomized Greedy
    uncovered_set = set(range(num_subsets))
    
    def random_block_containing(subset):
        remaining = [i for i in range(v) if i not in subset]
        chosen = random.sample(remaining, k - t)
        return tuple(sorted(list(subset) + chosen))

    initial_blocks = []
    while uncovered_set:
        s_idx = random.choice(list(uncovered_set))
        s_tuple = all_subsets[s_idx]
        
        best_block = None
        best_gain = -1
        for _ in range(20): # Reduced samples for speed
            b = random_block_containing(s_tuple)
            gain = 0
            for sub_idx in get_block_subsets(b):
                if sub_idx in uncovered_set:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = b
        
        if best_block is not None:
            initial_blocks.append(best_block)
            for sub_idx in get_block_subsets(best_block):
                uncovered_set.discard(sub_idx)
            
        if time.time() - start_time > time_limit - 10:
            break

    # 2. Simulated Annealing
    
    blocks = [list(b) for b in initial_blocks]
    
    coverage_counts = [0] * num_subsets
    for b in blocks:
        for sub_idx in get_block_subsets(b):
            coverage_counts[sub_idx] += 1
            
    uncovered_count = sum(1 for c in coverage_counts if c == 0)
    
    def calc_cost(blocks_len, unc_cnt):
        if unc_cnt > 0:
            return 10000 + blocks_len + unc_cnt * 10
        return blocks_len
        
    current_cost = calc_cost(len(blocks), uncovered_count)
    best_blocks = [list(b) for b in blocks]
    best_cost = current_cost
    
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.995
    
    rng = random.Random(42)
    
    def random_block():
        return tuple(sorted(random.sample(range(v), k)))
        
    for i in range(100000):
        if time.time() - start_time > time_limit:
            break
        
        r = rng.random()
        
        if r < 0.4 and len(blocks) > 0:
            # Remove block
            idx = rng.randint(0, len(blocks) - 1)
            b = blocks[idx]
            
            delta_uncovered = 0
            for sub_idx in get_block_subsets(b):
                if coverage_counts[sub_idx] == 1:
                    delta_uncovered += 1
                coverage_counts[sub_idx] -= 1
            
            new_uncovered = uncovered_count + delta_uncovered
            new_cost = calc_cost(len(blocks) - 1, new_uncovered)
            
            delta_cost = new_cost - current_cost
            if delta_cost < 0 or rng.random() < math.exp(-delta_cost / T):
                blocks.pop(idx)
                uncovered_count = new_uncovered
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_blocks = [list(b) for b in blocks]
            else:
                for sub_idx in get_block_subsets(b):
                    coverage_counts[sub_idx] += 1
                    
        elif r < 0.8:
            # Add block
            b = random_block()
            
            delta_uncovered = 0
            for sub_idx in get_block_subsets(b):
                if coverage_counts[sub_idx] == 0:
                    delta_uncovered -= 1
                coverage_counts[sub_idx] += 1
                
            new_uncovered = uncovered_count + delta_uncovered
            new_cost = calc_cost(len(blocks) + 1, new_uncovered)
            
            delta_cost = new_cost - current_cost
            if delta_cost < 0 or rng.random() < math.exp(-delta_cost / T):
                blocks.append(b)
                uncovered_count = new_uncovered
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_blocks = [list(b) for b in blocks]
            else:
                for sub_idx in get_block_subsets(b):
                    coverage_counts[sub_idx] -= 1
                    
        else:
            # Swap element in a block
            if len(blocks) == 0:
                continue
                
            idx = rng.randint(0, len(blocks) - 1)
            b = blocks[idx]
            
            pos = rng.randint(0, k - 1)
            old_point = b[pos]
            candidates = [p for p in range(v) if p not in b]
            if not candidates:
                continue
            new_point = rng.choice(candidates)
            
            if old_point == new_point:
                continue
                
            new_b = b.copy()
            new_b[pos] = new_point
            new_b.sort()
            
            # Remove old
            delta_uncovered_remove = 0
            for sub_idx in get_block_subsets(b):
                if coverage_counts[sub_idx] == 1:
                    delta_uncovered_remove += 1
                coverage_counts[sub_idx] -= 1
                
            # Add new
            delta_uncovered_add = 0
            for sub_idx in get_block_subsets(new_b):
                if coverage_counts[sub_idx] == 0:
                    delta_uncovered_add -= 1
                coverage_counts[sub_idx] += 1
                
            total_delta_uncovered = delta_uncovered_remove + delta_uncovered_add
            new_uncovered = uncovered_count + total_delta_uncovered
            new_cost = calc_cost(len(blocks), new_uncovered)
            
            delta_cost = new_cost - current_cost
            if delta_cost < 0 or rng.random() < math.exp(-delta_cost / T):
                blocks[idx] = new_b
                uncovered_count = new_uncovered
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_blocks = [list(b) for b in blocks]
            else:
                # Revert
                for sub_idx in get_block_subsets(new_b):
                    coverage_counts[sub_idx] -= 1
                for sub_idx in get_block_subsets(b):
                    coverage_counts[sub_idx] += 1

        T *= cooling_rate
        if T < T_min:
            T = T_min

    # Verify best solution
    final_coverage = [0] * num_subsets
    for b in best_blocks:
        for sub_idx in get_block_subsets(b):
            final_coverage[sub_idx] += 1
            
    if all(c > 0 for c in final_coverage):
        return [list(b) for b in best_blocks]
    else:
        # Fallback
        init_coverage = [0] * num_subsets
        for b in initial_blocks:
            for sub_idx in get_block_subsets(b):
                init_coverage[sub_idx] += 1
        if all(c > 0 for c in init_coverage):
            return [list(b) for b in initial_blocks]
        return [list(b) for b in best_blocks]
