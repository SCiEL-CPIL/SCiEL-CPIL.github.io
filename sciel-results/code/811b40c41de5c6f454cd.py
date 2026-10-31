import itertools
import time
import random
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Precompute
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_index = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    block_to_t_indices = {}
    for block in all_k_subsets:
        t_indices = [t_subset_to_index[ts] for ts in itertools.combinations(block, t)]
        block_to_t_indices[block] = t_indices
        
    # 1. Greedy Initialization
    cover_count = [0] * num_t_subsets
    uncovered = set(range(num_t_subsets))
    blocks = []
    used_blocks = set()
    
    while uncovered:
        best_block = None
        max_gain = -1
        for block in all_k_subsets:
            if block in used_blocks:
                continue
            t_indices = block_to_t_indices[block]
            gain = 0
            for idx in t_indices:
                if idx in uncovered:
                    gain += 1
            if gain > max_gain:
                max_gain = gain
                best_block = block
                if gain == 0:
                    break
            
        if best_block is None:
            break
            
        blocks.append(best_block)
        used_blocks.add(best_block)
        t_indices = block_to_t_indices[best_block]
        for idx in t_indices:
            if idx in uncovered:
                uncovered.discard(idx)
            cover_count[idx] += 1
            
        if time.time() > deadline:
            return [list(b) for b in blocks]

    # 2. Simulated Annealing
    current_blocks = blocks[:]
    best_blocks = current_blocks[:]
    best_cost = len(current_blocks)
    used_blocks = set(current_blocks)
    
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.999
    
    all_blocks_list = list(all_k_subsets)
    
    while time.time() < deadline:
        if T < T_min:
            T = T_min
            
        move_type = random.random()
        
        # 1. Remove Move (50%)
        if move_type < 0.5:
            if not current_blocks:
                continue
            idx = random.randint(0, len(current_blocks) - 1)
            block = current_blocks[idx]
            t_indices = block_to_t_indices[block]
            
            redundant = True
            for t_idx in t_indices:
                if cover_count[t_idx] == 1:
                    redundant = False
                    break
            
            if redundant:
                for t_idx in t_indices:
                    cover_count[t_idx] -= 1
                del current_blocks[idx]
                used_blocks.discard(block)
                new_cost = len(current_blocks)
                
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_blocks = current_blocks[:]
                    
                T = max(T * cooling_rate, T_min)
            else:
                T = max(T * cooling_rate, T_min)

        # 2. Swap Move (40%)
        elif move_type < 0.9:
            if not current_blocks:
                continue
            idx = random.randint(0, len(current_blocks) - 1)
            old_block = current_blocks[idx]
            old_t_indices = block_to_t_indices[old_block]
            
            criticals = []
            for t_idx in old_t_indices:
                if cover_count[t_idx] == 1:
                    criticals.append(t_idx)
            
            if len(criticals) > k:
                T = max(T * cooling_rate, T_min)
                continue
                
            union_points = set()
            for t_idx in criticals:
                union_points.update(all_t_subsets[t_idx])
            
            if len(union_points) > k:
                T = max(T * cooling_rate, T_min)
                continue
                
            available_points = [p for p in range(v) if p not in union_points]
            num_needed = k - len(union_points)
            
            if num_needed < 0:
                continue
                
            if num_needed == 0:
                new_block = tuple(sorted(union_points))
            else:
                extra = random.sample(available_points, num_needed)
                new_block = tuple(sorted(list(union_points) + extra))
                
            if new_block == old_block:
                T = max(T * cooling_rate, T_min)
                continue
                
            if new_block in used_blocks:
                T = max(T * cooling_rate, T_min)
                continue
                
            new_t_indices = block_to_t_indices[new_block]
            
            for t_idx in old_t_indices:
                cover_count[t_idx] -= 1
            for t_idx in new_t_indices:
                cover_count[t_idx] += 1
            
            current_blocks[idx] = new_block
            used_blocks.discard(old_block)
            used_blocks.add(new_block)
            
            T = max(T * cooling_rate, T_min)

        # 3. Add Move (10%)
        else:
            new_block = random.choice(all_blocks_list)
            if new_block in used_blocks:
                T = max(T * cooling_rate, T_min)
                continue
                
            new_t_indices = block_to_t_indices[new_block]
            
            for t_idx in new_t_indices:
                cover_count[t_idx] += 1
            
            current_blocks.append(new_block)
            used_blocks.add(new_block)
            
            delta_e = 1
            p_accept = math.exp(-delta_e / T)
            
            if random.random() > p_accept:
                for t_idx in new_t_indices:
                    cover_count[t_idx] -= 1
                current_blocks.pop()
                used_blocks.discard(new_block)
                
            T = max(T * cooling_rate, T_min)
            
    return [list(b) for b in best_blocks]
