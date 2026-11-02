import numpy as np
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    time_limit = 118.0  # Safety margin (evaluator kills at 150s)

    # 1. Precomputation
    points = list(range(v))
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    idx_map = {comb: i for i, comb in enumerate(all_ts)}
    
    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)
    
    # Precompute t-subset indices for each block
    block_t_indices = []
    for block in all_ks:
        subs = list(itertools.combinations(block, t))
        block_t_indices.append([idx_map[sub] for sub in subs])
        
    # Precompute inverse: triple -> list of blocks
    triple_to_blocks = [[] for _ in range(n_ts)]
    for bi, indices in enumerate(block_t_indices):
        for ti in indices:
            triple_to_blocks[ti].append(bi)

    # Precompute block t-indices as sets for faster membership testing
    block_t_sets = [set(indices) for indices in block_t_indices]

    # 2. Greedy Construction with temperature
    def greedy(seed, temperature=0.0):
        rng = random.Random(seed)
        covered = np.zeros(n_ts, dtype=bool)
        gain = np.array([len(indices) for indices in block_t_indices], dtype=np.int32)
        blocks = []
        
        while True:
            if time.time() - start_time > time_limit - 12:
                break
            
            max_gain = np.max(gain)
            if max_gain == 0:
                break
            
            # Temperature-based selection
            if temperature == 0.0:
                # Pure greedy: pick max gain
                candidates = np.where(gain == max_gain)[0]
                best_idx = candidates[rng.randint(0, len(candidates) - 1)]
            else:
                # Soft selection: probability proportional to exp(gain/temperature)
                # Only consider blocks with gain > 0
                positive_gain = gain > 0
                if not np.any(positive_gain):
                    break
                
                # Scale gains for numerical stability
                gains_pos = gain[positive_gain].astype(np.float64)
                max_g = np.max(gains_pos)
                weights = np.exp((gains_pos - max_g) / temperature)
                weights /= np.sum(weights)
                
                # Sample from positive gain blocks
                pos_indices = np.where(positive_gain)[0]
                best_idx = np.random.choice(pos_indices, p=weights)
            
            blocks.append(best_idx)
            
            subs = block_t_indices[best_idx]
            for ti in subs:
                if not covered[ti]:
                    covered[ti] = True
                    for other_bi in triple_to_blocks[ti]:
                        gain[other_bi] -= 1
            
            if np.all(covered):
                break
        
        return blocks

    # 3. Local Search: Remove redundant blocks
    def local_search_remove(blocks_list):
        current = list(blocks_list)
        improved = True
        while improved:
            if time.time() - start_time > time_limit - 5:
                break
            improved = False
            coverage_count = np.zeros(n_ts, dtype=np.int32)
            for b_idx in current:
                coverage_count[block_t_indices[b_idx]] += 1
            
            new_current = []
            for b_idx in current:
                subs = block_t_indices[b_idx]
                if np.all(coverage_count[subs] > 1):
                    improved = True
                    coverage_count[subs] -= 1
                else:
                    new_current.append(b_idx)
            current = new_current
        return current

    # 4. Local Search: 1-for-1 swap
    def local_search_1for1(blocks_list, time_budget):
        current = list(blocks_list)
        deadline = time.time() + time_budget
        
        while time.time() < deadline:
            if time.time() - start_time > time_limit - 5:
                break
                
            # Calculate coverage counts
            coverage_count = np.zeros(n_ts, dtype=np.int32)
            for b_idx in current:
                coverage_count[block_t_indices[b_idx]] += 1
            
            n_blocks = len(current)
            found = False
            
            # Try swapping each block with a better one
            # Sort by gain (uncovered triples it covers)
            gains = np.array([np.sum(coverage_count[block_t_indices[i]] == 1) for i in current], dtype=np.int32)
            
            # Try to replace low-gain blocks with high-gain blocks not in current
            current_set = set(current)
            
            # Sample some swaps
            indices = list(range(n_blocks))
            random.shuffle(indices)
            
            max_swaps_checked = 500
            swaps_checked = 0
            
            for i_idx in indices:
                if swaps_checked >= max_swaps_checked:
                    break
                if time.time() > deadline:
                    break
                
                a = current[i_idx]
                gain_a = gains[i_idx]
                
                if gain_a == 0:
                    continue
                
                # Find candidate blocks not in current that cover some of a's unique triples
                subs_a = block_t_indices[a]
                unique_a = [s for s in subs_a if coverage_count[s] == 1]
                
                if not unique_a:
                    continue
                
                # Try candidates from the first unique triple
                first_tri = unique_a[0]
                candidates = triple_to_blocks[first_tri]
                
                random.shuffle(candidates)
                for c in candidates:
                    if c in current_set:
                        continue
                    
                    # Check if c covers all of a's unique triples
                    c_set = block_t_sets[c]
                    if all(tri in c_set for tri in unique_a):
                        # Compute gain of c
                        gain_c = np.sum(coverage_count[block_t_indices[c]] == 1)
                        
                        if gain_c > gain_a:
                            # Perform swap
                            new_current = [x for x in current if x != a]
                            new_current.append(c)
                            current = new_current
                            current_set.discard(a)
                            current_set.add(c)
                            found = True
                            break
                
                if found:
                    break
            
            if not found:
                break
                
        return current

    # 5. Local Search: 2-for-1 replacement
    def local_search_2for1(blocks_list, time_budget):
        current = list(blocks_list)
        deadline = time.time() + time_budget
        
        while time.time() < deadline:
            if time.time() - start_time > time_limit - 5:
                break
                
            # Calculate coverage counts
            coverage_count = np.zeros(n_ts, dtype=np.int32)
            for b_idx in current:
                coverage_count[block_t_indices[b_idx]] += 1
            
            n_blocks = len(current)
            found = False
            
            # Sample pairs to check
            indices = list(range(n_blocks))
            random.shuffle(indices)
            
            max_pairs_checked = 800
            pairs_checked = 0
            
            for i_idx in range(len(indices)):
                for j_idx in range(i_idx + 1, len(indices)):
                    if pairs_checked >= max_pairs_checked:
                        break
                    if time.time() > deadline:
                        break
                    
                    a = current[indices[i_idx]]
                    b = current[indices[j_idx]]
                    
                    subs_a = block_t_indices[a]
                    subs_b = block_t_indices[b]
                    
                    unique_a = [s for s in subs_a if coverage_count[s] == 1]
                    unique_b = [s for s in subs_b if coverage_count[s] == 1]
                    
                    if not unique_a and not unique_b:
                        continue
                    
                    req_set = set(unique_a) | set(unique_b)
                    
                    if not req_set:
                        continue
                    
                    first_tri = next(iter(req_set))
                    candidates = triple_to_blocks[first_tri]
                    
                    random.shuffle(candidates)
                    best_c = -1
                    for c in candidates:
                        if c == a or c == b:
                            continue
                        
                        c_set = block_t_sets[c]
                        if all(tri in c_set for tri in req_set):
                            best_c = c
                            break
                    
                    if best_c != -1:
                        new_current = [x for x in current if x != a and x != b]
                        new_current.append(best_c)
                        current = new_current
                        found = True
                        break
                
                if found:
                    break
            
            if not found:
                break
                
        return current

    # 6. Main Loop
    best_blocks = None
    best_count = float('inf')
    
    # Try multiple seeds with different temperatures
    for seed in range(25):
        if time.time() - start_time > time_limit - 15:
            break
            
        # Vary temperature for diversity
        if seed % 3 == 0:
            temp = 0.0
        elif seed % 3 == 1:
            temp = 1.0
        else:
            temp = 2.0
        
        greedy_blocks = greedy(seed, temperature=temp)
        
        # Refine: remove redundant blocks
        refined = local_search_remove(greedy_blocks)
        
        # 1-for-1 swaps
        if time.time() - start_time < time_limit - 10:
            remaining = time_limit - (time.time() - start_time) - 10
            if remaining > 3.0:
                refined = local_search_1for1(refined, min(3.0, remaining * 0.3))
                refined = local_search_remove(refined)
        
        # 2-for-1 swaps
        if time.time() - start_time < time_limit - 8:
            remaining = time_limit - (time.time() - start_time) - 8
            if remaining > 3.0:
                refined = local_search_2for1(refined, min(4.0, remaining * 0.5))
                refined = local_search_remove(refined)
        
        count = len(refined)
        if count < best_count:
            best_count = count
            best_blocks = refined
            if best_count <= 69:
                pass

    if best_blocks is None:
        return []

    final_blocks = [list(all_ks[i]) for i in best_blocks]
    return final_blocks
