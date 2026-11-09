import itertools
import time
import random
import sys
from collections import defaultdict

def construct(v, k, t):
    """
    Solve the covering design problem C(v, k, t) using a structured probabilistic
    construction inspired by the Johnson scheme, followed by targeted local search.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    
    def time_ok():
        return time.time() < deadline - 5.0
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute all blocks and their covered t-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    block_covered_t = [None] * num_blocks
    t_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for i, b in enumerate(all_blocks):
        covered_indices = []
        for ts in itertools.combinations(b, t):
            idx = t_subset_to_idx[ts]
            covered_indices.append(idx)
            t_to_blocks[idx].append(i)
        block_covered_t[i] = covered_indices
    
    # Johnson scheme-inspired construction:
    # Use a random symmetric "weight" matrix to guide block selection
    def johnson_inspired_greedy(seed=None):
        if seed is not None:
            random.seed(seed)
        
        # Create a random weight structure over points
        # This approximates a probabilistic design
        point_weights = [1.0] * v
        for _ in range(3):
            for _ in range(v // 3):
                i, j = random.sample(range(v), 2)
                point_weights[i] += random.uniform(0.5, 2.0)
                point_weights[j] += random.uniform(0.5, 2.0)
        
        # Normalize
        total = sum(point_weights)
        point_weights = [w / total for w in point_weights]
        
        uncovered_set = set(range(num_t_subsets))
        selected = []
        
        while uncovered_set and time_ok():
            # Pick a random uncovered t-subset
            target_t_idx = random.choice(list(uncovered_set))
            target_t = all_t_subsets[target_t_idx]
            
            candidates = t_to_blocks[target_t_idx]
            
            # Score each candidate block using the point weights
            best_block = -1
            best_score = -1
            best_score_blocks = []
            
            for b_idx in candidates:
                block = all_blocks[b_idx]
                
                # Calculate gain (number of new t-subsets covered)
                gain = sum(1 for t_idx in block_covered_t[b_idx] if t_idx in uncovered_set)
                
                # Calculate structural score based on point weights
                struct_score = sum(point_weights[p] for p in block)
                
                # Combined score: primarily gain, secondarily structural fit
                score = gain * 100.0 + struct_score
                
                if score > best_score + 1e-9:
                    best_score = score
                    best_block = b_idx
                    best_score_blocks = [b_idx]
                elif abs(score - best_score) < 1e-9:
                    best_score_blocks.append(b_idx)
            
            if best_block == -1:
                break
            
            best_block = random.choice(best_score_blocks)
            selected.append(best_block)
            
            for t_idx in block_covered_t[best_block]:
                uncovered_set.discard(t_idx)
        
        return selected
    
    # Alternative: pure gain-based greedy with randomization
    def pure_greedy(seed=None):
        if seed is not None:
            random.seed(seed)
        
        uncovered_set = set(range(num_t_subsets))
        selected = []
        
        while uncovered_set and time_ok():
            target_t_idx = random.choice(list(uncovered_set))
            candidates = t_to_blocks[target_t_idx]
            
            best_block = -1
            best_gain = -1
            best_blocks_with_gain = []
            
            for b_idx in candidates:
                gain = sum(1 for t_idx in block_covered_t[b_idx] if t_idx in uncovered_set)
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
                    best_blocks_with_gain = [b_idx]
                elif gain == best_gain:
                    best_blocks_with_gain.append(b_idx)
            
            if best_block == -1:
                break
            
            best_block = random.choice(best_blocks_with_gain)
            selected.append(best_block)
            
            for t_idx in block_covered_t[best_block]:
                uncovered_set.discard(t_idx)
        
        return selected
    
    # Remove redundant blocks
    def try_remove_blocks(blocks):
        if not blocks:
            return blocks
        
        cover_count = [0] * num_t_subsets
        for b_idx in blocks:
            for t_idx in block_covered_t[b_idx]:
                cover_count[t_idx] += 1
        
        changed = True
        while changed and time_ok():
            changed = False
            for i in range(len(blocks)):
                if not time_ok():
                    return blocks
                
                b_idx = blocks[i]
                is_redundant = True
                for t_idx in block_covered_t[b_idx]:
                    if cover_count[t_idx] <= 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    for t_idx in block_covered_t[b_idx]:
                        cover_count[t_idx] -= 1
                    blocks.pop(i)
                    changed = True
                    break
        
        return blocks
    
    # 1-exchange local search with overlap optimization
    def one_exchange_search(blocks, max_iters=500):
        if not blocks:
            return blocks
        
        current_blocks = list(blocks)
        
        cover_count = [0] * num_t_subsets
        for b_idx in current_blocks:
            for t_idx in block_covered_t[b_idx]:
                cover_count[t_idx] += 1
        
        # Set of currently selected block indices for fast lookup
        selected_set = set(current_blocks)
        
        improved = True
        iterations = 0
        
        while improved and iterations < max_iters and time_ok():
            improved = False
            iterations += 1
            
            # First pass: remove redundant blocks
            removed_any = False
            for i in range(len(current_blocks)):
                if not time_ok():
                    break
                b_idx = current_blocks[i]
                is_redundant = True
                for t_idx in block_covered_t[b_idx]:
                    if cover_count[t_idx] <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for t_idx in block_covered_t[b_idx]:
                        cover_count[t_idx] -= 1
                    current_blocks.pop(i)
                    selected_set.discard(b_idx)
                    improved = True
                    removed_any = True
                    break
            
            if removed_any:
                continue
            
            # Try 1-exchanges
            for i in range(len(current_blocks)):
                if not time_ok():
                    break
                
                b_idx = current_blocks[i]
                
                # Find essential t-subsets (only covered by this block)
                essential_t = [t_idx for t_idx in block_covered_t[b_idx] if cover_count[t_idx] == 1]
                
                if not essential_t:
                    # Redundant - shouldn't happen after removal pass, but handle it
                    for t_idx in block_covered_t[b_idx]:
                        cover_count[t_idx] -= 1
                    current_blocks.pop(i)
                    selected_set.discard(b_idx)
                    improved = True
                    break
                
                # Try to find a replacement block
                # Sample candidates for efficiency
                num_candidates = min(3000, num_blocks)
                candidates = random.sample(range(num_blocks), num_candidates)
                candidates = [c for c in candidates if c not in selected_set]
                
                best_swap_block = -1
                best_swap_metric = -1
                
                for cand_idx in candidates:
                    # Check if candidate covers all essential t-subsets
                    covers_all = all(t_idx in block_covered_t[cand_idx] for t_idx in essential_t)
                    if not covers_all:
                        continue
                    
                    # Calculate overlap metric: we want to minimize overlap with existing blocks
                    # but also maximize new coverage
                    new_cover = 0
                    overlap_penalty = 0
                    for t_idx in block_covered_t[cand_idx]:
                        if cover_count[t_idx] == 0:
                            new_cover += 1
                        elif cover_count[t_idx] == 1:
                            overlap_penalty += 0.5  # mild penalty
                        else:
                            overlap_penalty += 1.0
                    
                    metric = new_cover * 2.0 - overlap_penalty
                    
                    if metric > best_swap_metric:
                        best_swap_metric = metric
                        best_swap_block = cand_idx
                
                if best_swap_block != -1 and best_swap_metric > 0:
                    # Perform swap
                    for t_idx in block_covered_t[b_idx]:
                        cover_count[t_idx] -= 1
                    for t_idx in block_covered_t[best_swap_block]:
                        cover_count[t_idx] += 1
                    selected_set.discard(b_idx)
                    selected_set.add(best_swap_block)
                    current_blocks[i] = best_swap_block
                    improved = True
                    break
        
        return current_blocks
    
    # Main search loop
    best_blocks = None
    best_len = float('inf')
    
    # Try structured construction first
    seeds = [42, 123, 999, 0, 1, 2, 3, 4, 5, 7, 11, 13, 17, 19, 23, 100, 200, 300, 400, 500]
    
    for seed in seeds:
        if not time_ok():
            break
        
        # Try Johnson-inspired construction
        greedy_res = johnson_inspired_greedy(seed)
        if not greedy_res:
            greedy_res = pure_greedy(seed)
        
        if not greedy_res:
            continue
        
        processed_res = try_remove_blocks(greedy_res)
        ls_res = one_exchange_search(processed_res)
        
        if len(ls_res) < best_len:
            best_len = len(ls_res)
            best_blocks = ls_res
        
        if best_len <= 69:
            break
    
    # If we have time, try more random seeds with both methods
    if time_ok():
        for _ in range(20):
            if not time_ok():
                break
            
            seed = random.randint(0, 1000000)
            
            # Alternate between the two methods
            if _ % 2 == 0:
                greedy_res = johnson_inspired_greedy(seed)
            else:
                greedy_res = pure_greedy(seed)
            
            if not greedy_res:
                continue
            
            processed_res = try_remove_blocks(greedy_res)
            ls_res = one_exchange_search(processed_res)
            
            if len(ls_res) < best_len:
                best_len = len(ls_res)
                best_blocks = ls_res
                
            if best_len <= 69:
                break
    
    if best_blocks is None:
        # Fallback: pure greedy
        greedy_res = pure_greedy(42)
        if greedy_res:
            best_blocks = try_remove_blocks(greedy_res)
        else:
            return []
    
    # Convert block indices back to actual blocks
    final_blocks = [list(all_blocks[b_idx]) for b_idx in best_blocks]
    
    # Ensure distinctness
    seen = set()
    unique_blocks = []
    for b in final_blocks:
        b.sort()
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            unique_blocks.append(b)
    
    return unique_blocks
