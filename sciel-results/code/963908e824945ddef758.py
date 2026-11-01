import time
import itertools
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin for 120s limit (kill at 150s)
    
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_t_subsets)
    t_subset_to_index = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_blocks = len(all_k_subsets)
    
    # Precompute coverage using lists
    block_to_tsubs = []
    block_to_tsubs_set = []
    for block in all_k_subsets:
        covered = []
        for tsub in itertools.combinations(block, t):
            idx = t_subset_to_index[tsub]
            covered.append(idx)
        block_to_tsubs.append(covered)
        block_to_tsubs_set.append(frozenset(covered))
        
    # Precompute inverse
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for b_idx, tsubs in enumerate(block_to_tsubs):
        for s_idx in tsubs:
            tsub_to_blocks[s_idx].append(b_idx)
            
    tsub_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_to_blocks]
    
    # Helper: Greedy Construction with faster gain updates
    def greedy_construction(randomize_ties=False, random_seed=None):
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random
        
        uncovered = np.ones(num_tsubs, dtype=bool)
        k_choose_t = len(block_to_tsubs[0])
        gains = np.full(num_blocks, k_choose_t, dtype=np.int32)
        
        selected_block_indices = []
        
        while uncovered.any():
            if time.time() > deadline:
                return None
            
            if randomize_ties and rng.random() < 0.3:
                top_k = 20
                if num_blocks > top_k:
                    top_indices = np.argpartition(gains, -top_k)[-top_k:]
                    b = rng.choice(top_indices)
                else:
                    b = np.argmax(gains)
            else:
                b = int(np.argmax(gains))
            
            if gains[b] == 0:
                break
            
            selected_block_indices.append(b)
            
            # Update: mark covered and decrement gains for blocks sharing those t-subsets
            for s_idx in block_to_tsubs[b]:
                if uncovered[s_idx]:
                    uncovered[s_idx] = False
                    gains[tsub_to_blocks_np[s_idx]] -= 1
                    
        if uncovered.any():
            return None
            
        return selected_block_indices

    # Helper: Local Optimization - improved with better data structures
    def local_optimize(blocks, time_budget=None):
        if not blocks:
            return []
        
        if time_budget is None:
            time_budget = deadline - time.time()
        
        selected_set = set(blocks)
        selected_list = list(blocks)
        
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b in selected_list:
            for s in block_to_tsubs[b]:
                cover_count[s] += 1
        
        # Precompute for each block: list of t-subsets it covers
        # And for quick redundancy check
        
        def is_redundant_fast(b_idx):
            # A block is redundant if all its t-subsets are covered by other blocks
            for s in block_to_tsubs[b_idx]:
                if cover_count[s] == 1:
                    return False
            return True
        
        # For swap: precompute candidate replacements for each t-subset
        # But that's expensive, so do it on-the-fly but efficiently
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Step A: Redundancy Removal
            redundant_blocks = []
            for b in selected_list:
                if is_redundant_fast(b):
                    redundant_blocks.append(b)
            
            if redundant_blocks:
                for b in redundant_blocks:
                    selected_list.remove(b)
                    selected_set.remove(b)
                    for s in block_to_tsubs[b]:
                        cover_count[s] -= 1
                improved = True
                continue
            
            # Step B: Swap - try to replace a block with one that covers its unique t-subsets
            # Compute unique t-subsets for each block
            unique_tsubs_map = {}
            unique_counts = {}
            
            for b in selected_list:
                u_list = []
                for s in block_to_tsubs[b]:
                    if cover_count[s] == 1:
                        u_list.append(s)
                unique_tsubs_map[b] = u_list
                unique_counts[b] = len(u_list)
            
            if not unique_counts:
                break
                
            min_unique = min(unique_counts.values())
            candidates_out = [b for b, c in unique_counts.items() if c == min_unique]
            random.shuffle(candidates_out)
            
            swap_made = False
            for b_out in candidates_out:
                s_uniq = unique_tsubs_map[b_out]
                
                # Find replacement: must cover all s_uniq, not in selected_set
                # Start from candidates covering the first unique t-subset
                if not s_uniq:
                    # No unique t-subsets, any unselected block works
                    continue
                    
                first_s = s_uniq[0]
                candidates = tsub_to_blocks_np[first_s]
                
                valid_candidates = []
                for c in candidates:
                    if c in selected_set:
                        continue
                    c_set = block_to_tsubs_set[c]
                    # Check covers all unique t-subsets
                    if all(s in c_set for s in s_uniq):
                        valid_candidates.append(c)
                
                if valid_candidates:
                    b_in = random.choice(valid_candidates)
                    # Perform swap
                    selected_list.remove(b_out)
                    selected_set.remove(b_out)
                    selected_list.append(b_in)
                    selected_set.add(b_in)
                    
                    for s in block_to_tsubs[b_out]:
                        cover_count[s] -= 1
                    for s in block_to_tsubs[b_in]:
                        cover_count[s] += 1
                    
                    swap_made = True
                    break
            
            if swap_made:
                improved = True
        
        return selected_list

    # Main logic
    best_blocks = None
    best_size = float('inf')
    
    # Adaptive number of restarts
    # First, do one greedy to estimate time
    t0 = time.time()
    blocks = greedy_construction(randomize_ties=False)
    if blocks is not None:
        t1 = time.time()
        greedy_time = t1 - t0
        
        # Estimate: local_optimize takes roughly 2-5x greedy time
        # Budget for local_optimize: 3x greedy_time
        # Total per restart: ~4x greedy_time
        # Available time: deadline - start_time
        # Max restarts: floor(available / (4 * greedy_time))
        
        remaining = deadline - start_time
        est_per_restart = max(1.0, 4.0 * greedy_time)
        max_restarts = max(1, int(remaining / est_per_restart))
        max_restarts = min(max_restarts, 15)  # Cap at 15
        
        # Store first result
        optimized = local_optimize(blocks)
        if len(optimized) < best_size:
            best_size = len(optimized)
            best_blocks = optimized
        
        # Additional restarts
        for r in range(1, max_restarts):
            if time.time() > deadline - 15:
                break
            
            blocks = greedy_construction(randomize_ties=True, random_seed=r)
            if blocks is None:
                continue
            
            optimized = local_optimize(blocks)
            if len(optimized) < best_size:
                best_size = len(optimized)
                best_blocks = optimized
    else:
        # Fallback
        return []
    
    if best_blocks is None:
        return []
    
    return [all_k_subsets[i] for i in best_blocks]

if __name__ == "__main__":
    v, k, t = 19, 7, 4
    t0 = time.time()
    result = construct(v, k, t)
    t1 = time.time()
    print(f"C({v},{k},{t}): {len(result)} blocks, time {t1 - t0:.1f}s")
