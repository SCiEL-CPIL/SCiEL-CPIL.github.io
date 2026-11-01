import time
import itertools
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_t_subsets)
    t_subset_to_index = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_blocks = len(all_k_subsets)
    
    # Precompute coverage: which t-subsets does each block cover?
    block_to_tsubs = []
    block_to_tsubs_set = []
    for block in all_k_subsets:
        covered = []
        for tsub in itertools.combinations(block, t):
            idx = t_subset_to_index[tsub]
            covered.append(idx)
        block_to_tsubs.append(covered)
        block_to_tsubs_set.append(set(covered))
        
    # Precompute inverse: which blocks cover each t-subset?
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for b_idx, tsubs in enumerate(block_to_tsubs):
        for s_idx in tsubs:
            tsub_to_blocks[s_idx].append(b_idx)
            
    # Convert to numpy for faster intersection if needed, but lists are fine for small intersections
    # We will use lists for tsub_to_blocks
    
    def greedy_construction(randomize_ties=False, random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            
        uncovered = np.ones(num_tsubs, dtype=bool)
        k_choose_t = len(block_to_tsubs[0])
        gains = np.full(num_blocks, k_choose_t, dtype=np.int32)
        
        selected_block_indices = []
        
        while uncovered.any():
            if time.time() > deadline:
                return None
            
            # Find block with max gain
            if randomize_ties:
                # Pick a random block from top 20 gains to break symmetry
                if random.random() < 0.5:
                    top_k = 20
                    if num_blocks > top_k:
                        top_indices = np.argpartition(gains, -top_k)[-top_k:]
                        b = random.choice(top_indices)
                    else:
                        b = np.argmax(gains)
                else:
                    b = np.argmax(gains)
            else:
                b = np.argmax(gains)
            
            if gains[b] == 0:
                break
            
            selected_block_indices.append(b)
            
            # Update uncovered and gains
            for s_idx in block_to_tsubs[b]:
                if uncovered[s_idx]:
                    uncovered[s_idx] = False
                    gains[tsub_to_blocks[s_idx]] -= 1
                    
        if uncovered.any():
            return None
            
        return selected_block_indices

    def local_optimize(blocks):
        if not blocks:
            return []
            
        selected_set = set(blocks)
        selected_list = list(blocks)
        
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b in selected_list:
            for s in block_to_tsubs[b]:
                cover_count[s] += 1
        
        def is_redundant(b_idx):
            for s in block_to_tsubs[b_idx]:
                if cover_count[s] == 1:
                    return False
            return True
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Step A: Redundancy Removal
            redundant_blocks = [b for b in selected_list if is_redundant(b)]
            if redundant_blocks:
                for b in redundant_blocks:
                    selected_list.remove(b)
                    selected_set.remove(b)
                    for s in block_to_tsubs[b]:
                        cover_count[s] -= 1
                improved = True
                continue
            
            # Step B: 1-for-1 Swap
            unique_tsubs_map = {}
            for b in selected_list:
                u_list = [s for s in block_to_tsubs[b] if cover_count[s] == 1]
                unique_tsubs_map[b] = u_list
            
            swap_made = False
            for b_out in selected_list:
                if time.time() > deadline:
                    return selected_list
                s_uniq = unique_tsubs_map[b_out]
                if not s_uniq:
                    continue
                
                # Find replacement
                # Candidates must cover all s in s_uniq
                # Intersect candidate lists for s in s_uniq
                candidate_sets = [set(tsub_to_blocks[s]) for s in s_uniq]
                if not candidate_sets:
                    continue
                common_candidates = set.intersection(*candidate_sets)
                # Exclude current selected blocks
                common_candidates -= selected_set
                
                if common_candidates:
                    b_in = random.choice(list(common_candidates))
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
                    improved = True
                    break # Restart loop to recalculate uniques
            
            if swap_made:
                continue

            # Step C: 2-for-1 Swap
            # This is more expensive. Only try if time permits.
            if len(selected_list) > 2:
                # Randomize order of pairs to avoid bias
                indices = list(range(len(selected_list)))
                random.shuffle(indices)
                
                pair_found = False
                for i_idx in range(len(indices)):
                    if time.time() > deadline:
                        return selected_list
                    b1 = selected_list[indices[i_idx]]
                    u1 = [s for s in block_to_tsubs[b1] if cover_count[s] == 1]
                    
                    for j_idx in range(i_idx + 1, len(indices)):
                        b2 = selected_list[indices[j_idx]]
                        u2 = [s for s in block_to_tsubs[b2] if cover_count[s] == 1]
                        
                        # The new block must cover all unique subsets of b1 and b2
                        # Note: If a subset is unique to b1, it's not covered by b2 (since b2 is also selected, if it covered it, count would be >=2? 
                        # Wait. cover_count[s] == 1 means only ONE selected block covers s.
                        # If s is in u1, only b1 covers it.
                        # If s is in u2, only b2 covers it.
                        # The union u1 U u2 must be covered by the new block.
                        
                        # Optimization: If u1 and u2 are large, intersection will be empty quickly.
                        # We need a block B_new such that u1 U u2 subset of Cover(B_new).
                        
                        # We can check if there's a block covering u1 and u2.
                        # Intersect candidates for u1 and u2.
                        # To speed up, start with the smaller list.
                        
                        all_uniq = u1 + u2
                        if not all_uniq:
                            continue
                            
                        # Sort all_uniq by frequency of covering blocks to minimize intersection size
                        # But we don't have frequency sorted. 
                        # Let's just intersect.
                        
                        # Get candidates for the first subset
                        first_s = all_uniq[0]
                        candidates = set(tsub_to_blocks[first_s])
                        
                        # Intersect with candidates for other subsets
                        for s in all_uniq[1:]:
                            candidates &= set(tsub_to_blocks[s])
                            if not candidates:
                                break
                            
                        if candidates:
                            candidates -= selected_set
                            if candidates:
                                b_in = random.choice(list(candidates))
                                # Perform 2-for-1 swap
                                selected_list.remove(b1)
                                selected_list.remove(b2)
                                selected_set.remove(b1)
                                selected_set.remove(b2)
                                selected_list.append(b_in)
                                selected_set.add(b_in)
                                
                                for s in block_to_tsubs[b1]:
                                    cover_count[s] -= 1
                                for s in block_to_tsubs[b2]:
                                    cover_count[s] -= 1
                                for s in block_to_tsubs[b_in]:
                                    cover_count[s] += 1
                                
                                pair_found = True
                                improved = True
                                break
                    if pair_found:
                        break
        
        return selected_list

    best_blocks = None
    best_size = float('inf')
    
    max_restarts = 8
    for r in range(max_restarts):
        if time.time() > deadline - 5:
            break
        
        blocks = greedy_construction(randomize_ties=(r > 0), random_seed=r*100)
        if blocks is None:
            continue
        
        optimized = local_optimize(blocks)
        
        if len(optimized) < best_size:
            best_size = len(optimized)
            best_blocks = optimized

    if best_blocks is None:
        blocks = greedy_construction(randomize_ties=False)
        if blocks is not None:
            best_blocks = local_optimize(blocks)
        else:
            return []
    
    return [all_k_subsets[i] for i in best_blocks]

if __name__ == "__main__":
    v, k, t = 19, 7, 4
    t0 = time.time()
    result = construct(v, k, t)
    t1 = time.time()
    print(f"C({v},{k},{t}): {len(result)} blocks, time {t1 - t0:.1f}s")
