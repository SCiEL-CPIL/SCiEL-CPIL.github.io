import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    time_limit = 145.0

    # Fallback for non-standard parameters
    if v != 19 or k != 7 or t != 4:
        return _greedy_generic(v, k, t)

    # --- Precomputation ---
    # 1. Map 4-subsets to IDs
    map4 = {}
    id_counter = 0
    for comb in itertools.combinations(range(v), t):
        map4[comb] = id_counter
        id_counter += 1
    n_4s = id_counter  # 3876

    # 2. For each 7-subset (block), get list of 4-subset IDs it covers
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)  # 50388
    block_4s = []
    for block in all_blocks:
        ids = []
        for sub in itertools.combinations(block, t):
            ids.append(map4[sub])
        block_4s.append(ids)
    
    # Stack into a single numpy array for vectorized operations
    block_4s_array = np.array(block_4s, dtype=np.int32)  # shape (n_blocks, 35)
    
    # 3. For each 4-subset, get list of block indices covering it
    sub_blocks = [[] for _ in range(n_4s)]
    for b_idx in range(n_blocks):
        for sid in block_4s_array[b_idx]:
            sub_blocks[sid].append(b_idx)

    # --- Helper: Greedy construction with randomization ---
    def greedy_construct(randomize_top_k=1):
        """Greedy set cover with optional randomization among top-k candidates."""
        counts = np.zeros(n_4s, dtype=np.int32)
        current_blocks = []
        uncovered_mask = np.ones(n_4s, dtype=bool)
        
        while np.any(uncovered_mask):
            if time.time() - start_time > time_limit - 15:
                break
            scores = uncovered_mask[block_4s_array].sum(axis=1)
            
            if randomize_top_k > 1 and np.any(scores > 0):
                max_score = scores.max()
                top_candidates = np.where(scores == max_score)[0]
                if len(top_candidates) > 1:
                    k_pick = min(randomize_top_k, len(top_candidates))
                    chosen = random.sample(top_candidates.tolist(), k_pick)
                    best_idx = random.choice(chosen)
                else:
                    best_idx = top_candidates[0]
            else:
                best_idx = np.argmax(scores)
            
            if scores[best_idx] == 0:
                break
            current_blocks.append(best_idx)
            counts[block_4s_array[best_idx]] += 1
            uncovered_mask = (counts == 0)
        
        return current_blocks, counts

    # --- Helper: Local search (removal + swap) ---
    def local_search(current_blocks, counts, max_time=5.0):
        """Perform local search: remove redundant blocks, then try swaps."""
        if not current_blocks:
            return current_blocks, counts
        
        block_set = set(current_blocks)
        ls_start = time.time()
        improved = True
        
        while improved and time.time() - ls_start < max_time:
            improved = False
            
            # Phase 1: Remove redundant blocks (vectorized)
            if current_blocks:
                cb_arr = np.array(current_blocks, dtype=np.int32)
                # For each block, count how many of its 4-subsets have count > 1
                block_subsets = block_4s_array[cb_arr]  # (N, 35)
                counts_for_subs = counts[block_subsets]  # (N, 35)
                redundancy = (counts_for_subs > 1).sum(axis=1)  # (N,)
                
                # Try removing blocks in order of highest redundancy
                order = np.argsort(redundancy)[::-1]
                for idx in order:
                    if time.time() - ls_start > max_time:
                        break
                    b_idx = current_blocks[idx]
                    # Try removing
                    counts[block_4s_array[b_idx]] -= 1
                    if np.any(counts == 0):
                        counts[block_4s_array[b_idx]] += 1
                    else:
                        # Successful removal
                        current_blocks.remove(b_idx)
                        block_set.discard(b_idx)
                        improved = True
                        break  # Restart pass since counts changed
            
            # Phase 2: Try swaps if no removals happened
            if not improved and current_blocks:
                # Pick a random block to try removing
                b_old = random.choice(current_blocks)
                counts[block_4s_array[b_old]] -= 1
                uncovered_now = (counts == 0)
                
                if not np.any(uncovered_now):
                    # b_old was redundant, remove it
                    current_blocks.remove(b_old)
                    block_set.discard(b_old)
                    improved = True
                else:
                    # Need to find a replacement
                    unc_idx = np.where(uncovered_now)[0]
                    if len(unc_idx) > 0:
                        # Find candidates that cover uncovered subsets
                        candidates = set()
                        for uid in unc_idx:
                            candidates.update(sub_blocks[uid])
                        candidates -= block_set
                        
                        if candidates:
                            # Find candidate covering most uncovered
                            best_cand = -1
                            best_cov = 0
                            # Sample if too many candidates
                            cand_list = list(candidates)
                            if len(cand_list) > 200:
                                cand_list = random.sample(cand_list, 200)
                            
                            for cand in cand_list:
                                cov = np.sum(uncovered_now[block_4s_array[cand]])
                                if cov > best_cov:
                                    best_cov = cov
                                    best_cand = cand
                            
                            if best_cand != -1 and best_cov > 0:
                                # Perform swap
                                current_blocks.remove(b_old)
                                block_set.discard(b_old)
                                current_blocks.append(best_cand)
                                block_set.add(best_cand)
                                counts[block_4s_array[best_cand]] += 1
                                # Check validity
                                if np.any(counts == 0):
                                    # Revert
                                    counts[block_4s_array[best_cand]] -= 1
                                    current_blocks.remove(best_cand)
                                    block_set.discard(best_cand)
                                    current_blocks.append(b_old)
                                    block_set.add(b_old)
                                    counts[block_4s_array[b_old]] += 1
                                else:
                                    improved = True
                    # Revert b_old if no swap
                    if not improved:
                        counts[block_4s_array[b_old]] += 1
        
        return current_blocks, counts

    # --- Helper: Repair to ensure validity ---
    def repair(current_blocks, counts):
        """Add blocks greedily until all 4-subsets are covered."""
        block_set = set(current_blocks)
        uncovered_mask = (counts == 0)
        while np.any(uncovered_mask):
            if time.time() - start_time > time_limit - 5:
                break
            scores = uncovered_mask[block_4s_array].sum(axis=1)
            best_idx = np.argmax(scores)
            if scores[best_idx] == 0:
                break
            if best_idx not in block_set:
                current_blocks.append(best_idx)
                block_set.add(best_idx)
                counts[block_4s_array[best_idx]] += 1
                uncovered_mask = (counts == 0)
            else:
                break
        return current_blocks, counts

    # --- Main: Multiple restarts with different randomization levels ---
    best_blocks = None
    best_len = float('inf')
    
    # Try several configurations of greedy + local search
    randomize_levels = [1, 2, 5, 10]
    
    for rlevel in randomize_levels:
        if time.time() - start_time > time_limit - 10:
            break
        
        # Run greedy with this randomization level
        current_blocks, counts = greedy_construct(randomize_top_k=rlevel)
        
        if not current_blocks:
            continue
        
        # Run local search
        current_blocks, counts = local_search(current_blocks, counts, max_time=8.0)
        
        # Repair if needed
        current_blocks, counts = repair(current_blocks, counts)
        
        if len(current_blocks) < best_len:
            best_len = len(current_blocks)
            best_blocks = list(current_blocks)
        
        # If we've hit the theoretical lower bound, stop
        if best_len <= 131:
            break

    # Final safety: if best_blocks is None or invalid, do one more greedy
    if best_blocks is None or len(best_blocks) == 0:
        current_blocks, counts = greedy_construct(randomize_top_k=1)
        current_blocks, counts = local_search(current_blocks, counts, max_time=5.0)
        current_blocks, counts = repair(current_blocks, counts)
        best_blocks = current_blocks

    # Final verification and repair
    final_counts = np.zeros(n_4s, dtype=np.int32)
    for b_idx in best_blocks:
        final_counts[block_4s_array[b_idx]] += 1
    
    if np.any(final_counts == 0):
        best_blocks, final_counts = repair(best_blocks, final_counts)

    result = [list(all_blocks[i]) for i in best_blocks]
    return result

def _greedy_generic(v, k, t):
    # Fallback for non-19,7,4
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        block = list(min(uncovered))
        while len(block) < k:
            def gain(x):
                return sum(1 for rest in itertools.combinations(block, t - 1)
                           if tuple(sorted(rest + (x,))) in uncovered)
            block.append(max((x for x in range(v) if x not in block), key=gain))
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    return blocks
