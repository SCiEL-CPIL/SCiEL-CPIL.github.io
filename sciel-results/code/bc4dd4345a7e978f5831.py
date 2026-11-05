import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 130.0

    # 1. Pre-computation
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k = len(all_k_blocks)

    block_t_idxs = []
    for block in all_k_blocks:
        t_idxs = []
        for sub in itertools.combinations(block, t):
            t_idxs.append(t_sub_to_idx[sub])
        block_t_idxs.append(t_idxs)

    t_sub_to_blocks = [[] for _ in range(n_t)]
    for b_idx, t_idxs in enumerate(block_t_idxs):
        for t_idx in t_idxs:
            t_sub_to_blocks[t_idx].append(b_idx)
    t_sub_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in t_sub_to_blocks]

    # 2. Greedy Construction
    gains = np.array([len(t_idxs) for t_idxs in block_t_idxs], dtype=np.int32)
    uncovered = np.ones(n_t, dtype=bool)
    selected_blocks = []

    while np.any(uncovered):
        if time.time() > deadline - 40:
            break
        best_idx = int(np.argmax(gains))
        if gains[best_idx] == 0:
            break
        selected_blocks.append(best_idx)
        newly_covered = []
        for t_idx in block_t_idxs[best_idx]:
            if uncovered[t_idx]:
                newly_covered.append(t_idx)
                uncovered[t_idx] = False
        if newly_covered:
            for t_idx in newly_covered:
                if len(t_sub_to_blocks_np[t_idx]) > 0:
                    gains[t_sub_to_blocks_np[t_idx]] -= 1

    # 3. Redundancy Removal (Initial)
    if selected_blocks and time.time() < deadline - 30:
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in selected_blocks:
            for t_idx in block_t_idxs[b_idx]:
                cover_count[t_idx] += 1

        changed = True
        while changed and time.time() < deadline - 25:
            changed = False
            new_blocks = []
            for i in range(len(selected_blocks) - 1, -1, -1):
                b_idx = selected_blocks[i]
                # Check if essential
                is_essential = False
                for t_idx in block_t_idxs[b_idx]:
                    if cover_count[t_idx] == 1:
                        is_essential = True
                        break
                if is_essential:
                    new_blocks.append(b_idx)
                else:
                    changed = True
                    for t_idx in block_t_idxs[b_idx]:
                        cover_count[t_idx] -= 1
            selected_blocks = new_blocks

    # 4. Optimized Local Search
    if selected_blocks and time.time() < deadline - 5:
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in selected_blocks:
            for t_idx in block_t_idxs[b_idx]:
                cover_count[t_idx] += 1

        is_selected = np.zeros(n_k, dtype=bool)
        for b_idx in selected_blocks:
            is_selected[b_idx] = True

        best_solution = list(selected_blocks)
        best_count = len(selected_blocks)
        selected_set = set(selected_blocks)

        search_time_limit = deadline - 5
        no_improve_count = 0
        max_no_improve = 20

        # Precompute sets for candidates? No, compute on fly.
        
        while time.time() < search_time_limit and no_improve_count < max_no_improve:
            improved = False

            # Step A: Remove redundant blocks
            # Vectorized check: is there any t in block with cover_count == 1?
            to_remove = []
            for b_idx in selected_blocks:
                # Use numpy for speed
                if not np.any(cover_count[block_t_idxs[b_idx]] == 1):
                    to_remove.append(b_idx)
            
            if to_remove:
                for b_idx in to_remove:
                    for t_idx in block_t_idxs[b_idx]:
                        cover_count[t_idx] -= 1
                    selected_blocks.remove(b_idx)
                    selected_set.discard(b_idx)
                    is_selected[b_idx] = False
                improved = True

            if not improved:
                # Step B: Add-Remove
                # Find critical t-subsets
                critical_t_idxs = np.where(cover_count == 1)[0]
                
                if len(critical_t_idxs) > 0:
                    critical_set = set(critical_t_idxs)
                    
                    # Select candidates
                    # We want candidates that cover many critical t-subsets
                    # Sample a few thousand non-selected blocks
                    candidates = [b for b in range(n_k) if not is_selected[b]]
                    if len(candidates) > 2000:
                        random.shuffle(candidates)
                        candidates = candidates[:2000]
                    
                    # Score candidates
                    scored_candidates = []
                    for cand in candidates:
                        # Count overlap with critical_set
                        # block_t_idxs[cand] is a list
                        # Intersection size
                        overlap = 0
                        # Optimization: if cand covers a critical t, it's a good candidate
                        # We can break early if overlap is high?
                        for t_idx in block_t_idxs[cand]:
                            if t_idx in critical_set:
                                overlap += 1
                                # Optional: break if overlap > some threshold?
                        if overlap > 0:
                            scored_candidates.append((overlap, cand))
                    
                    # Sort by overlap descending
                    scored_candidates.sort(key=lambda x: x[0], reverse=True)
                    
                    # Try top 10 candidates
                    best_add = None
                    best_net_gain = 0
                    
                    for overlap, cand in scored_candidates[:10]:
                        if time.time() > search_time_limit:
                            break
                        
                        cand_t_set = set(block_t_idxs[cand])
                        
                        # Check which selected blocks become removable
                        removable = []
                        for b_idx in selected_blocks:
                            # b_idx is removable if all its critical t-subsets are in cand_t_set
                            # Critical t-subsets of b_idx: {t in block_t_idxs[b_idx] if cover_count[t]==1}
                            # We can check: is there any critical t in b_idx NOT in cand_t_set?
                            is_removable = True
                            for t_idx in block_t_idxs[b_idx]:
                                if cover_count[t_idx] == 1 and t_idx not in cand_t_set:
                                    is_removable = False
                                    break
                            if is_removable:
                                removable.append(b_idx)
                        
                        net_gain = len(removable) - 1
                        if net_gain > best_net_gain:
                            best_net_gain = net_gain
                            best_add = (cand, removable)
                    
                    if best_add is not None:
                        cand, removable = best_add
                        # Apply
                        for t_idx in block_t_idxs[cand]:
                            cover_count[t_idx] += 1
                        selected_blocks.append(cand)
                        selected_set.add(cand)
                        is_selected[cand] = True

                        for b_idx in removable:
                            for t_idx in block_t_idxs[b_idx]:
                                cover_count[t_idx] -= 1
                            selected_blocks.remove(b_idx)
                            selected_set.discard(b_idx)
                            is_selected[b_idx] = False

                        improved = True

            if improved:
                no_improve_count = 0
                if len(selected_blocks) < best_count:
                    best_count = len(selected_blocks)
                    best_solution = list(selected_blocks)
            else:
                no_improve_count += 1

        selected_blocks = best_solution

        # Final deletion pass
        if time.time() < deadline - 2:
            cover_count = np.zeros(n_t, dtype=np.int32)
            for b_idx in selected_blocks:
                for t_idx in block_t_idxs[b_idx]:
                    cover_count[t_idx] += 1

            changed = True
            while changed and time.time() < deadline - 1:
                changed = False
                new_blocks = []
                for i in range(len(selected_blocks) - 1, -1, -1):
                    b_idx = selected_blocks[i]
                    is_essential = False
                    for t_idx in block_t_idxs[b_idx]:
                        if cover_count[t_idx] == 1:
                            is_essential = True
                            break
                    if is_essential:
                        new_blocks.append(b_idx)
                    else:
                        changed = True
                        for t_idx in block_t_idxs[b_idx]:
                            cover_count[t_idx] -= 1
                selected_blocks = new_blocks

    final_blocks = [list(all_k_blocks[i]) for i in selected_blocks]
    return final_blocks
