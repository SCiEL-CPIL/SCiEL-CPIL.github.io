import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # Precompute all k-subsets
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subs)

    # For each k-subset, compute the list of t-subset indices it covers
    k_sub_t_idxs = []
    for k_sub in all_k_subs:
        t_idxs = []
        for t_sub in itertools.combinations(k_sub, t):
            t_idxs.append(t_sub_to_idx[t_sub])
        k_sub_t_idxs.append(t_idxs)
    
    # Build reverse mapping: for each t-subset, which k-subsets cover it
    t_sub_to_k_idxs = [[] for _ in range(n_t)]
    for i, k_sub in enumerate(all_k_subs):
        for t_idx in k_sub_t_idxs[i]:
            t_sub_to_k_idxs[t_idx].append(i)

    def greedy_cover(seed):
        """
        Efficient greedy set cover using NumPy for gain updates.
        """
        rng = random.Random(seed)
        uncovered = np.ones(n_t, dtype=bool)
        
        # Initial gains: number of t-subsets covered by each block
        gains = np.array([len(k_sub_t_idxs[i]) for i in range(n_k)], dtype=np.int32)
        
        selected = []
        
        for _ in range(n_k): # Safety bound
            if not uncovered.any():
                break
            
            # Find best block
            # Add small noise to break ties deterministically based on seed
            if seed > 0:
                noise = rng.random() * 0.01
                scores = gains.astype(np.float64) + noise
            else:
                scores = gains.astype(np.float64)
                
            best_idx = int(np.argmax(scores))
            
            if gains[best_idx] < 1: # Gain is integer, so < 1 means 0
                break
                
            selected.append(best_idx)
            
            # Update gains
            for t_idx in k_sub_t_idxs[best_idx]:
                if uncovered[t_idx]:
                    uncovered[t_idx] = False
                    # Decrease gain for all blocks covering this t_subset
                    for b_idx in t_sub_to_k_idxs[t_idx]:
                        gains[b_idx] -= 1
                        
        return selected

    def reduce_blocks(blocks_idx, rng=None):
        """Remove redundant blocks via iterative pass."""
        if not blocks_idx:
            return blocks_idx
        
        if rng is None:
            rng = random.Random(42)

        # Build cover count
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in blocks_idx:
            for t_idx in k_sub_t_idxs[b_idx]:
                cover_count[t_idx] += 1

        changed = True
        while changed:
            changed = False
            new_blocks = []
            # Shuffle to remove different blocks in different runs
            if rng:
                rng.shuffle(blocks_idx)
            for b_idx in blocks_idx:
                is_essential = False
                for t_idx in k_sub_t_idxs[b_idx]:
                    if cover_count[t_idx] == 1:
                        is_essential = True
                        break
                if is_essential:
                    new_blocks.append(b_idx)
                else:
                    changed = True
                    for t_idx in k_sub_t_idxs[b_idx]:
                        cover_count[t_idx] -= 1
            blocks_idx = new_blocks

        return blocks_idx

    def local_search(initial_blocks, max_time, seed=0):
        """
        Local search focused on swaps and removals.
        """
        current_blocks = list(initial_blocks)
        if not current_blocks:
            return current_blocks

        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in current_blocks:
            for t_idx in k_sub_t_idxs[b_idx]:
                cover_count[t_idx] += 1
        
        # Ensure valid start
        if np.any(cover_count == 0):
            return current_blocks

        best_blocks = list(current_blocks)
        best_len = len(current_blocks)
        
        rng = random.Random(seed)
        no_improve_count = 0
        max_no_improve = 1000 # Increased patience
        
        while time.time() < max_time and no_improve_count < max_no_improve:
            move_type = rng.random()
            
            if move_type < 0.2:
                # Try removal
                if len(current_blocks) == 0:
                    no_improve_count += 1
                    continue
                i = rng.randint(0, len(current_blocks) - 1)
                b_idx = current_blocks[i]
                
                # Check if removable
                removable = True
                for t_idx in k_sub_t_idxs[b_idx]:
                    if cover_count[t_idx] == 1:
                        removable = False
                        break
                
                if removable:
                    for t_idx in k_sub_t_idxs[b_idx]:
                        cover_count[t_idx] -= 1
                    current_blocks.pop(i)
                    no_improve_count = 0
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best_blocks = list(current_blocks)
                else:
                    no_improve_count += 1

            elif move_type < 0.8:
                # Try swap
                if len(current_blocks) == 0:
                    no_improve_count += 1
                    continue
                i = rng.randint(0, len(current_blocks) - 1)
                old_b = current_blocks[i]
                
                # Find unique subsets
                unique_ts = []
                for t_idx in k_sub_t_idxs[old_b]:
                    if cover_count[t_idx] == 1:
                        unique_ts.append(t_idx)
                
                if unique_ts:
                    # Must cover all unique_ts
                    # Find intersection of candidates
                    candidates = set(t_sub_to_k_idxs[unique_ts[0]])
                    for t_idx in unique_ts[1:]:
                        candidates.intersection_update(t_sub_to_k_idxs[t_idx])
                    candidates.discard(old_b)
                    candidates -= set(current_blocks)
                else:
                    # No unique subsets, so old_b is redundant?
                    # If so, it should have been removed.
                    # But maybe due to race condition or logic, we just try to swap with any block
                    # to perturb the state.
                    candidates = set()
                    for _ in range(10):
                        c = rng.randint(0, n_k - 1)
                        if c not in current_blocks:
                            candidates.add(c)
                            break
                
                if not candidates:
                    no_improve_count += 1
                    continue
                
                new_b = rng.choice(list(candidates))
                
                # Perform swap
                for t_idx in k_sub_t_idxs[old_b]:
                    cover_count[t_idx] -= 1
                for t_idx in k_sub_t_idxs[new_b]:
                    cover_count[t_idx] += 1
                
                current_blocks[i] = new_b
                no_improve_count = 0
                
                # Quick reduction pass after swap
                changed = True
                limit = 3
                while changed and limit > 0:
                    changed = False
                    limit -= 1
                    rng.shuffle(current_blocks)
                    for j in range(len(current_blocks)):
                        b_idx = current_blocks[j]
                        is_essential = False
                        for t_idx in k_sub_t_idxs[b_idx]:
                            if cover_count[t_idx] == 1:
                                is_essential = True
                                break
                        if not is_essential:
                            for t_idx in k_sub_t_idxs[b_idx]:
                                cover_count[t_idx] -= 1
                            current_blocks.pop(j)
                            changed = True
                            if len(current_blocks) < best_len:
                                best_len = len(current_blocks)
                                best_blocks = list(current_blocks)
                            break # Restart loop as indices changed

            else:
                # Kick: Add a random block, then reduce
                if len(current_blocks) < n_k:
                    # Pick a random block not in current
                    new_b = rng.randint(0, n_k - 1)
                    if new_b not in current_blocks:
                        current_blocks.append(new_b)
                        for t_idx in k_sub_t_idxs[new_b]:
                            cover_count[t_idx] += 1
                        
                        # Aggressive reduction
                        changed = True
                        limit = 10
                        while changed and limit > 0:
                            changed = False
                            limit -= 1
                            rng.shuffle(current_blocks)
                            for j in range(len(current_blocks)):
                                b_idx = current_blocks[j]
                                is_essential = False
                                for t_idx in k_sub_t_idxs[b_idx]:
                                    if cover_count[t_idx] == 1:
                                        is_essential = True
                                        break
                                if not is_essential:
                                    for t_idx in k_sub_t_idxs[b_idx]:
                                        cover_count[t_idx] -= 1
                                    current_blocks.pop(j)
                                    changed = True
                                    if len(current_blocks) < best_len:
                                        best_len = len(current_blocks)
                                        best_blocks = list(current_blocks)
                                    break
                        
                        if len(current_blocks) < best_len:
                            no_improve_count = 0
                        else:
                            # Revert kick if it didn't help
                            # This is complex to revert exactly, so we just accept the state
                            # and continue. If it's worse, we rely on best_blocks.
                            no_improve_count += 1
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1

        return best_blocks

    # Run initial greedy
    blocks_idx = greedy_cover(0)
    blocks_idx = reduce_blocks(blocks_idx)
    
    best_blocks = blocks_idx
    best_len = len(blocks_idx)
    
    # Try randomized greedy if time allows
    remaining_time = deadline - time.time()
    if remaining_time > 30.0:
        seeds = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67]
        for seed in seeds:
            if time.time() >= deadline - 20.0:
                break
            b_idx = greedy_cover(seed)
            b_idx = reduce_blocks(b_idx, random.Random(seed))
            if b_idx and len(b_idx) < best_len:
                best_blocks = b_idx
                best_len = len(b_idx)

    # Run Local Search on the best solution found
    if best_blocks and time.time() < deadline - 10.0:
        # Run local search with different seeds
        ls_seeds = [100, 200, 300, 400, 500]
        for ls_seed in ls_seeds:
            if time.time() >= deadline - 5.0:
                break
            ls_result = local_search(best_blocks, deadline - 5.0, seed=ls_seed)
            if len(ls_result) < best_len:
                best_blocks = ls_result
                best_len = len(best_blocks)

    # Final reduction pass
    if best_blocks and time.time() < deadline - 2.0:
        best_blocks = reduce_blocks(best_blocks)

    # Convert to actual blocks
    if best_blocks is None:
        return []

    return [list(all_k_subs[b_idx]) for b_idx in best_blocks]
