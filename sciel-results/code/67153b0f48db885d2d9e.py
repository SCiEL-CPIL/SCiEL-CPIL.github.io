import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 138.0  # Safety margin before 150s kill

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

    def greedy_cover(seed, strategy="gain"):
        """
        Greedy set cover with different strategies.
        strategy: "gain" = max total uncovered covered, "hardest" = prioritize rare subsets
        """
        rng = random.Random(seed)
        uncovered = np.ones(n_t, dtype=bool)
        
        # Initial gains: number of t-subsets covered by each block
        gains = np.array([len(k_sub_t_idxs[i]) for i in range(n_k)], dtype=np.int32)
        
        # For "hardest" strategy: track how many blocks cover each t-subset
        t_cover_count = np.zeros(n_t, dtype=np.int32)
        for i in range(n_k):
            for t_idx in k_sub_t_idxs[i]:
                t_cover_count[t_idx] += 1
        
        selected = []
        
        for _ in range(n_k):  # Safety bound
            if not uncovered.any():
                break
            
            if strategy == "gain":
                scores = gains.astype(np.float64).copy()
                if seed > 0:
                    scores += rng.random() * 0.01
                best_idx = int(np.argmax(scores))
                if scores[best_idx] < 0.5:
                    break
            else:  # "hardest" strategy
                # Find the uncovered t-subset with minimum coverage (hardest to cover)
                # Then pick the block that covers it with max additional gain
                unc_idx = np.where(uncovered)[0]
                if len(unc_idx) == 0:
                    break
                # Among uncovered, find those with fewest covering blocks
                min_cov = np.min(t_cover_count[unc_idx])
                # Get all t-subsets with this minimum coverage among uncovered
                hard_ts = [i for i in unc_idx if t_cover_count[i] == min_cov]
                # Pick one randomly from hard ones
                target_t = rng.choice(hard_ts)
                # Find blocks covering target_t, pick the one with max gain
                candidates = [b for b in t_sub_to_k_idxs[target_t] if gains[b] > 0]
                if not candidates:
                    break
                best_idx = max(candidates, key=lambda b: gains[b] + rng.random() * 0.01)
            
            selected.append(best_idx)
            
            # Update gains and t_cover_count
            for t_idx in k_sub_t_idxs[best_idx]:
                if uncovered[t_idx]:
                    uncovered[t_idx] = False
                    t_cover_count[t_idx] -= 1
                    for b_idx in t_sub_to_k_idxs[t_idx]:
                        gains[b_idx] -= 1
                        
        return selected

    def reduce_blocks(blocks_idx):
        """Remove redundant blocks via iterative pass using vectorized operations."""
        if not blocks_idx:
            return blocks_idx

        # Build cover count
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in blocks_idx:
            for t_idx in k_sub_t_idxs[b_idx]:
                cover_count[t_idx] += 1

        changed = True
        while changed:
            changed = False
            new_blocks = []
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

    def local_search_swap(initial_blocks, max_time):
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
        
        if np.any(cover_count == 0):
            return current_blocks

        best_blocks = list(current_blocks)
        best_len = len(current_blocks)
        
        rng = random.Random(123)
        no_improve_count = 0
        max_no_improve = 800
        
        while time.time() < max_time and no_improve_count < max_no_improve:
            move_type = rng.random()
            
            if move_type < 0.35:
                # Try removal
                if len(current_blocks) == 0:
                    no_improve_count += 1
                    continue
                i = rng.randint(0, len(current_blocks) - 1)
                b_idx = current_blocks[i]
                
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

            elif move_type < 0.85:
                # Try swap
                if len(current_blocks) == 0:
                    no_improve_count += 1
                    continue
                i = rng.randint(0, len(current_blocks) - 1)
                old_b = current_blocks[i]
                
                unique_ts = []
                for t_idx in k_sub_t_idxs[old_b]:
                    if cover_count[t_idx] == 1:
                        unique_ts.append(t_idx)
                
                if unique_ts:
                    candidates = set(t_sub_to_k_idxs[unique_ts[0]])
                    for t_idx in unique_ts[1:]:
                        candidates.intersection_update(t_sub_to_k_idxs[t_idx])
                    candidates.discard(old_b)
                    candidates -= set(current_blocks)
                else:
                    candidates = set()
                    for _ in range(15):
                        c = rng.randint(0, n_k - 1)
                        if c not in current_blocks:
                            candidates.add(c)
                            break
                
                if not candidates:
                    no_improve_count += 1
                    continue
                
                new_b = rng.choice(list(candidates))
                
                for t_idx in k_sub_t_idxs[old_b]:
                    cover_count[t_idx] -= 1
                for t_idx in k_sub_t_idxs[new_b]:
                    cover_count[t_idx] += 1
                
                current_blocks[i] = new_b
                no_improve_count = 0
                
                # Aggressive reduction after swap
                changed = True
                limit = 8
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

            else:
                no_improve_count += 1

        return best_blocks

    # Run initial greedy with "gain" strategy
    blocks_idx = greedy_cover(0, "gain")
    blocks_idx = reduce_blocks(blocks_idx)
    
    best_blocks = blocks_idx
    best_len = len(blocks_idx)
    
    # Try multiple strategies and seeds if time allows
    remaining_time = deadline - time.time()
    if remaining_time > 15.0:
        # Try "hardest" strategy
        b_idx = greedy_cover(42, "hardest")
        b_idx = reduce_blocks(b_idx)
        if b_idx and len(b_idx) < best_len:
            best_blocks = b_idx
            best_len = len(b_idx)
        
        # Try randomized greedy with different seeds
        remaining_time = deadline - time.time()
        if remaining_time > 10.0:
            seeds = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23]
            for seed in seeds:
                if time.time() >= deadline - 8.0:
                    break
                b_idx = greedy_cover(seed, "gain")
                b_idx = reduce_blocks(b_idx)
                if b_idx and len(b_idx) < best_len:
                    best_blocks = b_idx
                    best_len = len(b_idx)
        
        # Try "hardest" with different seeds
        remaining_time = deadline - time.time()
        if remaining_time > 8.0:
            seeds = [10, 20, 30, 50, 70]
            for seed in seeds:
                if time.time() >= deadline - 6.0:
                    break
                b_idx = greedy_cover(seed, "hardest")
                b_idx = reduce_blocks(b_idx)
                if b_idx and len(b_idx) < best_len:
                    best_blocks = b_idx
                    best_len = len(b_idx)

    # Run Local Search on the best solution found
    if best_blocks and time.time() < deadline - 4.0:
        ls_result = local_search_swap(best_blocks, deadline - 4.0)
        if len(ls_result) < best_len:
            best_blocks = ls_result
            best_len = len(best_blocks)

    # Final reduction pass
    if best_blocks and time.time() < deadline - 1.0:
        best_blocks = reduce_blocks(best_blocks)

    if best_blocks is None:
        return []

    return [list(all_k_subs[b_idx]) for b_idx in best_blocks]
