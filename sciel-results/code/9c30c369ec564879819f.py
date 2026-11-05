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
        Local search focused on swaps and removals with improved candidate selection.
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
        max_no_improve = 1200
        
        # Precompute block membership sets for faster lookup
        current_set = set(current_blocks)
        
        while time.time() < max_time and no_improve_count < max_no_improve:
            move_type = rng.random()
            
            if move_type < 0.40:
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
                    current_set.discard(b_idx)
                    no_improve_count = 0
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best_blocks = list(current_blocks)
                else:
                    no_improve_count += 1

            elif move_type < 0.90:
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
                    # Find blocks that cover ALL unique t-subsets
                    candidates = set(t_sub_to_k_idxs[unique_ts[0]])
                    for t_idx in unique_ts[1:]:
                        candidates.intersection_update(t_sub_to_k_idxs[t_idx])
                    candidates.discard(old_b)
                    candidates -= current_set
                    
                    # If too many candidates, sample
                    if len(candidates) > 20:
                        candidates = set(rng.sample(list(candidates), 20))
                else:
                    # No unique constraints - try to find a block that improves coverage
                    # Sample random blocks not in current set
                    candidates = set()
                    for _ in range(30):
                        c = rng.randint(0, n_k - 1)
                        if c not in current_set:
                            candidates.add(c)
                
                if not candidates:
                    no_improve_count += 1
                    continue
                
                # Score candidates by gain
                scored = []
                for new_b in list(candidates):
                    gain = 0
                    for t_idx in k_sub_t_idxs[new_b]:
                        if cover_count[t_idx] == 0:
                            gain += 1
                    scored.append((gain, new_b))
                
                # Pick best or random from top
                scored.sort(reverse=True)
                if rng.random() < 0.7:
                    new_b = scored[0][1]
                else:
                    top_n = min(5, len(scored))
                    new_b = scored[rng.randint(0, top_n - 1)][1]
                
                # Perform swap
                for t_idx in k_sub_t_idxs[old_b]:
                    cover_count[t_idx] -= 1
                for t_idx in k_sub_t_idxs[new_b]:
                    cover_count[t_idx] += 1
                
                current_blocks[i] = new_b
                current_set.discard(old_b)
                current_set.add(new_b)
                no_improve_count = 0
                
                # Aggressive reduction after swap
                changed = True
                limit = 12
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
                            current_set.discard(b_idx)
                            changed = True
                            if len(current_blocks) < best_len:
                                best_len = len(current_blocks)
                                best_blocks = list(current_blocks)
                            break

            else:
                no_improve_count += 1

        return best_blocks

    def relaxation_based_init(seed, n_iter=50):
        """
        Generate initial solution from a continuous relaxation perspective.
        Uses soft-thresholding on a weighted scoring to create diverse starting points.
        """
        rng = random.Random(seed)
        
        # Create a random weighting scheme for blocks
        # Weight by how "rare" the t-subsets they cover are
        t_rarity = np.zeros(n_t, dtype=np.float64)
        for t_idx in range(n_t):
            t_rarity[t_idx] = 1.0 / len(t_sub_to_k_idxs[t_idx])
        
        # Random perturbation of weights
        weights = np.array([
            sum(t_rarity[t_idx] for t_idx in k_sub_t_idxs[i]) 
            for i in range(n_k)
        ], dtype=np.float64)
        
        # Add noise for diversity
        weights += rng.random() * 0.1
        
        # Soft-threshold: select blocks above a dynamic threshold
        # Start with a high threshold and lower it
        selected = []
        uncovered = np.ones(n_t, dtype=bool)
        gains = np.array([len(k_sub_t_idxs[i]) for i in range(n_k)], dtype=np.int32)
        
        # Sort by weight/gain ratio
        ratios = weights / (gains + 1)
        order = np.argsort(-ratios)
        
        threshold = np.max(ratios) * 0.8
        min_threshold = np.max(ratios) * 0.1
        
        for _ in range(n_iter):
            # Find blocks above threshold with positive gain
            candidates = [i for i in order if ratios[i] >= threshold and gains[i] > 0]
            
            if not candidates:
                threshold = max(min_threshold, threshold * 0.9)
                continue
            
            # Pick the best candidate
            best_idx = max(candidates, key=lambda i: gains[i] + rng.random() * 0.5)
            
            if best_idx in selected:
                threshold = max(min_threshold, threshold * 0.95)
                continue
            
            selected.append(best_idx)
            
            # Update
            for t_idx in k_sub_t_idxs[best_idx]:
                if uncovered[t_idx]:
                    uncovered[t_idx] = False
                    for b_idx in t_sub_to_k_idxs[t_idx]:
                        gains[b_idx] -= 1
            
            if not uncovered.any():
                break
            
            # Gradually lower threshold
            threshold = max(min_threshold, threshold * 0.98)
        
        # Fill remaining uncovered with greedy
        while uncovered.any():
            # Find block with max gain
            best_idx = -1
            best_gain = 0
            for i in range(n_k):
                if i in selected:
                    continue
                g = gains[i]
                if g > best_gain:
                    best_gain = g
                    best_idx = i
            
            if best_idx == -1 or best_gain == 0:
                break
            
            selected.append(best_idx)
            for t_idx in k_sub_t_idxs[best_idx]:
                if uncovered[t_idx]:
                    uncovered[t_idx] = False
                    for b_idx in t_sub_to_k_idxs[t_idx]:
                        gains[b_idx] -= 1
        
        return selected

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
        
        # Try relaxation-based initialization
        remaining_time = deadline - time.time()
        if remaining_time > 10.0:
            for seed in [100, 200, 300]:
                if time.time() >= deadline - 6.0:
                    break
                b_idx = relaxation_based_init(seed)
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
    
    # Run another local search pass if time allows
    if best_blocks and time.time() < deadline - 2.0:
        ls_result2 = local_search_swap(best_blocks, deadline - 2.0)
        if len(ls_result2) < best_len:
            best_blocks = ls_result2
            best_len = len(best_blocks)

    # Final reduction pass
    if best_blocks and time.time() < deadline - 1.0:
        best_blocks = reduce_blocks(best_blocks)

    if best_blocks is None:
        return []

    return [list(all_k_subs[b_idx]) for b_idx in best_blocks]
