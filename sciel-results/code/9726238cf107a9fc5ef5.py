import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

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
            idx = t_sub_to_idx[t_sub]
            t_idxs.append(idx)
        k_sub_t_idxs.append(t_idxs)

    # Build reverse mapping: for each t-subset, which k-subsets cover it
    t_sub_to_k_idxs = [[] for _ in range(n_t)]
    for i, k_sub in enumerate(all_k_subs):
        for t_idx in k_sub_t_idxs[i]:
            t_sub_to_k_idxs[t_idx].append(i)

    # Precompute block sizes (number of t-subsets each block covers)
    block_sizes = np.array([len(t_idxs) for t_idxs in k_sub_t_idxs], dtype=np.int32)

    def greedy_cover(seed, strategy='standard'):
        """Greedy set cover with different strategies."""
        rng = random.Random(seed)
        # Initialize uncovered mask using bytearray for speed
        uncovered = bytearray([1] * n_t)
        blocks_idx = []
        used_blocks = set()
        
        while any(uncovered):
            if time.time() >= deadline - 10.0:
                break
                
            if strategy == 'most_constrained':
                # Find uncovered t-subset with fewest candidates
                best_s_idx = -1
                min_candidates = float('inf')
                # Sample random uncovered t-subsets
                for _ in range(30):
                    while True:
                        s_idx = rng.randint(0, n_t - 1)
                        if uncovered[s_idx]:
                            break
                    candidates = [b for b in t_sub_to_k_idxs[s_idx] if b not in used_blocks]
                    if len(candidates) < min_candidates:
                        min_candidates = len(candidates)
                        best_s_idx = s_idx
                    if min_candidates <= 1:
                        break
                s_idx = best_s_idx
            elif strategy == 'random':
                # Pick a random uncovered t-subset
                while True:
                    s_idx = rng.randint(0, n_t - 1)
                    if uncovered[s_idx]:
                        break
            else:
                # Standard: pick first uncovered
                s_idx = -1
                for i in range(n_t):
                    if uncovered[i]:
                        s_idx = i
                        break
                if s_idx == -1:
                    break

            if s_idx < 0:
                break

            candidates = t_sub_to_k_idxs[s_idx]
            best_block_idx = -1
            best_gain = -1

            for b_idx in candidates:
                if b_idx in used_blocks:
                    continue
                # Calculate gain: number of uncovered t-subsets this block covers
                gain = 0
                for t_idx in k_sub_t_idxs[b_idx]:
                    if uncovered[t_idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx

            if best_block_idx == -1:
                break

            blocks_idx.append(best_block_idx)
            used_blocks.add(best_block_idx)
            
            # Update uncovered
            for t_idx in k_sub_t_idxs[best_block_idx]:
                uncovered[t_idx] = 0

        return blocks_idx

    def reduce_blocks(blocks_idx, max_passes=5):
        """Remove redundant blocks via iterative passes."""
        if not blocks_idx:
            return blocks_idx

        blocks_idx = list(blocks_idx)
        
        for _ in range(max_passes):
            if time.time() >= deadline - 5.0:
                break
            # Build cover count
            cover_count = np.zeros(n_t, dtype=np.int32)
            for b_idx in blocks_idx:
                cover_count[k_sub_t_idxs[b_idx]] += 1

            changed = False
            new_blocks = []
            for b_idx in blocks_idx:
                # Check if any t-subset in this block has cover_count == 1
                counts_for_block = cover_count[k_sub_t_idxs[b_idx]]
                if np.any(counts_for_block == 1):
                    new_blocks.append(b_idx)
                else:
                    changed = True
                    cover_count[k_sub_t_idxs[b_idx]] -= 1
            
            blocks_idx = new_blocks
            if not changed:
                break

        return blocks_idx

    def local_search(blocks_idx, max_iter=50):
        """
        Local search with delete, swap, and add operations.
        """
        if not blocks_idx:
            return blocks_idx
        
        blocks_idx = list(blocks_idx)
        
        for _ in range(max_iter):
            if time.time() >= deadline - 5.0:
                break
            
            # Rebuild cover count
            cover_count = np.zeros(n_t, dtype=np.int32)
            for b_idx in blocks_idx:
                cover_count[k_sub_t_idxs[b_idx]] += 1
            
            improved = False
            
            # Try deletions
            for i in range(len(blocks_idx)):
                b_idx = blocks_idx[i]
                counts_for_block = cover_count[k_sub_t_idxs[b_idx]]
                if not np.any(counts_for_block == 1):
                    # Safe to delete
                    cover_count[k_sub_t_idxs[b_idx]] -= 1
                    blocks_idx.pop(i)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swaps
            for i, b_idx in enumerate(blocks_idx):
                # Find essential t-subsets (covered only by this block)
                essential = []
                for t_idx in k_sub_t_idxs[b_idx]:
                    if cover_count[t_idx] == 1:
                        essential.append(t_idx)
                
                if not essential:
                    continue
                
                # Try to find a replacement block that covers all essential t-subsets
                candidates = set()
                for t_idx in essential:
                    candidates.update(t_sub_to_k_idxs[t_idx])
                candidates.discard(b_idx)
                
                if not candidates:
                    continue
                
                # Limit candidates
                if len(candidates) > 50:
                    candidates = set(random.Random(i).sample(list(candidates), 50))
                
                best_replace = -1
                best_gain = -1
                
                for cand_idx in candidates:
                    # Check if cand covers all essential t-subsets
                    cand_t_set = set(k_sub_t_idxs[cand_idx])
                    if not all(t in cand_t_set for t in essential):
                        continue
                    
                    # Calculate additional coverage
                    gain = 0
                    for t_idx in k_sub_t_idxs[cand_idx]:
                        if cover_count[t_idx] == 0:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_replace = cand_idx
                
                if best_replace != -1 and best_replace not in blocks_idx:
                    # Perform swap
                    cover_count[k_sub_t_idxs[b_idx]] -= 1
                    cover_count[k_sub_t_idxs[best_replace]] += 1
                    blocks_idx[i] = best_replace
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try adds: find uncovered t-subsets and add blocks to cover them
            uncovered = np.where(cover_count == 0)[0]
            if len(uncovered) > 0 and len(blocks_idx) < n_t:
                # Pick a random uncovered t-subset
                s_idx = random.Random(time.time()).randint(0, len(uncovered) - 1)
                s_idx = uncovered[s_idx]
                
                candidates = t_sub_to_k_idxs[s_idx]
                best_block_idx = -1
                best_gain = -1
                
                for b_idx in candidates:
                    if b_idx in blocks_idx:
                        continue
                    gain = 0
                    for t_idx in k_sub_t_idxs[b_idx]:
                        if cover_count[t_idx] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_block_idx = b_idx
                
                if best_block_idx != -1:
                    blocks_idx.append(best_block_idx)
                    cover_count[k_sub_t_idxs[best_block_idx]] += 1
                    improved = True
            
            if not improved:
                break
        
        return blocks_idx

    best_blocks = None
    best_len = float('inf')

    # 1. Deterministic Greedy (standard)
    blocks_idx = greedy_cover(0, 'standard')
    blocks_idx = reduce_blocks(blocks_idx)
    if blocks_idx:
        best_blocks = blocks_idx
        best_len = len(blocks_idx)

    # 2. Most constrained greedy
    blocks_idx = greedy_cover(1, 'most_constrained')
    blocks_idx = reduce_blocks(blocks_idx)
    if blocks_idx and len(blocks_idx) < best_len:
        best_blocks = blocks_idx
        best_len = len(blocks_idx)

    # 3. Randomized Greedy Restarts
    strategies = ['random', 'most_constrained', 'standard']
    for seed in range(2, 150):
        if time.time() >= deadline - 20.0:
            break
        strategy = strategies[seed % 3]
        blocks_idx = greedy_cover(seed, strategy)
        blocks_idx = reduce_blocks(blocks_idx)
        if blocks_idx and len(blocks_idx) < best_len:
            best_blocks = blocks_idx
            best_len = len(blocks_idx)

    # 4. Local search on best solution
    if best_blocks and time.time() < deadline - 10.0:
        improved_blocks = local_search(best_blocks)
        improved_blocks = reduce_blocks(improved_blocks)
        if improved_blocks and len(improved_blocks) < best_len:
            best_blocks = improved_blocks
            best_len = len(improved_blocks)

    # 5. Final reduction pass
    if best_blocks and time.time() < deadline - 3.0:
        best_blocks = reduce_blocks(best_blocks)

    # Convert to actual blocks
    if best_blocks is None:
        return []

    return [list(all_k_subs[b_idx]) for b_idx in best_blocks]
