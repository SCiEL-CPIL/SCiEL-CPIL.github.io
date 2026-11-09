import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using a fast greedy approach 
    selecting the best k-superset for a random uncovered t-subset, 
    followed by redundancy removal and multi-start optimization.
    """
    deadline = time.time() + 110.0  # Safety margin

    # Precompute t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    def greedy_once(seed):
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubs, dtype=bool)
        blocks = []
        
        # Precompute for each t-subset, the number of candidate blocks that contain it
        # This is C(v-t, k-t) for all t-subsets, but we can use it for weighted sampling
        # Actually all t-subsets have the same number of candidates: C(v-t, k-t)
        # So weighted sampling by "hardness" needs a different metric.
        # Instead, let's use: weight = 1 / (number of blocks already covering it + 1)
        # But at the start, no blocks cover anything, so all weights are equal.
        # A better approach: after each block is added, update weights.
        # weight[i] = 1 / cover_count[i] for uncovered i, else 0
        
        # Precompute candidate blocks for each t-subset
        # For t-subset S, candidates are all k-subsets containing S
        # Number: C(v-t, k-t)
        # For v=19, k=8, t=5: C(14,3) = 364 candidates per t-subset
        
        # Cache for block -> tsub indices
        block_tsub_cache = {}
        
        def get_block_tsubs(block):
            if block not in block_tsub_cache:
                idxs = np.array([tsub_to_idx[comb] for comb in itertools.combinations(block, t)], dtype=np.int32)
                block_tsub_cache[block] = idxs
            return block_tsub_cache[block]
        
        # Precompute for each t-subset, the list of candidate blocks
        # This is expensive: 11628 * 364 = 4.2M blocks to store
        # Let's generate on the fly but cache the list of candidates
        tsub_candidates_cache = {}
        
        def get_tsub_candidates(S):
            if S not in tsub_candidates_cache:
                S_set = set(S)
                remaining_points = [p for p in range(v) if p not in S_set]
                cands = []
                for extra in itertools.combinations(remaining_points, k - t):
                    block = tuple(sorted(S + tuple(extra)))
                    cands.append(block)
                tsub_candidates_cache[S] = cands
            return tsub_candidates_cache[S]
        
        # Weighted sampling: maintain cover counts for efficient weight updates
        cover_count = np.zeros(n_tsubs, dtype=np.int32)
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
            
            # Pick an uncovered t-subset with probability proportional to 1/cover_count
            # (harder to cover = lower cover count = higher probability)
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            
            # Compute weights: for uncovered, weight = 1 / max(cover_count, 1)
            # This biases toward t-subsets that are barely covered
            weights = np.zeros(len(uncovered_indices), dtype=np.float64)
            for j, idx in enumerate(uncovered_indices):
                cc = cover_count[idx]
                weights[j] = 1.0 / max(cc, 1)
            
            # Normalize and sample
            total_w = weights.sum()
            if total_w == 0:
                start_idx = rng.choice(uncovered_indices)
            else:
                probs = weights / total_w
                r = rng.random()
                cumsum = np.cumsum(probs)
                start_idx = uncovered_indices[np.searchsorted(cumsum, r)]
            
            S = all_tsubs[start_idx]
            
            # Get candidate blocks
            candidates = get_tsub_candidates(S)
            
            # Evaluate gain for each candidate
            best_block = None
            best_gain = -1
            
            # Shuffle to break ties
            rng.shuffle(candidates)
            
            for B in candidates:
                idxs = get_block_tsubs(B)
                gain = np.sum(uncovered[idxs])
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = B
            
            if best_block is None:
                break
                
            blocks.append(best_block)
            
            # Update uncovered mask and cover counts
            idxs = get_block_tsubs(best_block)
            uncovered[idxs] = False
            cover_count[idxs] += 1
        
        return blocks

    def remove_redundant(blocks):
        if not blocks:
            return blocks
        
        # Precompute t-subset indices for each block
        block_indices = []
        for b in blocks:
            idxs = np.array([tsub_to_idx[comb] for comb in itertools.combinations(b, t)], dtype=np.int32)
            block_indices.append(idxs)
        
        # Iterative removal: repeatedly remove redundant blocks until no more
        current_blocks = list(blocks)
        current_indices = list(block_indices)
        
        changed = True
        while changed:
            changed = False
            if time.time() > deadline:
                break
            
            new_blocks = []
            new_indices = []
            
            # Compute cover counts
            cover_count = np.zeros(n_tsubs, dtype=np.int32)
            for idxs in current_indices:
                cover_count[idxs] += 1
            
            for i in range(len(current_blocks)):
                if time.time() > deadline:
                    break
                idxs = current_indices[i]
                # Block i is redundant if all its tsubs are covered by at least one other block
                if np.all(cover_count[idxs] > 1):
                    # Remove it
                    cover_count[idxs] -= 1
                    changed = True
                else:
                    new_blocks.append(current_blocks[i])
                    new_indices.append(idxs)
            
            if not changed:
                break
            
            current_blocks = new_blocks
            current_indices = new_indices
        
        return current_blocks

    def local_search(blocks, indices, rng, max_iterations=50):
        """
        Try to reduce block count by swapping one block for another.
        For each block, try removing it and see if the covering is still valid.
        If not, try to find a replacement block that covers the missing t-subsets.
        """
        current_blocks = list(blocks)
        current_indices = list(indices)
        
        # Compute cover counts
        cover_count = np.zeros(n_tsubs, dtype=np.int32)
        for idxs in current_indices:
            cover_count[idxs] += 1
        
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations:
            if time.time() > deadline:
                break
            improved = False
            iterations += 1
            
            # Try removing each block and see if we can compensate
            for i in range(len(current_blocks)):
                if time.time() > deadline:
                    break
                
                idxs = current_indices[i]
                
                # What t-subsets would become uncovered if we remove this block?
                missing = idxs[cover_count[idxs] == 1]
                
                if len(missing) == 0:
                    # Block is redundant, remove it
                    cover_count[idxs] -= 1
                    current_blocks.pop(i)
                    current_indices.pop(i)
                    improved = True
                    break
                
                # Try to find a single block that covers all missing t-subsets
                # and also covers some already-covered t-subsets (for potential future benefit)
                # This is a set cover subproblem: find a block that contains all missing t-subsets
                
                # A block contains a t-subset S if S is a subset of the block
                # We need a block B such that for all S in missing, S ⊆ B
                # This means B must contain the union of all missing t-subsets
                # But the union might have more than k elements, in which case no single block works
                
                missing_set = set()
                for m in missing:
                    missing_set.update(all_tsubs[m])
                
                if len(missing_set) > k:
                    continue
                
                # Try all k-subsets that contain missing_set
                missing_sorted = tuple(sorted(missing_set))
                remaining = [p for p in range(v) if p not in missing_set]
                
                best_replacement = None
                best_gain = -1
                
                for extra in itertools.combinations(remaining, k - len(missing_set)):
                    B = tuple(sorted(missing_sorted + extra))
                    B_idxs = np.array([tsub_to_idx[comb] for comb in itertools.combinations(B, t)], dtype=np.int32)
                    
                    # Check if B covers all missing
                    missing_set_idx = set(missing)
                    if not missing_set_idx.issubset(set(B_idxs)):
                        continue
                    
                    # Gain: number of t-subsets in B that are currently uncovered
                    gain = np.sum(cover_count[B_idxs] == 0)
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_replacement = (B, B_idxs)
                
                if best_replacement is not None and best_gain > 0:
                    B, B_idxs = best_replacement
                    # Replace block i with B
                    cover_count[idxs] -= 1
                    cover_count[B_idxs] += 1
                    current_blocks[i] = B
                    current_indices[i] = B_idxs
                    improved = True
                    break
        
        return current_blocks, current_indices

    best_blocks = None
    best_len = float('inf')
    
    # Multi-start
    num_starts = 0
    start_time = time.time()
    
    # We will loop until time runs out
    while time.time() < deadline - 5.0: # Leave 5s for final reduction
        seed = num_starts
        blocks = greedy_once(seed)
        
        if not blocks:
            num_starts += 1
            continue
            
        # Reduce
        reduced_blocks = remove_redundant(blocks)
        
        # Local search
        if len(reduced_blocks) < best_len:
            # Only do local search if this is promising
            block_indices = []
            for b in reduced_blocks:
                idxs = np.array([tsub_to_idx[comb] for comb in itertools.combinations(b, t)], dtype=np.int32)
                block_indices.append(idxs)
            
            rng = random.Random(seed + 10000)
            ls_blocks, ls_indices = local_search(reduced_blocks, block_indices, rng)
            
            if len(ls_blocks) < best_len:
                best_len = len(ls_blocks)
                best_blocks = ls_blocks[:]
        elif len(reduced_blocks) < best_len:
            best_len = len(reduced_blocks)
            best_blocks = reduced_blocks[:]
        
        num_starts += 1
    
    if best_blocks is None:
        # Fallback
        best_blocks = greedy_once(0)
        best_blocks = remove_redundant(best_blocks)

    return best_blocks
