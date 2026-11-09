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
    deadline = time.time() + 112.0  # Safety margin (150s kill, leave 38s buffer)

    # Precompute t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Precompute: for each t-subset, the remaining points
    tsub_remaining = {}
    for i, S in enumerate(all_tsubs):
        S_set = set(S)
        tsub_remaining[i] = [p for p in range(v) if p not in S_set]

    # Cache: for each t-subset index, list of (block_tuple, tsub_idxs_numpy_array)
    candidate_cache = {}

    def get_candidates_for_tsub(tsub_idx):
        if tsub_idx in candidate_cache:
            return candidate_cache[tsub_idx]
        
        S = all_tsubs[tsub_idx]
        remaining = tsub_remaining[tsub_idx]
        
        candidates = []
        for extra in itertools.combinations(remaining, k - t):
            block = tuple(sorted(S + extra))
            tsub_idxs = np.array([tsub_to_idx[comb] for comb in itertools.combinations(block, t)], dtype=np.int32)
            candidates.append((block, tsub_idxs))
        
        candidate_cache[tsub_idx] = candidates
        return candidates

    def greedy_once(seed):
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubs, dtype=np.int8)
        blocks = []
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
            
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            
            start_idx = int(rng.choice(uncovered_indices))
            S = all_tsubs[start_idx]
            
            candidates = get_candidates_for_tsub(start_idx)
            
            best_block = None
            best_tsub_idxs = None
            best_gain = -1
            
            candidates_list = list(candidates)
            rng.shuffle(candidates_list)
            
            for B, tsub_idxs in candidates_list:
                gain = int(np.sum(uncovered[tsub_idxs]))
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = B
                    best_tsub_idxs = tsub_idxs
            
            if best_block is None:
                break
                
            blocks.append(best_block)
            
            uncovered[best_tsub_idxs] = 0
        
        return blocks

    def remove_redundant(blocks):
        if not blocks:
            return blocks
        
        cover_count = np.zeros(n_tsubs, dtype=np.int32)
        
        block_indices = []
        for b in blocks:
            idxs = np.array([tsub_to_idx[comb] for comb in itertools.combinations(b, t)], dtype=np.int32)
            block_indices.append(idxs)
            cover_count[idxs] += 1
        
        valid_blocks = []
        
        for i, b in enumerate(blocks):
            if time.time() > deadline:
                return blocks
            
            idxs = block_indices[i]
            if np.all(cover_count[idxs] > 1):
                cover_count[idxs] -= 1
            else:
                valid_blocks.append(b)
        
        # Second pass for more removals
        if len(valid_blocks) < len(blocks):
            cover_count2 = np.zeros(n_tsubs, dtype=np.int32)
            block_indices2 = []
            for b in valid_blocks:
                idxs = np.array([tsub_to_idx[comb] for comb in itertools.combinations(b, t)], dtype=np.int32)
                block_indices2.append(idxs)
                cover_count2[idxs] += 1
            
            valid_blocks2 = []
            for i, b in enumerate(valid_blocks):
                if time.time() > deadline:
                    return valid_blocks
                
                idxs = block_indices2[i]
                if np.all(cover_count2[idxs] > 1):
                    cover_count2[idxs] -= 1
                else:
                    valid_blocks2.append(b)
            
            if len(valid_blocks2) < len(valid_blocks):
                valid_blocks = valid_blocks2
        
        return valid_blocks

    best_blocks = None
    best_len = float('inf')
    
    num_starts = 0
    
    while time.time() < deadline - 8.0:
        seed = num_starts
        blocks = greedy_once(seed)
        
        if not blocks:
            num_starts += 1
            continue
            
        reduced_blocks = remove_redundant(blocks)
        
        if len(reduced_blocks) < best_len:
            best_len = len(reduced_blocks)
            best_blocks = reduced_blocks[:]
        
        num_starts += 1
    
    if best_blocks is None:
        best_blocks = greedy_once(0)
        best_blocks = remove_redundant(best_blocks)

    return best_blocks
