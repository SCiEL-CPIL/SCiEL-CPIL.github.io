import itertools
import time
import random
import numpy as np
from scipy import sparse

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Precomputation
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    num_t = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_k_blocks = list(itertools.combinations(points, k))
    num_k = len(all_k_blocks)
    
    # Build sparse matrix M: M[b, s] = 1 if block b covers t-subset s
    rows = []
    cols = []
    for b_idx, block in enumerate(all_k_blocks):
        for t_sub in itertools.combinations(block, t):
            s_idx = t_subset_to_idx[t_sub]
            rows.append(b_idx)
            cols.append(s_idx)
            
    M = sparse.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(num_k, num_t))
    
    # Precompute for each t-subset, which blocks cover it (column-wise)
    M_csc = M.tocsc()
    
    best_blocks = None
    best_score = float('inf')
    
    def reduce_blocks(blocks_idx, coverage):
        """Remove redundant blocks. Returns new list of block indices."""
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for b_idx in blocks_idx:
                idxs = M[b_idx].indices
                if np.all(coverage[idxs] > 1):
                    coverage[idxs] -= 1
                    changed = True
                else:
                    new_blocks.append(b_idx)
            blocks_idx = new_blocks
        return blocks_idx

    def greedy_construct(seed):
        random.seed(seed)
        coverage = np.zeros(num_t, dtype=np.int32)
        uncovered_count = num_t
        blocks_idx = []
        
        while uncovered_count > 0:
            if time.time() > deadline - 5.0:
                return None, coverage, blocks_idx
            
            mask = (coverage == 0)
            gains = M.dot(mask.astype(np.int8))
            
            max_gain = np.max(gains)
            if max_gain == 0:
                return None, coverage, blocks_idx
                
            candidates = np.where(gains == max_gain)[0]
            b_idx = random.choice(candidates)
            
            blocks_idx.append(b_idx)
            
            idxs = M[b_idx].indices
            # Track how many were newly covered
            newly_covered = np.sum(coverage[idxs] == 0)
            coverage[idxs] += 1
            uncovered_count -= newly_covered
            
        return blocks_idx, coverage, None

    def local_search(blocks_idx, coverage, max_steps=100):
        """Try to improve by removing blocks and covering gaps, or swapping."""
        blocks_idx = list(blocks_idx)
        coverage = coverage.copy()
        
        for step in range(max_steps):
            if time.time() > deadline - 3.0:
                break
            
            if not blocks_idx:
                break
                
            # Try removing a random block and seeing if we can cover gaps with fewer new blocks
            b_remove = random.choice(blocks_idx)
            
            # Temporarily remove
            idxs_removed = M[b_remove].indices
            coverage[idxs_removed] -= 1
            
            # Find uncovered t-subsets
            uncovered_mask = (coverage == 0)
            if not np.any(uncovered_mask):
                # Can remove this block
                blocks_idx.remove(b_remove)
                continue
            
            uncovered_indices = np.where(uncovered_mask)[0]
            
            # Find blocks that cover the most uncovered t-subsets
            gains = M.dot(uncovered_mask.astype(np.int8))
            max_gain = np.max(gains)
            
            if max_gain == 0:
                # Can't fix, restore
                coverage[idxs_removed] += 1
                continue
            
            # Try to cover with 1 block
            candidates = np.where(gains == max_gain)[0]
            
            # Filter out blocks already in our set
            block_set = set(blocks_idx)
            candidates = [c for c in candidates if c not in block_set]
            
            if not candidates:
                coverage[idxs_removed] += 1
                continue
                
            # Try the best candidate
            b_add = candidates[0]
            idxs_added = M[b_add].indices
            coverage[idxs_added] += 1
            
            # Check if all uncovered are now covered
            still_uncovered = np.sum(coverage == 0)
            
            if still_uncovered == 0:
                # Success: removed 1, added 1, net same or we can reduce
                blocks_idx.remove(b_remove)
                blocks_idx.append(b_add)
                
                # Try to reduce
                blocks_idx = reduce_blocks(blocks_idx, coverage)
            else:
                # Try with 2 blocks
                # Find second best block
                gains2 = M.dot((coverage == 0).astype(np.int8))
                max_gain2 = np.max(gains2)
                
                if max_gain2 > 0:
                    candidates2 = np.where(gains2 == max_gain2)[0]
                    candidates2 = [c for c in candidates2 if c not in block_set and c != b_add]
                    
                    if candidates2:
                        b_add2 = candidates2[0]
                        idxs_added2 = M[b_add2].indices
                        coverage[idxs_added2] += 1
                        
                        still_uncovered2 = np.sum(coverage == 0)
                        if still_uncovered2 == 0:
                            # Removed 1, added 2: net +1, not helpful unless we can reduce
                            blocks_idx.remove(b_remove)
                            blocks_idx.append(b_add)
                            blocks_idx.append(b_add2)
                            blocks_idx = reduce_blocks(blocks_idx, coverage)
                        else:
                            # Restore
                            coverage[idxs_added] -= 1
                            coverage[idxs_removed] += 1
                    else:
                        coverage[idxs_added] -= 1
                        coverage[idxs_removed] += 1
                else:
                    coverage[idxs_added] -= 1
                    coverage[idxs_removed] += 1
        return blocks_idx, coverage

    # Run multiple restarts
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
            
        res = greedy_construct(i)
        if res[0] is None:
            continue
            
        blocks_idx, coverage, _ = res
        
        # Reduce
        blocks_idx = reduce_blocks(blocks_idx, coverage)
        
        current_score = len(blocks_idx)
        
        # Local search
        if time.time() < deadline - 10.0:
            blocks_idx, coverage = local_search(blocks_idx, coverage, max_steps=80)
            current_score = len(blocks_idx)
        
        if current_score < best_score:
            best_score = current_score
            best_blocks = [list(all_k_blocks[b]) for b in blocks_idx]
            
    return best_blocks
