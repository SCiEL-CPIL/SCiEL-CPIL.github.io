import itertools
import time
import sys
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using ILP with greedy fallback.
    """
    deadline = time.monotonic() + 115.0
    points = list(range(v))
    
    # Precomputation
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)
    
    # Build sparse incidence matrix
    row_indices = []
    col_indices = []
    data = []
    
    for j, b in enumerate(all_kblocks):
        for ts in itertools.combinations(b, t):
            i = tsub_to_idx[ts]
            row_indices.append(i)
            col_indices.append(j)
            data.append(1)
    
    A = csr_matrix((data, (row_indices, col_indices)), shape=(num_tsubs, num_kblocks))
    
    # Precompute block masks for greedy
    block_masks = [0] * num_kblocks
    block_tsub_indices = [None] * num_kblocks
    
    for i, b in enumerate(all_kblocks):
        mask = 0
        indices = []
        for ts in itertools.combinations(b, t):
            idx = tsub_to_idx[ts]
            mask |= (1 << idx)
            indices.append(idx)
        block_masks[i] = mask
        block_tsub_indices[i] = indices
    
    full_mask = (1 << num_tsubs) - 1
    
    # Greedy construction
    greedy_selected = []
    uncovered_mask = full_mask
    
    while uncovered_mask and time.monotonic() < deadline - 20:
        best_gain = 0
        best_block_idx = -1
        
        for i in range(num_kblocks):
            common = block_masks[i] & uncovered_mask
            if common:
                gain = common.bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = i
                    
        if best_gain == 0:
            break
            
        greedy_selected.append(best_block_idx)
        uncovered_mask &= ~block_masks[best_block_idx]
        
        if uncovered_mask == 0:
            break
    
    # Prune greedy
    if greedy_selected:
        counts = [0] * num_tsubs
        for b_idx in greedy_selected:
            for t_idx in block_tsub_indices[b_idx]:
                counts[t_idx] += 1
                
        changed = True
        while changed:
            if time.monotonic() > deadline - 15:
                break
            changed = False
            new_selected = []
            for b_idx in greedy_selected:
                redundant = True
                for t_idx in block_tsub_indices[b_idx]:
                    if counts[t_idx] == 1:
                        redundant = False
                        break
                
                if redundant:
                    for t_idx in block_tsub_indices[b_idx]:
                        counts[t_idx] -= 1
                    changed = True
                else:
                    new_selected.append(b_idx)
            
            greedy_selected = new_selected
    
    greedy_blocks = [list(all_kblocks[i]) for i in greedy_selected]
    best_blocks = greedy_blocks
    best_count = len(greedy_blocks)
    
    # Try ILP if we have time
    remaining_time = deadline - time.monotonic() - 5
    if remaining_time > 10 and best_count > 0:
        try:
            c = np.ones(num_kblocks)
            constraint = LinearConstraint(A, lb=1, ub=np.inf)
            bounds = Bounds(0, 1)
            integrality = np.ones(num_kblocks)
            
            options = {'time_limit': int(remaining_time)}
            
            result = milp(c, constraints=[constraint], integrality=integrality,
                         bounds=bounds, options=options)
            
            if result.x is not None:
                x = result.x
                selected_indices = [i for i in range(num_kblocks) if x[i] > 0.5]
                
                if selected_indices:
                    covered_mask = 0
                    for b_idx in selected_indices:
                        covered_mask |= block_masks[b_idx]
                    
                    if covered_mask == full_mask:
                        ilp_count = len(selected_indices)
                        if ilp_count < best_count:
                            best_blocks = [list(all_kblocks[i]) for i in selected_indices]
                            best_count = ilp_count
                            
        except Exception:
            pass
    
    if not best_blocks:
        return []
    
    return best_blocks
