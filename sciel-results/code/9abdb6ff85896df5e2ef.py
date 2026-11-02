"""Improved covering algorithm using LP relaxation and greedy."""
import itertools
import time
import random
import numpy as np
from scipy import sparse
from scipy.optimize import linprog

def construct(v, k, t):
    deadline = time.time() + 100.0  # 20s safety margin
    
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Build sparse coverage matrix A (n_tsubsets x n_blocks)
    row_indices = []
    col_indices = []
    
    for j, block in enumerate(all_blocks):
        for ts in itertools.combinations(block, t):
            i = tsub_idx[ts]
            row_indices.append(i)
            col_indices.append(j)
            
    A = sparse.csc_matrix((np.ones(len(row_indices)), (row_indices, col_indices)), shape=(n_tsubsets, n_blocks))
    
    block_order = None
    try:
        # Solve LP
        c = np.ones(n_blocks)
        A_ub = -A.tocsr()
        b_ub = -np.ones(n_tsubsets)
        
        # Use HiGHS if available, else interior-point
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
        except TypeError:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, method='interior-point')
            
        if res.status == 0:
            x = res.x
            block_order = np.argsort(-x)
    except Exception:
        pass
        
    if block_order is None:
        # Fallback: Random order or frequency-based
        # Frequency-based: count how many t-subsets each block covers (all cover C(k,t) so it's uniform, useless)
        # So just random permutation
        block_order = np.random.permutation(n_blocks)
        
    uncovered = np.ones(n_tsubsets, dtype=bool)
    blocks_used = []
    
    # Precompute column ranges for A (CSC format)
    # A.indptr[j] to A.indptr[j+1] gives indices in A.indices for column j
    
    for j in block_order:
        if not np.any(uncovered):
            break
        if time.time() > deadline:
            break
            
        start = A.indptr[j]
        end = A.indptr[j+1]
        covered_tsubs = A.indices[start:end]
        
        # Check if any covered tsub is uncovered
        # This creates a boolean mask of length C(k,t)
        if np.any(uncovered[covered_tsubs]):
            blocks_used.append(all_blocks[j])
            uncovered[covered_tsubs] = False
            
    # Reduce redundant blocks
    if blocks_used:
        cover_count = np.zeros(n_tsubsets, dtype=np.int32)
        block_tsubs = []
        for b in blocks_used:
            tsubs = []
            for ts in itertools.combinations(b, t):
                idx = tsub_idx[ts]
                tsubs.append(idx)
                cover_count[idx] += 1
            block_tsubs.append(tsubs)
            
        kept = [True] * len(blocks_used)
        for i in range(len(blocks_used)):
            if not kept[i]:
                continue
            redundant = True
            for idx in block_tsubs[i]:
                if cover_count[idx] == 1:
                    redundant = False
                    break
            if redundant:
                kept[i] = False
                for idx in block_tsubs[i]:
                    cover_count[idx] -= 1
                    
        blocks_used = [blocks_used[i] for i in range(len(blocks_used)) if kept[i]]
        
    return blocks_used
