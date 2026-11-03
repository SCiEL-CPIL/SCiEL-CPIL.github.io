import itertools
import random
import numpy as np
from scipy import sparse

def construct(v, k, t):
    # Precompute all triples and blocks
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Build sparse incidence matrix
    rows = []
    cols = []
    data = []
    
    for b_idx, block in enumerate(all_blocks):
        for triple in itertools.combinations(block, t):
            t_idx = triple_to_idx[triple]
            rows.append(b_idx)
            cols.append(t_idx)
            data.append(1)
            
    M = sparse.csr_matrix((data, (rows, cols)), shape=(num_blocks, num_triples))
    
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple random greedy searches
    # Time limit is 120s. Let's do 10 runs to be safe, or dynamic?
    # 12M nnz. M @ v is fast.
    for run in range(15):
        np.random.seed(run)
        uncovered = np.ones(num_triples, dtype=bool)
        blocks = []
        
        while uncovered.any():
            scores = M @ uncovered
            
            max_score = scores.max()
            if max_score == 0:
                break
                
            candidates = np.where(scores == max_score)[0]
            best_idx = np.random.choice(candidates)
            
            blocks.append(list(all_blocks[best_idx]))
            
            row_mask = M[best_idx].toarray().flatten().astype(bool)
            uncovered &= ~row_mask
            
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
    return best_blocks
