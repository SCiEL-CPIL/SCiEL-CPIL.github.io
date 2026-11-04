import itertools
import time
import random
import numpy as np
from scipy.optimize import linprog

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0  # Safety margin from 120s
    
    # Pre-compute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_index = {sub: i for i, sub in enumerate(all_t_subs)}
    n_t_subs = len(all_t_subs)
    
    # Generate a pool of random blocks (in addition to some structured ones)
    random.seed(42)
    block_pool = set()
    
    # Add some structured blocks: all combinations with small offsets
    for i in range(v):
        for j in range(i+1, v):
            block = sorted(set([i, j] + random.sample([x for x in range(v) if x not in (i,j)], k-2)))
            block_pool.add(tuple(block))
    
    # Add more random blocks
    while len(block_pool) < min(10000, len(list(itertools.combinations(range(v), k)))):
        if time.time() - start_time > time_limit * 0.1:
            break
        block = tuple(sorted(random.sample(range(v), k)))
        block_pool.add(block)
    
    block_list = list(block_pool)
    n_blocks = len(block_list)
    
    # Build incidence matrix: A[i, j] = 1 if t-subset i is contained in block j
    # A has shape (n_t_subs, n_blocks)
    A = np.zeros((n_t_subs, n_blocks), dtype=np.float64)
    for j, block in enumerate(block_list):
        for sub in itertools.combinations(block, t):
            A[t_sub_index[sub], j] = 1.0
    
    # Solve LP relaxation: minimize sum(x_j) subject to A @ x >= 1, x >= 0
    # c = [1, 1, ..., 1] (minimize number of blocks)
    c = np.ones(n_blocks)
    # Constraints: A @ x >= 1, which is -A @ x <= -1
    A_ub = -A
    b_ub = -np.ones(n_t_subs)
    
    # Bounds: 0 <= x_j <= 1 (we can cap at 1 since we only need 0/1)
    bounds = [(0, 1)] * n_blocks
    
    # Solve LP
    try:
        lp_result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if not lp_result.success:
            raise ValueError("LP failed")
        x_lp = lp_result.x
    except Exception:
        # Fallback to greedy if LP fails
        x_lp = None
    
    if x_lp is not None and np.any(x_lp > 0):
        # Randomized rounding: sample blocks with probability proportional to LP weights
        best_blocks = None
        best_count = float('inf')
        
        # Multiple restarts of randomized rounding
        for restart in range(50):
            if time.time() - start_time > time_limit * 0.7:
                break
            
            # Sample blocks: include if random() < x_lp[j]
            sampled = set()
            for j in range(n_blocks):
                if random.random() < x_lp[j]:
                    sampled.add(j)
            
            # Check if sampled blocks cover everything
            cover_count = np.zeros(n_t_subs, dtype=np.int32)
            for j in sampled:
                cover_count += A[:, j]
            
            if np.all(cover_count >= 1):
                # Valid cover! Try to remove redundant blocks
                blocks_to_remove = set()
                for j in list(sampled):
                    # Check if block j is redundant
                    temp_cover = cover_count - A[:, j]
                    if np.all(temp_cover >= 1):
                        blocks_to_remove.add(j)
                
                for j in blocks_to_remove:
                    cover_count -= A[:, j]
                    sampled.remove(j)
                
                if len(sampled) < best_count:
                    best_count = len(sampled)
                    best_blocks = set(sampled)
            
            # Also try: start from LP solution, greedily add uncovered subsets
            # This is a hybrid approach
            if time.time() - start_time > time_limit * 0.85:
                break
    
    # Greedy fallback / improvement
    if best_blocks is None or time.time() - start_time < time_limit * 0.85:
        # Use greedy on the block pool
        uncovered = np.ones(n_t_subs, dtype=np.int32)
        total_uncovered = n_t_subs
        greedy_blocks = set()
        
        # Pre-compute which blocks contain which t-subset for fast lookup
        subset_to_blocks = [[] for _ in range(n_t_subs)]
        for j in range(n_blocks):
            for i in range(n_t_subs):
                if A[i, j] == 1:
                    subset_to_blocks[i].append(j)
        
        uncovered_count_per_block = np.sum(A, axis=0).astype(np.int32)
        
        while total_uncovered > 0:
            if time.time() - start_time > time_limit * 0.9:
                break
            
            # Find block with max uncovered count
            best_j = int(np.argmax(uncovered_count_per_block))
            if uncovered_count_per_block[best_j] == 0:
                break
            
            greedy_blocks.add(best_j)
            
            # Update uncovered counts
            for s in range(n_t_subs):
                if A[s, best_j] == 1 and uncovered[s] == 1:
                    uncovered[s] = 0
                    total_uncovered -= 1
                    for b_idx in subset_to_blocks[s]:
                        uncovered_count_per_block[b_idx] -= 1
        
        # Remove redundant blocks from greedy solution
        cover_count = np.zeros(n_t_subs, dtype=np.int32)
        for j in greedy_blocks:
            cover_count += A[:, j]
        
        improved = True
        while improved and time.time() - start_time < time_limit * 0.95:
            improved = False
            for j in list(greedy_blocks):
                temp_cover = cover_count - A[:, j]
                if np.all(temp_cover >= 1):
                    cover_count -= A[:, j]
                    greedy_blocks.remove(j)
                    improved = True
        
        if len(greedy_blocks) < best_count:
            best_count = len(greedy_blocks)
            best_blocks = greedy_blocks
    
    # Convert indices back to actual blocks
    if best_blocks is not None:
        final_blocks = [list(block_list[j]) for j in best_blocks]
    else:
        # Last resort: return empty (should not happen)
        final_blocks = []
    
    return final_blocks
