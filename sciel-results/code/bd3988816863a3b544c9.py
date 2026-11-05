import itertools
import numpy as np
from math import comb
import time
import scipy.sparse
from scipy.optimize import milp, LinearConstraint, Bounds
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # 1. Precomputation
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {trip: i for i, trip in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    k_t = comb(k, t)
    
    # Store block indices for fast access
    # Shape: (n_blocks, k_t)
    block_triple_indices = np.empty((n_blocks, k_t), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        for j, sub in enumerate(itertools.combinations(block, t)):
            block_triple_indices[i, j] = triple_to_idx[sub]
            
    def greedy_with_prune(random_seed):
        rng = np.random.RandomState(random_seed)
        
        # coverage_count[i] is the number of selected blocks covering triple i
        coverage_count = np.zeros(n_triples, dtype=np.int32)
        selected_indices = []
        
        while np.any(coverage_count == 0):
            if time.time() > deadline - 5:
                break
                
            uncovered_flags = (coverage_count == 0)
            gains = np.sum(uncovered_flags[block_triple_indices], axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            # Simple greedy: pick max gain
            # Add small random noise to break ties differently
            noise = rng.rand(n_blocks) * 0.1
            scores = gains + noise
            
            max_score = np.max(scores)
            candidates = np.where(scores == max_score)[0]
            best_idx = candidates[rng.randint(len(candidates))]
            
            selected_indices.append(best_idx)
            block_triples = block_triple_indices[best_idx]
            coverage_count[block_triples] += 1
            
        # Pruning Phase
        active_blocks = list(selected_indices)
        
        changed = True
        passes = 0
        max_passes = 5
        
        while changed and passes < max_passes:
            if time.time() > deadline - 5:
                break
            changed = False
            passes += 1
            rng.shuffle(active_blocks)
            
            new_active_blocks = []
            for b_idx in active_blocks:
                block_triples = block_triple_indices[b_idx]
                
                is_redundant = not np.any(coverage_count[block_triples] == 1)
                
                if is_redundant:
                    coverage_count[block_triples] -= 1
                    changed = True
                else:
                    new_active_blocks.append(b_idx)
            
            active_blocks = new_active_blocks
            
        return [list(all_blocks[i]) for i in active_blocks], set(active_blocks)

    best_blocks = None
    best_len = float('inf')
    pool_blocks = set()
    
    # Run multiple restarts
    n_restarts = 200
    
    for i in range(n_restarts):
        if time.time() > deadline - 20:  # Leave 20s for ILP
            break
            
        blocks, used_indices = greedy_with_prune(i)
        pool_blocks.update(used_indices)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        # If we hit the lower bound, we can stop early
        if best_len <= 69:
            break

    # If we didn't find 69, try ILP on the pool
    if best_len > 69 and time.time() < deadline - 5:
        pool_list = list(pool_blocks)
        if len(pool_list) > 0:
            # Build sparse matrix for the pool
            # Rows: triples (n_triples), Cols: blocks in pool (len(pool_list))
            # A[r, c] = 1 if triple r is in block c
            
            rows = []
            cols = []
            data = []
            
            for c_idx, b_idx in enumerate(pool_list):
                for t_idx in block_triple_indices[b_idx]:
                    rows.append(t_idx)
                    cols.append(c_idx)
                    data.append(1)
            
            A = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(n_triples, len(pool_list)))
            
            # Objective: Minimize sum(x)
            c = np.ones(len(pool_list))
            
            # Constraints: A @ x >= 1
            constraint = LinearConstraint(A, lb=np.ones(n_triples), ub=np.inf)
            
            # Bounds: x in {0, 1}
            bounds = Bounds(lb=0, ub=1)
            
            # Solve ILP
            try:
                res = milp(c, constraints=constraint, integrality=np.ones(len(pool_list)), bounds=bounds, 
                           options={"time_limit": 100})
                
                if res.status == 0: # Optimal or best found within time?
                    # res.status 0 means optimal. 1 means feasible.
                    # Check if it's a valid solution
                    x = res.x
                    selected_pool_indices = np.where(x > 0.5)[0]
                    
                    # Map back to original block indices
                    selected_orig_indices = [pool_list[i] for i in selected_pool_indices]
                    
                    # Verify coverage (just in case)
                    coverage_count = np.zeros(n_triples, dtype=np.int32)
                    for b_idx in selected_orig_indices:
                        coverage_count[block_triple_indices[b_idx]] += 1
                    
                    if np.all(coverage_count >= 1):
                        new_len = len(selected_orig_indices)
                        if new_len < best_len:
                            best_len = new_len
                            best_blocks = [list(all_blocks[i]) for i in selected_orig_indices]
                            
            except Exception as e:
                pass

    if best_blocks is None:
        # Fallback
        if time.time() < deadline - 1:
            blocks, _ = greedy_with_prune(0)
            best_blocks = blocks
        else:
            best_blocks = []
            
    return best_blocks
