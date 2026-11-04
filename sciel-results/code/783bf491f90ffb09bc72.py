import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, csr_matrix
import itertools
import time

def construct(v, k, t):
    # Parameters
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    
    # Map t-subsets to indices
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Build incidence matrix A (n_t_subsets x n_blocks)
    # A[i, j] = 1 if t_subset i is in block j
    # Using sparse matrix
    rows = []
    cols = []
    
    for j, block in enumerate(all_blocks):
        for t_sub in itertools.combinations(block, t):
            i = t_subset_to_idx[t_sub]
            rows.append(i)
            cols.append(j)
            
    A = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n_t_subsets, n_blocks))
    
    # ILP Setup
    c = np.ones(n_blocks)
    constraints = LinearConstraint(A, 1, np.inf)
    bounds = Bounds(0, 1)
    integrality = np.ones(n_blocks)
    
    start_time = time.time()
    ilp_solution = None
    
    try:
        # Use a relaxed gap to find a good solution quickly, not necessarily optimal
        # 110 seconds is a safe limit given 120s total timeout
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds, 
                   options={"time_limit": 110, "mip_rel_gap": 0.05})
        
        if res.x is not None:
            x_int = np.round(res.x).astype(int)
            # Verify validity
            if np.all(A @ x_int >= 1):
                ilp_solution = [list(all_blocks[i]) for i in np.where(x_int == 1)[0]]
                
    except Exception:
        pass
        
    # Fallback: If ILP failed or timed out without a solution, use a fast greedy heuristic
    if ilp_solution is None:
        # Fast Greedy Heuristic
        uncovered = np.ones(n_t_subsets, dtype=bool)
        selected_blocks = []
        
        # Precompute block covers
        block_covers = [[] for _ in range(n_blocks)]
        for i_idx, i in enumerate(rows):
            block_covers[cols[i_idx]].append(i)
            
        A_T = A.T.tocsr()
        
        while np.any(uncovered):
            gains = A_T @ uncovered.astype(np.float32)
            best_block = int(np.argmax(gains))
            
            if gains[best_block] == 0:
                break
                
            selected_blocks.append(best_block)
            
            for ts_idx in block_covers[best_block]:
                uncovered[ts_idx] = False
                
        # Simple redundancy removal
        final_selected = []
        coverage_count = np.zeros(n_t_subsets, dtype=np.int32)
        for b in selected_blocks:
            for ts_idx in block_covers[b]:
                coverage_count[ts_idx] += 1
                
        for b in selected_blocks:
            for ts_idx in block_covers[b]:
                coverage_count[ts_idx] -= 1
            if np.any(coverage_count < 1):
                for ts_idx in block_covers[b]:
                    coverage_count[ts_idx] += 1
                final_selected.append(b)
                
        ilp_solution = [list(all_blocks[i]) for i in final_selected]

    return ilp_solution
