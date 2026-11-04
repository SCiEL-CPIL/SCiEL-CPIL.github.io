import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix
import itertools
import time

def construct(v, k, t):
    # Parameters for the specific task: v=19, k=7, t=4
    # But we should use the passed v, k, t just in case, though the task is specific.
    
    # 1. Generate all possible blocks (k-subsets)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # 2. Generate all possible t-subsets (constraints)
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    
    # 3. Build the incidence matrix A
    # A[i, j] = 1 if t-subset i is contained in block j
    # We use a sparse matrix for efficiency
    A = lil_matrix((n_t_subsets, n_blocks), dtype=np.int8)
    
    # To speed up lookup, map t-subsets to indices
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    for j, block in enumerate(all_blocks):
        # Find all t-subsets within this block
        for t_sub in itertools.combinations(block, t):
            i = t_subset_to_idx[t_sub]
            A[i, j] = 1
            
    A = A.tocsr()
    
    # 4. Define the MILP
    # Minimize sum(x)
    c = np.ones(n_blocks)
    
    # Constraints: A @ x >= 1
    # LinearConstraint(A, 1, np.inf)
    constraints = LinearConstraint(A, 1, np.inf)
    
    # Bounds: 0 <= x <= 1
    bounds = Bounds(0, 1)
    
    # Integrality: all variables are integer
    integrality = np.ones(n_blocks)
    
    # Solve
    start_time = time.time()
    try:
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds, options={"time_limit": 100, "mip_rel_gap": 0.0})
        
        # Check if a solution was found
        if res.status == 0: # Optimal
            x = res.x
        elif res.status in [1, 2]: # Feasible or infeasible? 
            # status 1: Feasible (but maybe not optimal if time limit hit? No, status 1 is "Optimal" in some versions, 0 is "Optimal"? 
            # scipy docs: 
            # 0: Optimal
            # 1: Feasible (but not necessarily optimal? Actually, usually 0 is optimal, 1 is feasible, 2 is infeasible, 3 is unbounded, 4 is error)
            # Let's check if res.x is available and valid.
            if res.x is not None:
                x = res.x
            else:
                x = None
        else:
            x = None
            
        if x is not None:
            # Verify the solution
            # Round to nearest integer
            x_int = np.round(x).astype(int)
            
            # Check if all constraints are satisfied
            # A @ x_int >= 1
            coverage = A @ x_int
            if np.all(coverage >= 1):
                # Extract blocks
                selected_indices = np.where(x_int == 1)[0]
                blocks = [list(all_blocks[i]) for i in selected_indices]
                return blocks
            
    except Exception as e:
        pass
        
    # Fallback: Greedy Heuristic
    # If ILP fails or times out without a valid solution, use a greedy approach
    # This ensures we always return a valid covering.
    
    uncovered = set(all_t_subsets)
    blocks = []
    
    # Pre-compute coverage for each block to speed up greedy
    # block_covers[j] = set of t-subset indices covered by block j
    block_covers = [set() for _ in range(n_blocks)]
    for j, block in enumerate(all_blocks):
        for t_sub in itertools.combinations(block, t):
            block_covers[j].add(t_subset_to_idx[t_sub])
            
    while uncovered:
        best_block = -1
        best_gain = -1
        for j in range(n_blocks):
            if not uncovered:
                break
            gain = len(uncovered & block_covers[j])
            if gain > best_gain:
                best_gain = gain
                best_block = j
        
        if best_block == -1:
            break
            
        blocks.append(list(all_blocks[best_block]))
        uncovered.difference_update(block_covers[best_block])
        
    return blocks
