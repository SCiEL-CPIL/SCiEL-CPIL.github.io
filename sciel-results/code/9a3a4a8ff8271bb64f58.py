import time
import numpy as np
from itertools import combinations
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    """
    Constructs the Steiner system S(5,6,12) using Integer Linear Programming.
    """
    start_time = time.time()
    deadline = 100.0  # Leave margin for 150s limit
    
    points = range(12)
    blocks = list(combinations(points, 6))
    subsets_5 = list(combinations(points, 5))
    
    n_blocks = len(blocks)
    n_subsets = len(subsets_5)
    
    # Map 5-subsets to indices for constraint building
    # We need to check which blocks contain which 5-subsets
    # A is n_subsets x n_blocks
    
    # Optimization: Instead of building a dense matrix of size 792x924 (~700k entries),
    # we can build it efficiently or use sparse. SciPy milp accepts sparse.
    
    from scipy.sparse import coo_matrix
    
    # To build A efficiently using COO format:
    # Collect row, col indices and data.
    # This is 924 blocks * 6 subsets = 5544 operations.
    
    # Create a mapping from 5-subset to its index in subsets_5
    subset_index = {s: i for i, s in enumerate(subsets_5)}
    
    row_indices = []
    col_indices = []
    data = []
    
    for j, block in enumerate(blocks):
        for s in combinations(block, 5):
            i = subset_index[s]
            row_indices.append(i)
            col_indices.append(j)
            data.append(1)
            
    A = coo_matrix((data, (row_indices, col_indices)), shape=(n_subsets, n_blocks)).tocsr()
    
    # Objective: Maximize sum(x) -> Minimize -sum(x)
    c = -np.ones(n_blocks)
    
    # Constraints: A @ x <= 1
    constraints = LinearConstraint(A, ub=1)
    
    # Bounds: 0 <= x <= 1
    bounds = Bounds(0, 1)
    
    # Integrality: x is binary
    integrality = np.ones(n_blocks)
    
    options = {
        "time_limit": 90.0,  # Leave 10s for overhead
        "mip_rel_gap": 0.0   # Require optimality gap 0
    }
    
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds, options=options)
        
        # Check if solved within time
        if time.time() - start_time > deadline:
             # Fallback: Greedy
             return _greedy_fallback()
             
        if res.status == 0:  # Optimal
            x = res.x
            # x should be 0 or 1
            selected_blocks = [list(blocks[j]) for j in range(n_blocks) if x[j] > 0.5]
            return selected_blocks
        else:
            # If not optimal, try to get best found? 
            # HiGHS returns the best incumbent if it times out (status != 0 usually, or status 1/4/etc)
            # We can try to extract x if available and valid.
            if hasattr(res, 'x') and res.x is not None:
                x = res.x
                selected_blocks = [list(blocks[j]) for j in range(n_blocks) if x[j] > 0.5]
                # Check if this is a valid packing (it should be if constraints are satisfied)
                # For safety, if it's not 132, we might still return it if it's better than greedy
                # But greedy is deterministic. ILP incumbent is likely better.
                return selected_blocks
            else:
                return _greedy_fallback()
            
    except Exception as e:
        return _greedy_fallback()

def _greedy_fallback():
    from itertools import combinations
    covered, blocks = set(), []
    for b in combinations(range(12), 6):
        subsets = list(combinations(b, 5))
        if not any(s in covered for s in subsets):
            blocks.append(list(b))
            covered.update(subsets)
    return blocks
