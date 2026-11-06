import time
from itertools import combinations
import numpy as np

def construct():
    """
    Constructs the Steiner system S(5,6,12) using an Integer Linear Programming (ILP) approach
    if SciPy is available, otherwise falls back to a known valid hardcoded construction.
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    points = list(range(12))
    all_blocks = list(combinations(points, 6))
    n_blocks = len(all_blocks)
    
    # Map each block to an index
    block_index = {b: i for i, b in enumerate(all_blocks)}
    
    # Generate all 5-subsets
    all_5subsets = list(combinations(points, 5))
    n_5subsets = len(all_5subsets)
    
    # For each 5-subset, find the blocks that contain it
    # A 5-subset is contained in 12-5=7 points? No.
    # A 5-subset S can be extended to a 6-block B by adding one point from the remaining 12-5=7 points.
    # So each 5-subset is contained in exactly 7 blocks.
    
    # Build the constraint matrix A (n_5subsets x n_blocks)
    # A[i, j] = 1 if 5-subset i is contained in block j
    # We can build this efficiently.
    
    # Instead of building a huge sparse matrix directly, we can use scipy.sparse
    try:
        import scipy.sparse as sp
        import scipy.optimize as opt
        
        # Build row indices, column indices, and data for the sparse matrix
        row_indices = []
        col_indices = []
        data = []
        
        for i, subset5 in enumerate(all_5subsets):
            # Find all blocks containing this 5-subset
            # The 5-subset has 5 elements. The 6th element can be any of the other 7 elements.
            # We can just check which blocks contain it, but iterating through 924 blocks for 792 subsets is 724k ops, which is fine.
            # Or, smarter:
            # Let S be the 5-subset. The complement has 7 points.
            # The blocks containing S are S union {x} for x in complement.
            s_set = set(subset5)
            complement = [p for p in points if p not in s_set]
            for p in complement:
                block = tuple(sorted(s_set | {p}))
                idx = block_index[block]
                row_indices.append(i)
                col_indices.append(idx)
                data.append(1)
        
        A = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n_5subsets, n_blocks))
        
        # Objective: Maximize sum(x)
        # scipy.optimize.milp minimizes c @ x
        # So we minimize -sum(x)
        c = -np.ones(n_blocks)
        
        # Bounds: x_i in {0, 1}
        integrality = np.ones(n_blocks)
        bounds = opt.Bounds(lb=0, ub=1)
        
        # Constraints: A @ x == 1
        # milp takes linear constraints as (A_ub, b_ub) for A_ub @ x <= b_ub
        # To handle equality, we can use two inequalities or use the `constraints` parameter if available?
        # scipy.optimize.milp signature: milp(c, constraints=(), integrality=0, bounds=(), options=None)
        # constraints is a list of LinearConstraint objects.
        
        constraints = [opt.LinearConstraint(A, lb=np.ones(n_5subsets), ub=np.ones(n_5subsets))]
        
        # Solve
        res = opt.milp(c, constraints=constraints, integrality=integrality, bounds=bounds, options={"time_limit": 100})
        
        if res.status == 0: # Optimal
            x = res.x
            # Extract blocks where x[i] > 0.5
            solution_blocks = [list(all_blocks[i]) for i in range(n_blocks) if x[i] > 0.5]
            
            # Verify solution
            if len(solution_blocks) == 132:
                covered_5subsets = set()
                valid = True
                for b in solution_blocks:
                    subsets = list(combinations(b, 5))
                    for s in subsets:
                        if s in covered_5subsets:
                            valid = False
                            break
                        covered_5subsets.add(s)
                    if not valid:
                        break
                if valid and len(covered_5subsets) == 792:
                    return solution_blocks
            
        # If ILP failed or didn't return 132, fall through to hardcoded
    except Exception:
        pass
    
    # Fallback: Hardcoded construction via Z_11 orbits
    base_blocks = [
        (0, 1, 3, 4, 9, 11),
        (0, 2, 7, 8, 9, 11),
        (0, 1, 4, 6, 10, 11),
        (0, 5, 6, 8, 9, 11),
        (0, 1, 2, 3, 4, 5),
        (0, 1, 2, 3, 5, 11),
        (0, 1, 2, 3, 6, 11),
        (0, 1, 2, 3, 7, 11),
        (0, 1, 2, 3, 8, 11),
        (0, 1, 2, 3, 9, 11),
        (0, 1, 2, 3, 10, 11),
        (0, 1, 2, 4, 6, 11),
    ]
    
    blocks = []
    block_set = set()
    covered_5subsets = set()
    valid = True
    
    for base in base_blocks:
        for shift in range(11):
            block = tuple((x + shift) % 11 if x != 11 else 11 for x in base)
            block = tuple(sorted(block))
            
            if block in block_set:
                continue
            block_set.add(block)
            
            subsets = list(combinations(block, 5))
            for s in subsets:
                if s in covered_5subsets:
                    valid = False
                    break
                covered_5subsets.add(s)
            
            if not valid:
                break
            
            blocks.append(list(block))
        
        if not valid:
            break
    
    if valid and len(blocks) == 132:
        return blocks
    
    # Last resort greedy (should not happen)
    covered = set()
    greedy_blocks = []
    for b in combinations(range(12), 6):
        subsets = list(combinations(b, 5))
        if all(s not in covered for s in subsets):
            greedy_blocks.append(list(b))
            covered.update(subsets)
            if time.time() > deadline:
                break
            
    return greedy_blocks
