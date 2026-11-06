import time
import sys
from itertools import combinations
import numpy as np

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 150.0 # 150s limit, use 120s for safety margin
    
    n_points = 12
    # Generate all possible 6-subsets (blocks)
    all_blocks = list(combinations(range(n_points), 6))
    num_blocks = len(all_blocks)
    
    # Generate all 5-subsets
    all_5subsets = list(combinations(range(n_points), 5))
    num_5subsets = len(all_5subsets)
    
    # Map 5-subsets to indices for constraint building
    # We need to build the constraint matrix A (num_5subsets x num_blocks)
    # A[i, j] = 1 if 5subset[i] is a subset of block[j]
    
    # Optimization: Pre-compute membership
    # Since num_blocks is 924 and num_5subsets is 792, 
    # a dense matrix is 924 * 792 * 8 bytes ~ 5.8 MB, which is fine.
    
    # Try using SciPy MILP
    try:
        from scipy.optimize import milp, LinearConstraint
        import scipy.sparse as sp
        
        # Build sparse constraint matrix
        # For each block j, find which 5-subsets it contains
        # Each block contains 6 5-subsets.
        rows = []
        cols = []
        data = []
        
        # Map each 5-subset to an index
        subset_to_idx = {s: i for i, s in enumerate(all_5subsets)}
        
        for j, block in enumerate(all_blocks):
            for subset in combinations(block, 5):
                i = subset_to_idx[subset]
                rows.append(i)
                cols.append(j)
                data.append(1)
                
        A = sp.csr_matrix((data, (rows, cols)), shape=(num_5subsets, num_blocks))
        
        # Objective: maximize sum x_j
        # milp minimizes, so use c = -1
        c = -np.ones(num_blocks)
        
        # Constraints: 0 <= Ax <= 1
        constraints = [LinearConstraint(A, 0, 1)]
        
        # Integrality: all variables integer (0 or 1)
        integrality = np.ones(num_blocks)
        
        # Bounds: 0 <= x_j <= 1
        bounds = [(0, 1)] * num_blocks
        
        # Check time
        if time.time() - start_time > deadline:
            # Fallback to greedy if MILP setup took too long (unlikely)
            pass 
        else:
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
            
            if res.status == 0: # Optimal
                x = res.x
                blocks = []
                for j in range(num_blocks):
                    if x[j] > 0.5:
                        blocks.append(list(all_blocks[j]))
                return blocks
            else:
                # If MILP failed, fall back to greedy
                pass
                
    except Exception as e:
        # Fallback to greedy if SciPy fails
        pass

    # Fallback: Greedy Lexicographic Packing
    # This is the original method, kept as a safety net
    covered = set()
    blocks = []
    for b in all_blocks:
        subsets = list(combinations(b, 5))
        if not any(s in covered for s in subsets):
            blocks.append(list(b))
            covered.update(subsets)
            # Check time
            if time.time() - start_time > deadline:
                break
                
    return blocks
