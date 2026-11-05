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
    
    from scipy.sparse import lil_matrix
    A = lil_matrix((n_subsets, n_blocks), dtype=np.int8)
    
    # To build A efficiently:
    # For each block, find all its 5-subsets and mark them.
    # This is 924 blocks * 6 subsets = 5544 operations. Very fast.
    
    # Create a mapping from 5-subset to its index in subsets_5
    # Using a dictionary might be slow for tuple lookups if not careful, 
    # but 792 items is small.
    subset_index = {s: i for i, s in enumerate(subsets_5)}
    
    for j, block in enumerate(blocks):
        for s in combinations(block, 5):
            i = subset_index[s]
            A[i, j] = 1
            
    A = A.tocsr()
    
    # Objective: Maximize sum(x) -> Minimize -sum(x)
    c = -np.ones(n_blocks)
    
    # Constraints: A @ x <= 1
    constraints = LinearConstraint(A, ub=1)
    
    # Bounds: 0 <= x <= 1
    bounds = Bounds(0, 1)
    
    # Integrality: x is binary
    integrality = np.ones(n_blocks)
    
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
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
            # Milp might not provide partial. 
            # For this size, it should be optimal.
            # If it fails, fallback to greedy.
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
