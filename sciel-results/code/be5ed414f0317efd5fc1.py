import time
import random
import numpy as np
from itertools import combinations

def construct():
    """
    Constructs the Witt design S(5,6,12) by solving an Integer Linear Programming (ILP) problem.
    If the ILP solver is not available or fails, it falls back to a random greedy heuristic.
    """
    deadline = time.time() + 140
    
    # Precompute all 6-subsets (blocks) and 5-subsets (constraints)
    points = range(12)
    all_blocks = list(combinations(points, 6))
    all_5subsets = list(combinations(points, 5))
    
    n_blocks = len(all_blocks)
    n_constraints = len(all_5subsets)
    
    # Map 5-subsets to indices
    subset_to_idx = {s: i for i, s in enumerate(all_5subsets)}
    
    # Build the constraint matrix A (n_constraints x n_blocks)
    # A[i, j] = 1 if all_5subsets[i] is a subset of all_blocks[j]
    # Since all_5subsets are size 5 and all_blocks are size 6,
    # a 5-subset is in a 6-block if and only if it is one of the 6 subsets of that block.
    
    # We can build this efficiently.
    # For each block, find its 5-subsets and mark them in the matrix.
    # Using a dense matrix for 792x924 is ~700k entries, which is fine.
    A = np.zeros((n_constraints, n_blocks), dtype=np.int8)
    
    for j, block in enumerate(all_blocks):
        # Generate the 6 five-subsets of this block
        for k in range(6):
            # Remove the k-th element to get a 5-subset
            sub = tuple(list(block[:k]) + list(block[k+1:]))
            idx = subset_to_idx[sub]
            A[idx, j] = 1
            
    # Objective: Maximize sum of x_j
    c = np.ones(n_blocks)
    
    # Constraints: sum_{j} A[i, j] x_j <= 1 for all i
    # This is equivalent to: -1 <= sum A x <= 1? No, 0 <= sum A x <= 1.
    # Since x >= 0, sum A x >= 0 is automatic.
    # So we need sum A x <= 1.
    # In scipy.optimize.milp, constraints are defined as lb <= A x <= ub.
    # We can use a single LinearConstraint with A, lb=0, ub=1.
    
    try:
        from scipy.optimize import LinearConstraint, milp
        
        cons = LinearConstraint(A, lb=0, ub=1)
        integrality = np.ones(n_blocks)  # All variables are binary/integer
        bounds = [(0, 1)] * n_blocks
        
        # Solve
        res = milp(c=c, constraints=cons, integrality=integrality, bounds=bounds, 
                   options={'time_limit': 120})
        
        if res.success and res.x is not None:
            blocks = []
            for j in range(n_blocks):
                if res.x[j] > 0.5:
                    blocks.append(list(all_blocks[j]))
            # Verify validity just in case
            if len(blocks) == 132:
                return blocks
            # If not 132, it might be a sub-optimal solution or solver issue
            # But ILP should find 132.
            
    except ImportError:
        pass
    except Exception:
        pass
    
    # Fallback: Random Greedy
    # This is the logic from the current program, but we ensure it runs if ILP fails.
    
    block_subsets = {}
    for b in all_blocks:
        # Precompute the 5-subsets for greedy check
        subs = []
        for k in range(6):
            sub = tuple(list(b[:k]) + list(b[k+1:]))
            subs.append(sub)
        block_subsets[b] = subs
        
    best_blocks = []
    best_count = 0
    
    # Use a fixed seed for reproducibility in fallback, or random
    # Given the deadline, we do as many restarts as possible.
    for attempt in range(1000):
        if time.time() > deadline:
            break
        
        shuffled = all_blocks[:]
        random.shuffle(shuffled)
        
        covered = set()
        blocks = []
        
        for b in shuffled:
            subsets = block_subsets[b]
            # Check if any 5-subset is already covered
            conflict = False
            for s in subsets:
                if s in covered:
                    conflict = True
                    break
            if not conflict:
                blocks.append(b)
                covered.update(subsets)
        
        if len(blocks) > best_count:
            best_count = len(blocks)
            best_blocks = blocks[:]
            if best_count == 132:
                break
                
    return [list(b) for b in best_blocks]

if __name__ == "__main__":
    blocks = construct()
    print(f"Found {len(blocks)} blocks")
