import numpy as np
from itertools import combinations
from scipy.optimize import milp, LinearConstraint, Bounds
import time

def construct():
    best_edges = None
    best_count = float('inf')
    best_n = 9
    
    start_time = time.time()
    
    # Try n from 9 to 13
    # n=9 is guaranteed to have a solution (K_9^5 is not 2-colorable)
    for n in range(9, 14):
        if time.time() - start_time > 100:
            break
            
        edges = list(combinations(range(n), 5))
        M = len(edges)
        
        # Precompute edge masks
        edge_masks = np.array([sum(1 << v for v in e) for e in edges])
        
        # Generate constraints
        # We fix vertex 0 to color 0.
        # Other vertices 1..n-1 can be 0 or 1.
        # Total colorings: 2^(n-1)
        
        cons_rows = []
        
        # Iterate over all subsets of {1, ..., n-1}
        for mask in range(1 << (n - 1)):
            # Construct full coloring mask
            full_mask = 1 # Vertex 0 is 0
            # Bits 1 to n-1 are determined by 'mask'
            # mask bit i corresponds to vertex i+1
            # We can compute this efficiently
            # full_mask = 1 | (mask << 1)
            full_mask = 1 | (mask << 1)
            
            # Find edges that are monochromatic under this coloring
            # An edge is monochromatic if (edge_mask & full_mask) == edge_mask (all 0)
            # OR (edge_mask & full_mask) == 0 (all 1)
            
            # Vectorized check
            # all_0 = (edge_masks & full_mask) == edge_masks
            # all_1 = (edge_masks & full_mask) == 0
            # mono = all_0 | all_1
            
            all_0 = (edge_masks & full_mask) == edge_masks
            all_1 = (edge_masks & full_mask) == 0
            mono = all_0 | all_1
            
            if np.any(mono):
                # Create a row for this constraint
                # We only need to add the row if there is at least one monochromatic edge
                # If no edge is monochromatic, this coloring is a valid 2-coloring of the complete graph,
                # which means no subset can be non-2-colorable. (But this doesn't happen for n>=9)
                cons_rows.append(mono.astype(float))
            else:
                # If no edge is monochromatic, the constraint is 0 >= 1, which is infeasible.
                # This means K_n^5 is 2-colorable, so no solution for this n.
                break
        
        if len(cons_rows) == 0:
            continue
            
        A = np.array(cons_rows)
        
        # Solve ILP
        c = np.ones(M)
        constraints = LinearConstraint(A, lb=1, ub=np.inf)
        integrality = np.ones(M)
        bounds = Bounds(lb=0, ub=1)
        
        try:
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
            if res.status == 0:
                selected = res.x > 0.5
                count = int(selected.sum())
                if count < best_count:
                    best_count = count
                    best_edges = [list(edges[i]) for i in range(M) if selected[i]]
                    best_n = n
        except Exception:
            continue
            
    if best_edges:
        return {'n': best_n, 'edges': best_edges}
    else:
        # Fallback
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
