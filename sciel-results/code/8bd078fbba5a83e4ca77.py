# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def ring_routing() -> np.ndarray:
    # Problem data
    N = 10
    D = 15
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2),
    ]
    weights = np.array([d[2] for d in demands], dtype=float)
    sources = np.array([d[0] for d in demands], dtype=int)
    targets = np.array([d[1] for d in demands], dtype=int)

    # Precompute edge usage matrices
    # cw_matrix[i, e] = 1 if demand i uses edge e when going CW
    # ccw_matrix[i, e] = 1 if demand i uses edge e when going CCW
    cw_matrix = np.zeros((D, N), dtype=int)
    ccw_matrix = np.zeros((D, N), dtype=int)
    
    for i in range(D):
        s, t = sources[i], targets[i]
        e = s
        while e != t:
            cw_matrix[i, e] = 1
            e = (e + 1) % N
        ccw_matrix[i] = 1 - cw_matrix[i]
    
    # Precompute for fast load calculation
    # When demand i goes CW, it adds weights[i] to edges where cw_matrix[i,e]=1
    # When demand i goes CCW, it adds weights[i] to edges where ccw_matrix[i,e]=1
    
    # For each demand i, precompute the vector of load contributions
    # If demand i goes CW: contribution to edge loads = weights[i] * cw_matrix[i]
    # If demand i goes CCW: contribution to edge loads = weights[i] * ccw_matrix[i]
    
    cw_contribs = weights[:, np.newaxis] * cw_matrix  # shape (D, N)
    ccw_contribs = weights[:, np.newaxis] * ccw_matrix  # shape (D, N)
    
    def compute_loads_fast(assignment):
        """Compute edge loads given a boolean assignment (True = CCW)."""
        # For each demand, if assignment[i] is True (CCW), use ccw_contribs[i]
        # otherwise use cw_contribs[i]
        # This is: sum over i of (assignment[i] * ccw_contribs[i] + (1-assignment[i]) * cw_contribs[i])
        # = sum_i cw_contribs[i] + sum_i assignment[i] * (ccw_contribs[i] - cw_contribs[i])
        
        # Precompute base = sum of all cw_contribs
        # delta = ccw_contribs - cw_contribs
        # loads = base + assignment @ delta
        
        base_loads = np.sum(cw_contribs, axis=0)  # shape (N,)
        delta = ccw_contribs - cw_contribs  # shape (D, N)
        # assignment is boolean array of shape (D,)
        loads = base_loads + assignment.astype(float) @ delta
        return loads
    
    def max_load(assignment):
        return compute_loads_fast(assignment).max()
    
    # Exhaustive search over all 2^15 = 32768 possible assignments
    # This is fast enough and guarantees finding the global optimum
    
    best_assign = None
    best_ml = float('inf')
    
    # Use Gray code for efficient enumeration
    # But for 32768 iterations, even simple enumeration is fine
    
    for mask in range(1 << D):
        # Convert mask to boolean array
        # Bit i of mask corresponds to demand i: 1 = CCW, 0 = CW
        assignment = np.array([(mask >> i) & 1 for i in range(D)], dtype=bool)
        
        ml = max_load(assignment)
        if ml < best_ml:
            best_ml = ml
            best_assign = assignment
    
    # Convert to output format: >= 0.5 means CCW
    result = best_assign.astype(float) * 1.0
    return result

# EVOLVE-BLOCK-END
