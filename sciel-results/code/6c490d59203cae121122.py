"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x4 matrix.
    
    Strategy: Try to find a decomposition with rank less than 36 (schoolbook) by
    searching over possible rank-1 tensor combinations with half-integer coefficients.
    """
    n, m, p = DIMENSIONS  # 3, 3, 4
    nm, mp, pn = n * m, m * p, p * n  # 9, 12, 12
    
    # Build the multiplication tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Try ranks from a reasonable lower bound up to schoolbook rank
    # Lower bound: at least max(nm, mp, pn) / min(nm, mp, pn) * something
    # Practical lower bound for 3x3x3x4: try from 24 to 36
    
    best_rank = nm * mp  # 36
    best_decomp = None
    
    # First, return schoolbook as fallback
    U_school = np.zeros((nm, best_rank))
    V_school = np.zeros((mp, best_rank))
    W_school = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    best_decomp = (U_school, V_school, W_school)
    
    # Try to find lower rank decompositions using a structured approach
    # We'll try ranks from 24 to 35
    
    for target_rank in range(24, 36):
        # Try to find a decomposition with this rank
        # Use a heuristic: start with schoolbook and try to merge terms
        
        # For each pair of schoolbook terms, check if they can be merged
        # into a single term with half-integer coefficients
        
        # Actually, let's try a different approach: use known constructions
        # For 3x3 x 3x4, try to build from 3x3 x 3x3 decomposition
        
        # Let's try a direct search: for small ranks, try random initializations
        # and use iterative refinement
        
        found = False
        for attempt in range(50):
            # Initialize with random vectors scaled to half-integers
            np.random.seed(attempt * 1000 + target_rank)
            
            # Try to construct a decomposition
            U = np.zeros((nm, target_rank))
            V = np.zeros((mp, target_rank))
            W = np.zeros((pn, target_rank))
            
            # Initialize with schoolbook-like structure but with random perturbations
            for c in range(target_rank):
                # Pick a random (i,j,k) to base this term on
                i = np.random.randint(n)
                j = np.random.randint(m)
                k = np.random.randint(p)
                
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                
                # Add some half-integer perturbations
                for _ in range(3):
                    idx_u = np.random.randint(nm)
                    idx_v = np.random.randint(mp)
                    idx_w = np.random.randint(pn)
                    U[idx_u, c] += np.random.choice([-0.5, 0.5])
                    V[idx_v, c] += np.random.choice([-0.5, 0.5])
                    W[idx_w, c] += np.random.choice([-0.5, 0.5])
            
            # Check if this decomposition is exact
            T_recon = np.zeros((nm, mp, pn))
            for c in range(target_rank):
                T_recon += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
            
            if np.allclose(T_recon, T, atol=1e-10):
                found = True
                best_rank = target_rank
                best_decomp = (U, V, W)
                break
        
        if found:
            break
    
    U, V, W = best_decomp
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
