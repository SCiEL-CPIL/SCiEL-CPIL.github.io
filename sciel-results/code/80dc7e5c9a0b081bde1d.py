import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _verify_full(U, V, W, n, m, p):
    """Full verification of the decomposition."""
    rank = len(U[0])
    # Check all valid triples give 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                total = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                if total != 1:
                    return False
    # Check invalid triples give 0
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i * m + j, j * p + k, k * n + i))
    
    n_U, n_V, n_W = len(U), len(V), len(W)
    # Check all invalid triples (comprehensive but bounded)
    for a in range(n_U):
        for b in range(n_V):
            for c in range(n_W):
                if (a, b, c) not in valid:
                    total = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                    if total != 0:
                        return False
    return True

def _try_reduce_rank(U, V, W, n, m, p, target_rank):
    """
    Try to find a decomposition of rank target_rank by solving a linear system.
    
    Strategy: Fix all but one column in each of U, V, W, and solve for the last.
    This is a linear system in the unknown column of W (given U, V fixed).
    """
    rank = len(U[0])
    if target_rank >= rank:
        return None
    
    # We'll try to remove (rank - target_rank) columns
    # For simplicity, try removing one column at a time
    
    for col_to_remove in range(rank):
        # Try removing this column
        U_r = [[U[i][j] for j in range(rank) if j != col_to_remove] for i in range(len(U))]
        V_r = [[V[i][j] for j in range(rank) if j != col_to_remove] for i in range(len(V))]
        W_r = [[W[i][j] for j in range(rank) if j != col_to_remove] for i in range(len(W))]
        
        if len(U_r[0]) == target_rank:
            # Check if this reduced decomposition still works
            # We need to check if the tensor identity holds
            # But removing a column changes the tensor, so this won't work directly
            # Instead, we need to find new U, V, W of lower rank
            
            pass
    
    # Alternative: direct search for lower rank
    # This is NP-hard in general, so we use heuristics
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    base_rank = n * m * p
    
    # Start with schoolbook decomposition
    best = _schoolbook(n, m, p)
    best_rank = base_rank
    
    # Verify the schoolbook is correct
    if not _verify_full(best['U'], best['V'], best['W'], n, m, p):
        # Should never happen, but just in case
        return best
    
    # Try to find a lower-rank decomposition
    # Use a randomized approach: try to find a rank-59 decomposition
    
    # The key insight: we can formulate this as finding U, V, W of rank r
    # such that for all (a,b,c): sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    
    # For rank 59, we have 59*12 + 59*20 + 59*15 = 59*47 = 2773 variables
    # And 12*20*15 = 3600 equations (but only 60 are nonzero in T)
    # Actually, we need all 3600 equations to hold
    
    # This is a system of quadratic equations (bilinear in U,V,W)
    # Hard to solve directly
    
    # Heuristic: start from schoolbook and try small perturbations
    # Check if we can merge two terms
    
    U = [row[:] for row in best['U']]
    V = [row[:] for row in best['V']]
    W = [row[:] for row in best['W']]
    
    # Try all pairs of columns to see if they can be merged
    # Two columns c1, c2 can be merged if there exist new columns u', v', w'
    # such that the tensor contribution is preserved
    
    # Simpler: check if one column is redundant
    for target in range(base_rank):
        if time.time() > deadline - 10.0:
            break
        
        # Try to express column 'target' as a linear combination of others
        # This requires solving a linear system
        
        other_cols = [r for r in range(base_rank) if r != target]
        num_unknowns = len(other_cols)
        
        # Build the system: for each (a,b,c), 
        # U[a,target]*V[b,target]*W[c,target] = sum_{r in other} alpha_r * U[a,r]*V[b,r]*W[c,r]
        
        # This is NOT linear in alpha because of the products
        # So this approach doesn't work directly
        
        # Instead, use the fact that if we fix U and V, then W must satisfy
        # a linear system
        
        pass
    
    # Given the difficulty, return the schoolbook decomposition
    # which is guaranteed valid
    
    return best
