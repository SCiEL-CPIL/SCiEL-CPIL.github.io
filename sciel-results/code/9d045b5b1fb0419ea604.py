import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    total_terms = n * m * p
    rank_limit = 114  # Initial guess based on Strassen hybrid estimate

    # We formulate a MILP to find a sparse tensor decomposition.
    # Variables:
    # x[t, r] in {0, 1}: whether term t is used in rank r
    # c[t, r] in R: coefficient of term t in rank r
    #
    # The tensor T is defined by T[i, j, k] = 1 for all i, j, k.
    # Decomposition: sum_r sum_t x[t, r] c[t, r] * (e_i^U e_j^V e_k^W) = T
    # This implies for each (i, j, k):
    # sum_r c[t_{ijk}, r] = 1, where t_{ijk} is the index of the basis term for (i, j, k).
    #
    # Wait, the basis terms are the unit tensors.
    # The "term" t corresponds to a specific (i, j, k) in the trivial basis.
    # However, Strassen uses linear combinations.
    # A general rank-1 term is u ⊗ v ⊗ w.
    # If we restrict to "sparse" rank-1 terms where u, v, w have small support,
    # we can enumerate candidate rank-1 tensors.
    
    # Given the 120s limit and complexity, a full MILP over all possible rank-1 tensors
    # is infeasible. Instead, we use a constructive heuristic inspired by Strassen
    # for the smaller dimensions and trivial for the rest, which is a known improvement.
    
    # Let's try the specific hybrid construction:
    # Split M dimension (5) into 2 + 3.
    # A is 4x5, B is 5x6.
    # A = [A1 (4x2), A2 (4x3)]
    # B = [B1 (2x6); B2 (3x6)]
    # C = A1 B1 + A2 B2
    
    # For A1 B1 (4x2x6): Use Strassen-like or trivial?
    # Trivial rank for 4x2x6 is 4*2*6 = 48.
    # Strassen for 2x2x2 has rank 7.
    # Can we apply Strassen to 4x2x6?
    # View 4x2 as 2x(2x2)? No.
    # View 2x6 as 2x(3x2)?
    # Let's stick to the trivial decomposition for the 3-part and try to optimize the 2-part.
    # Actually, the previous estimate of 114 was 42 (for 4x2x6?) + 72 (for 4x3x6).
    # 4*3*6 = 72.
    # 4*2*6 = 48.
    # 72 + 48 = 120.
    # So the 114 estimate implies we saved 6 ranks in the 4x2x6 part.
    # Is there a known rank < 48 for 4x2x6?
    # Rank of 4x2x6 tensor.
    # It is known that R(2,2,2)=7.
    # R(2,3,2) ?
    # Let's try a simple block decomposition.
    
    # We will implement the trivial decomposition but with a twist:
    # We will check if a simple Strassen application on a 2x2x2 subblock reduces rank.
    # But 5 is not a power of 2.
    
    # Let's try a different approach:
    # Use the trivial decomposition for all terms except where we can apply a local Strassen.
    # For example, if we have a 2x2x2 subproblem, we can replace 8 terms with 7.
    # In 4x5x6, can we find disjoint 2x2x2 subproblems?
    # Dimensions: 4 (i), 5 (j), 6 (k).
    # We can partition i into 2+2, j into 2+3, k into 2+4?
    # Subblock (i in 0,1; j in 0,1; k in 0,1) is 2x2x2.
    # Subblock (i in 0,1; j in 0,1; k in 2,3) is 2x2x2.
    # Subblock (i in 0,1; j in 0,1; k in 4,5) is 2x2x2.
    # Subblock (i in 2,3; j in 0,1; k in 0,1) is 2x2x2.
    # Subblock (i in 2,3; j in 0,1; k in 2,3) is 2x2x2.
    # Subblock (i in 2,3; j in 0,1; k in 4,5) is 2x2x2.
    # So for j in {0,1}, we have 6 subblocks of 2x2x2.
    # Each 2x2x2 block has rank 8 trivially, rank 7 with Strassen.
    # So for j=0,1, we have 6 blocks * 7 = 42 terms.
    # For j=2,3,4, we have trivial terms.
    # j=2: 4*6 = 24 terms.
    # j=3: 4*6 = 24 terms.
    # j=4: 4*6 = 24 terms.
    # Total = 42 + 72 = 114.
    
    # This matches the estimate!
    # We will implement this construction.
    
    rank = 114
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # We need to map these to the global indices.
    # For a subblock defined by i0, i1, j0, j1, k0, k1:
    # The local indices are (li, lj, lk) in {0,1}^3.
    # Global indices: i = i0 + li, j = j0 + lj, k = k0 + lk.
    
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        # The 7 terms
        # 1. (A00+A11)(B00+B11)
        # U: A00, A11
        # V: B00, B11
        # W: C00, C11
        # Coefficients:
        # U[0,0]=1, U[1,1]=1
        # V[0,0]=1, V[1,1]=1
        # W[0,0]=1, W[1,1]=1
        
        # Term 1
        U[i0 * m + j0, col] = 1
        U[(i0+1) * m + (j0+1), col] = 1
        V[j0 * p + k0, col] = 1
        V[(j0+1) * p + (k0+1), col] = 1
        W[k0 * n + i0, col] = 1
        W[(k0+1) * n + (i0+1), col] = 1
        col += 1
        
        # Term 2: (A10+A11)B00
        U[i0 * m + (j0+1), col] = 1
        U[(i0+1) * m + (j0+1), col] = 1
        V[j0 * p + k0, col] = 1
        W[k0 * n + (i0+1), col] = 1
        col += 1
        
        # Term 3: A00(B10-B11)
        U[i0 * m + j0, col] = 1
        V[(j0+1) * p + k0, col] = 1
        V[(j0+1) * p + (k0+1), col] = -1
        W[k0 * n + i0, col] = 1
        W[(k0+1) * n + i0, col] = -1
        col += 1
        
        # Term 4: A11(B01-B00)
        U[(i0+1) * m + (j0+1), col] = 1
        V[j0 * p + (k0+1), col] = 1
        V[j0 * p + k0, col] = -1
        W[(k0+1) * n + (i0+1), col] = 1
        W[k0 * n + (i0+1), col] = -1
        col += 1
        
        # Term 5: (A00+A01)B11
        U[i0 * m + j0, col] = 1
        U[i0 * m + (j0+1), col] = 1
        V[(j0+1) * p + (k0+1), col] = 1
        W[(k0+1) * n + i0, col] = 1
        col += 1
        
        # Term 6: (A10-A00)(B00+B01)
        U[i0 * m + (j0+1), col] = 1
        U[i0 * m + j0, col] = -1
        V[j0 * p + k0, col] = 1
        V[j0 * p + (k0+1), col] = 1
        W[k0 * n + (i0+1), col] = 1
        W[(k0+1) * n + (i0+1), col] = 1
        col += 1
        
        # Term 7: (A01-A11)(B10+B11)
        U[i0 * m + (j0+1), col] = 1
        U[(i0+1) * m + (j0+1), col] = -1
        V[(j0+1) * p + k0, col] = 1
        V[(j0+1) * p + (k0+1), col] = 1
        W[k0 * n + i0, col] = 1
        W[(k0+1) * n + i0, col] = 1
        col += 1

    # Apply Strassen to blocks where j in {0,1} and i,k are partitioned into pairs
    for i0 in range(0, n, 2):
        for k0 in range(0, p, 2):
            for j0 in [0, 1]:
                add_strassen_block(i0, j0, k0)
                
    # Apply trivial to remaining blocks (j in {2,3,4})
    for i in range(n):
        for j in [2, 3, 4]:
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    # Verify col == rank
    assert col == rank, f"Rank mismatch: {col} vs {rank}"

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
