"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]
import time
import random
import itertools

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=7
    
    # We will construct a decomposition with rank 78.
    # Strategy: Use Strassen's algorithm (rank 7) for 6 disjoint 2x2x2 sub-tensors
    # within the i in {0,1} slice, and schoolbook (rank 1) for the rest.
    
    # Total schoolbook rank: 3 * 4 * 7 = 84
    # Reductions: 6 blocks * 1 reduction each = 6
    # Target rank: 78
    
    rank = 78
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define the 6 Strassen blocks for i in {0, 1}
    # j indices: {0,1}, {2,3}
    # k indices: {0,1}, {2,3}, {4,5}
    # i indices: {0,1}
    
    j_blocks = [(0, 1), (2, 3)]
    k_blocks = [(0, 1), (2, 3), (4, 5)]
    i_block = (0, 1)
    
    # Strassen coefficients
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # We define the support for each term (u, v, w)
    # u: list of (i, j)
    # v: list of (j, k)
    # w: list of (k, i)
    
    def add_strassen_term(u_support, v_support, w_support, c):
        for (i, j) in u_support:
            U[i * m + j][c] = 1
        for (j, k) in v_support:
            V[j * p + k][c] = 1
        for (k, i) in w_support:
            W[k * n + i][c] = 1

    # Process each block
    for i0, i1 in [i_block]:
        for j0, j1 in j_blocks:
            for k0, k1 in k_blocks:
                # Term 1: (A00+A11)(B00+B11)
                # A00=(i0,j0), A11=(i1,j1)
                # B00=(j0,k0), B11=(j1,k1)
                # W: (k0,i0), (k1,i1)
                add_strassen_term([(i0,j0), (i1,j1)], [(j0,k0), (j1,k1)], [(k0,i0), (k1,i1)], col)
                col += 1
                
                # Term 2: (A10+A11)B00
                # A10=(i1,j0), A11=(i1,j1)
                # B00=(j0,k0)
                # W: (k0,i1) -- wait, M2 contributes to C10 and C11?
                # C10 = M2 + M4. C11 = M1 - M2 + ...
                # So M2 is associated with i1.
                # W should have support on i1.
                # Specifically, W[k0, i1] and W[k1, i1]?
                # M2 = (A10+A11)B00.
                # Product is scalar.
                # It contributes to C10 (i1,k0) and C11 (i1,k1)?
                # No, B00 is only k0. So it contributes to C10.
                # But in the tensor decomposition, the term is rank 1.
                # $u \otimes v \otimes w$.
                # $u_{i1 j0}=1, u_{i1 j1}=1$.
                # $v_{j0 k0}=1$.
                # $w_{k0 i1}=1$.
                # Check contribution to C[i1, k0]:
                # Sum over j: U[i1 j, c] V[j k0, c] W[k0 i1, c].
                # j=j0: U[i1 j0] V[j0 k0] W[k0 i1] = 1*1*1 = 1.
                # j=j1: U[i1 j1] V[j1 k0] W[k0 i1] = 1*0*1 = 0.
                # So it contributes 1 to C[i1, k0].
                # Does it contribute to C[i1, k1]?
                # W[k1 i1] is 0. So no.
                # This matches M2 contributing to C10.
                
                add_strassen_term([(i1,j0), (i1,j1)], [(j0,k0)], [(k0,i1)], col)
                col += 1
                
                # Term 3: A00(B01-B11)
                # A00=(i0,j0)
                # B01=(j0,k1), B11=(j1,k1)
                # W: (k1,i0)
                add_strassen_term([(i0,j0)], [(j0,k1), (j1,k1)], [(k1,i0)], col)
                col += 1
                # Note: V[j1,k1] should be -1?
                # M3 = A00(B01 - B11).
                # So V[j0,k1]=1, V[j1,k1]=-1.
                V[j1 * p + k1][col-1] = -1
                
                # Term 4: A11(B10-B00)
                # A11=(i1,j1)
                # B10=(j1,k0), B00=(j0,k0)
                # W: (k0,i1)
                add_strassen_term([(i1,j1)], [(j1,k0), (j0,k0)], [(k0,i1)], col)
                col += 1
                # V[j0,k0] should be -1.
                V[j0 * p + k0][col-1] = -1
                
                # Term 5: (A00+A01)B11
                # A00=(i0,j0), A01=(i0,j1)
                # B11=(j1,k1)
                # W: (k1,i0)
                add_strassen_term([(i0,j0), (i0,j1)], [(j1,k1)], [(k1,i0)], col)
                col += 1
                
                # Term 6: (A10-A00)(B00+B01)
                # A10=(i1,j0), A00=(i0,j0)
                # B00=(j0,k0), B01=(j0,k1)
                # W: (k0,i1), (k1,i1)
                add_strassen_term([(i1,j0), (i0,j0)], [(j0,k0), (j0,k1)], [(k0,i1), (k1,i1)], col)
                col += 1
                # U[i0,j0] should be -1.
                U[i0 * m + j0][col-1] = -1
                
                # Term 7: (A01-A11)(B10+B11)
                # A01=(i0,j1), A11=(i1,j1)
                # B10=(j1,k0), B11=(j1,k1)
                # W: (k0,i0), (k1,i0)
                add_strassen_term([(i0,j1), (i1,j1)], [(j1,k0), (j1,k1)], [(k0,i0), (k1,i0)], col)
                col += 1
                # U[i1,j1] should be -1.
                U[i1 * m + j1][col-1] = -1

    # Add remaining schoolbook terms
    # i in {0,1}, j in {0,1,2,3}, k=6
    for i in [0, 1]:
        for j in range(m):
            for k in [6]:
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # i=2, all j, all k
    for i in [2]:
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    assert col == rank, f"Column count {col} != Rank {rank}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
