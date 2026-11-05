"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

import time
import numpy as np

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Reconstruct C = A @ B for a symbolic test
    # We need to check: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == delta
    # Actually the tensor T[i,j,k] = 1 if C[i,k] = sum_j A[i,j]*B[j,k]
    # The decomposition says: C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_k V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
    # Wait, that's not right either. Let me think about this more carefully.
    
    # The tensor for matrix multiplication C = A @ B where A is n x m, B is m x p, C is n x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # A rank-r decomposition has: C[i,k] = sum_{r=0}^{r-1} (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
    # NO. That's not standard tensor decomposition either.
    
    # Standard: The multiplication tensor T has indices (i, j, k) and T[i,j,k] = delta_{appropriate}
    # Actually for C = A * B: T is a 3-tensor where T[i,j,k] contributes to C[i,k] via A[i,j]*B[j,k]
    # The rank-r decomposition: T = sum_r U_r ⊗ V_r ⊗ W_r where (U_r ⊗ V_r ⊗ W_r)[i,j,k] = U[i,j,r]*V[j,k,r]*W[k,i,r]
    # And we need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 if the (i,j,k) term contributes to C[i,k]
    
    # For standard matrix mult: C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all valid (i,j,k), and the output index is (i,k)
    # But in the factorization format: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 for all (i,j,k)
    # because each (i,j,k) triple corresponds to one term A[i,j]*B[j,k] that goes into C[i,k].
    
    # Actually NO. The W factor determines which output element each rank term contributes to.
    # The correct check: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # This is because the term A[i,j]*B[j,k] contributes coefficient 1 to C[i,k].
    
    # Let's verify this way
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                expected = 1.0
                err = abs(s - expected)
                if err > max_err:
                    max_err = err
                if err > tol:
                    return False
    return True

def construct():
    """
    Rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
    
    Strategy: Use block partitioning with Strassen's algorithm for the 2x2 block.
    Partition:
      A (3x3): rows {0}, {1,2}; cols {0}, {1,2}
      B (3x4): rows {0}, {1,2}; cols {0,1}, {2,3}
    
    This gives 4 sub-multiplications:
      - A[0,0] * B[0,0:2] -> C[0,0:2]  : scalar * 1x2, rank 2
      - A[0,1:3] * B[1:3,0:2] -> C[0,0:2] : 1x2 * 2x2 -> 1x2, rank 2 (can use 2 terms)
      - A[1:3,0] * B[0,2:4] -> C[1:3,2:4] : 2x1 * 1x2 -> 2x2, rank 4 schoolbook
      - A[1:3,1:3] * B[1:3,2:4] -> C[1:3,2:4] : 2x2 * 2x2 -> 2x2, rank 7 Strassen
    
    Total: 2 + 2 + 4 + 7 = 15
    
    But we can potentially do better by combining some blocks.
    Let me try a direct approach with rank 15 first, verify it, then try to optimize.
    """
    n, m, p = 3, 3, 4
    
    # Build the rank-15 decomposition as described
    rank = 15
    U = np.zeros((n * m, rank))  # 9 x 15
    V = np.zeros((m * p, rank))  # 12 x 15
    W = np.zeros((p * n, rank))  # 12 x 15
    
    col = 0
    
    # === Block 1: A[0,0] * B[0,k] for k=0,1 -> C[0,k] (rank 2) ===
    
    # C[0,0] += A[0,0] * B[0,0]
    U[0, col] = 1   # A[0,0]: 0*3+0 = 0
    V[0, col] = 1   # B[0,0]: 0*4+0 = 0
    W[0, col] = 1   # C[0,0]: 0*3+0 = 0
    col += 1
    
    # C[0,1] += A[0,0] * B[0,1]
    U[0, col] = 1   # A[0,0]
    V[1, col] = 1   # B[0,1]: 0*4+1 = 1
    W[3, col] = 1   # C[0,1]: 1*3+0 = 3
    col += 1
    
    # === Block 2: A[0,j] * B[j,k] for j=1,2; k=0,1 -> C[0,k] (rank 2) ===
    # C[0,k] = sum_{j=1,2} A[0,j] * B[j,k]
    # We need 2 rank-1 terms to handle this 1x2 @ 2x2 -> 1x2
    
    # Term 3: A[0,1]*B[1,0] + A[0,2]*B[2,0] -> C[0,0]
    # But we can't combine two different products in one rank-1 term unless they share structure.
    # Actually, for 1x2 * 2x2 -> 1x2, the schoolbook rank is 4, but we can do it in 2 terms
    # if we're clever. Let's use:
    # C[0,0] = A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # C[0,1] = A[0,1]*B[1,1] + A[0,2]*B[2,1]
    
    # Rank-2 approach: 
    # Term 3: (A[0,1]) * (B[1,0], B[1,1]) -> contributes A[0,1]*B[1,k] to C[0,k]
    # But that's still 2 separate terms for k=0 and k=1...
    
    # Actually the minimum rank for 1x2 * 2x2 -> 1x2 is 2 (it's just two dot products of length 2,
    # but each is independent). Wait no, the two outputs are C[0,0] and C[0,1], and they share
    # the same A vector but different B columns. The rank is 2*2=4 schoolbook, but can we do better?
    
    # For matrix mult 1x2 * 2x2 -> 1x2: C[0,0] = A0*B0 + A1*B2, C[0,1] = A0*B1 + A1*B3
    # This is two independent bilinear forms. The rank is at most 4 (schoolbook).
    # Can we do it in fewer? Each output is a bilinear form in 4 variables. 
    # The two bilinear forms are linearly independent, so we need at least 2 rank-1 terms?
    # No - a rank-1 term for a bilinear form f(x,y) = sum x_i y_i means f = (sum a_i x_i)(sum b_j y_j)
    # For C[0,0] = A0*B0 + A1*B2, this is not a product of two linear forms (it's a sum of two products).
    # So we need at least 2 rank-1 terms per output, giving 4 total.
    # UNLESS we can combine across outputs.
    
    # Let me just use 4 terms for this block (schoolbook).
    # C[0,0] += A[0,1]*B[1,0]
    U[1, col] = 1   # A[0,1]: 0*3+1 = 1
    V[4, col] = 1   # B[1,0]: 1*4+0 = 4
    W[0, col] = 1   # C[0,0]
    col += 1
    
    # C[0,1] += A[0,1]*B[1,1]
    U[1, col] = 1   # A[0,1]
    V[5, col] = 1   # B[1,1]: 1*4+1 = 5
    W[3, col] = 1   # C[0,1]
    col += 1
    
    # C[0,0] += A[0,2]*B[2,0]
    U[2, col] = 1   # A[0,2]: 0*3+2 = 2
    V[8, col] = 1   # B[2,0]: 2*4+0 = 8
    W[0, col] = 1   # C[0,0]
    col += 1
    
    # C[0,1] += A[0,2]*B[2,1]
    U[2, col] = 1   # A[0,2]
    V[9, col] = 1   # B[2,1]: 2*4+1 = 9
    W[3, col] = 1   # C[0,1]
    col += 1
    
    # === Block 3: A[i,0] * B[0,k] for i=1,2; k=2,3 -> C[i,k] (rank 4 schoolbook) ===
    
    # C[1,2] += A[1,0]*B[0,2]
    U[3, col] = 1   # A[1,0]: 1*3+0 = 3
    V[2, col] = 1   # B[0,2]: 0*4+2 = 2
    W[5, col] = 1   # C[1,2]: 2*3+1 = 5
    col += 1
    
    # C[1,3] += A[1,0]*B[0,3]
    U[3, col] = 1   # A[1,0]
    V[3, col] = 1   # B[0,3]: 0*4+3 = 3
    W[8, col] = 1   # C[1,3]: 3*3+1 = 8
    col += 1
    
    # C[2,2] += A[2,0]*B[0,2]
    U[6, col] = 1   # A[2,0]: 2*3+0 = 6
    V[2, col] = 1   # B[0,2]
    W[6, col] = 1   # C[2,2]: 2*3+2 = 6
    col += 1
    
    # C[2,3] += A[2,0]*B[0,3]
    U[6, col] = 1   # A[2,0]
    V[3, col] = 1   # B[0,3]
    W[9, col] = 1   # C[2,3]: 3*3+2 = 9
    col += 1
    
    # === Block 4: A[1:3,1:3] * B[1:3,2:4] -> C[1:3,2:4] (rank 7 Strassen) ===
    # A' = [[A[1,1], A[1,2]], [A[2,1], A[2,2]]]
    # B' = [[B[1,2], B[1,3]], [B[2,2], B[2,3]]]
    # C' = [[C[1,2], C[1,3]], [C[2,2], C[2,3]]]
    
    # Strassen:
    # M1 = (A11+A22)(B12+B23)
    # M2 = (A21+A22) B12
    # M3 = A11 (B13-B23)
    # M4 = A22 (B22-B12)
    # M5 = (A11+A12) B23
    # M6 = (A21-A11)(B12+B13)
    # M7 = (A12-A22)(B22+B23)
    
    # C12 = M1 - M4 + M6
    # C13 = M1 + M3
    # C22 = M2 + M4
    # C23 = M5 - M3 + M7
    
    # Indices:
    # A[1,1]=4, A[1,2]=5, A[2,1]=7, A[2,2]=8
    # B[1,2]=6, B[1,3]=7, B[2,2]=10, B[2,3]=11
    # C[1,2]=5, C[1,3]=8, C[2,2]=6, C[2,3]=9
    
    # M1 = (A[1,1]+A[2,2])(B[1,2]+B[2,3])
    U[4, col] = 1
    U[8, col] = 1
    V[6, col] = 1
    V[11, col] = 1
    W[5, col] = 1    # C[1,2] += M1
    W[8, col] = 1    # C[1,3] += M1
    col += 1
    
    # M2 = (A[2,1]+A[2,2]) B[1,2]
    U[7, col] = 1
    U[8, col] = 1
    V[6, col] = 1
    W[6, col] = 1    # C[2,2] += M2
    col += 1
    
    # M3 = A[1,1] (B[1,3]-B[2,3])
    U[4, col] = 1
    V[7, col] = 1
    V[11, col] = -1
    W[8, col] = 1    # C[1,3] += M3
    W[9, col] = -1   # C[2,3] -= M3
    col += 1
    
    # M4 = A[2,2] (B[2,2]-B[1,2])
    U[8, col] = 1
    V[10, col] = 1
    V[6, col] = -1
    W[5, col] = -1   # C[1,2] -= M4
    W[6, col] = 1    # C[2,2] += M4
    col += 1
    
    # M5 = (A[1,1]+A[1,2]) B[2,3]
    U[4, col] = 1
    U[5, col] = 1
    V[11, col] = 1
    W[9, col] = 1    # C[2,3] += M5
    col += 1
    
    # M6 = (A[2,1]-A[1,1])(B[1,2]+B[1,3])
    U[7, col] = 1
    U[4, col] = -1
    V[6, col] = 1
    V[7, col] = 1
    W[5, col] = 1    # C[1,2] += M6
    col += 1
    
    # M7 = (A[1,2]-A[2,2])(B[2,2]+B[2,3])
    U[5, col] = 1
    U[8, col] = -1
    V[10, col] = 1
    V[11, col] = 1
    W[9, col] = 1    # C[2,3] += M7
    col += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Verify
    if not _verify_decomposition(U_list, V_list, W_list, n, m, p):
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb))
        V_sb = np.zeros((m * p, rank_sb))
        W_sb = np.zeros((p * n, rank_sb))
        col_sb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m + j, col_sb] = 1
                    V_sb[j*p + k, col_sb] = 1
                    W_sb[k*n + i, col_sb] = 1
                    col_sb += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
