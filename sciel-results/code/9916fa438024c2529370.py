"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def _strassen_2x2_factors():
    """
    Return (U, V, W) for 2x2 matrix multiplication using Strassen's algorithm.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    n = 2
    rank = 7
    U = np.zeros((n * n, rank))
    V = np.zeros((n * n, rank))
    W = np.zeros((n * n, rank))

    # Strassen's 7 products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A10)B11
    # M6 = (A11 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # Index mapping: U row = i*2+j, V row = j*2+k, W row = k*2+i
    
    # M1 = (A00+A11)(B00+B11)
    # U rows: A00 -> row 0, A11 -> row 3
    U[0, 0] = 1
    U[3, 0] = 1
    # V rows: B00 -> row 0, B11 -> row 3
    V[0, 0] = 1
    V[3, 0] = 1
    # W rows: C00 needs M1, C11 needs M1
    # C00 = k=0,i=0 -> W[0*2+0] = W[0]
    # C11 = k=1,i=1 -> W[1*2+1] = W[3]
    W[0, 0] = 1
    W[3, 0] = 1

    # M2 = (A10+A11)B00
    # U rows: A10 -> row 2, A11 -> row 3
    U[2, 1] = 1
    U[3, 1] = 1
    # V rows: B00 -> row 0
    V[0, 1] = 1
    # W: C10 needs M2 -> k=0,i=1 -> W[0*2+1] = W[1]
    W[1, 1] = 1

    # M3 = A00(B10-B11)
    # U rows: A00 -> row 0
    U[0, 2] = 1
    # V rows: B10 -> row 2, B11 -> row 3
    V[2, 2] = 1
    V[3, 2] = -1
    # W: C01 needs M3 -> k=1,i=0 -> W[1*2+0] = W[2]
    # C11 needs M3 -> k=1,i=1 -> W[1*2+1] = W[3]
    W[2, 2] = 1
    W[3, 2] = 1

    # M4 = A11(B01-B00)
    # U rows: A11 -> row 3
    U[3, 3] = 1
    # V rows: B01 -> row 1, B00 -> row 0
    V[1, 3] = 1
    V[0, 3] = -1
    # W: C00 needs M4 -> W[0]
    # C10 needs M4 -> W[1]
    W[0, 3] = 1
    W[1, 3] = 1

    # M5 = (A00+A10)B11
    # U rows: A00 -> row 0, A10 -> row 2
    U[0, 4] = 1
    U[2, 4] = 1
    # V rows: B11 -> row 3
    V[3, 4] = 1
    # W: C00 needs -M5 -> W[0]
    # C01 needs M5 -> W[2]
    W[0, 4] = -1
    W[2, 4] = 1

    # M6 = (A11-A00)(B00+B01)
    # U rows: A11 -> row 3, A00 -> row 0
    U[3, 5] = 1
    U[0, 5] = -1
    # V rows: B00 -> row 0, B01 -> row 1
    V[0, 5] = 1
    V[1, 5] = 1
    # W: C11 needs M6 -> W[3]
    W[3, 5] = 1

    # M7 = (A01-A11)(B10+B11)
    # U rows: A01 -> row 1, A11 -> row 3
    U[1, 6] = 1
    U[3, 6] = -1
    # V rows: B10 -> row 2, B11 -> row 3
    V[2, 6] = 1
    V[3, 6] = 1
    # W: C00 needs M7 -> W[0]
    W[0, 6] = 1

    return U, V, W


def _expand_2x2_to_4x4(U2, V2, W2):
    """
    Expand 2x2 Strassen factors to 4x4 factors.
    The 4x4 matrix is partitioned into four 2x2 blocks.
    We apply Strassen to the block structure.
    
    Actually, for 4x4 we can directly use the recursive Strassen approach:
    Split 4x4 into 2x2 blocks, apply Strassen at block level.
    
    U4 shape: (16, 7), V4 shape: (16, 7), W4 shape: (16, 7)
    """
    n = 4
    rank = 7
    U4 = np.zeros((n * n, rank))
    V4 = np.zeros((n * n, rank))
    W4 = np.zeros((n * n, rank))
    
    # Block indices for 4x4 split into 2x2 blocks:
    # Block (0,0): rows 0-1, cols 0-1 -> local indices (0,0)
    # Block (0,1): rows 0-1, cols 2-3 -> local indices (0,1)
    # Block (1,0): rows 2-3, cols 0-1 -> local indices (1,0)
    # Block (1,1): rows 2-3, cols 2-3 -> local indices (1,1)
    
    # For each Strassen product r (0..6):
    # U2 has entries at rows corresponding to A-blocks
    # V2 has entries at rows corresponding to B-blocks
    # W2 has entries at rows corresponding to C-blocks
    
    for r in range(rank):
        # U2 column r: which A-blocks are involved and their coefficients
        # U2 row index = i2*2+j2 where (i2,j2) is the block position in A
        for u2_row in range(4):
            coeff = U2[u2_row, r]
            if coeff != 0:
                i2, j2 = u2_row // 2, u2_row % 2
                # This corresponds to A-block (i2, j2)
                # In the 4x4 matrix, A-block (i2,j2) occupies rows i2*2..i2*2+1, cols j2*2..j2*2+1
                # U4 row index = i*4+j where i is row in 4x4, j is col in 4x4
                for di in range(2):
                    for dj in range(2):
                        i = i2 * 2 + di
                        j = j2 * 2 + dj
                        U4[i * 4 + j, r] = coeff
        
        # V2 column r: which B-blocks are involved
        for v2_row in range(4):
            coeff = V2[v2_row, r]
            if coeff != 0:
                j2, k2 = v2_row // 2, v2_row % 2
                # This corresponds to B-block (j2, k2)
                # In the 4x4 matrix, B-block (j2,k2) occupies rows j2*2..j2*2+1, cols k2*2..k2*2+1
                for dj in range(2):
                    for dk in range(2):
                        j = j2 * 2 + dj
                        k = k2 * 2 + dk
                        V4[j * 4 + k, r] = coeff
        
        # W2 column r: which C-blocks are involved
        for w2_row in range(4):
            coeff = W2[w2_row, r]
            if coeff != 0:
                k2, i2 = w2_row // 2, w2_row % 2
                # This corresponds to C-block (k2, i2)
                # In the 4x4 matrix, C-block (k2,i2) occupies rows k2*2..k2*2+1, cols i2*2..i2*2+1
                for dk in range(2):
                    for di in range(2):
                        k = k2 * 2 + dk
                        i = i2 * 2 + di
                        W4[k * 4 + i, r] = coeff
    
    return U4, V4, W4


def _verify_decomposition(U, V, W, n=4, m=4, p=4, tol=1e-10):
    """
    Verify that the tensor decomposition is exact.
    For all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 if i,j,k form a valid triple
    Actually, the matrix multiplication tensor T[i,j,k] = 1 for all i,j,k in range(n), and the
    decomposition should satisfy:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Wait, let me reconsider. The matrix multiplication tensor is:
    C[i,k] = sum_j A[i,j] * B[j,k]
    So T[i,j,k] = 1 for all i,j,k (it's a 3-tensor of all ones? No...)
    
    Actually, the tensor is T[i,j,k] where C_ik = sum_j A_ij B_jk.
    The tensor T has T[i,j,k] = delta... no.
    
    Let me think again. The bilinear form is:
    C_ik = sum_j A_ij * B_jk
    
    This can be written as:
    C_ik = sum_r (sum_j U[i,j,r] * A_ij) * (sum_k V[j,k,r] * B_jk) ... no that's not right either.
    
    The correct decomposition is:
    C_ik = sum_r U[i*m+j... 
    
    Let me use the standard formulation. For matrix multiplication C = AB:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    A rank-r decomposition means:
    C[i,k] = sum_{r=1}^{r} (sum_j U[i,j,r] * A[i,j]) * (sum_k V[j,k,r] * B[j,k]) ... 
    
    No, the standard form is:
    C[i,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] ... 
    
    I think the convention here is:
    U has shape (n*m, rank), indexed as U[i*m+j, r]
    V has shape (m*p, rank), indexed as V[j*p+k, r]  
    W has shape (p*n, rank), indexed as W[k*n+i, r]
    
    And the decomposition satisfies:
    For all i, j, k:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    Because the matrix multiplication tensor T[i,j,k] = 1 for all i,j,k? No, that's not right.
    
    Actually, I think the tensor is such that:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    And the decomposition is:
    sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_k V[j*p+k, r] * B[j,k]) ... 
    
    I'm getting confused. Let me just verify empirically by checking that the decomposition
    reproduces matrix multiplication.
    """
    rank = U.shape[1]
    # Build the tensor T[i,j,k] from the decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This should equal 1 for all i,j,k if it's the all-ones tensor
    # But that's not the matrix multiplication tensor...
    
    # Actually, I think the correct check is:
    # The matrix multiplication tensor T has T[i,j,k] = 1 for all i,j,k
    # No wait, that would be a rank-1 tensor.
    
    # Let me reconsider. The matrix multiplication tensor is defined as:
    # T[i,j,k] = 1 if the indices form a valid "edge" in the multiplication
    # Actually, I think T[i,j,k] = 1 for all i,j,k, and the bilinear form is:
    # C[i,k] = sum_j A[i,j] * B[j,k] = sum_j sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] A[i,j] B[j,k]
    
    # Hmm, I think the correct interpretation is that the decomposition gives:
    # C[i,k] = sum_r (sum_j U[i*m+j,r] * A[i,j]) * (sum_k V[j*p+k,r] * B[j,k])
    # But that doesn't make sense because j is free in the second sum.
    
    # Let me just do an empirical check: create random matrices A and B, compute C = A*B,
    # and also compute C' using the decomposition, and check they match.
    
    # Using the decomposition:
    # C'[i,k] = sum_r (sum_j U[i*m+j,r] * A[i,j]) * (sum_j V[j*p+k,r] * B[j,k])
    # Wait, that's not right either because both sums are over j.
    
    # I think the correct formula is:
    # C'[i,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] * A[i,j] * B[j,k] summed over j
    # But that has j appearing in U, V, W indices...
    
    # OK let me just look at the schoolbook example:
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 where c = i*m*p + j*p + k
    # So for each (i,j,k), there's one rank-1 term.
    # The product U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 only for the specific (i,j,k)
    # corresponding to column c.
    
    # So the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For schoolbook, T[i,j,k] = 1 for all i,j,k.
    
    # So the matrix multiplication tensor is indeed the all-ones tensor? That seems wrong.
    
    # Wait, I think I'm misunderstanding. Let me re-read the task.
    # "Find a rank-minimal exact tensor decomposition for multiplying a 4 by 4 matrix by a 4 by 4 matrix."
    
    # The matrix multiplication tensor T is a 3-tensor where T[i,j,k] represents the coefficient
    # of A[i,j] * B[j,k] in C[i,k]. Since C[i,k] = sum_j A[i,j] * B[j,k], we have
    # T[i,j,k] = 1 for all i,j,k.
    
    # So yes, the tensor is all-ones, and we need to decompose it as:
    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    
    # For schoolbook, each (i,j,k) gets its own rank-1 term, giving rank n*m*p.
    # For Strassen, we group terms cleverly to reduce the rank.
    
    # So my verification should check:
    # For all i,j,k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True


def construct():
    start_time = time.time()
    deadline = 150.0 - 5.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with Strassen 4x4 (rank 7)
    U2, V2, W2 = _strassen_2x2_factors()
    U, V, W = _expand_2x2_to_4x4(U2, V2, W2)
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        # Convert to lists for the return format
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
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
# EVOLVE-BLOCK-END
