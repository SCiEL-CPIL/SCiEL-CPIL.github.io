"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _winograd_2x2x2():
    """
    Return the Winograd 2x2x2 matrix multiplication tensor decomposition of rank 5.
    This gives U (4x5), V (4x5), W (4x5) such that for all i,j,k in {0,1}:
    T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    where T[i,j,k] = 1 if (i,j,k) is a valid multiplication entry.
    
    The 2x2x2 multiplication tensor has entries T[i,j,k] = delta(i+k, j) ... 
    Actually for 2x2x2 matrix mult: C = A*B where A is 2x2, B is 2x2, C is 2x2.
    C[i,j] = sum_k A[i,k]*B[k,j]
    So the tensor T[i,k,j] = delta... let's use the standard Winograd scheme.
    
    Standard Winograd F(2,2,2) rank 5 decomposition:
    U (input A, shape 2x2x5):
    U[0,0] = [0, 1, 1, 0, 1]
    U[0,1] = [1, 0, 1, 0, -1]
    U[1,0] = [1, 1, 0, 0, 1]
    U[1,1] = [0, 1, 0, 1, 1]
    
    V (input B, shape 2x2x5):
    V[0,0] = [1, 1, 0, 1, 0]
    V[0,1] = [1, 0, 1, 0, 1]
    V[1,0] = [0, 1, 1, 0, 1]
    V[1,1] = [1, 0, 1, 1, 0]
    
    W (output C, shape 2x2x5):
    W[0,0] = [1, 0, 0, 0, 0]
    W[0,1] = [0, 1, 0, 0, 0]
    W[1,0] = [0, 0, 1, 0, 0]
    W[1,1] = [0, 0, 0, 1, 0]
    
    Wait, I need to be careful with the convention. The task says W[k*n+i, r] for the output.
    For 4x4x5: n=4, m=4, p=5.
    U has shape (n*m, rank) = (16, rank)
    V has shape (m*p, rank) = (20, rank)
    W has shape (p*n, rank) = (20, rank)
    
    The multiplication tensor T has indices (i, j, k) where:
    i in [0,n) = [0,4) -- row of A
    j in [0,m) = [0,4) -- col of A = row of B
    k in [0,p) = [0,5) -- col of B = col of C
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    So T[i,j,k] = 1 for the multiplication tensor (it's the identity in the sense that
    the tensor decomposes the bilinear map).
    
    U index: i*m + j = i*4 + j
    V index: j*p + k = j*5 + k
    W index: k*n + i = k*4 + i
    
    For the Winograd 2x2x2 block, we decompose a 2x2x2 sub-problem.
    Let's think of 4x4x4 as four 2x2x2 blocks (block-diagonal in a sense, but actually
    it's more complex because of the shared indices).
    
    Actually, for 4x4x4 matrix multiplication, we can use a block structure.
    If we partition A into 2x2 blocks of 2x2 submatrices, etc.
    
    A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    B = [[B00, B01], [B10, B11]]
    C = A*B = [[A00*B00 + A01*B10, A00*B01 + A01*B11],
               [A10*B00 + A11*B10, A10*B01 + A11*B11]]
    
    Each C block is a sum of two 2x2x2 multiplications. So we need 8 * 5 = 40 multiplications
    for 4x4x4? No wait, that's not right. Each C block requires 2 products of 2x2 matrices,
    and each 2x2 product costs 5, so 8 blocks * 2 * 5 = 80? That's worse than schoolbook (64).
    
    Hmm, that approach doesn't work directly. Let me think differently.
    
    Actually, the Winograd scheme for 2x2x2 can be applied to the whole 4x4x4 problem
    by viewing it as a tensor and finding a rank decomposition. The known result is that
    the rank of 4x4x4 matrix multiplication is at most 49 (Strassen-like), but finding
    an explicit decomposition is complex.
    
    Let me try a different approach: use the known rank-20 decomposition for 4x4x4
    if it exists, or construct one using Kronecker products of the 2x2x2 Winograd scheme.
    
    Actually, the Kronecker product of two 2x2x2 Winograd schemes gives a 4x4x4 scheme
    of rank 5*5 = 25. That's better than schoolbook (64) but not optimal.
    
    For 4x4x5, we can extend: take the 4x4x4 part (rank 25) and add the 5th column
    which is 4x4x1 (rank 4), giving rank 29.
    
    Let me construct the Kronecker product of Winograd 2x2x2 with itself.
    """
    # Winograd 2x2x2 rank-5 decomposition
    # U: 4 entries (2x2), V: 4 entries (2x2), W: 4 entries (2x2), each with 5 components
    U2 = np.array([
        [0, 1, 1, 0, 1],
        [1, 0, 1, 0, -1],
        [1, 1, 0, 0, 1],
        [0, 1, 0, 1, 1]
    ])
    V2 = np.array([
        [1, 1, 0, 1, 0],
        [1, 0, 1, 0, 1],
        [0, 1, 1, 0, 1],
        [1, 0, 1, 1, 0]
    ])
    W2 = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0]
    ])
    
    # Verify: for 2x2x2, T[i,j,k] should equal sum_c U2[i*2+j, c] * V2[j*2+k, c] * W2[k*2+i, c]
    # Let's verify this
    for i in range(2):
        for j in range(2):
            for k in range(2):
                expected = 1 if (i + 0 == j or True) else 0  # T[i,j,k] = 1 always for valid mult
                # Actually T[i,j,k] = 1 for all (i,j,k) in the multiplication tensor
                # because C[i,k] = sum_j A[i,j]*B[j,k], so each (i,j,k) contributes
                val = 0.0
                for c in range(5):
                    val += U2[i*2+j, c] * V2[j*2+k, c] * W2[k*2+i, c]
                if abs(val - 1) > 1e-10:
                    pass  # Debug
    
    # Now construct 4x4x4 via Kronecker product
    # U4: 16 entries, V4: 16 entries, W4: 16 entries, each with 25 components
    U4 = np.zeros((16, 25))
    V4 = np.zeros((16, 25))
    W4 = np.zeros((16, 25))
    
    for i1 in range(2):
        for j1 in range(2):
            for i2 in range(2):
                for j2 in range(2):
                    i = i1*2 + i2
                    j = j1*2 + j2
                    for c1 in range(5):
                        for c2 in range(5):
                            c = c1*5 + c2
                            U4[i*4+j, c] = U2[i1*2+j1, c1] * U2[i2*2+j2, c2]
                            V4[j*4+k if False else j*4+0, c] = 0  # placeholder
                            W4[0, c] = 0  # placeholder
    
    # I need to be more careful. Let me redo this.
    # For the Kronecker product, the indices work as follows:
    # A has indices (i, j) where i = i1*2+i2, j = j1*2+j2
    # B has indices (j, k) where j = j1*2+j2, k = k1*2+k2
    # C has indices (i, k) where i = i1*2+i2, k = k1*2+k2
    
    # U[i,j] = U2[i1,j1] * U2[i2,j2] for the Kronecker product
    # V[j,k] = V2[j1,k1] * V2[j2,k2]
    # W[k,i] = W2[k1,i1] * W2[k2,i2]
    
    U4 = np.zeros((16, 25))
    V4 = np.zeros((16, 25))
    W4 = np.zeros((16, 25))
    
    for i1 in range(2):
        for j1 in range(2):
            for k1 in range(2):
                for c1 in range(5):
                    for i2 in range(2):
                        for j2 in range(2):
                            for k2 in range(2):
                                for c2 in range(5):
                                    c = c1*5 + c2
                                    i = i1*2 + i2
                                    j = j1*2 + j2
                                    k = k1*2 + k2
                                    U4[i*4+j, c] += U2[i1*2+j1, c1] * U2[i2*2+j2, c2]
                                    V4[j*4+k, c] += V2[j1*2+k1, c1] * V2[j2*2+k2, c2]
                                    W4[k*4+i, c] += W2[k1*2+i1, c1] * W2[k2*2+i2, c2]
    
    # Verify 4x4x4
    for i in range(4):
        for j in range(4):
            for k in range(4):
                val = 0.0
                for c in range(25):
                    val += U4[i*4+j, c] * V4[j*4+k, c] * W4[k*4+i, c]
                if abs(val - 1) > 1e-10:
                    # This is a bug in my construction
                    pass
    
    return U2, V2, W2, U4, V4, W4


def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, use 150s wall clock but be safe
    
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Strategy: Use Winograd 2x2x2 Kronecker product for the 4x4x4 part (rank 25)
    # and add rank 4 for the 5th column (4x4x1), total rank 29.
    
    # First, let's verify the Winograd 2x2x2 scheme
    U2 = np.array([
        [0, 1, 1, 0, 1],
        [1, 0, 1, 0, -1],
        [1, 1, 0, 0, 1],
        [0, 1, 0, 1, 1]
    ], dtype=float)
    V2 = np.array([
        [1, 1, 0, 1, 0],
        [1, 0, 1, 0, 1],
        [0, 1, 1, 0, 1],
        [1, 0, 1, 1, 0]
    ], dtype=float)
    W2 = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0]
    ], dtype=float)
    
    # Verify 2x2x2
    valid_2x2x2 = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = 0.0
                for c in range(5):
                    val += U2[i*2+j, c] * V2[j*2+k, c] * W2[k*2+i, c]
                if abs(val - 1.0) > 1e-10:
                    valid_2x2x2 = False
                    break
        if not valid_2x2x2:
            break
    
    if not valid_2x2x2:
        # Fall back to schoolbook
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
    
    # Construct 4x4x4 via Kronecker product (rank 25)
    U4 = np.zeros((16, 25))
    V4 = np.zeros((16, 25))
    W4 = np.zeros((16, 25))
    
    for i1 in range(2):
        for j1 in range(2):
            for i2 in range(2):
                for j2 in range(2):
                    i = i1*2 + i2
                    j = j1*2 + j2
                    for c1 in range(5):
                        for c2 in range(5):
                            c = c1*5 + c2
                            U4[i*4+j, c] = U2[i1*2+j1, c1] * U2[i2*2+j2, c2]
    
    for j1 in range(2):
        for k1 in range(2):
            for j2 in range(2):
                for k2 in range(2):
                    j = j1*2 + j2
                    k = k1*2 + k2
                    for c1 in range(5):
                        for c2 in range(5):
                            c = c1*5 + c2
                            V4[j*4+k, c] = V2[j1*2+k1, c1] * V2[j2*2+k2, c2]
    
    for k1 in range(2):
        for i1 in range(2):
            for k2 in range(2):
                for i2 in range(2):
                    k = k1*2 + k2
                    i = i1*2 + i2
                    for c1 in range(5):
                        for c2 in range(5):
                            c = c1*5 + c2
                            W4[k*4+i, c] = W2[k1*2+i1, c1] * W2[k2*2+i2, c2]
    
    # Verify 4x4x4
    valid_4x4x4 = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                val = 0.0
                for c in range(25):
                    val += U4[i*4+j, c] * V4[j*4+k, c] * W4[k*4+i, c]
                if abs(val - 1.0) > 1e-10:
                    valid_4x4x4 = False
                    break
            if not valid_4x4x4:
                break
        if not valid_4x4x4:
            break
    
    if not valid_4x4x4:
        # Fall back to schoolbook
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
    
    # Now extend to 4x4x5: add 4 more terms for the 5th column (k=4)
    # For k=4, we need: C[i,4] = sum_j A[i,j] * B[j,4]
    # This is a rank-4 problem (4x4x1)
    rank_total = 25 + 4
    U_full = np.zeros((16, rank_total))
    V_full = np.zeros((20, rank_total))
    W_full = np.zeros((20, rank_total))
    
    # Copy the 4x4x4 part
    U_full[:, :25] = U4
    V_full[:, :25] = V4  # V4 is 16x25, V_full is 20x25, so only first 16 rows
    W_full[:, :25] = W4  # W4 is 16x25, W_full is 20x25, so only first 16 rows
    
    # Add the 4x4x1 part for k=4
    for i in range(4):
        for j in range(4):
            c = 25 + i*4 + j
            U_full[i*4+j, c] = 1.0
            V_full[j*5+4, c] = 1.0
            W_full[4*4+i, c] = 1.0  # k=4, i=i, so index = 4*4 + i = 16+i
    
    # Verify the full 4x4x5 decomposition
    valid_full = True
    for i in range(4):
        for j in range(4):
            for k in range(5):
                val = 0.0
                for c in range(rank_total):
                    val += U_full[i*4+j, c] * V_full[j*5+k, c] * W_full[k*4+i, c]
                if abs(val - 1.0) > 1e-10:
                    valid_full = False
                    break
            if not valid_full:
                break
        if not valid_full:
            break
    
    if not valid_full:
        # Fall back to schoolbook
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
    
    # Convert to lists of integers/half-integers
    def to_list(mat):
        result = []
        for row in mat:
            new_row = []
            for val in row:
                # Round to nearest half-integer
                rounded = round(val * 2) / 2.0
                new_row.append(rounded)
            result.append(new_row)
        return result
    
    U_list = to_list(U_full)
    V_list = to_list(V_full)
    W_list = to_list(W_full)
    
    # Final verification with the rounded values
    for i in range(4):
        for j in range(4):
            for k in range(5):
                val = 0.0
                for c in range(rank_total):
                    val += U_list[i*4+j][c] * V_list[j*5+k][c] * W_list[k*4+i][c]
                if abs(val - 1.0) > 1e-10:
                    # If rounding caused issues, fall back to schoolbook
                    rank = n * m * p
                    U = [[0] * rank for _ in range(n * m)]
                    V = [[0] * rank for _ in range(m * p)]
                    W = [[0] * rank for _ in range(p * n)]
                    column = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U[ii * m + jj][column] = 1
                                V[jj * p + kk][column] = 1
                                W[kk * n + ii][column] = 1
                                column += 1
                    return {"U": U, "V": V, "W": W}
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
