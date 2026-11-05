"""Matrix multiplication 4x4x8 via Strassen-based tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def _strassen_222():
    """Return (U,V,W) each a list of 4 vectors of length 7 for 2x2x2 Strassen."""
    # 7 linear forms (M1..M7); each row is [a11,a12,a21,a22] for A,
    # [b11,b12,b21,b22] for B, [c11,c12,c21,c22] for C.
    M = [
        # M1 = (a11+a22)(b11+b22)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2 = (a21+a22)b11
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1]),
        # M3 = a11(b21-b22)
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 0, 1, 0]),
        # M4 = a22(b12-b11)
        ([0, 0, 0, 1], [1, -1, 0, 0], [0, 1, 0, 0]),
        # M5 = (a11+a12)b22
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1]),
        # M6 = (a21-a11)(b11+b12)
        ([-1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]),
        # M7 = (a12-a22)(b21+b22)
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 0, 1, 0]),
    ]
    U = [list(a) for a, b, c in M]
    V = [list(b) for a, b, c in M]
    W = [list(c) for a, b, c in M]
    return U, V, W

def _rank2_212():
    """Return (U,V,W) for 2x1x2 multiplication, rank 2."""
    # C = A * B where A is 2x1, B is 1x2.
    # C11 = a1*b1, C12 = a1*b2, C21 = a2*b1, C22 = a2*b2.
    # Rank-2: M1 = a1*(b1,b2), M2 = a2*(b1,b2)
    U = [[1, 0], [0, 1]]           # a1, a2
    V = [[1, 1], [1, 1]]           # b1, b2 (same for both)
    W = [[1, 0], [1, 0], [0, 1], [0, 1]]  # c11,c12,c21,c22
    return U, V, W

def _embed_224_into_222(U222, V222, W222, U212, V212, W212):
    """
    Build a rank-6 decomposition of 2x2x4 (A:2x2, B:2x4, C:2x4).
    Split B into two 2x2 blocks: B = [B_left, B_right].
    Use Strassen for the 2x2x2 left part (rank 7? No - we use rank 6 by sharing).
    Actually, let's just use Strassen for each 2x2x2 half: rank 7+7=14.
    But we can do better: the 2x2x4 tensor has rank 6.
    Let me construct it directly.
    
    C = A * B where A is 2x2, B is 2x4.
    C is 2x4. C[i,k] = sum_j A[i,j]*B[j,k].
    
    Rank-6 decomposition:
    M1 = A11 * B1 (row 1 of B) -> contributes to C[0,:]*A11
    M2 = A12 * B2 (row 2 of B) -> contributes to C[0,:]*A12
    M3 = A21 * B1 -> contributes to C[1,:]*A21
    M4 = A22 * B2 -> contributes to C[1,:]*A22
    That's only rank 4! Because C[i,k] = A[i,1]*B[1,k] + A[i,2]*B[2,k].
    So rank is 4.
    """
    # Actually 2x2x4 has rank 4.
    # M1: A11 * B1row, M2: A12 * B2row, M3: A21 * B1row, M4: A22 * B2row
    U = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]  # A11,A12,A21,A22
    V = [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]  # B1row for M1,M2; B2row for M3,M4
    # Wait, V has 4 rows (B is 2x4, so m*p = 2*4=8 entries). Let me be careful.
    # B is 2x4: B[0,0],B[0,1],B[0,2],B[0,3],B[1,0],B[1,1],B[1,2],B[1,3]
    # M1: A11 * B[0,:] -> V row for M1: [1,1,1,1,0,0,0,0]
    # M2: A12 * B[1,:] -> V row for M2: [0,0,0,0,1,1,1,1]
    # M3: A21 * B[0,:] -> V row for M3: [1,1,1,1,0,0,0,0]
    # M4: A22 * B[1,:] -> V row for M4: [0,0,0,0,1,1,1,1]
    V = [[1,1,1,1,0,0,0,0], [0,0,0,0,1,1,1,1], [1,1,1,1,0,0,0,0], [0,0,0,0,1,1,1,1]]
    # W: C is 2x4, so p*n = 4*2 = 8 entries: C[0,0],C[0,1],C[0,2],C[0,3],C[1,0],C[1,1],C[1,2],C[1,3]
    # M1: A11*B[0,:] -> C[0,:] += A11*B[0,:] -> W row: [1,1,1,1,0,0,0,0]
    # M2: A12*B[1,:] -> C[0,:] += A12*B[1,:] -> W row: [1,1,1,1,0,0,0,0]
    # M3: A21*B[0,:] -> C[1,:] += A21*B[0,:] -> W row: [0,0,0,0,1,1,1,1]
    # M4: A22*B[1,:] -> C[1,:] += A22*B[1,:] -> W row: [0,0,0,0,1,1,1,1]
    W = [[1,1,1,1,0,0,0,0], [1,1,1,1,0,0,0,0], [0,0,0,0,1,1,1,1], [0,0,0,0,1,1,1,1]]
    return U, V, W

def _build_4x4x8_decomposition():
    """
    Build the 4x4x8 decomposition.
    Split A (4x4) into 2x2 blocks: A = [[A00, A01], [A10, A11]]
    Split B (4x8) into 2x4 blocks: B = [[B00, B01], [B10, B11]] where each is 2x4
    C (4x8) = A*B: C = [[A00*B00+A01*B10, A00*B01+A01*B11], [A10*B00+A11*B10, A10*B01+A11*B11]]
    
    Each 2x2x4 product has rank 4. There are 4 such products, so total rank = 4*4 = 16?
    Wait, but the products are independent. Let me count:
    C00 = A00*B00 + A01*B10 (2x2 * 2x4 + 2x2 * 2x4)
    C01 = A00*B01 + A01*B11
    C10 = A10*B00 + A11*B10
    C11 = A10*B01 + A11*B11
    
    Each C block is 2x4. The product A_ij * B_jk is 2x2 * 2x4 = rank 4.
    So each C block requires 2 rank-4 decompositions = rank 8.
    Total: 4 blocks * 8 = 32.
    
    Hmm, that's rank 32. Let me verify with Strassen.
    
    Actually, let me just use the direct rank-4 for each 2x2x4 and sum them up.
    Total rank = 4 (blocks) * 2 (products per block) * 4 (rank per 2x2x4) = 32.
    
    But wait, can I do better? The 4x4x8 tensor... Let me think about using Strassen on the 4x4x4 part.
    
    Alternative: Use Strassen 2x2x2 (rank 7) for the 4x4x4 subproblem (rank 7*4=28 for 4x4x4),
    and handle the 4x4x4 extra part separately.
    
    4x4x8 = 4x4x4 + 4x4x4 (two halves of B).
    Each 4x4x4 via Strassen: rank 28. Total: 56. Worse than 32.
    
    So the direct block approach gives rank 32. Let me verify this is correct.
    """
    n, m, p = 4, 4, 8
    # U: (n*m, rank) = (16, rank)
    # V: (m*p, rank) = (32, rank)
    # W: (p*n, rank) = (32, rank)
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    # A indices: A[i,j] where i in 0..3, j in 0..3. Index in U: i*m+j = i*4+j
    # B indices: B[j,k] where j in 0..3, k in 0..7. Index in V: j*p+k = j*8+k
    # C indices: C[k*n+i, r]... wait, the convention is W[k*n+i, r] for C[i,k]?
    # Let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
    # So W is indexed by [k*n+i, r] where the C entry is C[i,k]? Or C[k,i]?
    # Standard: C = A*B, C[i,k] = sum_j A[i,j]*B[j,k].
    # The convention W[k*n+i, r] suggests the C matrix is stored as C[k,i] in row-major?
    # Actually, I think it means: the (i,k) entry of C is at row k*n+i in W.
    # So W row index = k*n + i, where C[i,k] is the entry.
    
    # Let me just build it and verify.
    
    # For each 2x2x4 product A_sub * B_sub:
    # A_sub is 2x2, B_sub is 2x4, C_sub is 2x4.
    # Rank-4: M1=A11*B1row, M2=A12*B2row, M3=A21*B1row, M4=A22*B2row
    
    # Block (bi, bj) of A (2x2), block (bj, bk) of B (2x4), contributing to block (bi, bk) of C (2x4)
    # bi in {0,1}, bj in {0,1}, bk in {0,1}
    
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                # A_sub: rows 2*bi, 2*bi+1; cols 2*bj, 2*bj+1
                # B_sub: rows 2*bj, 2*bj+1; cols 4*bk .. 4*bk+3
                # C_sub: rows 2*bi, 2*bi+1; cols 4*bk .. 4*bk+3
                
                # M1: A[2bi, 2bj] * B[2bj, 4bk:4bk+4]
                # M2: A[2bi, 2bj+1] * B[2bj+1, 4bk:4bk+4]
                # M3: A[2bi+1, 2bj] * B[2bj, 4bk:4bk+4]
                # M4: A[2bi+1, 2bj+1] * B[2bj+1, 4bk:4bk+4]
                
                for mi in range(4):
                    U_row = [0] * 16
                    V_row = [0] * 32
                    W_row = [0] * 32
                    
                    if mi == 0:
                        # A[2bi, 2bj]
                        U_row[2*bi * 4 + 2*bj] = 1
                        # B[2bj, 4bk:4bk+4]
                        for dk in range(4):
                            V_row[2*bj * 8 + 4*bk + dk] = 1
                        # C[2bi, 4bk:4bk+4]
                        for dk in range(4):
                            # W index: k*n + i = (4bk+dk)*4 + 2bi
                            W_row[(4*bk + dk) * 4 + 2*bi] = 1
                    elif mi == 1:
                        # A[2bi, 2bj+1]
                        U_row[2*bi * 4 + 2*bj + 1] = 1
                        # B[2bj+1, 4bk:4bk+4]
                        for dk in range(4):
                            V_row[(2*bj + 1) * 8 + 4*bk + dk] = 1
                        # C[2bi, 4bk:4bk+4]
                        for dk in range(4):
                            W_row[(4*bk + dk) * 4 + 2*bi] = 1
                    elif mi == 2:
                        # A[2bi+1, 2bj]
                        U_row[(2*bi + 1) * 4 + 2*bj] = 1
                        # B[2bj, 4bk:4bk+4]
                        for dk in range(4):
                            V_row[2*bj * 8 + 4*bk + dk] = 1
                        # C[2bi+1, 4bk:4bk+4]
                        for dk in range(4):
                            W_row[(4*bk + dk) * 4 + 2*bi + 1] = 1
                    elif mi == 3:
                        # A[2bi+1, 2bj+1]
                        U_row[(2*bi + 1) * 4 + 2*bj + 1] = 1
                        # B[2bj+1, 4bk:4bk+4]
                        for dk in range(4):
                            V_row[(2*bj + 1) * 8 + 4*bk + dk] = 1
                        # C[2bi+1, 4bk:4bk+4]
                        for dk in range(4):
                            W_row[(4*bk + dk) * 4 + 2*bi + 1] = 1
                    
                    U_rows.append(U_row)
                    V_rows.append(V_row)
                    W_rows.append(W_row)
    
    return U_rows, V_rows, W_rows

def _verify_decomposition(U, V, W):
    """Verify that the decomposition is exact."""
    n, m, p = 4, 4, 8
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # The tensor T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Let's check a few entries.
    rank = len(U[0])
    T = np.zeros((n, m, p))
    for r in range(rank):
        U_col = U[:, r]
        V_col = V[:, r]
        W_col = W[:, r]
        # Reshape
        U_mat = U_col.reshape(n, m)
        V_mat = V_col.reshape(m, p)
        W_mat = W_col.reshape(p, n)
        # T += U_mat * V_mat * W_mat (outer product style)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += U_mat[i, j] * V_mat[j, k] * W_mat[k, i]
    
    # The multiplication tensor: T[i,j,k] = 1 if k is the output index, i is row of A, j is col of A = row of B
    # Actually, the tensor for matrix mult: T[i,j,k] = delta_{j,k}... no.
    # C[i,k] = sum_j A[i,j] B[j,k]. So the tensor has T[i,j,k] = 1 for all i,j,k? No.
    # The tensor is T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k].
    # So T[i,j,k] = 1 for all valid i,j,k.
    expected = np.ones((n, m, p))
    
    max_err = np.max(np.abs(T - expected))
    return max_err

def construct():
    start_time = time.time()
    deadline = 120.0 - 10.0  # 10s safety margin
    
    U, V, W = _build_4x4x8_decomposition()
    
    # Verify
    err = _verify_decomposition(U, V, W)
    if err > 1e-10:
        # Fallback to schoolbook
        n, m, p = DIMENSIONS
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
