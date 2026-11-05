"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify that the tensor decomposition is exact."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    # Build the full tensor T[i,j,k] which should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Actually, the convention is: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T[i,j,k] = 1 for all i,j,k
    # And the decomposition says: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x5 matrix.
    
    Strategy: Use Strassen-based decomposition for the 2x2x2 blocks, and schoolbook for the remaining dimensions.
    
    We decompose C = A * B where A is 2x4 and B is 4x5.
    Split A into two 2x2 blocks: A = [A1, A2]
    Split B into columns: B = [B1; B2; B3; B4; B5] where B1,B2 are 2x1, B3,B4 are 2x1, B5 is 1x1... 
    
    Actually, let me think of this differently.
    
    C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    We can split the 4 columns of B into two groups of 2, and the 5 rows of B into groups.
    
    Let me use a more systematic approach:
    - Split j into j1 in {0,1} and j2 in {2,3}
    - Split k into k1 in {0,1,2} and k2 in {3,4}
    
    For the 2x2x2 sub-problems, we can use Strassen (rank 7).
    For the 2x2x1 sub-problems, we can use rank 4 (schoolbook).
    
    Sub-problems:
    1. A[:,0:2] * B[0:2, 0:2] -> 2x2x2, rank 7 (Strassen)
    2. A[:,2:4] * B[2:4, 0:2] -> 2x2x2, rank 7 (Strassen)
    3. A[:,0:2] * B[0:2, 2:4] -> 2x2x2, rank 7 (Strassen)
    4. A[:,2:4] * B[2:4, 2:4] -> 2x2x2, rank 7 (Strassen)
    5. A[:,0:2] * B[0:2, 4] -> 2x2x1, rank 4
    6. A[:,2:4] * B[2:4, 4] -> 2x2x1, rank 4
    
    Total: 7+7+7+7+4+4 = 36
    
    This gives rank 36, which is better than the schoolbook rank of 40.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Build the rank-36 Strassen-based decomposition
    # We'll construct U, V, W matrices of shape (n*m, rank), (m*p, rank), (p*n, rank)
    
    # First, let's define the Strassen decomposition for 2x2x2
    # Strassen's 7 products:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # For 2x2x1, we use 4 products (schoolbook):
    # P1 = A11*B1
    # P2 = A12*B2
    # P3 = A21*B1
    # P4 = A22*B2
    
    rank = 36
    U = [[0.0] * rank for _ in range(n * m)]  # 8 rows
    V = [[0.0] * rank for _ in range(m * p)]  # 20 rows
    W = [[0.0] * rank for _ in range(p * n)]  # 10 rows
    
    col = 0
    
    # Helper: add a Strassen 2x2x2 decomposition
    # a_rows: indices of the 2 rows of A (in the n*m space)
    # b_cols: indices of the 2 cols of B (in the m*p space, these are the j indices)
    # k_rows: indices of the 2 rows of C output (in the p*n space, these are the k indices)
    
    def add_strassen_222(a_row_base, b_col_base, k_row_base):
        nonlocal col
        # A is indexed by (i, j) where i in {0,1}, j in {0,1} relative to base
        # B is indexed by (j, k) where j in {0,1}, k in {0,1} relative to base
        # C is indexed by (i, k) where i in {0,1}, k in {0,1} relative to base
        
        # In the global indexing:
        # A[i+m*j] for i in {0,1}, j in {0,1} -> a_row_base + i + 2*j (but we need to be careful)
        # Actually, A has shape (n, m) = (2, 4). So A[i, j] is at row i*m + j = i*4 + j
        # B has shape (m, p) = (4, 5). So B[j, k] is at row j*p + k = j*5 + k
        # C has shape (n, p) = (2, 5). The output W is indexed by k*n + i
        
        # For the sub-problem A[:, j0:j0+2] * B[j0:j0+2, k0:k0+2]:
        # A rows: i*4 + (j0 + j) for i in {0,1}, j in {0,1}
        # B rows: (j0 + j)*5 + (k0 + k) for j in {0,1}, k in {0,1}
        # W indices: (k0 + k)*2 + i for k in {0,1}, i in {0,1}
        
        j0 = b_col_base  # This is the starting j index
        k0 = k_row_base  # This is the starting k index
        
        # Strassen products
        # M1 = (A11 + A22)(B11 + B22)
        # A11 = A[0, j0], A22 = A[1, j0+1]
        # B11 = B[j0, k0], B22 = B[j0+1, k0+1]
        
        # M1
        U[a_row_base + 0][col] = 1.0  # A[0, j0]
        U[a_row_base + 2][col] = 1.0  # A[1, j0+1] -> wait, A[1, j0+1] is at row 1*4 + (j0+1) = 4 + j0 + 1
        # Hmm, I need to be more careful with indexing.
        
        # Let me redefine. A has rows indexed by i*4 + j for i in {0,1}, j in {0,1,2,3}
        # So A[0, j0] is at row 0*4 + j0 = j0
        # A[1, j0+1] is at row 1*4 + (j0+1) = 4 + j0 + 1
        
        # Let me restart with explicit indices.
        pass
    
    # This is getting complicated. Let me just construct it explicitly.
    
    # Clear and rebuild
    for row in range(n * m):
        for col in range(rank):
            U[row][col] = 0.0
    for row in range(m * p):
        for col in range(rank):
            V[row][col] = 0.0
    for row in range(p * n):
        for col in range(rank):
            W[row][col] = 0.0
    
    col = 0
    
    # Strassen for 2x2x2: A[:, j0:j0+2] * B[j0:j0+2, k0:k0+2]
    def add_strassen_222(j0, k0):
        nonlocal col
        # A indices: i*4 + (j0 + dj) for i in {0,1}, dj in {0,1}
        # B indices: (j0 + dj)*5 + (k0 + dk) for dj in {0,1}, dk in {0,1}
        # W indices: (k0 + dk)*2 + i for dk in {0,1}, i in {0,1}
        
        # M1 = (A[0,j0] + A[1,j0+1]) * (B[j0,k0] + B[j0+1,k0+1])
        U[0*4 + j0][col] += 1.0
        U[1*4 + (j0+1)][col] += 1.0
        V[j0*5 + k0][col] += 1.0
        V[(j0+1)*5 + (k0+1)][col] += 1.0
        # C[0,k0] += M1, C[1,k0+1] += M1
        # Actually, in Strassen: C11 = M1 + M4 - M5 + M7, C12 = M3 + M5, C21 = M2 + M4, C22 = M1 - M2 + M3 + M6
        # So M1 contributes to C[0,k0] and C[1,k0+1]
        W[k0*2 + 0][col] += 1.0
        W[(k0+1)*2 + 1][col] += 1.0
        col += 1
        
        # M2 = (A[1,j0] + A[1,j0+1]) * B[j0,k0]
        U[1*4 + j0][col] += 1.0
        U[1*4 + (j0+1)][col] += 1.0
        V[j0*5 + k0][col] += 1.0
        # C[1,k0] += M2, C[1,k0+1] -= M2
        W[k0*2 + 1][col] += 1.0
        W[(k0+1)*2 + 1][col] -= 1.0
        col += 1
        
        # M3 = A[0,j0] * (B[j0,k0+1] - B[j0+1,k0+1])
        U[0*4 + j0][col] += 1.0
        V[j0*5 + (k0+1)][col] += 1.0
        V[(j0+1)*5 + (k0+1)][col] -= 1.0
        # C[0,k0+1] += M3, C[1,k0+1] += M3
        W[(k0+1)*2 + 0][col] += 1.0
        W[(k0+1)*2 + 1][col] += 1.0
        col += 1
        
        # M4 = A[1,j0+1] * (B[j0,k0+1] - B[j0,k0])
        U[1*4 + (j0+1)][col] += 1.0
        V[j0*5 + (k0+1)][col] += 1.0
        V[j0*5 + k0][col] -= 1.0
        # C[0,k0] += M4, C[1,k0] += M4
        W[k0*2 + 0][col] += 1.0
        W[k0*2 + 1][col] += 1.0
        col += 1
        
        # M5 = (A[0,j0] + A[0,j0+1]) * B[j0+1,k0+1]
        U[0*4 + j0][col] += 1.0
        U[0*4 + (j0+1)][col] += 1.0
        V[(j0+1)*5 + (k0+1)][col] += 1.0
        # C[0,k0] -= M5, C[0,k0+1] += M5
        W[k0*2 + 0][col] -= 1.0
        W[(k0+1)*2 + 0][col] += 1.0
        col += 1
        
        # M6 = (A[1,j0] - A[0,j0]) * (B[j0,k0] + B[j0,k0+1])
        U[1*4 + j0][col] += 1.0
        U[0*4 + j0][col] -= 1.0
        V[j0*5 + k0][col] += 1.0
        V[j0*5 + (k0+1)][col] += 1.0
        # C[1,k0+1] += M6
        W[(k0+1)*2 + 1][col] += 1.0
        col += 1
        
        # M7 = (A[0,j0+1] - A[1,j0+1]) * (B[j0+1,k0] + B[j0+1,k0+1])
        U[0*4 + (j0+1)][col] += 1.0
        U[1*4 + (j0+1)][col] -= 1.0
        V[(j0+1)*5 + k0][col] += 1.0
        V[(j0+1)*5 + (k0+1)][col] += 1.0
        # C[0,k0] += M7
        W[k0*2 + 0][col] += 1.0
        col += 1
    
    # Schoolbook for 2x2x1: A[:, j0:j0+2] * B[j0:j0+2, k0]
    def add_schoolbook_221(j0, k0):
        nonlocal col
        # A indices: i*4 + (j0 + dj) for i in {0,1}, dj in {0,1}
        # B indices: (j0 + dj)*5 + k0 for dj in {0,1}
        # W indices: k0*2 + i for i in {0,1}
        
        # P1 = A[0,j0] * B[j0,k0]
        U[0*4 + j0][col] += 1.0
        V[j0*5 + k0][col] += 1.0
        W[k0*2 + 0][col] += 1.0
        col += 1
        
        # P2 = A[0,j0+1] * B[j0+1,k0]
        U[0*4 + (j0+1)][col] += 1.0
        V[(j0+1)*5 + k0][col] += 1.0
        W[k0*2 + 0][col] += 1.0
        col += 1
        
        # P3 = A[1,j0] * B[j0,k0]
        U[1*4 + j0][col] += 1.0
        V[j0*5 + k0][col] += 1.0
        W[k0*2 + 1][col] += 1.0
        col += 1
        
        # P4 = A[1,j0+1] * B[j0+1,k0]
        U[1*4 + (j0+1)][col] += 1.0
        V[(j0+1)*5 + k0][col] += 1.0
        W[k0*2 + 1][col] += 1.0
        col += 1
    
    # Now add all sub-problems:
    # 1. A[:,0:2] * B[0:2, 0:2] -> Strassen
    add_strassen_222(0, 0)
    # 2. A[:,2:4] * B[2:4, 0:2] -> Strassen
    add_strassen_222(2, 0)
    # 3. A[:,0:2] * B[0:2, 2:4] -> Strassen
    add_strassen_222(0, 2)
    # 4. A[:,2:4] * B[2:4, 2:4] -> Strassen
    add_strassen_222(2, 2)
    # 5. A[:,0:2] * B[0:2, 4] -> Schoolbook 2x2x1
    add_schoolbook_221(0, 4)
    # 6. A[:,2:4] * B[2:4, 4] -> Schoolbook 2x2x1
    add_schoolbook_221(2, 4)
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank_school = n * m * p
    U_s = [[0.0] * rank_school for _ in range(n * m)]
    V_s = [[0.0] * rank_school for _ in range(m * p)]
    W_s = [[0.0] * rank_school for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i * m + j][c] = 1.0
                V_s[j * p + k][c] = 1.0
                W_s[k * n + i][c] = 1.0
                c += 1
    return {"U": U_s, "V": V_s, "W": W_s}

# EVOLVE-BLOCK-END
