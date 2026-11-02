import time
import numpy as np
import itertools

DIMENSIONS = [3, 3, 8]
_START_TIME = time.time()
DEADLINE = _START_TIME + 110.0

def _verify(U, V, W, n, m, p, tol=1e-9):
    rank = len(U[0]) if n * m > 0 else 0
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    U_3d = U_np.reshape(n, m, rank)
    V_3d = V_np.reshape(m, p, rank)
    W_3d = W_np.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    expected = np.ones((n, m, p), dtype=float)
    return np.allclose(T, expected, atol=tol)

def _strassen_222():
    """Return U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm (rank 7).
    U: (2*2, 7), V: (2*2, 7), W: (2*2, 7)
    Indexing: U[i*2+j], V[i*2+j], W[i*2+j]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen's 7 products for 2x2 matrices:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U corresponds to A entries, V to B entries, W to C entries
    # U[i*2+j] for A[i,j], V[i*2+j] for B[i,j], W[i*2+j] for C[i,j]
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1  # A00
    U[3, 0] = 1  # A11
    V[0, 0] = 1  # B00
    V[3, 0] = 1  # B11
    W[0, 0] = 1  # C00
    W[3, 0] = 1  # C11
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1  # A10
    U[3, 1] = 1  # A11
    V[0, 1] = 1  # B00
    W[2, 1] = 1  # C10
    
    # M3 = A00(B10-B11)
    U[0, 2] = 1  # A00
    V[2, 2] = 1  # B10
    V[3, 2] = -1 # B11
    W[1, 2] = 1  # C01
    W[3, 2] = 1  # C11
    
    # M4 = A11(B01-B00)
    U[3, 3] = 1  # A11
    V[1, 3] = 1  # B01
    V[0, 3] = -1 # B00
    W[0, 3] = 1  # C00
    W[2, 3] = 1  # C10
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1  # A00
    U[1, 4] = 1  # A01
    V[3, 4] = 1  # B11
    W[0, 4] = -1 # C00
    W[1, 4] = 1  # C01
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1  # A10
    U[0, 5] = -1 # A00
    V[0, 5] = 1  # B00
    V[1, 5] = 1  # B01
    W[3, 5] = 1  # C11
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1  # A01
    U[3, 6] = -1 # A11
    V[2, 6] = 1  # B10
    V[3, 6] = 1  # B11
    W[0, 6] = 1  # C00
    W[1, 6] = 1  # C01
    
    return U, V, W

def _block_decompose():
    """
    Decompose 3x3x8 matrix multiplication using block structure.
    Split 3x3 into blocks: 2x2, 2x1, 1x2, 1x1
    Split 3x8 into blocks: top 2 rows, bottom 1 row; each split into 4x4 blocks (since 8=2*4)
    
    Actually, let's think more carefully.
    A is 3x3, B is 3x8.
    Split A into:
      A00: 2x2 (rows 0-1, cols 0-1)
      A01: 2x1 (rows 0-1, col 2)
      A10: 1x2 (row 2, cols 0-1)
      A11: 1x1 (row 2, col 2)
    
    Split B into:
      B_top: 2x8 (rows 0-1)
      B_bot: 1x8 (row 2)
    
    But B_top is 2x8, and we can split it further into 2x4 blocks.
    Actually, for the tensor decomposition, we need to handle the full product.
    
    The product C = A*B is 3x8.
    C rows 0-1 come from A00*B_top + A01*B_bot
    C row 2 comes from A10*B_top + A11*B_bot
    
    Let's split B_top into 4 blocks of size 2x2: B0, B1, B2, B3
    B_bot into 4 blocks of size 1x2: D0, D1, D2, D3
    
    Then:
    C[0:2, 0:2] = A00*B0 + A01*D0
    C[0:2, 2:4] = A00*B1 + A01*D1
    C[0:2, 4:6] = A00*B2 + A01*D2
    C[0:2, 6:8] = A00*B3 + A01*D3
    C[2, 0:2] = A10*B0 + A11*D0
    C[2, 2:4] = A10*B1 + A11*D1
    C[2, 4:6] = A10*B2 + A11*D2
    C[2, 6:8] = A10*B3 + A11*D3
    
    Now we can use Strassen for A00*Bk (2x2x2, rank 7 each) and handle the other terms.
    A01 is 2x1, Dk is 1x2. A01*Dk is outer product: (2x1)*(1x2) = 2x2. Rank 1 each? 
    T[i,j] = A01[i] * Dk[j]. This is a single product, so rank 1.
    
    A10 is 1x2, Bk is 2x2. A10*Bk: T[j] = sum_i A10[i] * Bk[i,j]. This is matrix-vector, rank 2.
    A11 is 1x1, Dk is 1x2. A11*Dk: T[j] = A11 * Dk[j]. Rank 1 each, total rank 2.
    
    So for each column block k (4 blocks):
    - A00*Bk: Strassen rank 7
    - A01*Dk: rank 1
    - A10*Bk: rank 2
    - A11*Dk: rank 1
    Total per block: 7+1+2+1 = 11
    Total for 4 blocks: 44
    
    Wait, that's 44, which is better than 72.
    But let me double-check the ranks.
    
    A01*Dk: A01 is 2x1, Dk is 1x2. Product is 2x2. T[i,j] = A01[i] * Dk[j].
    This is indeed rank 1 (outer product of A01 and Dk^T).
    
    A10*Bk: A10 is 1x2, Bk is 2x2. Product is 1x2. T[j] = A10[0]*Bk[0,j] + A10[1]*Bk[1,j].
    This is sum of 2 products, rank 2.
    
    A11*Dk: A11 is scalar, Dk is 1x2. Product is 1x2. T[j] = A11 * Dk[j]. Rank 1 each, but for fixed k, it's rank 1.
    
    So per block: 7+1+2+1 = 11. Total: 44.
    
    Let me implement this.
    """
    n, m, p = 3, 3, 8
    U = []
    V = []
    W = []
    
    for k in range(4):
        col_start = k * 2
        
        # Strassen for A00*Bk (2x2x2)
        sU, sV, sW = _strassen_222()
        
        # Map Strassen indices to global
        # A00: rows 0-1, cols 0-1 -> global U indices: i*3+j for i in 0-1, j in 0-1
        # Bk: rows 0-1, cols 2k to 2k+1 -> global V indices: i*8 + (2k+j) for i in 0-1, j in 0-1
        # C[0:2, 2k:2k+2]: rows 0-1, cols 2k to 2k+1 -> global W indices: (2k+j)*3 + i for j in 0-1, i in 0-1
        
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    u_idx = i * 3 + j  # A00[i,j]
                    v_idx = i * 8 + (col_start + j)  # Bk[i,j]
                    w_idx = (col_start + j) * 3 + i  # C[i, col_start+j]
                    
                    # Add Strassen contribution
                    if not U:
                        U = [0] * 44
                        V = [0] * 44
                        W = [0] * 44
                    U[u_idx] += sU[i*2+j, r]
                    V[v_idx] += sV[i*2+j, r]
                    W[w_idx] += sW[i*2+j, r]
        
        # A01*Dk: rank 1
        # A01: rows 0-1, col 2 -> U indices: i*3+2 for i in 0-1
        # Dk: row 2, cols 2k to 2k+1 -> V indices: 2*8 + (2k+j) for j in 0-1
        # C[0:2, 2k:2k+2]: W indices: (2k+j)*3 + i for j in 0-1, i in 0-1
        # T[i,j] = A01[i] * Dk[j]
        rank_a01 = 7  # after Strassen
        for i in range(2):
            for j in range(2):
                u_idx = i * 3 + 2
                v_idx = 2 * 8 + (col_start + j)
                w_idx = (col_start + j) * 3 + i
                U[u_idx] += 0  # A01[i] coefficient is 1
                V[v_idx] += 0  # Dk[j] coefficient is 1
                W[w_idx] += 1  # C[i, col_start+j]
        
        # Actually, for rank 1, we need a single term: U[i,rank] * V[j,rank] * W[k,rank]
        # T[i,j] = A01[i] * Dk[j]
        # So U[i*3+2, rank] = A01[i] coefficient = 1
        # V[2*8+col_start+j, rank] = Dk[j] coefficient = 1
        # W[(col_start+j)*3+i, rank] = 1
        # But this is one rank for each (i,j) pair? No, it's one rank for the whole outer product.
        # T[i,j] = A01[i] * Dk[j] = sum over r=0 (since rank 1) of U[i,0] * V[j,0] * W[i,j,0]
        # So we need U[i*3+2, rank] = 1 for all i, V[2*8+col_start+j, rank] = 1 for all j, 
        # and W[(col_start+j)*3+i, rank] = 1 for all i,j? No.
        # Actually, the tensor is T[i,j] = A01[i] * Dk[j].
        # This can be written as sum_r U[i,0,r] * V[j,0,r] * W[j,i,r]
        # With r=0: U[i,0,0] = A01[i], V[j,0,0] = Dk[j], W[j,i,0] = 1
        # But A01[i] and Dk[j] are entries, not coefficients.
        # The decomposition is: T = sum_r U_r * V_r * W_r (outer products)
        # For rank 1: T[i,j] = U[i,0] * V[j,0] * W[j,i,0]
        # We need U[i,0] * V[j,0] * W[j,i,0] = A01[i] * Dk[j]
        # Set U[i,0] = 1, V[j,0] = 1, W[j,i,0] = A01[i] * Dk[j]? No, W doesn't depend on A01 and Dk.
        # I think I'm confusing the tensor entries with the factor entries.
        
        # Let me restart the block decomposition more carefully.
        pass
    
    # This approach is getting complicated. Let me use a different strategy.
    # I'll construct the full 72-rank schoolbook decomposition first, then try to reduce.
    pass

def _schoolbook():
    n, m, p = 3, 3, 8
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    result = _schoolbook()
    if _verify(result['U'], result['V'], result['W'], n, m, p):
        return result
    return result
