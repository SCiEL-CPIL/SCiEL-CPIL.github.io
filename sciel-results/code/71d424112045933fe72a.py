import time
import numpy as np
from fractions import Fraction

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_2x2x2():
    """
    Return U, V, W for 2x2x2 Strassen decomposition.
    U: (4, 7), V: (4, 7), W: (4, 7)
    
    C = A * B where A, B, C are 2x2.
    C[0,0] = M1 + M4 - M5 - M7
    C[0,1] = M3 + M5
    C[1,0] = M2 + M4
    C[1,1] = M1 - M3 + M6 + M7
    
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B01-B11)
    M4 = A11*(B10-B00)
    M5 = (A00+A01)*B11
    M6 = (A10-A00)*(B00+B01)
    M7 = (A01-A11)*(B10+B11)
    
    Tensor convention: T[i,j,k] = sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r]
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Index: A[i,j] -> U[i*m+j], B[j,k] -> V[j*p+k], C[k,i] -> W[k*n+i]
    # For 2x2: A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[2], A[1,1]=U[3]
    # B[0,0]=V[0], B[0,1]=V[1], B[1,0]=V[2], B[1,1]=V[3]
    # C[0,0]=W[0], C[0,1]=W[1], C[1,0]=W[2], C[1,1]=W[3]
    
    # M1 = (A00+A11)(B00+B11)
    # U: A00=1, A11=1 -> U[0]=1, U[3]=1
    # V: B00=1, B11=1 -> V[0]=1, V[3]=1
    # W: contributes to C00(+1), C11(+1)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C[0,0] and C[1,1]
    
    # M2 = (A10+A11)*B00
    # U: A10=1, A11=1 -> U[2]=1, U[3]=1
    # V: B00=1 -> V[0]=1
    # W: contributes to C10(+1)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1  # C[1,0]
    
    # M3 = A00*(B01-B11)
    # U: A00=1 -> U[0]=1
    # V: B01=1, B11=-1 -> V[1]=1, V[3]=-1
    # W: contributes to C01(+1), C11(-1)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1  # C[0,1] and C[1,1]
    
    # M4 = A11*(B10-B00)
    # U: A11=1 -> U[3]=1
    # V: B10=1, B00=-1 -> V[2]=1, V[0]=-1
    # W: contributes to C00(+1), C10(+1)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1  # C[0,0] and C[1,0]
    
    # M5 = (A00+A01)*B11
    # U: A00=1, A01=1 -> U[0]=1, U[1]=1
    # V: B11=1 -> V[3]=1
    # W: contributes to C00(-1), C01(+1)
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1  # C[0,0] and C[0,1]
    
    # M6 = (A10-A00)*(B00+B01)
    # U: A10=1, A00=-1 -> U[2]=1, U[0]=-1
    # V: B00=1, B01=1 -> V[0]=1, V[1]=1
    # W: contributes to C11(+1)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C[1,1]
    
    # M7 = (A01-A11)*(B10+B11)
    # U: A01=1, A11=-1 -> U[1]=1, U[3]=-1
    # V: B10=1, B11=1 -> V[2]=1, V[3]=1
    # W: contributes to C00(-1), C11(+1)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[3, 6] = 1  # C[0,0] and C[1,1]
    
    return U, V, W

def _build_strassen_based(n, m, p):
    """
    Build a decomposition for n x m times m x p by applying Strassen to 2x2 blocks.
    For m=3, p=10: we have 3 rows of B, 10 columns.
    
    Strategy: 
    - For each pair of output columns (2k, 2k+1), apply a 2x2x2 Strassen on the 
      corresponding 2-column slice of B and a 2-row slice of A.
    - Since n=2, the A dimension is fully used (2 rows).
    - For m=3, we need to handle 3 input rows. We can use Strassen on (rows 0,1) 
      and add the contribution from row 2 separately, OR use a 2x3x2 decomposition.
    
    Actually, let's think more carefully.
    
    The tensor M_{2,3,10} has entries T[i,j,k] = 1 for all i,j,k.
    
    A simpler approach: use the fact that we can decompose the m=3 dimension.
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    We can group: C[i,k] = (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + A[i,2]*B[2,k]
    
    The first part is a 2x2x2 problem (rows 0,1 of A and B) for each pair of columns.
    The second part is a 2x1x1 problem (row 2) for each column.
    
    For 10 columns = 5 pairs:
    - 5 * Strassen(2x2x2) = 5 * 7 = 35 terms for the (row 0,1) part
    - 5 * 2 * 1 = 10 terms for the row 2 part (2 output rows * 1 input row per column)
    Total: 45 terms
    
    Wait, let me reconsider. For each column pair (2k, 2k+1):
    - Strassen handles A[0:2, 0:2] * B[0:2, 2k:2k+2] -> gives C[0:2, 2k:2k+2]
      This uses 7 rank-1 terms.
    - Then we add A[0:2, 2] * B[2, 2k:2k+2] -> this is 2x1x2, rank 4 (schoolbook)
    Total per column pair: 7 + 4 = 11
    For 5 column pairs: 5 * 11 = 55
    
    That gives rank 55!
    """
    n, m, p = 2, 3, 10
    num_pairs = p // 2  # 5 pairs
    
    # Each column pair contributes:
    # - 7 Strassen terms for (A rows 0,1) x (B rows 0,1) x (2 output cols)
    # - 4 schoolbook terms for (A rows 0,1) x (B row 2) x (2 output cols)
    terms_per_pair = 7 + 4
    total_rank = num_pairs * terms_per_pair
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Get Strassen decomposition
    Su, Sv, Sw = _strassen_2x2x2()
    # Su: (4,7), Sv: (4,7), Sw: (4,7)
    # Su[i*2+j, r] corresponds to A[i,j] for i,j in {0,1}
    # Sv[j*2+k, r] corresponds to B[j,k] for j in {0,1}, k in {0,1}
    # Sw[k*2+i, r] corresponds to C[k,i] for k in {0,1}, i in {0,1}
    
    for pair in range(num_pairs):
        col_offset = pair * terms_per_pair
        k0 = pair * 2  # first column index in this pair
        
        # --- Strassen part: A[0:2, 0:2] * B[0:2, k0:k0+2] ---
        for r in range(7):
            # U: A[i,j] for i in {0,1}, j in {0,1}
            for i in range(2):
                for j in range(2):
                    val = Su[i*2 + j, r]
                    if val != 0:
                        U[i * m + j, col_offset + r] = val
            # V: B[j, k] for j in {0,1}, k in {0,1} -> actual B[j, k0+k]
            for j in range(2):
                for k in range(2):
                    val = Sv[j*2 + k, r]
                    if val != 0:
                        V[j * p + (k0 + k), col_offset + r] = val
            # W: C[k, i] for k in {0,1}, i in {0,1} -> actual C[k0+k, i]
            for k in range(2):
                for i in range(2):
                    val = Sw[k*2 + i, r]
                    if val != 0:
                        W[(k0 + k) * n + i, col_offset + r] = val
        
        # --- Schoolbook part: A[0:2, 2] * B[2, k0:k0+2] ---
        # C[i, k0+k] += A[i, 2] * B[2, k0+k]
        # This is 2*1*2 = 4 terms
        sb_offset = 7
        for i in range(2):
            for k in range(2):
                idx = col_offset + sb_offset + i * 2 + k
                U[i * m + 2, idx] = 1  # A[i, 2]
                V[2 * p + (k0 + k), idx] = 1  # B[2, k0+k]
                W[(k0 + k) * n + i, idx] = 1  # C[k0+k, i]
    
    return U, V, W

def construct():
    n, m, p = 2, 3, 10
    deadline = time.time() + 140
    
    # Build the Strassen-based decomposition
    U, V, W = _build_strassen_based(n, m, p)
    rank = U.shape[1]
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
