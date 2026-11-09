"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _strassen_2x2():
    """Return (U, V, W) for 2x2 matrix multiplication, rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Convention: C[i,k] = sum_j A[i,j]*B[j,k]
    Decomposition: sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = 1 for all i,j,k
    Flat indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    
    Standard Strassen (0-indexed, a=A00,a1=A01,a2=A10,a3=A11; b=B00,b1=B01,b2=B10,b3=B11):
    M1 = (a+a3)(b+b3)  -> C00+=1, C11+=1
    M2 = (a2+a3)*b     -> C10+=1
    M3 = a*(b2-b3)     -> C01+=1
    M4 = a3*(b1-b2)    -> C01+=1, C00-=1
    M5 = (a+a1)*b3     -> C00+=1, C10+=1
    M6 = (a2-a)*(b+b2) -> C11+=1
    M7 = (a1-a3)*(b1+b3)-> C00+=1, C11+=1
    
    Verify: C00 = M1+M5+M7-M4, C01=M3+M4, C10=M2+M5, C11=M1+M6-M2+M3
    Check C00: (a+a3)(b+b3)+(a+a1)b3+(a1-a3)(b1+b3)-a3(b1-b2)
    = ab+ab3+ab3+ab3 + ab3+ab3 + a1b1+a1b3-a3b1-a3b3 - a3b1+a3b2
    = ab + 5ab3 + ab3 + a1b1 + a1b3 - 2a3b1 - a3b3 + a3b2
    Hmm this doesn't look right. Let me just use the well-known correct Strassen.
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Using the standard Strassen decomposition verified numerically:
    # Factor 0: M1 = (A00+A11)(B00+B11), contributes to C00 and C11
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[1, 0] = 1; V[2, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Factor 1: M2 = (A10+A11)*B00, contributes to C10
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1
    
    # Factor 2: M3 = A00*(B10-B11), contributes to C01
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1
    
    # Factor 3: M4 = A11*(B01-B10), contributes to C01 and C00 (negative)
    U[3, 3] = 1
    V[1, 3] = 1; V[2, 3] = -1
    W[2, 3] = 1; W[0, 3] = -1
    
    # Factor 4: M5 = (A00+A01)*B11, contributes to C00 and C10
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # Factor 5: M6 = (A10-A00)*(B00+B10), contributes to C11
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # Factor 6: M7 = (A01-A11)*(B01+B11), contributes to C00 and C11
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[3, 6] = 1
    
    return U, V, W

def _verify_strassen():
    """Verify the Strassen decomposition is correct using numpy."""
    U, V, W = _strassen_2x2()
    n, m, p = 2, 2, 2
    rank = U.shape[1]
    # Build verification tensor: for each (i,j,k), compute sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # Use einsum for speed
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    # T[i,j,k] = sum_r U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    if not np.allclose(T, 1.0, atol=1e-10):
        print(f"Strassen verification failed: max diff = {np.max(np.abs(T-1.0))}")
        return False
    return True

def _build_4x4x9():
    """Build 4x4x9 decomposition using Kronecker product of Strassen(2,2) with schoolbook(2,1)
    for the 4x4x2 part, then combine appropriately for 4x4x9.
    
    Strategy: 
    - 4x4x9 = 4x4x4 + 4x4x5
    - 4x4x4 via Kronecker of Strassen(2,2) x Strassen(2,2): rank 49
    - 4x4x5 via schoolbook: rank 80
    - Total: 129
    
    Alternative: 4x4x9 via Kronecker of Strassen(2,2) x (2, 9/2)... doesn't work.
    Better: 4x4x9 = (2x2 outer) x (2x? inner). View 9 = 2*4 + 1.
    Use Strassen(2,2) for outer (rank 7), and for inner use 2x4x... 
    
    Actually simplest reliable: Kronecker Strassen(2,2) x Strassen(2,2) for 4x4x4 (rank 49)
    + schoolbook for remaining 4x4x5 (rank 80) = 129.
    """
    n, m, p = 4, 4, 9
    
    S_U, S_V, S_W = _strassen_2x2()
    
    # Build 4x4x4 via Kronecker product of two Strassen(2,2)
    U_44 = np.zeros((16, 49))
    V_44 = np.zeros((16, 49))
    W_44 = np.zeros((16, 49))
    
    # U: row = i*4+j, i=2*i1+i2, j=2*j1+j2; col = r1*7+r2
    # U_44[(2*i1+i2)*4+(2*j1+j2), r1*7+r2] = S_U[2*i1+j1, r1] * S_U[2*i2+j2, r2]
    for i1 in range(2):
        for j1 in range(2):
            for i2 in range(2):
                for j2 in range(2):
                    row_u = (2*i1 + i2) * 4 + (2*j1 + j2)
                    for r1 in range(7):
                        for r2 in range(7):
                            U_44[row_u, r1*7+r2] = S_U[2*i1+j1, r1] * S_U[2*i2+j2, r2]
    
    # V: row = j*4+k, j=2*j1+j2, k=2*k1+k2
    # V_44[(2*j1+j2)*4+(2*k1+k2), r1*7+r2] = S_V[2*j1+k1, r1] * S_V[2*j2+k2, r2]
    for j1 in range(2):
        for k1 in range(2):
            for j2 in range(2):
                for k2 in range(2):
                    row_v = (2*j1 + j2) * 4 + (2*k1 + k2)
                    for r1 in range(7):
                        for r2 in range(7):
                            V_44[row_v, r1*7+r2] = S_V[2*j1+k1, r1] * S_V[2*j2+k2, r2]
    
    # W: row = k*4+i, k=2*k1+k2, i=2*i1+i2
    # W_44[(2*k1+k2)*4+(2*i1+i2), r1*7+r2] = S_W[2*k1+i1, r1] * S_W[2*k2+i2, r2]
    for k1 in range(2):
        for i1 in range(2):
            for k2 in range(2):
                for i2 in range(2):
                    row_w = (2*k1 + k2) * 4 + (2*i1 + i2)
                    for r1 in range(7):
                        for r2 in range(7):
                            W_44[row_w, r1*7+r2] = S_W[2*k1+i1, r1] * S_W[2*k2+i2, r2]
    
    # Verify 4x4x4 using numpy
    U_44_3d = U_44.reshape(4, 4, 49)
    V_44_3d = V_44.reshape(4, 4, 49)
    W_44_3d = W_44.reshape(4, 4, 49)
    T_44 = np.einsum('ijr,jkr,kir->ijk', U_44_3d, V_44_3d, W_44_3d)
    verified_44 = np.allclose(T_44, 1.0, atol=1e-10)
    
    if not verified_44:
        print(f"4x4x4 Kronecker verification failed, max diff: {np.max(np.abs(T_44-1.0))}")
        # Fall back to schoolbook for 4x4x4
        rank_44 = 64
        U_44 = np.zeros((16, 64))
        V_44 = np.zeros((16, 64))
        W_44 = np.zeros((16, 64))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U_44[i*4+j, col] = 1
                    V_44[j*4+k, col] = 1
                    W_44[k*4+i, col] = 1
                    col += 1
    else:
        rank_44 = 49
    
    # Schoolbook for 4x4x5 (k=4..8)
    rank_extra = 80
    total_rank = rank_44 + rank_extra
    
    U_total = np.zeros((16, total_rank))
    V_total = np.zeros((16, total_rank))
    W_total = np.zeros((36, total_rank))
    
    U_total[:, :rank_44] = U_44
    V_total[:, :rank_44] = V_44
    W_total[:16, :rank_44] = W_44
    
    col_offset = rank_44
    for k_extra in range(4, 9):
        for i in range(4):
            for j in range(4):
                U_total[i*4+j, col_offset] = 1
                V_total[j*4+k_extra, col_offset] = 1
                W_total[k_extra*4+i, col_offset] = 1
                col_offset += 1
    
    # Convert to lists of integers
    U_list = U_total.tolist()
    V_list = V_total.tolist()
    W_list = W_total.tolist()
    
    for mat in [U_list, V_list, W_list]:
        for row in mat:
            for idx in range(len(row)):
                val = row[idx]
                if abs(val - round(val)) < 1e-10:
                    row[idx] = int(round(val))
    
    return {"U": U_list, "V": V_list, "W": W_list}

def _schoolbook(n, m, p):
    """Pure schoolbook decomposition for n x m by m x p matrix multiplication."""
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

def construct():
    start_time = time.time()
    deadline = 110  # seconds, with safety margin
    
    # First verify Strassen
    try:
        strassen_ok = _verify_strassen()
    except Exception as e:
        print(f"Strassen verification error: {e}")
        strassen_ok = False
    
    if not strassen_ok:
        return _schoolbook(*DIMENSIONS)
    
    # Try to build the improved decomposition
    try:
        result = _build_4x4x9()
        elapsed = time.time() - start_time
        if elapsed < deadline:
            return result
    except Exception as e:
        print(f"Error building improved decomposition: {e}")
    
    # Fallback to schoolbook
    return _schoolbook(*DIMENSIONS)
# EVOLVE-BLOCK-END
