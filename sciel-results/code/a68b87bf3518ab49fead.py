"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _strassen_2x2():
    """Return (U, V, W) for 2x2 matrix multiplication, rank 7.
    Flat indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Standard Strassen decomposition
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
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    if not np.allclose(T, 1.0, atol=1e-10):
        print(f"Strassen verification failed: max diff = {np.max(np.abs(T-1.0))}")
        return False
    return True

def _kron_strassen_4x4x4():
    """Build 4x4x4 via Kronecker product of two Strassen(2,2), rank 49.
    Uses vectorized numpy for speed.
    """
    S_U, S_V, S_W = _strassen_2x2()
    # S_U shape: (4, 7) = (2*2, 7)
    # Reshape to (2, 2, 7) for outer/inner decomposition
    S_U_3d = S_U.reshape(2, 2, 7)  # [i1, j1, r1]
    S_V_3d = S_V.reshape(2, 2, 7)  # [j1, k1, r1]
    S_W_3d = S_W.reshape(2, 2, 7)  # [k1, i1, r1]
    
    rank = 49
    U_44 = np.zeros((16, rank))
    V_44 = np.zeros((16, rank))
    W_44 = np.zeros((16, rank))
    
    # U_44[(2*i1+i2)*4+(2*j1+j2), r1*7+r2] = S_U_3d[i1,j1,r1] * S_U_3d[i2,j2,r2]
    # Build using einsum-style broadcasting
    # For each (i1,j1,i2,j2,r1,r2): row = (2*i1+i2)*4+(2*j1+j2), col = r1*7+r2
    # We can vectorize by building the full tensor and then reshaping
    
    # U contribution: outer product of S_U_3d with itself
    # U_44_tensor[i1,i2,j1,j2,r1,r2] = S_U_3d[i1,j1,r1] * S_U_3d[i2,j2,r2]
    U_tensor = S_U_3d[:, None, :, None, :] * S_U_3d[None, :, None, :, :]
    # U_tensor shape: (2, 2, 2, 2, 7, 7) = [i1, i2, j1, j2, r1, r2]
    # Reshape to (16, 49): row = (i1*2+i2)*4 + (j1*2+j2), col = r1*7+r2
    U_44 = U_tensor.reshape(16, 49)
    
    # V contribution: V_44[(2*j1+j2)*4+(2*k1+k2), r1*7+r2] = S_V_3d[j1,k1,r1] * S_V_3d[j2,k2,r2]
    V_tensor = S_V_3d[:, None, :, None, :] * S_V_3d[None, :, None, :, :]
    # V_tensor shape: (2, 2, 2, 2, 7, 7) = [j1, j2, k1, k2, r1, r2]
    V_44 = V_tensor.reshape(16, 49)
    
    # W contribution: W_44[(2*k1+k2)*4+(2*i1+i2), r1*7+r2] = S_W_3d[k1,i1,r1] * S_W_3d[k2,i2,r2]
    W_tensor = S_W_3d[:, None, :, None, :] * S_W_3d[None, :, None, :, :]
    # W_tensor shape: (2, 2, 2, 2, 7, 7) = [k1, k2, i1, i2, r1, r2]
    W_44 = W_tensor.reshape(16, 49)
    
    return U_44, V_44, W_44

def _verify_4x4x4(U_44, V_44, W_44):
    """Verify 4x4x4 decomposition."""
    rank = U_44.shape[1]
    U_3d = U_44.reshape(4, 4, rank)
    V_3d = V_44.reshape(4, 4, rank)
    W_3d = W_44.reshape(4, 4, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    if not np.allclose(T, 1.0, atol=1e-10):
        print(f"4x4x4 verification failed: max diff = {np.max(np.abs(T-1.0))}")
        return False
    return True

def _build_4x4x9():
    """Build 4x4x9 decomposition: 4x4x4 (rank 49) + 4x4x4 (rank 49) + 4x4x1 (rank 16) = 114."""
    n, m, p = 4, 4, 9
    
    # Build 4x4x4 via Kronecker Strassen
    U_44, V_44, W_44 = _kron_strassen_4x4x4()
    
    # Verify
    if not _verify_4x4x4(U_44, V_44, W_44):
        # Fall back to schoolbook for the 4x4x4 blocks
        print("Kronecker 4x4x4 failed, using schoolbook")
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
        rank_44 = 64
    else:
        rank_44 = 49
    
    # Total: 4x4x4 (k=0..3) + 4x4x4 (k=4..7) + 4x4x1 (k=8)
    # First 4x4x4 block uses columns 0..rank_44-1
    # Second 4x4x4 block uses columns rank_44..2*rank_44-1
    # 4x4x1 block uses columns 2*rank_44..2*rank_44+15
    
    rank_441 = 16  # schoolbook for 4x4x1
    total_rank = 2 * rank_44 + rank_441
    
    U_total = np.zeros((16, total_rank))
    V_total = np.zeros((16, total_rank))
    W_total = np.zeros((36, total_rank))
    
    # First 4x4x4 block (k=0,1,2,3)
    U_total[:, :rank_44] = U_44
    V_total[:, :rank_44] = V_44
    W_total[:16, :rank_44] = W_44  # W for k=0..3: W[k*4+i, r]
    
    # Second 4x4x4 block (k=4,5,6,7): same U,V but W shifted
    U_total[:, rank_44:2*rank_44] = U_44
    V_total[:, rank_44:2*rank_44] = V_44
    # W for second block: W[(k+4)*4+i, r] for k=0..3 -> rows 16..31
    W_total[16:32, rank_44:2*rank_44] = W_44
    
    # 4x4x1 block (k=8): schoolbook
    col_offset = 2 * rank_44
    for i in range(4):
        for j in range(4):
            U_total[i*4+j, col_offset] = 1
            V_total[j*4+8, col_offset] = 1
            W_total[8*4+i, col_offset] = 1
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
