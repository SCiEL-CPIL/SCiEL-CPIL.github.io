"""Find rank-minimal exact tensor decomposition for 2x2 * 2x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 2, 7]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U,V,W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            lhs = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            rhs = 0.0
                            for r in range(rank):
                                rhs += U[i*m+j, r] * V[jp*p+kp, r] * W[kp*n+ip, r]
                            if abs(lhs - rhs) > 1e-9:
                                return False
    return True

def _construct_strassen_block(U, V, W, col_start, k_start, n=2, m=2, p=7):
    """
    Add Strassen rank-7 terms for a 2x2 block of B at columns k_start, k_start+1.
    """
    # M1 = (A00 + A11) * (B0,ks + B1,ks+1)
    r = col_start + 0
    U[0, r] += 1   # A00
    U[3, r] += 1   # A11
    V[7*0 + k_start, r] += 1       # B0,ks
    V[7*1 + (k_start+1), r] += 1   # B1,ks+1
    W[2*k_start + 0, r] += 1       # C0,ks
    W[2*(k_start+1) + 1, r] += 1   # C1,ks+1

    # M2 = (A10 + A11) * B0,ks
    r = col_start + 1
    U[2, r] += 1   # A10
    U[3, r] += 1   # A11
    V[7*0 + k_start, r] += 1       # B0,ks
    W[2*k_start + 1, r] += 1       # C1,ks
    W[2*(k_start+1) + 1, r] -= 1   # C1,ks+1

    # M3 = A00 * (B0,ks+1 - B1,ks+1)
    r = col_start + 2
    U[0, r] += 1   # A00
    V[7*0 + (k_start+1), r] += 1   # B0,ks+1
    V[7*1 + (k_start+1), r] -= 1   # B1,ks+1
    W[2*(k_start+1) + 0, r] += 1   # C0,ks+1
    W[2*(k_start+1) + 1, r] += 1   # C1,ks+1

    # M4 = A11 * (B1,ks - B0,ks)
    r = col_start + 3
    U[3, r] += 1   # A11
    V[7*1 + k_start, r] += 1       # B1,ks
    V[7*0 + k_start, r] -= 1       # B0,ks
    W[2*k_start + 0, r] += 1       # C0,ks
    W[2*k_start + 1, r] += 1       # C1,ks

    # M5 = (A00 + A01) * B1,ks+1
    r = col_start + 4
    U[0, r] += 1   # A00
    U[1, r] += 1   # A01
    V[7*1 + (k_start+1), r] += 1   # B1,ks+1
    W[2*k_start + 0, r] -= 1       # C0,ks
    W[2*(k_start+1) + 0, r] += 1   # C0,ks+1

    # M6 = (A10 - A00) * (B0,ks + B0,ks+1)
    r = col_start + 5
    U[2, r] += 1   # A10
    U[0, r] -= 1   # A00
    V[7*0 + k_start, r] += 1       # B0,ks
    V[7*0 + (k_start+1), r] += 1   # B0,ks+1
    W[2*(k_start+1) + 1, r] += 1   # C1,ks+1

    # M7 = (A01 - A11) * (B1,ks + B1,ks+1)
    r = col_start + 6
    U[1, r] += 1   # A01
    U[3, r] -= 1   # A11
    V[7*1 + k_start, r] += 1       # B1,ks
    V[7*1 + (k_start+1), r] += 1   # B1,ks+1
    W[2*k_start + 0, r] += 1       # C0,ks

def _construct_schoolbook_column(U, V, W, col_start, k, n=2, m=2, p=7):
    """Add schoolbook rank-4 terms for a single column k of B."""
    # A00 * B0,k -> C0,k
    r = col_start + 0
    U[0, r] += 1
    V[7*0 + k, r] += 1
    W[2*k + 0, r] += 1

    # A01 * B1,k -> C0,k
    r = col_start + 1
    U[1, r] += 1
    V[7*1 + k, r] += 1
    W[2*k + 0, r] += 1

    # A10 * B0,k -> C1,k
    r = col_start + 2
    U[2, r] += 1
    V[7*0 + k, r] += 1
    W[2*k + 1, r] += 1

    # A11 * B1,k -> C1,k
    r = col_start + 3
    U[3, r] += 1
    V[7*1 + k, r] += 1
    W[2*k + 1, r] += 1

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s safety margin
    
    n, m, p = DIMENSIONS
    
    # Strategy: Use Strassen for 2x2 blocks of B, schoolbook for remaining column
    # 7 columns -> 3 blocks of 2 (cols 0-1, 2-3, 4-5) + 1 single column (col 6)
    # Rank: 7 + 7 + 7 + 4 = 25
    
    rank = 25
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    # Add Strassen for columns 0,1
    _construct_strassen_block(U, V, W, 0, 0, n, m, p)
    
    # Add Strassen for columns 2,3
    _construct_strassen_block(U, V, W, 7, 2, n, m, p)
    
    # Add Strassen for columns 4,5
    _construct_strassen_block(U, V, W, 14, 4, n, m, p)
    
    # Add schoolbook for column 6
    _construct_schoolbook_column(U, V, W, 21, 6, n, m, p)
    
    # Verify the decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        schoolbook_rank = n * m * p
        U_sb = np.zeros((n*m, schoolbook_rank), dtype=int)
        V_sb = np.zeros((m*p, schoolbook_rank), dtype=int)
        W_sb = np.zeros((p*n, schoolbook_rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
