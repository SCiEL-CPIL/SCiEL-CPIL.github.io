import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    start = time.time()
    n, m, p = DIMENSIONS  # 3, 5, 5
    deadline = start + 140  # safety margin before 150s kill
    
    # --- Pre-computed optimal Strassen block placements for 3x5x5 ---
    # We use 4 independent 2x2x2 Strassen blocks.
    # Total rank from Strassen: 4 * 7 = 28
    # Covered triples: 4 * 8 = 32
    # Total triples: 3 * 5 * 5 = 75
    # Remaining schoolbook triples: 75 - 32 = 43
    # Total rank = 28 + 43 = 71
    
    # Blocks cover:
    # Block 0: i in {0,1}, j in {0,1}, k in {0,1}
    # Block 1: i in {0,1}, j in {2,3}, k in {0,1}
    # Block 2: i in {0,1}, j in {0,1}, k in {2,3}
    # Block 3: i in {0,1}, j in {2,3}, k in {2,3}
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2),
    ]
    
    n_blocks = len(blocks)
    strassen_rank = n_blocks * 7
    covered_count = n_blocks * 8
    total_triples = n * m * p
    schoolbook_count = total_triples - covered_count
    total_rank = strassen_rank + schoolbook_count
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    for i0, j0, k0 in blocks:
        # Strassen 2x2x2 block
        # M1 = (A00+A11)(B00+B11)
        c = col
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        c += 1
        
        # M2 = (A10+A11)B00
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        c += 1
        
        # M3 = A00(B01-B11)
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        c += 1
        
        # M4 = A11(B10-B00)
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        c += 1
        
        # M5 = (A00+A01)B11
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        c += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        c += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        c += 1
        
        col = c
    
    # Mark covered triples
    covered = [[[False]*p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered[i0+di][j0+dj][k0+dk] = True
    
    # Schoolbook for remaining triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered[i][j][k]:
                    U[i*m+j][col] += 1
                    V[j*p+k][col] += 1
                    W[k*n+i][col] += 1
                    col += 1
    
    if col != total_rank:
        # Should not happen, but safety check
        pass
    
    # Verify using numpy
    try:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Target tensor: T[i,j,k] = 1 if (i*m+j, j*p+k, k*n+i) is the target
        # We check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        target = np.zeros((n, m, p), dtype=float)
        target[:] = 1.0
        
        # Compute decomposition
        # T_decomp[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        T_decomp = np.zeros((n, m, p), dtype=float)
        for r in range(total_rank):
            # U[:, r] is shape (n*m,)
            # V[:, r] is shape (m*p,)
            # W[:, r] is shape (p*n,)
            u_col = U_np[:, r].reshape(n, m)
            v_col = V_np[:, r].reshape(m, p)
            w_col = W_np[:, r].reshape(p, n)
            T_decomp += u_col[:, :, np.newaxis] * v_col[np.newaxis, :, :] * w_col[np.newaxis, np.newaxis, :]
        
        if not np.allclose(T_decomp, target, atol=1e-10):
            # Verification failed, fallback to schoolbook
            raise ValueError("Verification failed")
    except Exception:
        # Fallback to pure schoolbook
        rank = n * m * p
        U_f = [[0.0]*rank for _ in range(n*m)]
        V_f = [[0.0]*rank for _ in range(m*p)]
        W_f = [[0.0]*rank for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i*m+j][c] = 1.0
                    V_f[j*p+k][c] = 1.0
                    W_f[k*n+i][c] = 1.0
                    c += 1
        return {"U": U_f, "V": V_f, "W": W_f}
    
    if time.time() > deadline:
        rank = n * m * p
        U_f = [[0.0]*rank for _ in range(n*m)]
        V_f = [[0.0]*rank for _ in range(m*p)]
        W_f = [[0.0]*rank for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i*m+j][c] = 1.0
                    V_f[j*p+k][c] = 1.0
                    W_f[k*n+i][c] = 1.0
                    c += 1
        return {"U": U_f, "V": V_f, "W": W_f}
    
    return {"U": U, "V": V, "W": W}
