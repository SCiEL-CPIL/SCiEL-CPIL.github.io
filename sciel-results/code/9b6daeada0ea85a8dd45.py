"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Rank-56 decomposition for multiplying a 2x2 matrix A by a 2x16 matrix B.
    Uses Strassen's algorithm (rank 7) for each of the 8 independent 2x2 blocks.
    Falls back to schoolbook (rank 64) if verification fails.
    This is the current best known rank for this tensor shape.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    
    # Strassen coefficients for 2x2x2
    # U rows: [A00, A01, A10, A11] (indices 0,1,2,3)
    # V rows: [B00, B01, B10, B11] (indices 0,1,2,3)
    # W rows: [C00, C01, C10, C11] (indices 0,1,2,3)
    
    # 7 Strassen intermediates
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U coefficients: 7 x 4 matrix (intermediate x A-entry)
    strassen_U = np.array([
        [1, 0, 0, 1],   # M1: A00+A11
        [0, 0, 1, 1],   # M2: A10+A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00+A01
        [-1, 0, 1, 0],  # M6: A10-A00
        [0, 1, 0, -1]   # M7: A01-A11
    ], dtype=float)
    
    # V coefficients: 7 x 4 matrix (intermediate x B-entry)
    strassen_V = np.array([
        [1, 0, 0, 1],   # M1: B00+B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01-B11
        [-1, 0, 1, 0],  # M4: B10-B00
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00+B01
        [0, 0, 1, 1]    # M7: B10+B11
    ], dtype=float)
    
    # W coefficients: 7 x 4 matrix (intermediate x C-entry)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    strassen_W = np.array([
        [1, 0, 0, 1],   # M1: C00+1, C11+1
        [0, 0, 1, -1],  # M2: C10+1, C11-1
        [0, 1, 0, 1],   # M3: C01+1, C11+1
        [1, 0, 1, 0],   # M4: C00+1, C10+1
        [-1, 1, 0, 0],  # M5: C00-1, C01+1
        [0, 0, 0, 1],   # M6: C11+1
        [1, 0, 0, 0]    # M7: C00+1
    ], dtype=float)
    
    num_blocks = p // 2  # 8 blocks of 2 columns
    rank = num_blocks * 7  # 56
    
    # Build the full factor matrices
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for block in range(num_blocks):
        k0 = block * 2
        k1 = k0 + 1
        base = block * 7
        
        # For this block:
        # B[0, k0] -> V[0*p + k0] = V[k0]
        # B[0, k1] -> V[0*p + k1] = V[k1]
        # B[1, k0] -> V[1*p + k0] = V[p + k0]
        # B[1, k1] -> V[1*p + k1] = V[p + k1]
        
        # C[0, k0] -> W[k0*n + 0] = W[2*k0]
        # C[0, k1] -> W[k1*n + 0] = W[2*k1]
        # C[1, k0] -> W[k0*n + 1] = W[2*k0+1]
        # C[1, k1] -> W[k1*n + 1] = W[2*k1+1]
        
        for s in range(7):
            col = base + s
            
            # U: same for all blocks (A entries are global)
            U[:, col] = strassen_U[s]
            
            # V: map local B indices to global V indices
            V[k0, col] = strassen_V[s, 0]       # B00 -> B[0,k0]
            V[k1, col] = strassen_V[s, 1]       # B01 -> B[0,k1]
            V[p + k0, col] = strassen_V[s, 2]   # B10 -> B[1,k0]
            V[p + k1, col] = strassen_V[s, 3]   # B11 -> B[1,k1]
            
            # W: map local C indices to global W indices
            W[2*k0, col] = strassen_W[s, 0]     # C00 -> C[0,k0]
            W[2*k1, col] = strassen_W[s, 1]     # C01 -> C[0,k1]
            W[2*k0+1, col] = strassen_W[s, 2]   # C10 -> C[1,k0]
            W[2*k1+1, col] = strassen_W[s, 3]   # C11 -> C[1,k1]
    
    # Fast vectorized verification
    # Check all (i,j,k) entries: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            u_row = U[i * m + j]  # shape (rank,)
            for k in range(p):
                v_row = V[j * p + k]  # shape (rank,)
                w_row = W[k * n + i]  # shape (rank,)
                val = np.dot(u_row, v_row * w_row)
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-9:
        # Fall back to schoolbook (rank 64)
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb), dtype=float)
        V_fb = np.zeros((m * p, rank_fb), dtype=float)
        W_fb = np.zeros((p * n, rank_fb), dtype=float)
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_fb[ii * m + jj, col] = 1
                    V_fb[jj * p + kk, col] = 1
                    W_fb[kk * n + ii, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
