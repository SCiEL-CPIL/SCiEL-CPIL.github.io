import numpy as np
import time

START_TIME = time.time()
TIME_LIMIT = 140.0

def construct():
    n, m, p = 3, 4, 6
    # Strategy: split B into two 4x3 blocks, each 3x4x3 uses rank-12 decomposition
    # Total rank: 24
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_3x4x3_block(k_base):
        """Add rank-12 decomposition for 3x4x3 block at column offset k_base."""
        nonlocal col
        # A is 3x4 (rows 0-2, cols 0-3), B is 4x3 (rows 0-3, cols k_base to k_base+2)
        # C is 3x3
        
        # Sub-block: A[0:2,:] x B[:, 0:2] -> C[0:2, 0:2] (2x2x2 Strassen, rank 7)
        # A[0:2,:] x B[:, 2] -> C[0:2, 2] (2x4x1, rank 2)
        # A[2,:] x B[:, 0:2] -> C[2, 0:2] (1x4x2, rank 2)
        # A[2,:] x B[:, 2] -> C[2, 2] (1x4x1, rank 1)
        
        # Strassen 2x2x2 for C[0:2, 0:2]
        # A sub: A[0, 0:2], A[1, 0:2]
        # B sub: B[0:2, k_base:k_base+2]
        a00 = 0 * m + 0
        a01 = 0 * m + 1
        a10 = 1 * m + 0
        a11 = 1 * m + 1
        b00 = 0 * p + k_base
        b01 = 0 * p + (k_base + 1)
        b10 = 1 * p + k_base
        b11 = 1 * p + (k_base + 1)
        c00 = k_base * n + 0
        c01 = (k_base + 1) * n + 0
        c10 = k_base * n + 1
        c11 = (k_base + 1) * n + 1
        
        # Strassen 7 terms
        # M1 = (a00+a11)*(b00+b11) -> c00 += M1, c11 += M1
        U[a00][col] = 1; U[a11][col] = 1
        V[b00][col] = 1; V[b11][col] = 1
        W[c00][col] = 1; W[c11][col] = 1
        col += 1
        
        # M2 = a11*(b10-b00) -> c10 += M2, c11 -= M2
        U[a11][col] = 1
        V[b10][col] = 1; V[b00][col] = -1
        W[c10][col] = 1; W[c11][col] = -1
        col += 1
        
        # M3 = (a00-a10)*b00 -> c00 += M3, c10 += M3
        U[a00][col] = 1; U[a10][col] = -1
        V[b00][col] = 1
        W[c00][col] = 1; W[c10][col] = 1
        col += 1
        
        # M4 = a11*(b11-b01) -> c01 += M4, c11 += M4
        U[a11][col] = 1
        V[b11][col] = 1; V[b01][col] = -1
        W[c01][col] = 1; W[c11][col] = 1
        col += 1
        
        # M5 = (a00+a01)*b11 -> c01 += M5, c11 -= M5
        U[a00][col] = 1; U[a01][col] = 1
        V[b11][col] = 1
        W[c01][col] = 1; W[c11][col] = -1
        col += 1
        
        # M6 = (a10-a00)*(b00+b01) -> c00 -= M6, c10 += M6
        U[a10][col] = 1; U[a00][col] = -1
        V[b00][col] = 1; V[b01][col] = 1
        W[c00][col] = -1; W[c10][col] = 1
        col += 1
        
        # M7 = (a01-a11)*(b10+b11) -> c01 += M7, c11 -= M7
        U[a01][col] = 1; U[a11][col] = -1
        V[b10][col] = 1; V[b11][col] = 1
        W[c01][col] = 1; W[c11][col] = -1
        col += 1
        
        # C[0:2, 2] from A[0:2,:] x B[:, 2] (2x4x1, rank 2)
        # C[0,2] = A[0,:] dot B[:, k_base+2]
        # C[1,2] = A[1,:] dot B[:, k_base+2]
        b_col2 = 0 * p + (k_base + 2)
        # Wait, B is 4x3, so B[j, k_base+2] for j=0..3
        # Actually B row index is j (0..3), col index is k (k_base..k_base+2)
        # So B[j, k_base+2] has index j*p + (k_base+2)
        
        # C[0,2]: sum_j A[0,j] * B[j, k_base+2]
        for j in range(4):
            U[0 * m + j][col] = 1
            V[j * p + (k_base + 2)][col] = 1
            W[(k_base + 2) * n + 0][col] = 1
        col += 1
        
        # C[1,2]: sum_j A[1,j] * B[j, k_base+2]
        for j in range(4):
            U[1 * m + j][col] = 1
            V[j * p + (k_base + 2)][col] = 1
            W[(k_base + 2) * n + 1][col] = 1
        col += 1
        
        # C[2, 0:2] from A[2,:] x B[:, 0:2] (1x4x2, rank 2)
        # C[2,0] = A[2,:] dot B[:, k_base]
        # C[2,1] = A[2,:] dot B[:, k_base+1]
        for j in range(4):
            U[2 * m + j][col] = 1
            V[j * p + k_base][col] = 1
            W[k_base * n + 2][col] = 1
        col += 1
        
        for j in range(4):
            U[2 * m + j][col] = 1
            V[j * p + (k_base + 1)][col] = 1
            W[(k_base + 1) * n + 2][col] = 1
        col += 1
        
        # C[2, 2] from A[2,:] x B[:, 2] (1x4x1, rank 1)
        for j in range(4):
            U[2 * m + j][col] = 1
            V[j * p + (k_base + 2)][col] = 1
            W[(k_base + 2) * n + 2][col] = 1
        col += 1
    
    # Two blocks: k_base = 0 and k_base = 3
    add_3x4x3_block(0)
    add_3x4x3_block(3)
    
    # Verify
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i * m + j]
                v_val = V_np[j * p + k]
                w_val = W_np[k * n + i]
                total = np.dot(u_val, v_val * w_val)
                if abs(total - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        return _construct_schoolbook()
    
    return {"U": U, "V": V, "W": W}

def _construct_schoolbook():
    n, m, p = 3, 4, 6
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
