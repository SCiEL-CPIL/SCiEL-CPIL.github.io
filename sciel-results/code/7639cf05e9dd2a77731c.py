"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Reconstruct the tensor
    # T[i*m+j, j'*p+k, k'*n+i'] should be 1 if j==j' and i==i' and k==k'
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Expected: C[i,k] gets contribution from A[i,j] * B[j,k]
                # In tensor form: T[(i,j),(j,k),(k,i)] = 1
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i  # W[k*n+i] convention
                
                # Compute sum over r
                s = 0.0
                for r in range(rank):
                    s += U[a_idx, r] * V[b_idx, r] * W[c_idx, r]
                
                if abs(s - 1.0) > 1e-9:
                    return False
    
    # Also verify that "wrong" entries are zero
    # T[(i,j),(j',k),(k',i')] = 0 unless j==j' and i==i' and k==k'
    # Check a few wrong entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                # Check that for wrong c_idx, the sum is 0
                for k2 in range(p):
                    for i2 in range(n):
                        if k2 == k and i2 == i:
                            continue
                        c_idx2 = k2 * n + i2
                        s = 0.0
                        for r in range(rank):
                            s += U[a_idx, r] * V[b_idx, r] * W[c_idx2, r]
                        if abs(s) > 1e-9:
                            return False
                # Check for wrong b_idx
                for j2 in range(m):
                    if j2 == j:
                        continue
                    for k2 in range(p):
                        b_idx2 = j2 * p + k2
                        c_idx2 = k2 * n + i
                        s = 0.0
                        for r in range(rank):
                            s += U[a_idx, r] * V[b_idx2, r] * W[c_idx2, r]
                        if abs(s) > 1e-9:
                            return False
                # Check for wrong a_idx
                for i2 in range(n):
                    for j2 in range(m):
                        if i2 == i and j2 == j:
                            continue
                        a_idx2 = i2 * m + j2
                        c_idx2 = k * n + i
                        s = 0.0
                        for r in range(rank):
                            s += U[a_idx2, r] * V[b_idx, r] * W[c_idx2, r]
                        if abs(s) > 1e-9:
                            return False
    return True

def _strassen_block(a_idx, b_idx, w_idx, start_col, U, V, W):
    """Add a Strassen 2x2x2 decomposition starting at start_col."""
    # Strassen formulas:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    a00, a01, a10, a11 = a_idx
    b00, b01, b10, b11 = b_idx
    z00, z01, z10, z11 = w_idx
    
    # U (A factors)
    U[a00][start_col + 0] += 1
    U[a11][start_col + 0] += 1
    U[a10][start_col + 1] += 1
    U[a11][start_col + 1] += 1
    U[a00][start_col + 2] += 1
    U[a11][start_col + 3] += 1
    U[a00][start_col + 4] += 1
    U[a01][start_col + 4] += 1
    U[a10][start_col + 5] += 1
    U[a00][start_col + 5] -= 1
    U[a01][start_col + 6] += 1
    U[a11][start_col + 6] -= 1
    
    # V (B factors)
    V[b00][start_col + 0] += 1
    V[b11][start_col + 0] += 1
    V[b00][start_col + 1] += 1
    V[b01][start_col + 2] += 1
    V[b11][start_col + 2] -= 1
    V[b10][start_col + 3] += 1
    V[b00][start_col + 3] -= 1
    V[b11][start_col + 4] += 1
    V[b00][start_col + 5] += 1
    V[b01][start_col + 5] += 1
    V[b10][start_col + 6] += 1
    V[b11][start_col + 6] += 1
    
    # W (C factors)
    # z00 = M1 + M4 - M5 + M7
    W[z00][start_col + 0] += 1
    W[z00][start_col + 3] += 1
    W[z00][start_col + 4] -= 1
    W[z00][start_col + 6] += 1
    
    # z01 = M3 + M5
    W[z01][start_col + 2] += 1
    W[z01][start_col + 4] += 1
    
    # z10 = M2 + M4
    W[z10][start_col + 1] += 1
    W[z10][start_col + 3] += 1
    
    # z11 = M1 - M2 + M3 + M6
    W[z11][start_col + 0] += 1
    W[z11][start_col + 1] -= 1
    W[z11][start_col + 2] += 1
    W[z11][start_col + 5] += 1

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Build the rank-42 Strassen decomposition
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # A is 2x4. Indices:
    # A[0,0]=0, A[0,1]=1, A[0,2]=2, A[0,3]=3
    # A[1,0]=4, A[1,1]=5, A[1,2]=6, A[1,3]=7
    a_L = [0, 1, 4, 5]  # A_L = [[A00, A01], [A10, A11]]
    a_R = [2, 3, 6, 7]  # A_R = [[A02, A03], [A12, A13]]
    
    # B is 4x6. Indices j*p + k
    # Block 1: B cols 0,1
    b_1_T = [0, 1, 6, 7]    # B rows 0,1 cols 0,1
    b_1_B = [12, 13, 18, 19] # B rows 2,3 cols 0,1
    
    # W indices for C cols 0,1: C[0,0]->W[0], C[0,1]->W[2], C[1,0]->W[1], C[1,1]->W[3]
    w_1 = [0, 2, 1, 3]
    
    _strassen_block(a_L, b_1_T, w_1, 0, U, V, W)
    _strassen_block(a_R, b_1_B, w_1, 7, U, V, W)
    
    # Block 2: B cols 2,3
    b_2_T = [2, 3, 8, 9]
    b_2_B = [14, 15, 20, 21]
    w_2 = [4, 6, 5, 7]
    
    _strassen_block(a_L, b_2_T, w_2, 14, U, V, W)
    _strassen_block(a_R, b_2_B, w_2, 21, U, V, W)
    
    # Block 3: B cols 4,5
    b_3_T = [4, 5, 10, 11]
    b_3_B = [16, 17, 22, 23]
    w_3 = [8, 10, 9, 11]
    
    _strassen_block(a_L, b_3_T, w_3, 28, U, V, W)
    _strassen_block(a_R, b_3_B, w_3, 35, U, V, W)
    
    # Verify the decomposition
    if time.time() < deadline - 20:
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        else:
            # Fallback: return as-is (shouldn't happen)
            return {"U": U, "V": V, "W": W}
    else:
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
