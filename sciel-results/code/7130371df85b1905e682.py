"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # 5s safety margin
    
    # Rank 22 decomposition (current best)
    # 7 for A_L B_T1 (2x2x2 Strassen)
    # 7 for A_L B_T2 (2x2x2 Strassen)
    # 8 for A_R B_B  (2x1x4 Schoolbook)
    rank = 22
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val, col):
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
    
    # Indices:
    # A: 2x3. A[i,j] index i*3+j
    # B: 3x4. B[j,k] index j*4+k
    # C: 2x4. C[i,k] index k*2+i (W[k*n+i, r] -> W index is k*n+i)
    # n=2. So W index is 2*k + i.
    
    # Block 1: A_L B_T1
    # A_L is A[:, 0:2] (2x2)
    # B_T1 is B[0:2, 0:2] (2x2)
    # Strassen for C1 (2x2) corresponding to C[:, 0:2]
    # C1[i,k] = C[i, k] for k in 0,1.
    
    # Strassen terms for 2x2x2:
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # Map to global indices
    # A_L indices:
    # A00 -> A[0,0] -> idx 0
    # A01 -> A[0,1] -> idx 1
    # A10 -> A[1,0] -> idx 3
    # A11 -> A[1,1] -> idx 4
    
    # B_T1 indices:
    # B00 -> B[0,0] -> idx 0
    # B01 -> B[0,1] -> idx 1
    # B10 -> B[1,0] -> idx 4
    # B11 -> B[1,1] -> idx 5
    
    # C1 indices (W):
    # C00 -> C[0,0] -> W idx 0*2+0 = 0
    # C01 -> C[0,1] -> W idx 1*2+0 = 2
    # C10 -> C[1,0] -> W idx 0*2+1 = 1
    # C11 -> C[1,1] -> W idx 1*2+1 = 3
    
    col = 0
    # M1
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[5][col] = 1
    W[0][col] = 1
    W[2][col] = 0
    W[1][col] = 0
    W[3][col] = 1
    col += 1
    
    # M2
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[0][col] = 0
    W[2][col] = 0
    W[1][col] = 1
    W[3][col] = -1
    col += 1
    
    # M3
    U[0][col] = 1
    V[1][col] = 1; V[5][col] = -1
    W[0][col] = 0
    W[2][col] = 1
    W[1][col] = 0
    W[3][col] = 1
    col += 1
    
    # M4
    U[4][col] = 1
    V[4][col] = 1; V[0][col] = -1
    W[0][col] = 1
    W[2][col] = 0
    W[1][col] = 1
    W[3][col] = 0
    col += 1
    
    # M5
    U[0][col] = 1; U[1][col] = 1
    V[5][col] = 1
    W[0][col] = -1
    W[2][col] = 1
    W[1][col] = 0
    W[3][col] = 0
    col += 1
    
    # M6
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[0][col] = 0
    W[2][col] = 0
    W[1][col] = 0
    W[3][col] = 1
    col += 1
    
    # M7
    U[1][col] = 1; U[4][col] = -1
    V[4][col] = 1; V[5][col] = 1
    W[0][col] = 1
    W[2][col] = 0
    W[1][col] = 0
    W[3][col] = 0
    col += 1
    
    # Block 2: A_L B_T2
    # B_T2 is B[0:2, 2:4] (2x2)
    # B indices:
    # B02 -> B[0,2] -> idx 2
    # B03 -> B[0,3] -> idx 3
    # B12 -> B[1,2] -> idx 6
    # B13 -> B[1,3] -> idx 7
    
    # C2 indices (W):
    # C02 -> C[0,2] -> W idx 2*2+0 = 4
    # C03 -> C[0,3] -> W idx 3*2+0 = 6
    # C12 -> C[1,2] -> W idx 2*2+1 = 5
    # C13 -> C[1,3] -> W idx 3*2+1 = 7
    
    # M1
    U[0][col] = 1; U[4][col] = 1
    V[2][col] = 1; V[7][col] = 1
    W[4][col] = 1
    W[6][col] = 0
    W[5][col] = 0
    W[7][col] = 1
    col += 1
    
    # M2
    U[3][col] = 1; U[4][col] = 1
    V[2][col] = 1
    W[4][col] = 0
    W[6][col] = 0
    W[5][col] = 1
    W[7][col] = -1
    col += 1
    
    # M3
    U[0][col] = 1
    V[3][col] = 1; V[7][col] = -1
    W[4][col] = 0
    W[6][col] = 1
    W[5][col] = 0
    W[7][col] = 1
    col += 1
    
    # M4
    U[4][col] = 1
    V[6][col] = 1; V[2][col] = -1
    W[4][col] = 1
    W[6][col] = 0
    W[5][col] = 1
    W[7][col] = 0
    col += 1
    
    # M5
    U[0][col] = 1; U[1][col] = 1
    V[7][col] = 1
    W[4][col] = -1
    W[6][col] = 1
    W[5][col] = 0
    W[7][col] = 0
    col += 1
    
    # M6
    U[3][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[4][col] = 0
    W[6][col] = 0
    W[5][col] = 0
    W[7][col] = 1
    col += 1
    
    # M7
    U[1][col] = 1; U[4][col] = -1
    V[6][col] = 1; V[7][col] = 1
    W[4][col] = 1
    W[6][col] = 0
    W[5][col] = 0
    W[7][col] = 0
    col += 1
    
    # Block 3: A_R B_B
    # A_R is A[:, 2] (2x1)
    # A02 -> idx 2
    # A12 -> idx 5
    
    # B_B is B[2, :] (1x4)
    # B20 -> idx 8
    # B21 -> idx 9
    # B22 -> idx 10
    # B23 -> idx 11
    
    # C indices:
    # C00 -> W 0
    # C01 -> W 2
    # C02 -> W 4
    # C03 -> W 6
    # C10 -> W 1
    # C11 -> W 3
    # C12 -> W 5
    # C13 -> W 7
    
    # Z1: A02 B20 -> C00
    U[2][col] = 1
    V[8][col] = 1
    W[0][col] = 1
    col += 1
    
    # Z2: A02 B21 -> C01
    U[2][col] = 1
    V[9][col] = 1
    W[2][col] = 1
    col += 1
    
    # Z3: A02 B22 -> C02
    U[2][col] = 1
    V[10][col] = 1
    W[4][col] = 1
    col += 1
    
    # Z4: A02 B23 -> C03
    U[2][col] = 1
    V[11][col] = 1
    W[6][col] = 1
    col += 1
    
    # Z5: A12 B20 -> C10
    U[5][col] = 1
    V[8][col] = 1
    W[1][col] = 1
    col += 1
    
    # Z6: A12 B21 -> C11
    U[5][col] = 1
    V[9][col] = 1
    W[3][col] = 1
    col += 1
    
    # Z7: A12 B22 -> C12
    U[5][col] = 1
    V[10][col] = 1
    W[5][col] = 1
    col += 1
    
    # Z8: A12 B23 -> C13
    U[5][col] = 1
    V[11][col] = 1
    W[7][col] = 1
    col += 1
    
    # Verify the decomposition
    # Convert to numpy for verification
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 if i,j,k match, 0 otherwise
    # Actually, the tensor decomposition is:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need T[i,j,k] = 1 for all valid i,j,k (since matrix multiplication is exact)
    # Wait, no. The matrix multiplication tensor T is defined such that:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has entries T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # Let's verify:
    # For each r, the term is U[:,r] * V[:,r] * W[:,r] (outer product in a sense)
    # The sum over r should give the tensor T where T[i,j,k] = 1
    
    # Reshape U, V, W to 3D for verification
    # U: (n*m, rank) -> (n, m, rank)
    # V: (m*p, rank) -> (m, p, rank)
    # W: (p*n, rank) -> (p, n, rank)
    
    U_3d = U_np.reshape(n, m, rank)
    V_3d = V_np.reshape(m, p, rank)
    W_3d = W_np.reshape(p, n, rank)
    
    # Compute tensor: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Note: W is indexed by [k*n+i], so W_3d[k, i, r]
    
    # Use einsum for efficiency
    # T[i,j,k] = sum_r U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
    
    # This is tricky because of the index permutations. Let's do it explicitly.
    T = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] += U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
    
    # Check if T is all ones
    if not np.allclose(T, 1.0):
        # Fallback to schoolbook
        rank_school = n * m * p
        U_school = [[0] * rank_school for _ in range(n * m)]
        V_school = [[0] * rank_school for _ in range(m * p)]
        W_school = [[0] * rank_school for _ in range(p * n)]
        col_school = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j][col_school] = 1
                    V_school[j * p + k][col_school] = 1
                    W_school[k * n + i][col_school] = 1
                    col_school += 1
        return {"U": U_school, "V": V_school, "W": W_school}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
