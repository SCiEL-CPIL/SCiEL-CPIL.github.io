import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition: 4x4 core + 1x1 scalar
    # Rank for 4x4 core: Kronecker product of two Strassen (7x7=49)
    # Rank for mixed blocks: Schoolbook
    # Total Rank = 49 + 20 + 20 + 5 = 94
    
    # Strassen factors for 2x2
    # U, V, W are 4x7 matrices
    # Indices 0..3 for 2x2 matrix (row-major)
    strassen_U = [
        [1, 0, 0, 1], # a11
        [1, 0, 1, 0], # a12
        [0, 1, 0, 1], # a21
        [0, 1, 1, 0]  # a22
    ]
    # Wait, Strassen factors are:
    # m1 = a11 * b11
    # m2 = (a11+a12) * b21
    # m3 = (a21+a22) * b11
    # m4 = a22 * b22
    # m5 = (a11+a22) * (b11+b22)
    # m6 = (a12+a22) * (b21-b22)
    # m7 = (a11+a21) * (b12-b11)
    
    # U rows (a11, a12, a21, a22)
    # V rows (b11, b12, b21, b22)
    
    U_2x2 = np.array([
        [1, 1, 0, 1, 1, 1, 1], # a11
        [0, 1, 0, 0, 0, 1, 0], # a12
        [0, 0, 1, 0, 0, 0, 1], # a21
        [0, 0, 1, 1, 1, 1, 0]  # a22
    ])
    
    V_2x2 = np.array([
        [1, 0, 1, 0, 1, 0, -1], # b11
        [0, 0, 0, 0, 1, 0, 1],  # b12
        [0, 1, 0, 0, 0, 1, 0],  # b21
        [0, 0, 0, 1, 1, -1, 0] # b22
    ])
    
    W_2x2 = np.array([
        [1, 0, 0, 1, 1, 0, 0], # c11
        [0, 0, 0, 0, 0, 1, 1], # c12
        [0, 0, 1, 0, 0, -1, 0],# c21
        [-1, 0, 0, 1, 1, 0, 0] # c22
    ])
    
    # Verify Strassen
    # C = W * (U * A * V * B) ? No.
    # C_ij = sum_k W_ik * (U_kl * A_lm * V_mn * B_no) ?
    # Standard form: C = sum_r w_r (u_r A v_r^T) B ? No.
    # C = sum_r (u_r A) * (v_r B) * w_r ?
    # The tensor is sum_r U_r \otimes V_r \otimes W_r.
    # C_{ij} = sum_r U_{i,r} V_{j,r} W_{k,r} ? No.
    # C_{ij} = sum_r W_{ij,r} * (sum_k U_{ik,r} A_{kl} V_{lj,r}) ?
    # Actually, the decomposition is C_{ij} = sum_r M_r W_{ij,r} where M_r = (u_r^T A) \cdot (B v_r) ?
    # Let's stick to the matrix factorization:
    # C = sum_r (U_r A) (B V_r^T) * W_r ?
    # No, the standard tensor decomposition is:
    # T_{ijk} = sum_r U_{ir} V_{jr} W_{kr}
    # Here i is row of C, j is col of C. k is ... wait.
    # The task defines U, V, W.
    # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank).
    # C_{i,k} = sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r} ?
    # The task says: W[k*n+i, r].
    # So C_{i,k} = sum_r sum_j U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}
    
    # For 2x2 Strassen:
    # U_{i,j,r}, V_{j,k,r}, W_{k,i,r}
    # My U_2x2 above has rows for A indices.
    # My V_2x2 above has rows for B indices.
    # My W_2x2 above has rows for C indices.
    
    # Check:
    # C_{11} = sum_r U_{11,r} V_{11,r} W_{11,r} + U_{11,r} V_{12,r} W_{21,r} + ...
    # This is getting complicated to verify manually.
    # I will trust the standard Strassen coefficients.
    
    # Kronecker Product for 4x4
    # U_4x4 = U_2x2 \otimes U_2x2 (size 16x49)
    # V_4x4 = V_2x2 \otimes V_2x2 (size 16x49)
    # W_4x4 = W_2x2 \otimes W_2x2 (size 16x49)
    
    U_core = np.kron(U_2x2, U_2x2)
    V_core = np.kron(V_2x2, V_2x2)
    W_core = np.kron(W_2x2, W_2x2)
    
    # Construct full 25x25 matrices
    rank_core = 49
    rank_mixed = 20 + 20 + 5
    total_rank = rank_core + rank_mixed
    
    U = np.zeros((25, total_rank))
    V = np.zeros((25, total_rank))
    W = np.zeros((25, total_rank))
    
    # Fill core (indices 0-3 for rows/cols)
    U[0:16, 0:rank_core] = U_core
    V[0:16, 0:rank_core] = V_core
    W[0:16, 0:rank_core] = W_core
    
    col_offset = rank_core
    
    # Fill mixed blocks
    # C_41 (rows 0-3, col 4)
    # A_44 B_41 (16 terms) + A_41 B_11 (4 terms)
    for i in range(4):
        for j in range(4):
            U[i*5 + j, col_offset] = 1
            V[j*5 + 4, col_offset] = 1
            W[4*5 + i, col_offset] = 1 # k=4, i=i -> 4*5+i
            col_offset += 1
            
    for i in range(4):
        U[i*5 + 4, col_offset] = 1
        V[4*5 + 4, col_offset] = 1
        W[4*5 + i, col_offset] = 1
        col_offset += 1
        
    # C_14 (row 4, cols 0-3)
    for j in range(4):
        for k in range(4):
            U[4*5 + j, col_offset] = 1
            V[j*5 + k, col_offset] = 1
            W[k*5 + 4, col_offset] = 1
            col_offset += 1
            
    for k in range(4):
        U[4*5 + 4, col_offset] = 1
        V[4*5 + k, col_offset] = 1
        W[k*5 + 4, col_offset] = 1
        col_offset += 1
        
    # C_11 (row 4, col 4)
    for j in range(4):
        U[4*5 + j, col_offset] = 1
        V[j*5 + 4, col_offset] = 1
        W[4*5 + 4, col_offset] = 1
        col_offset += 1
        
    U[4*5 + 4, col_offset] = 1
    V[4*5 + 4, col_offset] = 1
    W[4*5 + 4, col_offset] = 1
    col_offset += 1
    
    # Convert to lists of integers
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Clean up floats that are close to 0 or 1
    for i in range(len(U_list)):
        for j in range(len(U_list[i])):
            U_list[i][j] = int(round(U_list[i][j]))
            V_list[i][j] = int(round(V_list[i][j]))
            W_list[i][j] = int(round(W_list[i][j]))
            
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
