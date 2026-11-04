import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # 1. Strassen 2x2 decomposition (Rank 7)
    # U_S: 4x7, V_S: 4x7, W_S: 4x7
    # Flattening: row-major. A = [a00, a01, a10, a11]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    U_S = np.array([
        [1, 0, 1, 0, 1, -1, 0], # a00
        [0, 0, 0, 0, 1, 0, 1],  # a01
        [0, 1, 0, 0, 0, 1, 0],  # a10
        [1, 1, 0, 1, 0, 0, -1]  # a11
    ], dtype=float)
    
    V_S = np.array([
        [1, 1, 0, -1, 0, 1, 0], # b00
        [0, 0, 1, 0, 0, 1, 0],  # b01
        [1, 0, -1, 1, 0, 0, 1], # b10
        [1, 0, 0, 0, 1, 0, 1]   # b11
    ], dtype=float)
    
    W_S = np.array([
        [1, 0, 0, 1, -1, 0, 1], # c00
        [0, 0, 1, 0, 1, 0, 0],  # c01
        [0, 1, 0, 1, 0, 0, 0],  # c10
        [1, -1, 1, 0, 0, 1, 0]  # c11
    ], dtype=float)

    # 2. Standard 3x3 decomposition (Rank 9)
    # U_L, V_L, W_L are 9x9 identity-like matrices for schoolbook.
    # For standard multiplication, U_L, V_L, W_L are just basis vectors.
    # c_ij = sum_k a_ik b_kj
    # We can represent this with U_L, V_L, W_L of size 9x9.
    # U_L[i*3+j, k*3+l] = 1 if i==k and j==l? No.
    # The tensor is T[i,j,k] = delta_ik delta_jl? No.
    # T[i,j,k] where i is row of A, j is col of B, k is intermediate.
    # Standard: C[i,j] = sum_k A[i,k] B[k,j].
    # Tensor index (i, j, k).
    # U[i*3+k, c] = 1 if c == i*3+k? No.
    # In the schoolbook decomposition, each term is A[i,k] * B[k,j].
    # The "rank" term is indexed by (i,k,j).
    # U[i*3+k, (i,k,j)] = 1
    # V[k*3+j, (i,k,j)] = 1
    # W[j*3+i, (i,k,j)] = 1 (Output C[i,j] is at index i*3+j)
    
    U_L = np.zeros((9, 9))
    V_L = np.zeros((9, 9))
    W_L = np.zeros((9, 9))
    for i in range(3):
        for k in range(3):
            for j in range(3):
                idx = i*9 + k*3 + j # Rank index
                U_L[i*3 + k, idx] = 1
                V_L[k*3 + j, idx] = 1
                W_L[j*3 + i, idx] = 1 # Wait, output is C[i,j]. Index is i*3+j.
                # My W_S above used row-major for C. c00, c01, c10, c11.
                # So W_L[j*3+i] is wrong. It should be W_L[i*3+j, idx] = 1.
                
    # Correct W_L
    W_L = np.zeros((9, 9))
    for i in range(3):
        for k in range(3):
            for j in range(3):
                idx = i*9 + k*3 + j
                W_L[i*3 + j, idx] = 1

    # 3. Tensor Product (Kronecker Product)
    # U_6 = U_S (x) U_L
    # Size: (4*9) x (7*9) = 36 x 63
    U_6 = np.kron(U_S, U_L)
    V_6 = np.kron(V_S, V_L)
    W_6 = np.kron(W_S, W_L)
    
    # 4. Restrict to 5x5
    # The 6x6 indices are mapped from (i_S, i_L) to i = i_S*3 + i_L.
    # We want to keep only indices i < 5.
    # i=0: (0,0)
    # i=1: (0,1)
    # i=2: (0,2)
    # i=3: (1,0)
    # i=4: (1,1)
    # i=5: (1,2) -> Zero out
    
    # Create a mask for valid indices in 6x6 space
    valid_indices = [0, 1, 2, 3, 4]
    
    # U_6 rows correspond to A_6 indices.
    # V_6 rows correspond to B_6 indices.
    # W_6 rows correspond to C_6 indices.
    
    # Filter rows
    U_5 = U_6[valid_indices, :]
    V_5 = V_6[valid_indices, :]
    W_5 = W_6[valid_indices, :]
    
    # Convert to list of lists as required
    return {
        "U": U_5.tolist(),
        "V": V_5.tolist(),
        "W": W_5.tolist()
    }
# EVOLVE-BLOCK-END
