"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Fallback: Schoolbook Rank 28
    def get_schoolbook():
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

    # Attempt Rank 25 Decomposition
    # Structure: 3 blocks of 2x2x2 (Strassen, Rank 7) + 1 block of 2x2x1 (Schoolbook, Rank 4)
    # Total Rank: 7 * 3 + 4 = 25
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(U_vec, V_vec, W_vec):
        U_list.append(U_vec)
        V_list.append(V_vec)
        W_list.append(W_vec)

    # Strassen Block for cols [c0, c1]
    def add_strassen_block(c0, c1):
        # A indices: 0:a, 1:b, 2:c, 3:d
        # B indices for this block:
        # x is B[0, c0] -> idx 0*p + c0
        # y is B[0, c1] -> idx 0*p + c1
        # z is B[1, c0] -> idx 1*p + c0
        # w is B[1, c1] -> idx 1*p + c1
        
        # C indices for this block:
        # C[0, c0] is W[0*p + c0] (Row 0, Col c0)
        # C[0, c1] is W[0*p + c1]
        # C[1, c0] is W[1*p + c0]
        # C[1, c1] is W[1*p + c1]
        
        # Strassen Multiplications
        # M1 = (a+d)(x+w)
        U1 = [1, 0, 0, 1]
        V1 = [0]*14; V1[0*p+c0]=1; V1[0*p+c1]=1
        # W1 contributes to C11 and C22
        # C11 coeff: +1, C22 coeff: +1
        W1 = [0]*14; W1[0*p+c0]=1; W1[1*p+c1]=1
        add_term(U1, V1, W1)
        
        # M2 = (c+d)(x)
        U2 = [0, 0, 1, 1]
        V2 = [0]*14; V2[0*p+c0]=1
        # W2 contributes to C21 (+1) and C22 (-1)
        W2 = [0]*14; W2[1*p+c0]=1; W2[1*p+c1]=-1
        add_term(U2, V2, W2)
        
        # M3 = (a-b)(y+w)
        U3 = [1, -1, 0, 0]
        V3 = [0]*14; V3[0*p+c1]=1; V3[1*p+c1]=1
        # W3 contributes to C21 (+1)
        W3 = [0]*14; W3[1*p+c0]=1
        add_term(U3, V3, W3)
        
        # M4 = (d-b)(z+w)
        U4 = [0, -1, 0, 1]
        V4 = [0]*14; V4[1*p+c0]=1; V4[1*p+c1]=1
        # W4 contributes to C11 (-1) and C12 (+1)
        W4 = [0]*14; W4[0*p+c0]=-1; W4[0*p+c1]=1
        add_term(U4, V4, W4)
        
        # M5 = (a+b)(x)
        U5 = [1, 1, 0, 0]
        V5 = [0]*14; V5[0*p+c0]=1
        # W5 contributes to C11 (+1)
        W5 = [0]*14; W5[0*p+c0]=1
        add_term(U5, V5, W5)
        
        # M6 = (c+d)(w-y)
        U6 = [0, 0, 1, 1]
        V6 = [0]*14; V6[1*p+c1]=1; V6[0*p+c1]=-1
        # W6 contributes to C22 (+1)
        W6 = [0]*14; W6[1*p+c1]=1
        add_term(U6, V6, W6)
        
        # M7 = (a-b)(z+w)
        U7 = [1, -1, 0, 0]
        V7 = [0]*14; V7[1*p+c0]=1; V7[1*p+c1]=1
        # W7 contributes to C11 (-1), C12 (+1), C22 (+1)
        W7 = [0]*14; W7[0*p+c0]=-1; W7[0*p+c1]=1; W7[1*p+c1]=1
        add_term(U7, V7, W7)

    # Schoolbook Block for col c
    def add_schoolbook_block(c):
        # C[0, c] = a*B[0,c] + b*B[1,c]
        # C[1, c] = c*B[0,c] + d*B[1,c]
        
        # Term 1: a * B[0,c] -> C[0,c]
        U1 = [1, 0, 0, 0]
        V1 = [0]*14; V1[0*p+c]=1
        W1 = [0]*14; W1[0*p+c]=1
        add_term(U1, V1, W1)
        
        # Term 2: b * B[1,c] -> C[0,c]
        U2 = [0, 1, 0, 0]
        V2 = [0]*14; V2[1*p+c]=1
        W2 = [0]*14; W2[0*p+c]=1
        add_term(U2, V2, W2)
        
        # Term 3: c * B[0,c] -> C[1,c]
        U3 = [0, 0, 1, 0]
        V3 = [0]*14; V3[0*p+c]=1
        W3 = [0]*14; W3[1*p+c]=1
        add_term(U3, V3, W3)
        
        # Term 4: d * B[1,c] -> C[1,c]
        U4 = [0, 0, 0, 1]
        V4 = [0]*14; V4[1*p+c]=1
        W4 = [0]*14; W4[1*p+c]=1
        add_term(U4, V4, W4)

    # Build the decomposition
    add_strassen_block(0, 1)
    add_strassen_block(2, 3)
    add_strassen_block(4, 5)
    add_schoolbook_block(6)
    
    # Verify validity
    # Convert to numpy arrays for quick check
    U_np = np.array(U_list).T # Shape (rank, n*m)
    V_np = np.array(V_list).T # Shape (rank, m*p)
    W_np = np.array(W_list).T # Shape (rank, p*n)
    
    # Random test
    np.random.seed(42)
    A_test = np.random.rand(2, 2)
    B_test = np.random.rand(2, 7)
    C_true = A_test @ B_test
    
    # Compute C via decomposition
    # C_flat = sum_k (U_k dot A_flat) * (V_k dot B_flat) * W_k
    A_flat = A_test.flatten()
    B_flat = B_test.flatten()
    
    # (U_np @ A_flat) gives vector of length rank
    # (V_np @ B_flat) gives vector of length rank
    # Element-wise product, then dot with W_np?
    # C_flat = W_np.T @ ((U_np @ A_flat) * (V_np @ B_flat))
    
    C_decomp_flat = W_np.T @ ((U_np @ A_flat) * (V_np @ B_flat))
    C_decomp = C_decomp_flat.reshape(2, 7)
    
    if np.allclose(C_true, C_decomp, atol=1e-9):
        # Valid Rank 25
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Invalid, fallback to Schoolbook
        return get_schoolbook()

# EVOLVE-BLOCK-END
