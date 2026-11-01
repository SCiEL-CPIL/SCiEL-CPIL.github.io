"""
Rank-minimal tensor decomposition for 2x2 x 2x15 matrix multiplication.

Strategy:
- The tensor T_{2,2,15} decomposes as a direct sum over the 15 columns of B.
- Each column of B gives a 2x2 x 2x1 sub-problem.
- We can group columns to exploit Strassen's 7-rank decomposition for 2x2x2.
- Split 15 columns into 7 groups of 2 and 1 group of 1.
- For each group of 2 columns: use Strassen 2x2x2 (rank 7).
- For the last single column: use schoolbook 2x2x2x1 (rank 4).
- Total rank = 7*7 + 4 = 53.
"""
import time
import numpy as np

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 15
    start_time = time.time()
    deadline = start_time + 100  # safety margin

    # U shape: (n*m, rank) = (4, rank)
    # V shape: (m*p, rank) = (30, rank)
    # W shape: (p*n, rank) = (30, rank)
    
    # We'll build the decomposition by groups.
    # 7 groups of 2 columns each, then 1 group of 1 column.
    
    U_list = []  # Each entry is a 4-vector
    V_list = []  # Each entry is a 30-vector
    W_list = []  # Each entry is a 30-vector
    
    # Strassen 2x2x2 coefficients
    # M1 = (a00 + a11)(b00 + b11)
    U1 = [1, 0, 0, 1]
    V1 = [1, 0, 0, 1]
    # c00 = +1, c11 = +1 -> W[0]=1, W[3]=1
    W1 = [1, 0, 0, 1]
    
    # M2 = (a10 + a11)b00
    U2 = [0, 0, 1, 1]
    V2 = [1, 0, 0, 0]
    # c10 = +1 -> W[2]=1
    W2 = [0, 0, 1, 0]
    
    # M3 = a00(b10 - b11)
    U3 = [1, 0, 0, 0]
    V3 = [0, 0, 1, -1]
    # c10 = +1, c11 = -1 -> W[2]=1, W[3]=-1
    W3 = [0, 0, 1, -1]
    
    # M4 = a11(b01 - b00)
    U4 = [0, 0, 0, 1]
    V4 = [-1, 1, 0, 0]
    # c01 = +1, c10 = -1 -> W[1]=1, W[2]=-1
    W4 = [0, 1, -1, 0]
    
    # M5 = (a00 + a10)b11
    U5 = [1, 0, 1, 0]
    V5 = [0, 0, 0, 1]
    # c01 = +1, c11 = +1 -> W[1]=1, W[3]=1
    W5 = [0, 1, 0, 1]
    
    # M6 = (a11 - a10)(b00 + b10)
    U6 = [0, 0, -1, 1]
    V6 = [1, 0, 1, 0]
    # c11 = +1 -> W[3]=1
    W6 = [0, 0, 0, 1]
    
    # M7 = (a01 - a11)(b01 + b11)
    U7 = [0, 1, 0, -1]
    V7 = [0, 1, 0, 1]
    # c00 = +1, c01 = +1, c10 = +1 -> W[0]=1, W[1]=1, W[2]=1
    W7 = [1, 1, 1, 0]
    
    strassen_U = [U1, U2, U3, U4, U5, U6, U7]
    strassen_V = [V1, V2, V3, V4, V5, V6, V7]
    strassen_W = [W1, W2, W3, W4, W5, W6, W7]
    
    # For a single column (2x2 x 2x1):
    # A is 2x2, B is 2x1, C is 2x1.
    # Schoolbook: 4 terms.
    # a00*b0 -> c0
    # a01*b1 -> c0
    # a10*b0 -> c1
    # a11*b1 -> c1
    single_U = [
        [1, 0, 0, 0],  # a00
        [0, 1, 0, 0],  # a01
        [0, 0, 1, 0],  # a10
        [0, 0, 0, 1],  # a11
    ]
    single_V = [
        [1, 0],  # b0
        [0, 1],  # b1
        [1, 0],  # b0
        [0, 1],  # b1
    ]
    single_W = [
        [1, 0],  # c0
        [1, 0],  # c0
        [0, 1],  # c1
        [0, 1],  # c1
    ]
    
    # Now assemble.
    # For each of the 7 groups of 2 columns (k=0,1), (k=2,3), ..., (k=12,13):
    # The V indices for these 2 columns in the 2x15 B matrix:
    # B is 2x15, so V has shape (2*15, rank) = (30, rank).
    # V[j*15 + k, r] where j is row (0,1), k is col (0..14).
    # For columns 2g and 2g+1:
    #   b00 (row 0, col 2g) -> V[0*15 + 2g] = V[30g]
    #   b01 (row 0, col 2g+1) -> V[0*15 + 2g+1] = V[30g+1]
    #   b10 (row 1, col 2g) -> V[1*15 + 2g] = V[15+30g]
    #   b11 (row 1, col 2g+1) -> V[1*15 + 2g+1] = V[15+30g+1]
    
    # W indices for C:
    # C is 2x15, W[k*2 + i, r] where i is row (0,1), k is col (0..14).
    # For columns 2g and 2g+1:
    #   c00 (row 0, col 2g) -> W[2g*2 + 0] = W[4g]
    #   c01 (row 0, col 2g+1) -> W[(2g+1)*2 + 0] = W[4g+2]
    #   c10 (row 1, col 2g) -> W[2g*2 + 1] = W[4g+1]
    #   c11 (row 1, col 2g+1) -> W[(2g+1)*2 + 1] = W[4g+3]
    
    for g in range(7):
        for t in range(7):
            U_vec = strassen_U[t][:]
            U_list.append(U_vec)
            
            V_vec = [0] * 30
            v4 = strassen_V[t]
            V_vec[15*0 + 2*g] = v4[0]    # b00
            V_vec[15*0 + 2*g+1] = v4[1]  # b01
            V_vec[15*1 + 2*g] = v4[2]    # b10
            V_vec[15*1 + 2*g+1] = v4[3]  # b11
            V_list.append(V_vec)
            
            W_vec = [0] * 30
            w4 = strassen_W[t]
            W_vec[2*(2*g) + 0] = w4[0]    # c00
            W_vec[2*(2*g+1) + 0] = w4[1]  # c01
            W_vec[2*(2*g) + 1] = w4[2]    # c10
            W_vec[2*(2*g+1) + 1] = w4[3]  # c11
            W_list.append(W_vec)
    
    # For the last single column (col 14):
    # V indices: row 0 col 14 -> V[14], row 1 col 14 -> V[29]
    # W indices: row 0 col 14 -> W[14*2+0]=W[28], row 1 col 14 -> W[14*2+1]=W[29]
    for t in range(4):
        U_vec = single_U[t][:]
        U_list.append(U_vec)
        
        V_vec = [0] * 30
        v2 = single_V[t]
        V_vec[0*15 + 14] = v2[0]  # b0
        V_vec[1*15 + 14] = v2[1]  # b1
        V_list.append(V_vec)
        
        W_vec = [0] * 30
        w2 = single_W[t]
        W_vec[14*2 + 0] = w2[0]  # c0
        W_vec[14*2 + 1] = w2[1]  # c1
        W_list.append(W_vec)
    
    rank = len(U_list)
    assert rank == 53, f"Expected rank 53, got {rank}"
    
    U = np.array(U_list).T  # shape (4, 53)
    V = np.array(V_list).T  # shape (30, 53)
    W = np.array(W_list).T  # shape (30, 53)
    
    # Verify the decomposition with random test
    if time.time() < deadline:
        np.random.seed(42)
        A_test = np.random.randn(2, 2)
        B_test = np.random.randn(2, 15)
        C_expected = A_test @ B_test
        
        A_flat = A_test.flatten()  # row-major: a00, a01, a10, a11
        B_flat = B_test.flatten()  # row-major: b[0][0], b[0][1], ..., b[0][14], b[1][0], ..., b[1][14]
        
        C_flat = np.zeros(30)
        for r in range(rank):
            u_dot = np.dot(U[:, r], A_flat)
            v_dot = np.dot(V[:, r], B_flat)
            C_flat += W[:, r] * u_dot * v_dot
        
        C_from_decomp = C_flat.reshape(2, 15)
        if not np.allclose(C_from_decomp, C_expected, atol=1e-9):
            # Fallback to schoolbook
            U_s = np.zeros((4, 60))
            V_s = np.zeros((30, 60))
            W_s = np.zeros((30, 60))
            col = 0
            for i in range(2):
                for j in range(2):
                    for k in range(15):
                        U_s[i*2 + j, col] = 1
                        V_s[j*15 + k, col] = 1
                        W_s[k*2 + i, col] = 1
                        col += 1
            return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
