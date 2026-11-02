import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = 2, 3, 8
    
    # Target rank: 44
    # Structure: 4 blocks of (2x3 * 3x2)
    # Each block uses rank 11:
    #   - 4 for the first column/row (j=0)
    #   - 7 for the remaining 2x2 (j=1,2) using Strassen
    
    rank = 44
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)
    
    col = 0
    
    # Helper to add a rank-1 term
    # u_indices: list of (index, coeff) for U
    # v_indices: list of (index, coeff) for V
    # w_indices: list of (index, coeff) for W
    def add_term(u_idx, v_idx, w_idx):
        nonlocal col
        for idx, c in u_idx:
            U[idx, col] += c
        for idx, c in v_idx:
            V[idx, col] += c
        for idx, c in w_idx:
            W[idx, col] += c
        col += 1

    # Iterate over 4 blocks of k
    for block in range(4):
        k0 = block * 2
        k1 = block * 2 + 1
        
        # --- Part 1: j=0 (Rank 4) ---
        # A indices: i*3 + 0
        # B indices: 0*8 + k
        # C indices: k*2 + i
        
        # Term 1: A[0,0] * B[0,k0] -> C[0,k0]
        add_term([(0, 1)], [(k0, 1)], [(k0, 1)])
        
        # Term 2: A[0,0] * B[0,k1] -> C[0,k1]
        add_term([(0, 1)], [(k1, 1)], [(k1, 1)])
        
        # Term 3: A[1,0] * B[0,k0] -> C[1,k0]
        add_term([(3, 1)], [(k0, 1)], [(k0 + 1, 1)])
        
        # Term 4: A[1,0] * B[0,k1] -> C[1,k1]
        add_term([(3, 1)], [(k1, 1)], [(k1 + 1, 1)])
        
        # --- Part 2: j=1,2 (Rank 7, Strassen) ---
        # A_sub: [[A[0,1], A[0,2]], [A[1,1], A[1,2]]]
        # Indices: 
        # A[0,1] -> 1
        # A[0,2] -> 2
        # A[1,1] -> 4
        # A[1,2] -> 5
        #
        # B_sub: [[B[1,k0], B[1,k1]], [B[2,k0], B[2,k1]]]
        # Indices:
        # B[1,k0] -> 8+k0
        # B[1,k1] -> 8+k1
        # B[2,k0] -> 16+k0
        # B[2,k1] -> 16+k1
        #
        # C_sub: [[C[0,k0], C[0,k1]], [C[1,k0], C[1,k1]]]
        # Indices:
        # C[0,k0] -> 2*k0
        # C[0,k1] -> 2*k1
        # C[1,k0] -> 2*k0 + 1
        # C[1,k1] -> 2*k1 + 1
        
        a00, a01, a10, a11 = 1, 2, 4, 5
        b00, b01, b10, b11 = 8 + k0, 8 + k1, 16 + k0, 16 + k1
        c00, c01, c10, c11 = 2 * k0, 2 * k1, 2 * k0 + 1, 2 * k1 + 1
        
        # Strassen M1 = (A11 + A22) * (B11 + B22)
        # A11=a00, A22=a11, B11=b00, B22=b11
        add_term([(a00, 1), (a11, 1)], [(b00, 1), (b11, 1)], 
                 [(c00, 1), (c11, 1)])
        
        # Strassen M2 = (A21 + A22) * B11
        # A21=a10, A22=a11, B11=b00
        add_term([(a10, 1), (a11, 1)], [(b00, 1)], 
                 [(c10, 1), (c11, -1)])
        
        # Strassen M3 = A11 * (B12 - B22)
        # A11=a00, B12=b01, B22=b11
        add_term([(a00, 1)], [(b01, 1), (b11, -1)], 
                 [(c01, 1), (c11, 1)])
        
        # Strassen M4 = A22 * (B21 - B11)
        # A22=a11, B21=b10, B11=b00
        add_term([(a11, 1)], [(b10, 1), (b00, -1)], 
                 [(c00, 1), (c10, 1)])
        
        # Strassen M5 = (A11 + A12) * B22
        # A11=a00, A12=a01, B22=b11
        add_term([(a00, 1), (a01, 1)], [(b11, 1)], 
                 [(c00, -1), (c01, 1)])
        
        # Strassen M6 = (A21 - A11) * (B11 + B12)
        # A21=a10, A11=a00, B11=b00, B12=b01
        add_term([(a10, 1), (a00, -1)], [(b00, 1), (b01, 1)], 
                 [(c11, 1)])
        
        # Strassen M7 = (A12 - A22) * (B21 + B22)
        # A12=a01, A22=a11, B21=b10, B22=b11
        add_term([(a01, 1), (a11, -1)], [(b10, 1), (b11, 1)], 
                 [(c00, 1)])

    # Verify the solution
    def verify(U, V, W):
        r = U.shape[1]
        T_check = np.einsum('ad,bd,cd->abc', U, V, W)
        T_exp = np.zeros((n * m, m * p, p * n), dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    T_exp[a_idx, b_idx, c_idx] = 1
        return np.array_equal(T_check, T_exp)

    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook if construction failed
        rank_schoolbook = n * m * p
        U_sb = np.zeros((n*m, rank_schoolbook), dtype=np.int8)
        V_sb = np.zeros((m*p, rank_schoolbook), dtype=np.int8)
        W_sb = np.zeros((p*n, rank_schoolbook), dtype=np.int8)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    U_sb[a_idx, col] = 1
                    V_sb[b_idx, col] = 1
                    W_sb[c_idx, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
