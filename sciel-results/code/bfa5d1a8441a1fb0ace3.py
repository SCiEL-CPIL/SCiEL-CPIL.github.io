import time
import numpy as np

DIMENSIONS = [3, 3, 8]
DEADLINE_S = 110  # safety margin from 120s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    def time_left():
        return DEADLINE_S - (time.time() - start_time)
    
    def schoolbook():
        rank = n*m*p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def validate(U, V, W, tol=1e-10):
        """Check that sum_c U[:,c] V[:,c] W[:,c] == multiplication tensor."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        
        # Compute T_computed efficiently
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We can use einsum or loops.
        # Loops are safe and fast for small sizes.
        T_computed = np.zeros((n, m, p))
        for c in range(rank):
            u_c = U[:, c]
            v_c = V[:, c]
            w_c = W[:, c]
            
            u_mat = u_c.reshape(n, m)
            v_mat = v_c.reshape(m, p)
            w_mat = w_c.reshape(p, n)
            
            # T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
            # Vectorized update for this c
            for i in range(n):
                for j in range(m):
                    if abs(u_mat[i,j]) < 1e-15:
                        continue
                    # v_mat[j, :] is length p
                    # w_mat[:, i] is length p
                    # w_mat[k, i] * v_mat[j, k]
                    T_computed[i, j, :] += u_mat[i,j] * v_mat[j, :] * w_mat[:, i]
        
        # Expected T is all 1s
        T_expected = np.ones((n, m, p))
        return np.max(np.abs(T_computed - T_expected)) < tol

    def get_17_rank_block():
        """Returns U, V, W for a 3x3x2 block with rank 17."""
        rank = 17
        U = np.zeros((9, rank))
        V = np.zeros((6, rank))
        W = np.zeros((6, rank))
        
        col = 0
        # Strassen part (7 terms)
        # Standard Strassen for 2x2x2 embedded in 3x3x2
        # Indices for U: i (0..1), j (0..1) -> i*2+j
        # Indices for V: j (0..1), k (0..1) -> j*2+k
        # Indices for W: k (0..1), i (0..1) -> k*2+i
        # We map these to the 3x3x2 block indices:
        # U: i_local*3 + j_local
        # V: j_local*2 + k_local
        # W: k_local*3 + i_local
        
        # Strassen coefficients
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # U coeffs (i,j)
        U_coeffs = [
            {0: 1, 3: 1},       # M1: A11+A22 -> (0,0)+(1,1)
            {2: 1, 3: 1},       # M2: A21+A22 -> (1,0)+(1,1)
            {0: 1},             # M3: A11
            {3: 1},             # M4: A22
            {0: 1, 1: 1},       # M5: A11+A12
            {2: 1, 0: -1},      # M6: A21-A11
            {1: 1, 3: -1}       # M7: A12-A22
        ]
        # V coeffs (j,k)
        V_coeffs = [
            {0: 1, 3: 1},       # M1: B11+B22 -> (0,0)+(1,1)
            {0: 1},             # M2: B11
            {1: 1, 3: -1},      # M3: B21-B22 -> (1,0)-(1,1)
            {2: 1, 0: -1},      # M4: B12-B11 -> (0,1)-(0,0)
            {3: 1},             # M5: B22
            {0: 1, 1: 1},       # M6: B11+B12
            {2: 1, 3: 1}        # M7: B21+B22
        ]
        # W coeffs (k,i)
        W_coeffs = [
            {0: 1, 3: 1},       # M1: C11+C22 -> (0,0)+(1,1)
            {2: 1, 3: -1},      # M2: C21-C22 -> (1,0)-(1,1)
            {1: 1, 3: 1},       # M3: C12+C22 -> (0,1)+(1,1)
            {0: 1, 2: 1},       # M4: C11+C12 -> (0,0)+(0,1)
            {0: -1, 1: 1},      # M5: C12-C11 -> (0,1)-(0,0)
            {3: 1},             # M6: C21
            {0: 1}              # M7: C11
        ]
        
        for l in range(7):
            for li, coeff in U_coeffs[l].items():
                i_local = li // 2
                j_local = li % 2
                U[i_local*3 + j_local, col] += coeff
            for lj, coeff in V_coeffs[l].items():
                j_local = lj // 2
                k_local = lj % 2
                V[j_local*2 + k_local, col] += coeff
            for lc, coeff in W_coeffs[l].items():
                i_local = lc // 2
                k_local = lc % 2
                W[k_local*3 + i_local, col] += coeff
            col += 1
        
        # Schoolbook part (10 terms)
        # Cover the rest of the 3x3x2 multiplication
        # i in {0,1}, j=2, k in {0,1}
        for i in range(2):
            for k in range(2):
                U[i*3 + 2, col] = 1.0
                V[2*2 + k, col] = 1.0
                W[k*3 + i, col] = 1.0
                col += 1
        # i=2, j in {0,1}, k in {0,1}
        for j in range(2):
            for k in range(2):
                U[2*3 + j, col] = 1.0
                V[j*2 + k, col] = 1.0
                W[k*3 + 2, col] = 1.0
                col += 1
        # i=2, j=2, k in {0,1}
        for k in range(2):
            U[2*3 + 2, col] = 1.0
            V[2*2 + k, col] = 1.0
            W[k*3 + 2, col] = 1.0
            col += 1
            
        return U, V, W

    # Assemble full 3x3x8 decomposition
    U_block, V_block, W_block = get_17_rank_block()
    rank_block = 17
    rank_total = 4 * rank_block
    
    U = np.zeros((9, rank_total))
    V = np.zeros((24, rank_total)) # 3*8
    W = np.zeros((24, rank_total)) # 8*3
    
    for g in range(4):
        k0 = 2 * g
        col_start = g * rank_block
        col_end = (g + 1) * rank_block
        
        # U
        U[:, col_start:col_end] = U_block
        
        # V
        # V_block has shape (6, rank_block). Rows are j*2 + k_local.
        # Full V has shape (24, rank_total). Rows are j*8 + k_global.
        for c in range(rank_block):
            for j in range(3):
                for k_local in range(2):
                    k_global = k0 + k_local
                    V[j*8 + k_global, col_start + c] = V_block[j*2 + k_local, c]
                    
        # W
        # W_block has shape (6, rank_block). Rows are k_local*3 + i.
        # Full W has shape (24, rank_total). Rows are k_global*3 + i.
        for c in range(rank_block):
            for k_local in range(2):
                k_global = k0 + k_local
                for i in range(3):
                    W[k_global*3 + i, col_start + c] = W_block[k_local*3 + i, c]

    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if validate(U, V, W):
        return result
    else:
        return schoolbook()
