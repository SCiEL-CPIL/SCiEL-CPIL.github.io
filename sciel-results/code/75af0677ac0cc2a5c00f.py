import numpy as np
import time

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 32: 4 Strassen blocks (rank 7 each) + 1 Schoolbook block (rank 4)
    rank = 32
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col_idx = 0
    
    # Strassen blocks for column pairs
    # Pairs: (0,1), (2,3), (4,5), (6,7)
    for k1 in range(0, 8, 2):
        k2 = k1 + 1
        
        # Local indices for U (4 entries)
        # u = i*m + j
        # Local V indices (4 entries)
        # v = j*p + k
        # But we map local k' in {0,1} to global k1, k2
        # Local W indices (4 entries)
        # w = k*n + i
        # Map local k' to global k1, k2
        
        # Define U columns for Strassen (7 products)
        # A = [[a, b], [c, d]]
        # u0=a, u1=b, u2=c, u3=d
        U_local = np.zeros((4, 7), dtype=float)
        U_local[0, 0] = 1; U_local[3, 0] = 1  # a+d
        U_local[2, 1] = 1; U_local[3, 1] = 1  # c+d
        U_local[0, 2] = 1                     # a
        U_local[3, 3] = 1                     # d
        U_local[0, 4] = 1; U_local[1, 4] = 1  # a+b
        U_local[2, 5] = 1; U_local[3, 5] = 1  # c+d
        U_local[1, 6] = 1; U_local[3, 6] = 1  # b+d
        
        # Define V columns for Strassen (7 products)
        # B = [[x, y], [z, w]]
        # x = B[0, k1] -> global index k1
        # y = B[0, k2] -> global index k2
        # z = B[1, k1] -> global index p + k1
        # w = B[1, k2] -> global index p + k2
        V_local = np.zeros((4, 7), dtype=float)
        # v indices 0,1,2,3 correspond to x,y,z,w
        V_local[0, 0] = 1; V_local[3, 0] = 1  # x+w
        V_local[3, 1] = 1                     # w
        V_local[1, 2] = 1; V_local[3, 2] = -1 # y-w
        V_local[0, 3] = 1; V_local[1, 3] = -1 # x-y
        V_local[0, 4] = 1                     # x
        V_local[2, 5] = 1; V_local[0, 5] = -1 # z-x
        V_local[1, 6] = 1; V_local[2, 6] = -1 # y-z
        
        # Map local V indices to global V indices
        # local 0 -> global k1
        # local 1 -> global k2
        # local 2 -> global p + k1
        # local 3 -> global p + k2
        v_map = [k1, k2, p + k1, p + k2]
        
        # Define W columns (to be solved)
        # Local W indices 0,1,2,3 correspond to:
        # w0 = C[0, k1] -> global 2*k1
        # w1 = C[1, k1] -> global 2*k1 + 1
        # w2 = C[0, k2] -> global 2*k2
        # w3 = C[1, k2] -> global 2*k2 + 1
        w_map = [2*k1, 2*k1+1, 2*k2, 2*k2+1]
        
        # Construct the tensor T_local (4x4x4)
        # T[u, v, w] = 1 if u=i*2+j, v=j*2+k', w=k'*2+i
        T_local = np.zeros((4, 4, 4), dtype=float)
        for i in range(2):
            for j in range(2):
                for kp in range(2):
                    u = i * 2 + j
                    v = j * 2 + kp
                    w = kp * 2 + i
                    T_local[u, v, w] = 1
        
        # Solve for W_local (4x7)
        # Equation: sum_c U[u,c] V[v,c] W[:,c] = T_local[u,v,:]
        # We have 8 equations (for non-zero T entries) and 28 unknowns.
        # We can set up a linear system A x = b
        # x is W_local.flatten()
        
        A_mat = np.zeros((8, 28), dtype=float)
        b_vec = np.zeros(8, dtype=float)
        
        eq_idx = 0
        for i in range(2):
            for j in range(2):
                for kp in range(2):
                    u = i * 2 + j
                    v = j * 2 + kp
                    w = kp * 2 + i
                    
                    # Coefficients for each (w', c)
                    for c in range(7):
                        coeff = U_local[u, c] * V_local[v, c]
                        if coeff != 0:
                            # x index is w*7 + c
                            A_mat[eq_idx, w * 7 + c] = coeff
                    
                    b_vec[eq_idx] = 1
                    eq_idx += 1
        
        # Solve using least squares
        W_local_flat, residuals, rank_ls, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        W_local = W_local_flat.reshape(4, 7)
        
        # Assign to global U, V, W
        for c in range(7):
            U[:, col_idx + c] = U_local[:, c]
            
            # Map V
            for lv in range(4):
                if V_local[lv, c] != 0:
                    gv = v_map[lv]
                    V[gv, col_idx + c] = V_local[lv, c]
            
            # Map W
            for lw in range(4):
                if W_local[lw, c] != 0:
                    gw = w_map[lw]
                    W[gw, col_idx + c] = W_local[lw, c]
        
        col_idx += 7
    
    # Schoolbook block for column 8
    k1 = 8
    # 4 terms
    for i in range(2):
        for j in range(2):
            u = i * m + j
            v = j * p + k1
            w = k1 * n + i
            
            U[u, col_idx] = 1
            V[v, col_idx] = 1
            W[w, col_idx] = 1
            col_idx += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
