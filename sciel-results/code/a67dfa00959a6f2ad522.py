import numpy as np
import time
import random

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110.0  # Safety margin

    dim_u = n * m
    dim_v = m * p
    dim_w = p * n

    # Build the target tensor T
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def get_strassen_2x2x2():
        """
        Return Strassen's rank-7 decomposition for 2x2x2 matrix multiplication.
        U is 4x7, V is 4x7, W is 4x7.
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Term 1: M1 = (A11 + A22)(B11 + B22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # Term 2: M2 = (A21 + A22)B11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # Term 3: M3 = A11(B12 - B22)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # Term 4: M4 = A22(B21 - B11)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # Term 5: M5 = (A11 + A12)B22
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # Term 6: M6 = (A21 - A11)(B11 + B12)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # Term 7: M7 = (A12 - A22)(B21 + B22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    def get_standard_2x1x2():
        """
        Standard rank-4 decomposition for 2x1x2 matrix multiplication.
        """
        U = np.zeros((2, 4))
        V = np.zeros((2, 4))
        W = np.zeros((4, 4))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[0, 1] = 1; V[1, 1] = 1; W[1, 1] = 1
        U[1, 2] = 1; V[0, 2] = 1; W[2, 2] = 1
        U[1, 3] = 1; V[1, 3] = 1; W[3, 3] = 1
        
        return U, V, W

    def build_structured_decomposition():
        """
        Build structured decomposition: Strassen for 2x2x2 + standard for 2x1x2 per block.
        Total rank: 5 * (7 + 4) = 55
        """
        U = np.zeros((dim_u, 55))
        V = np.zeros((dim_v, 55))
        W = np.zeros((dim_w, 55))
        
        col_offset = 0
        
        for block in range(5):
            U_str, V_str, W_str = get_strassen_2x2x2()
            U_std, V_std, W_std = get_standard_2x1x2()
            
            # Strassen part (first 2 rows of B block)
            for r in range(7):
                # A: 2x2 submatrix -> full A indices 0,1,2,3
                for a_idx in range(4):
                    if U_str[a_idx, r] != 0:
                        U[a_idx, col_offset + r] = U_str[a_idx, r]
                
                # B: rows 0,1 of 3x2 block -> full B
                b_map = [2*block, 2*block+1, 10+2*block, 10+2*block+1]
                for b_idx in range(4):
                    if V_str[b_idx, r] != 0:
                        V[b_map[b_idx], col_offset + r] = V_str[b_idx, r]
                
                # C: 2x2 submatrix -> full C (index = k*2 + i)
                c_map = [4*block, 4*block+2, 4*block+1, 4*block+3]
                for c_idx in range(4):
                    if W_str[c_idx, r] != 0:
                        W[c_map[c_idx], col_offset + r] = W_str[c_idx, r]
            
            col_offset += 7
            
            # Standard part (last row of B block)
            for r in range(4):
                # A: 2x1 -> full A indices 0,2
                a_map = [0, 2]
                for a_idx in range(2):
                    if U_std[a_idx, r] != 0:
                        U[a_map[a_idx], col_offset + r] = U_std[a_idx, r]
                
                # B: 1x2 -> full B (row 2 of block)
                b_map = [20+2*block, 20+2*block+1]
                for b_idx in range(2):
                    if V_std[b_idx, r] != 0:
                        V[b_map[b_idx], col_offset + r] = V_std[b_idx, r]
                
                # C: 2x2 -> full C
                c_map = [4*block, 4*block+2, 4*block+1, 4*block+3]
                for c_idx in range(4):
                    if W_std[c_idx, r] != 0:
                        W[c_map[c_idx], col_offset + r] = W_std[c_idx, r]
            
            col_offset += 4
        
        return U, V, W, 55

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx, col] = 1.0
                    V[v_idx, col] = 1.0
                    W[w_idx, col] = 1.0
                    col += 1
        return U, V, W, rank

    # Try the structured decomposition first
    U_struct, V_struct, W_struct, rank_struct = build_structured_decomposition()
    
    # Verify the structured decomposition
    T_rec = np.einsum('ur,vr,wr->uvw', U_struct, V_struct, W_struct)
    if np.allclose(T_rec, T, atol=1e-9):
        # Check if it's already half-integer
        U_r = np.round(U_struct * 2) / 2.0
        V_r = np.round(V_struct * 2) / 2.0
        W_r = np.round(W_struct * 2) / 2.0
        T_rec_r = np.einsum('ur,vr,wr->uvw', U_r, V_r, W_r)
        if np.allclose(T_rec_r, T, atol=1e-9):
            U_list = U_r.tolist()
            V_list = V_r.tolist()
            W_list = W_r.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to Schoolbook
    U, V, W, rank = get_schoolbook()
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
