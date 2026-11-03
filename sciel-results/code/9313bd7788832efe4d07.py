"""Program for 3x4x5 matrix multiplication tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build schoolbook decomposition as fallback
    U_school = np.zeros((n * m, rank_schoolbook), dtype=np.float64)
    V_school = np.zeros((m * p, rank_schoolbook), dtype=np.float64)
    W_school = np.zeros((p * n, rank_schoolbook), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
                
    # Attempt block-diagonal Strassen decomposition
    # Split m=4 into 2+2
    # A = [A_L, A_R], B = [B_L; B_R]
    # C = A_L B_L + A_R B_R
    
    # We construct the decomposition for A_L B_L and A_R B_R separately and concatenate
    # A_L is 3x2, B_L is 2x5
    # A_R is 3x2, B_R is 2x5
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    def add_term(u, v, w, i_offset=0, j_offset=0, k_offset=0):
        """
        u: scalar (U[i, r])
        v: scalar (V[j, r])
        w: scalar (W[k, r])
        i_offset: row index in A
        j_offset: col index in A / row index in B
        k_offset: col index in B
        """
        # We need to add to global U, V, W
        # U has shape (n*m, rank). Index is i*m + j.
        # V has shape (m*p, rank). Index is j*p + k.
        # W has shape (p*n, rank). Index is k*n + i.
        # 
        # Wait, the decomposition is T_{i,j,k} = sum U_{i,r} V_{j,r} W_{k,r}
        # But U, V, W in the output are matrices.
        # U[i*m+j, r] = U_{i,r} ? No.
        # The tensor is T_{i,j,k}.
        # The output U is (n*m, rank).
        # The output V is (m*p, rank).
        # The output W is (p*n, rank).
        #
        # The identity is:
        # C_{i,k} = sum_j A_{i,j} B_{j,k} T_{i,j,k}
        # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} ?
        # No, the standard mapping is:
        # U[i*m+j, r] corresponds to coefficient of A_{i,j}
        # V[j*p+k, r] corresponds to coefficient of B_{j,k}
        # W[k*n+i, r] corresponds to coefficient of C_{k,i} (or C_{i,k} depending on convention)
        #
        # Task says: W[k*n+i, r]. So W index is k*n + i.
        # This implies W contributes to C_{i,k} with index k*n+i?
        # Usually C_{i,k} is stored at i*p+k or k*n+i.
        # If W is (p*n, rank), index k*n+i suggests row k, col i?
        # Or row index = k*n+i.
        # If we sum U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r],
        # does this give C_{i,k}?
        #
        # Let's check the schoolbook code:
        # U[i*m+j, col] = 1
        # V[j*p+k, col] = 1
        # W[k*n+i, col] = 1
        # Sum = 1.
        # This term is added to C_{i,k}.
        # So yes, the product U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # contributes to C_{i,k}.
        
        pass # Placeholder

    # We will build the matrices directly
    # Rank estimate: 32
    max_rank = 40
    U_new = np.zeros((n * m, max_rank), dtype=np.float64)
    V_new = np.zeros((m * p, max_rank), dtype=np.float64)
    W_new = np.zeros((p * n, max_rank), dtype=np.float64)
    
    rank = 0
    
    def add_decomposition(i_range, j_range, k_range, U_sub, V_sub, W_sub):
        nonlocal rank
        # U_sub: (len(i_range)*len(j_range), r)
        # V_sub: (len(j_range)*len(k_range), r)
        # W_sub: (len(k_range)*len(i_range), r)
        r = U_sub.shape[1]
        for r_idx in range(r):
            if rank >= max_rank:
                return
            # Copy columns
            # Map indices
            for ii in range(len(i_range)):
                for jj in range(len(j_range)):
                    global_i = i_range[ii]
                    global_j = j_range[jj]
                    U_new[global_i * m + global_j, rank] = U_sub[ii * len(j_range) + jj, r_idx]
            
            for jj in range(len(j_range)):
                for kk in range(len(k_range)):
                    global_j = j_range[jj]
                    global_k = k_range[kk]
                    V_new[global_j * p + global_k, rank] = V_sub[jj * len(k_range) + kk, r_idx]
                    
            for kk in range(len(k_range)):
                for ii in range(len(i_range)):
                    global_k = k_range[kk]
                    global_i = i_range[ii]
                    W_new[global_k * n + global_i, rank] = W_sub[kk * len(i_range) + ii, r_idx]
            rank += 1

    # Helper to get Strassen 2x2x2 decomposition
    def get_strassen_2x2x2():
        # Returns U, V, W of shape (4, 7), (4, 7), (4, 7)
        # Indices: i in 0..1, j in 0..1, k in 0..1
        # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        U_s = np.zeros((4, 7))
        V_s = np.zeros((4, 7))
        W_s = np.zeros((4, 7))
        
        # M1 = (a00+a11)(b00+b11)
        # U: a00, a11
        U_s[0, 0] = 1; U_s[3, 0] = 1 # i0j0, i1j1
        V_s[0, 0] = 1; V_s[3, 0] = 1 # j0k0, j1k1
        W_s[0, 0] = 1; W_s[3, 0] = 1 # k0i0, k1i1
        # C00 += M1, C11 += M1
        # W[k*n+i]
        # k=0, i=0 -> W[0]
        # k=1, i=1 -> W[3]
        
        # M2 = (a10+a11)b00
        U_s[2, 1] = 1; U_s[3, 1] = 1 # i1j0, i1j1
        V_s[0, 1] = 1 # j0k0
        W_s[1, 1] = 1 # k1i1? No.
        # C10 = M2 + M6. C00 = M1 - M2 + M3 - M4.
        # M2 contributes to C00 (-1) and C10 (+1)?
        # Strassen:
        # C00 = M1 - M2 + M3 - M4
        # C01 = M4 + M5
        # C10 = M2 + M6
        # C11 = M1 - M5 + M7
        
        # Let's fill W carefully.
        # W[k*2+i, r]
        
        # M1: C00(+1), C11(+1)
        W_s[0, 0] = 1  # k0 i0
        W_s[3, 0] = 1  # k1 i1
        
        # M2: C00(-1), C10(+1)
        W_s[0, 1] = -1 # k0 i0
        W_s[2, 1] = 1  # k1 i0
        
        # M3: C00(+1), C10(-1)
        W_s[0, 2] = 1  # k0 i0
        W_s[2, 2] = -1 # k1 i0
        
        # M4: C00(-1), C01(+1)
        W_s[0, 3] = -1 # k0 i0
        W_s[1, 3] = 1  # k0 i1
        
        # M5: C01(-1), C11(-1)
        W_s[1, 4] = -1 # k0 i1
        W_s[3, 4] = -1 # k1 i1
        
        # M6: C10(+1)
        W_s[2, 5] = 1  # k1 i0
        
        # M7: C11(+1)
        W_s[3, 6] = 1  # k1 i1
        
        # Now U and V
        # M1: (a00+a11) (b00+b11)
        U_s[0, 0] = 1; U_s[3, 0] = 1
        V_s[0, 0] = 1; V_s[3, 0] = 1
        
        # M2: (a10+a11) b00
        U_s[2, 1] = 1; U_s[3, 1] = 1
        V_s[0, 1] = 1
        
        # M3: a00 (b10-b11)
        U_s[0, 2] = 1
        V_s[2, 2] = 1; V_s[3, 2] = -1
        
        # M4: a11 (b01-b00)
        U_s[3, 3] = 1
        V_s[1, 3] = 1; V_s[0, 3] = -1
        
        # M5: (a00+a01) b11
        U_s[0, 4] = 1; U_s[1, 4] = 1
        V_s[3, 4] = 1
        
        # M6: (a10-a00) (b00+b01)
        U_s[2, 5] = 1; U_s[0, 5] = -1
        V_s[0, 5] = 1; V_s[1, 5] = 1
        
        # M7: (a01-a11) (b10+b11)
        U_s[1, 6] = 1; U_s[3, 6] = -1
        V_s[2, 6] = 1; V_s[3, 6] = 1
        
        return U_s, V_s, W_s

    # Helper for 1x2x5 (Rank 1)
    def get_1x2x5():
        # i in 0..0, j in 0..1, k in 0..4
        # U (2, 1), V (10, 1), W (5, 1)
        U_1 = np.ones((2, 1))
        V_1 = np.ones((10, 1))
        W_1 = np.ones((5, 1))
        return U_1, V_1, W_1

    # Helper for 2x2x1 (Rank 1)
    def get_2x2x1():
        # i in 0..1, j in 0..1, k in 0..0
        # U (4, 1), V (2, 1), W (2, 1)
        U_2 = np.ones((4, 1))
        V_2 = np.ones((2, 1))
        W_2 = np.ones((2, 1))
        return U_2, V_2, W_2

    # Process A_L B_L
    # i: 0..2, j: 0..1, k: 0..4
    # Split i: 0 (1x2x5) and 1..2 (2x2x5)
    
    # Part 1: i=0, j=0..1, k=0..4
    U1, V1, W1 = get_1x2x5()
    add_decomposition([0], [0, 1], [0, 1, 2, 3, 4], U1, V1, W1)
    
    # Part 2: i=1..2, j=0..1, k=0..4
    # Split k: 0..1 (2x2x2) and 2..4 (2x2x3)
    
    # Part 2a: i=1..2, j=0..1, k=0..1
    U2a, V2a, W2a = get_strassen_2x2x2()
    add_decomposition([1, 2], [0, 1], [0, 1], U2a, V2a, W2a)
    
    # Part 2b: i=1..2, j=0..1, k=2..4
    # Split k: 2..3 (2x2x2) and 4 (2x2x1)
    
    # Part 2b-i: i=1..2, j=0..1, k=2..3
    U2b_i, V2b_i, W2b_i = get_strassen_2x2x2()
    add_decomposition([1, 2], [0, 1], [2, 3], U2b_i, V2b_i, W2b_i)
    
    # Part 2b-ii: i=1..2, j=0..1, k=4
    U2b_ii, V2b_ii, W2b_ii = get_2x2x1()
    add_decomposition([1, 2], [0, 1], [4], U2b_ii, V2b_ii, W2b_ii)
    
    # Process A_R B_R
    # i: 0..2, j=2..3, k=0..4
    
    # Part 3: i=0, j=2..3, k=0..4
    U3, V3, W3 = get_1x2x5()
    add_decomposition([0], [2, 3], [0, 1, 2, 3, 4], U3, V3, W3)
    
    # Part 4: i=1..2, j=2..3, k=0..4
    
    # Part 4a: i=1..2, j=2..3, k=0..1
    U4a, V4a, W4a = get_strassen_2x2x2()
    add_decomposition([1, 2], [2, 3], [0, 1], U4a, V4a, W4a)
    
    # Part 4b: i=1..2, j=2..3, k=2..4
    
    # Part 4b-i: i=1..2, j=2..3, k=2..3
    U4b_i, V4b_i, W4b_i = get_strassen_2x2x2()
    add_decomposition([1, 2], [2, 3], [2, 3], U4b_i, V4b_i, W4b_i)
    
    # Part 4b-ii: i=1..2, j=2..3, k=4
    U4b_ii, V4b_ii, W4b_ii = get_2x2x1()
    add_decomposition([1, 2], [2, 3], [4], U4b_ii, V4b_ii, W4b_ii)

    # Trim to actual rank
    U_out = U_new[:, :rank]
    V_out = V_new[:, :rank]
    W_out = W_new[:, :rank]
    
    # Convert to list
    U_list = U_out.tolist()
    V_list = V_out.tolist()
    W_list = W_out.tolist()
    
    # Ensure integers
    for i in range(len(U_list)):
        for j in range(len(U_list[i])):
            U_list[i][j] = int(round(U_list[i][j]))
    for i in range(len(V_list)):
        for j in range(len(V_list[i])):
            V_list[i][j] = int(round(V_list[i][j]))
    for i in range(len(W_list)):
        for j in range(len(W_list[i])):
            W_list[i][j] = int(round(W_list[i][j]))
            
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
