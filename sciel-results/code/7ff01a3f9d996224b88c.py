import numpy as np
import time

def construct():
    n, m, p = 4, 5, 5
    deadline = time.time() + 145.0
    
    def verify(U, V, W):
        T_calc = np.einsum('ut,vt,wt->uvw', U, V, W, optimize=True)
        T_exp = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_exp[i*m+j, j*p+k, k*n+i] = 1.0
        return np.allclose(T_calc, T_exp, atol=1e-10)

    def to_lists(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Total rank estimate: 60
    total_rank = 60
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col = 0
    
    # Helper to add a sub-decomposition
    def add_sub(u_start, u_len, v_start, v_len, w_start, w_len, sub_U, sub_V, sub_W):
        nonlocal col
        rank_sub = sub_U.shape[1]
        for t in range(rank_sub):
            # Map local indices to global
            for i in range(u_len):
                if sub_U[i, t] != 0:
                    U[u_start + i, col + t] = sub_U[i, t]
            for j in range(v_len):
                if sub_V[j, t] != 0:
                    V[v_start + j, col + t] = sub_V[j, t]
            for k in range(w_len):
                if sub_W[k, t] != 0:
                    W[w_start + k, col + t] = sub_W[k, t]
        col += rank_sub

    # 1. A[0:2, 0:2] * B[0:2, :] -> 2x2x5
    # U indices: i in 0..1, j in 0..1 -> global i*5+j
    # V indices: j in 0..1, k in 0..4 -> global j*5+k
    # W indices: k in 0..4, i in 0..1 -> global k*4+i
    # Sub U: 2x2, Sub V: 2x5, Sub W: 5x2
    
    sub_U_1 = np.zeros((2, 10))
    sub_V_1 = np.zeros((10, 10)) # Wait, V is (m_sub * p, rank). m_sub=2, p=5. Size 10.
    sub_W_1 = np.zeros((5, 10)) # W is (p * n_sub, rank). p=5, n_sub=2. Size 10.
    
    # Schoolbook for 2x2x5
    # C[i,k] = sum_j A[i,j] B[j,k]
    t = 0
    for i in range(2):
        for j in range(2):
            for k in range(5):
                sub_U_1[i, t] = 1.0
                sub_V_1[j*5+k, t] = 1.0
                sub_W_1[k*2+i, t] = 1.0
                t += 1
    # Rank 10
    add_sub(0*5+0, 2, 0*5+0, 10, 0*4+0, 5, sub_U_1, sub_V_1, sub_W_1)
    
    # 2. A[2:4, 0:2] * B[0:2, :] -> 2x2x5
    # U indices: i in 2..3, j in 0..1
    # V indices: same
    # W indices: k in 0..4, i in 2..3
    sub_U_2 = np.zeros((2, 10))
    sub_V_2 = np.zeros((10, 10))
    sub_W_2 = np.zeros((5, 10))
    t = 0
    for i in range(2):
        for j in range(2):
            for k in range(5):
                sub_U_2[i, t] = 1.0
                sub_V_2[j*5+k, t] = 1.0
                sub_W_2[k*2+i, t] = 1.0
                t += 1
    add_sub(2*5+0, 2, 0*5+0, 10, 0*4+2, 5, sub_U_2, sub_V_2, sub_W_2)
    
    # 3. A[0:1, 2:5] * B[2:5, :] -> 1x3x5
    # U: i=0, j=2..4
    # V: j=2..4, k=0..4
    # W: k=0..4, i=0
    sub_U_3 = np.zeros((1, 15))
    sub_V_3 = np.zeros((15, 15))
    sub_W_3 = np.zeros((5, 15))
    t = 0
    for i in range(1):
        for j in range(3):
            for k in range(5):
                sub_U_3[i, t] = 1.0
                sub_V_3[j*5+k, t] = 1.0 # Local j is 0..2, global j is 2..4
                sub_W_3[k*1+i, t] = 1.0
                t += 1
    # Mapping for V: local j 0..2 corresponds to global j 2..4
    # So V index in sub is j_local * 5 + k. Global is (j_local+2)*5 + k.
    # My add_sub function assumes contiguous indices.
    # I must fix the mapping in add_sub or construct the global matrices directly.
    
    # Let's rewrite the addition logic to be explicit about index maps.
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    col = 0
    
    def add_block(u_map, v_map, w_map, sub_U, sub_V, sub_W):
        nonlocal col
        rank_sub = sub_U.shape[1]
        for t in range(rank_sub):
            for i_local, i_global in u_map:
                val = sub_U[i_local, t]
                if val != 0:
                    U[i_global, col + t] = val
            for j_local, j_global in v_map:
                val = sub_V[j_local, t]
                if val != 0:
                    V[j_global, col + t] = val
            for k_local, k_global in w_map:
                val = sub_W[k_local, t]
                if val != 0:
                    W[k_global, col + t] = val
        col += rank_sub

    # 1. A[0:2, 0:2] * B[0:2, :]
    u_map_1 = [(i, i*5 + j) for i in range(2) for j in range(2)] # i_local, j_local -> i*5+j
    # Wait, sub_U is 2x2. Indices (i, j).
    # My sub_U construction above used i, j, k loops.
    # sub_U[i, t] was set.
    # The map should be from sub_U row index to global U index.
    # sub_U row index is i. Global is i*5 + j? No, j is in V.
    # U index is i*m + j.
    # In sub-problem, j is 0..1.
    # So u_map should map sub_U row i to global U index i*5 + j?
    # No, sub_U depends on j.
    # In the construction of sub_U_1, I did:
    # for i, j, k: sub_U_1[i, t] = 1.
    # This means sub_U_1[i, t] is 1 for specific t corresponding to j.
    # So the index in sub_U is just i.
    # The global index is i*5 + j.
    # But j is determined by t.
    # This implies I cannot use a simple map if the value depends on j which is encoded in t.
    
    # Actually, the standard way is:
    # U_global[i*m+j, t_global] = sub_U[i, j, t_local]
    # V_global[j*p+k, t_global] = sub_V[j, k, t_local]
    # W_global[k*n+i, t_global] = sub_W[k, i, t_local]
    
    # I will construct the full 3D tensors for the sub-problems and then flatten.
    
    def get_sub_tensors_2x2x5():
        # Returns U (2,2,10), V (2,5,10), W (5,2,10)
        r = 10
        U_s = np.zeros((2, 2, r))
        V_s = np.zeros((2, 5, r))
        W_s = np.zeros((5, 2, r))
        t = 0
        for i in range(2):
            for j in range(2):
                for k in range(5):
                    U_s[i, j, t] = 1.0
                    V_s[j, k, t] = 1.0
                    W_s[k, i, t] = 1.0
                    t += 1
        return U_s, V_s, W_s

    def get_sub_tensors_1x3x5():
        r = 15
        U_s = np.zeros((1, 3, r))
        V_s = np.zeros((3, 5, r))
        W_s = np.zeros((5, 1, r))
        t = 0
        for i in range(1):
            for j in range(3):
                for k in range(5):
                    U_s[i, j, t] = 1.0
                    V_s[j, k, t] = 1.0
                    W_s[k, i, t] = 1.0
                    t += 1
        return U_s, V_s, W_s

    def get_sub_tensors_3x3x2():
        r = 18
        U_s = np.zeros((3, 3, r))
        V_s = np.zeros((3, 2, r))
        W_s = np.zeros((2, 3, r))
        t = 0
        for i in range(3):
            for j in range(3):
                for k in range(2):
                    U_s[i, j, t] = 1.0
                    V_s[j, k, t] = 1.0
                    W_s[k, i, t] = 1.0
                    t += 1
        return U_s, V_s, W_s

    def get_sub_tensors_3x3x3_laderman():
        # Laderman decomposition for 3x3x3
        # Rank 7
        r = 7
        U_s = np.zeros((3, 3, r))
        V_s = np.zeros((3, 3, r))
        W_s = np.zeros((3, 3, r))
        # Known Laderman coefficients
        # M1 = A11 B11
        # M2 = A11 B22
        # ...
        # I will use a known integer solution or fall back to schoolbook if unsure.
        # To be safe, use Schoolbook (27) if I don't have Laderman hardcoded.
        # But 7 is much better.
        # I'll implement a simple search for 3x3x3 rank 7 at startup if needed,
        # or use a known one.
        
        # Known Laderman:
        # P1 = A11 B11
        # P2 = A11 B22
        # P3 = A22 B11
        # P4 = A22 B22
        # P5 = (A11 + A22)(B11 + B22)
        # P6 = (A11 + A12 + A21 + A22)(B11 + B21 + B12 + B22) - P1 - P2 - P3 - P4 - P5 ... complex.
        
        # I will use Schoolbook for 3x3x3 to be safe and robust. Rank 27.
        # Total Rank becomes 10 + 10 + 15 + 18 + 27 = 80.
        # Still better than 100.
        
        r = 27
        U_s = np.zeros((3, 3, r))
        V_s = np.zeros((3, 3, r))
        W_s = np.zeros((3, 3, r))
        t = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U_s[i, j, t] = 1.0
                    V_s[j, k, t] = 1.0
                    W_s[k, i, t] = 1.0
                    t += 1
        return U_s, V_s, W_s

    def merge_sub(U_global, V_global, W_global, U_s, V_s, W_s, i_off, j_off, k_off):
        # i_off: offset for A rows
        # j_off: offset for A cols / B rows
        # k_off: offset for B cols
        r = U_s.shape[2]
        for i in range(U_s.shape[0]):
            for j in range(U_s.shape[1]):
                for t in range(r):
                    U_global[(i + i_off)*n + (j + j_off), t] += U_s[i, j, t]
        for j in range(V_s.shape[0]):
            for k in range(V_s.shape[1]):
                for t in range(r):
                    V_global[(j + j_off)*p + (k + k_off), t] += V_s[j, k, t]
        for k in range(W_s.shape[0]):
            for i in range(W_s.shape[1]):
                for t in range(r):
                    W_global[(k + k_off)*n + (i + i_off), t] += W_s[k, i, t]

    # Recalculate Total Rank with Schoolbook 3x3x3
    # 1. 2x2x5 (10)
    # 2. 2x2x5 (10)
    # 3. 1x3x5 (15)
    # 4. 3x3x2 (18)
    # 5. 3x3x3 (27)
    # Total 80
    
    U = np.zeros((n*m, 80))
    V = np.zeros((m*p, 80))
    W = np.zeros((p*n, 80))
    
    # Block 1: A[0:2, 0:2] B[0:2, :]
    U1, V1, W1 = get_sub_tensors_2x2x5()
    merge_sub(U, V, W, U1, V1, W1, 0, 0, 0)
    
    # Block 2: A[2:4, 0:2] B[0:2, :]
    U2, V2, W2 = get_sub_tensors_2x2x5()
    merge_sub(U, V, W, U2, V2, W2, 2, 0, 0)
    
    # Block 3: A[0:1, 2:5] B[2:5, :]
    U3, V3, W3 = get_sub_tensors_1x3x5()
    merge_sub(U, V, W, U3, V3, W3, 0, 2, 0)
    
    # Block 4: A[1:4, 2:5] B[2:5, 0:2] -> 3x3x2
    # A rows 1,2,3. A cols 2,3,4. B rows 2,3,4. B cols 0,1.
    U4, V4, W4 = get_sub_tensors_3x3x2()
    merge_sub(U, V, W, U4, V4, W4, 1, 2, 0)
    
    # Block 5: A[1:4, 2:5] B[2:5, 2:5] -> 3x3x3
    # A rows 1,2,3. A cols 2,3,4. B rows 2,3,4. B cols 2,3,4.
    U5, V5, W5 = get_sub_tensors_3x3x3_laderman()
    merge_sub(U, V, W, U5, V5, W5, 1, 2, 2)
    
    if verify(U, V, W):
        return to_lists(U, V, W)
    
    # Fallback
    U = np.zeros((n*m, n*m*p))
    V = np.zeros((m*p, n*m*p))
    W = np.zeros((p*n, n*m*p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return to_lists(U, V, W)

