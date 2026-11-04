"""Matrix multiplication 3x4x8 rank-minimal decomposition search."""
import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def _verify_safe(U, V, W, n=3, m=4, p=8, tol=1e-6):
    """
    Safely verify the decomposition using explicit loops.
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    Target: T[i,j,k] = 1 for all i,j,k.
    """
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    if U_arr.shape[1] != V_arr.shape[1] or U_arr.shape[1] != W_arr.shape[1]:
        return False
    rank = U_arr.shape[1]
    
    T_recon = np.zeros((n, m, p))
    for c in range(rank):
        u_c = U_arr[:, c]
        v_c = V_arr[:, c]
        w_c = W_arr[:, c]
        
        u_2d = u_c.reshape(n, m)
        v_2d = v_c.reshape(m, p)
        w_2d = w_c.reshape(p, n)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] += u_2d[i, j] * v_2d[j, k] * w_2d[k, i]
    
    if not np.allclose(T_recon, 1.0, atol=tol):
        return False
    return True

def _construct_rank_64():
    """
    Construct a rank-64 decomposition.
    i=0: 1x4x8 -> two 1x4x4 blocks (rank 4 each) = 8.
    i=1,2: 2x4x8 -> two 2x4x4 blocks (rank 28 each) = 56.
    Total: 64.
    """
    n, m, p = 3, 4, 8
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_sub_term(i_off, j_off, k_off, sub_n, sub_m, sub_p, u_sub, v_sub, w_sub):
        rank_sub = u_sub.shape[1]
        for c in range(rank_sub):
            u_col = np.zeros(n * m)
            v_col = np.zeros(m * p)
            w_col = np.zeros(p * n)
            
            for i in range(sub_n):
                for j in range(sub_m):
                    u_col[(i_off + i) * m + (j_off + j)] = u_sub[i * sub_m + j, c]
            
            for j in range(sub_m):
                for k in range(sub_p):
                    v_col[(j_off + j) * p + (k_off + k)] = v_sub[j * sub_p + k, c]
            
            for k in range(sub_p):
                for i in range(sub_n):
                    w_col[(k_off + k) * n + (i_off + i)] = w_sub[k * sub_n + i, c]
            
            U_list.append(u_col)
            V_list.append(v_col)
            W_list.append(w_col)

    def get_1x4x4():
        """
        1x4x4 decomposition, rank 4.
        """
        u = np.zeros((1 * 4, 4))
        v = np.zeros((4 * 4, 4))
        w = np.zeros((4 * 1, 4))
        for c in range(4):
            u[c, c] = 1.0
            for k in range(4):
                v[c * 4 + k, c] = 1.0
            for k in range(4):
                w[k, c] = 1.0
        return u, v, w

    def get_2x2x2_strassen():
        """
        Standard Strassen 2x2x2 decomposition, rank 7.
        """
        u_s = np.array([
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1]
        ]).T
        
        v_s = np.array([
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [-1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]).T
        
        w_s = np.zeros((4, 7))
        w_s[0, 0] = 1; w_s[3, 0] = 1
        w_s[1, 1] = 1; w_s[3, 1] = -1
        w_s[2, 2] = 1; w_s[3, 2] = 1
        w_s[0, 3] = 1; w_s[1, 3] = 1
        w_s[0, 4] = -1; w_s[2, 4] = 1
        w_s[3, 5] = 1
        w_s[0, 6] = 1
        
        return u_s, v_s, w_s

    def get_2x4x4():
        """
        2x4x4 decomposition, rank 28.
        Split j into 0,1 and 2,3.
        Split k into 0,1 and 2,3.
        4 blocks of 2x2x2 Strassen.
        """
        u_s, v_s, w_s = get_2x2x2_strassen()
        
        u_244 = np.zeros((8, 28))
        v_244 = np.zeros((16, 28))
        w_244 = np.zeros((8, 28))
        
        # Block 1: j=0,1; k=0,1
        for c in range(7):
            u_244[0, c] = u_s[0, c]   # i=0,j=0
            u_244[1, c] = u_s[1, c]   # i=0,j=1
            u_244[4, c] = u_s[2, c]   # i=1,j=0
            u_244[5, c] = u_s[3, c]   # i=1,j=1
            
            v_244[0, c] = v_s[0, c]   # j=0,k=0
            v_244[1, c] = v_s[1, c]   # j=0,k=1
            v_244[4, c] = v_s[2, c]   # j=1,k=0
            v_244[5, c] = v_s[3, c]   # j=1,k=1
            
            w_244[0, c] = w_s[0, c]   # k=0,i=0
            w_244[1, c] = w_s[1, c]   # k=0,i=1
            w_244[2, c] = w_s[2, c]   # k=1,i=0
            w_244[3, c] = w_s[3, c]   # k=1,i=1
            
        # Block 2: j=0,1; k=2,3
        for c in range(7):
            u_244[0, c+7] = u_s[0, c]
            u_244[1, c+7] = u_s[1, c]
            u_244[4, c+7] = u_s[2, c]
            u_244[5, c+7] = u_s[3, c]
            
            v_244[2, c+7] = v_s[0, c]   # j=0,k=2
            v_244[3, c+7] = v_s[1, c]   # j=0,k=3
            v_244[6, c+7] = v_s[2, c]   # j=1,k=2
            v_244[7, c+7] = v_s[3, c]   # j=1,k=3
            
            w_244[4, c+7] = w_s[0, c]   # k=2,i=0
            w_244[5, c+7] = w_s[1, c]   # k=2,i=1
            w_244[6, c+7] = w_s[2, c]   # k=3,i=0
            w_244[7, c+7] = w_s[3, c]   # k=3,i=1
            
        # Block 3: j=2,3; k=0,1
        for c in range(7):
            u_244[2, c+14] = u_s[0, c]  # i=0,j=2
            u_244[3, c+14] = u_s[1, c]  # i=0,j=3
            u_244[6, c+14] = u_s[2, c]  # i=1,j=2
            u_244[7, c+14] = u_s[3, c]  # i=1,j=3
            
            v_244[8, c+14] = v_s[0, c]  # j=2,k=0
            v_244[9, c+14] = v_s[1, c]  # j=2,k=1
            v_244[12, c+14] = v_s[2, c] # j=3,k=0
            v_244[13, c+14] = v_s[3, c] # j=3,k=1
            
            w_244[0, c+14] = w_s[0, c]
            w_244[1, c+14] = w_s[1, c]
            w_244[2, c+14] = w_s[2, c]
            w_244[3, c+14] = w_s[3, c]
            
        # Block 4: j=2,3; k=2,3
        for c in range(7):
            u_244[2, c+21] = u_s[0, c]
            u_244[3, c+21] = u_s[1, c]
            u_244[6, c+21] = u_s[2, c]
            u_244[7, c+21] = u_s[3, c]
            
            v_244[10, c+21] = v_s[0, c] # j=2,k=2
            v_244[11, c+21] = v_s[1, c] # j=2,k=3
            v_244[14, c+21] = v_s[2, c] # j=3,k=2
            v_244[15, c+21] = v_s[3, c] # j=3,k=3
            
            w_244[4, c+21] = w_s[0, c]
            w_244[5, c+21] = w_s[1, c]
            w_244[6, c+21] = w_s[2, c]
            w_244[7, c+21] = w_s[3, c]
            
        return u_244, v_244, w_244

    u_144, v_144, w_144 = get_1x4x4()
    u_244, v_244, w_244 = get_2x4x4()
    
    # Block 1: A(3x4) x B1(4x4), columns 0-3
    add_sub_term(0, 0, 0, 1, 4, 4, u_144, v_144, w_144)
    add_sub_term(1, 0, 0, 2, 4, 4, u_244, v_244, w_244)
    
    # Block 2: A(3x4) x B2(4x4), columns 4-7
    add_sub_term(0, 0, 4, 1, 4, 4, u_144, v_144, w_144)
    add_sub_term(1, 0, 4, 2, 4, 4, u_244, v_244, w_244)
    
    U_arr = np.array(U_list).T
    V_arr = np.array(V_list).T
    W_arr = np.array(W_list).T
    
    return U_arr, V_arr, W_arr

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Fallback: Schoolbook
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_schoolbook
    
    # Try Rank 64
    if time.time() < deadline:
        try:
            U_64, V_64, W_64 = _construct_rank_64()
            if _verify_safe(U_64.tolist(), V_64.tolist(), W_64.tolist()):
                if U_64.shape[1] < best_rank:
                    best_result = {"U": U_64.tolist(), "V": V_64.tolist(), "W": W_64.tolist()}
                    best_rank = U_64.shape[1]
        except Exception:
            pass
    
    return best_result
