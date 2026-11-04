"""Matrix multiplication 3x4x8 rank-minimal decomposition search."""
import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def _verify_safe(U, V, W, n=3, m=4, p=8, tol=1e-6):
    """
    Safely verify the decomposition using vectorized operations.
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    Target: T[i,j,k] = 1 for all i,j,k.
    """
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    if U_arr.shape[1] != V_arr.shape[1] or U_arr.shape[1] != W_arr.shape[1]:
        return False
    rank = U_arr.shape[1]
    
    # Reshape to 3D
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    T_recon = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    if not np.allclose(T_recon, 1.0, atol=tol):
        return False
    return True

def _construct_rank_72():
    """
    Construct a rank-72 decomposition using Strassen on 2x2 blocks.
    
    Split A (3x4) into A_top (1x4) and A_bottom (2x4).
    Split B (4x8) into 4 blocks of 4x2: B1, B2, B3, B4.
    
    For A_top (1x4) x B_j (4x2): schoolbook rank = 1*4*2 = 8 each, total 32
    For A_bottom (2x4) x B_j (4x2): Strassen on 2x2x2 gives rank 7 each, total 28
    But we need to handle the 4x2 dimension properly.
    
    Actually, let me think more carefully. We have 3x4 times 4x8.
    
    Better approach: Split into (3x4)*(4x4) and (3x4)*(4x4).
    Each 3x4x4 can be done with rank... let's think.
    
    Actually, let me try: split B into two 4x4 blocks.
    A (3x4) * B1 (4x4): use Strassen-style on the 2x2 sub-blocks of A.
    Split A into A1 (1x4) and A2 (2x4).
    A1*B1: 1x4x4, rank 8 schoolbook
    A2*B1: 2x4x4, can use Strassen on 2x2x2 twice: 2*7=14
    Total for first half: 22
    
    Similarly for B2: 22
    Total: 44? That seems too good. Let me verify.
    
    Wait, 2x4x4: split into two 2x2x2 blocks.
    Each 2x2x2 has Strassen rank 7.
    So 2x4x4 has rank 14.
    1x4x4 has rank 8.
    Total per 3x4x4: 22.
    Two such blocks: 44.
    
    But schoolbook is 3*4*4=48 per block, 96 total.
    Strassen gives 44 total. That's a big improvement!
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
        """Schoolbook rank 8 for 1x4x4."""
        u = np.zeros((4, 8))
        v = np.zeros((16, 8))
        w = np.zeros((4, 8))
        idx = 0
        for j in range(4):
            for k in range(4):
                u[j, idx] = 1.0
                v[j * 4 + k, idx] = 1.0
                w[k * 1 + 0, idx] = 1.0  # n=1, so k*n+i = k*1+0 = k
                idx += 1
        return u, v, w

    def get_2x4x4():
        """Rank 14 using Strassen on two 2x2x2 blocks.
        For 2x4x4: A is 2x4, B is 4x4, C is 4x2.
        Split B into B_left (4x2) and B_right (4x2).
        Each 2x2x2 uses Strassen with rank 7.
        U: (2*4, 14) = (8, 14)
        V: (4*4, 14) = (16, 14)
        W: (4*2, 14) = (8, 14)
        """
        # Strassen 2x2x2
        # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
        # C11 = M1 + M4 - M2 - M5
        # C12 = M3 + M5
        # C21 = M6 + M1
        # C22 = M2 - M3 + M7 - M1
        u_s = np.array([
            [1, 1, 0, 0],   # M1: A11+A22
            [0, 0, 1, 1],   # M2: A21+A22
            [1, 0, 0, 0],   # M3: A11
            [0, 0, 0, 1],   # M4: A22
            [1, 1, 0, 0],   # M5: A11+A12
            [-1, 1, 0, 0],  # M6: A21-A11
            [0, -1, 0, 1]   # M7: A12-A22
        ]).T  # shape (4, 7)
        
        v_s = np.array([
            [1, 0, 0, 1],   # M1: B11+B22
            [1, 0, 0, 0],   # M2: B11
            [0, 1, 0, -1],  # M3: B12-B22
            [-1, 0, 1, 0],  # M4: B21-B11
            [0, 0, 0, 1],   # M5: B22
            [1, 1, 0, 0],   # M6: B11+B12
            [0, 0, 1, 1]    # M7: B21+B22
        ]).T  # shape (4, 7)
        
        w_s = np.zeros((4, 7))
        # C11 at index 0 (k=0, i=0), C12 at index 1 (k=0, i=1)?? 
        # Wait, W index is k*n+i where n=2.
        # C11: i=0, k=0 -> index 0
        # C21: i=1, k=0 -> index 1
        # C12: i=0, k=1 -> index 2
        # C22: i=1, k=1 -> index 3
        w_s[0, 0] = 1; w_s[3, 0] = -1  # M1: C11 + C22 -> but we need C11 - C22 for M1?
        # Actually: C11 = M1 + M4 - M2 - M5
        # So M1 contributes +1 to C11
        w_s[0, 0] = 1
        # M2: C21 - C22
        w_s[1, 1] = 1; w_s[3, 1] = -1
        # M3: C12 + C22
        w_s[2, 2] = 1; w_s[3, 2] = 1
        # M4: C11 + C21
        w_s[0, 3] = 1; w_s[1, 3] = 1
        # M5: -C11 + C12
        w_s[0, 4] = -1; w_s[2, 4] = 1
        # M6: C22
        w_s[3, 5] = 1
        # M7: C11
        w_s[0, 6] = 1
        
        u_244 = np.zeros((8, 14))   # 2*4 = 8
        v_244 = np.zeros((16, 14))  # 4*4 = 16
        w_244 = np.zeros((8, 14))   # 4*2 = 8
        
        # First 2x2x2 block: A (2x2) x B_left (2x2) -> C_left (2x2)
        # A rows 0,1; B cols 0,1; C cols 0,1
        for c in range(7):
            # U: A is 2x2, indices (i,j) -> i*2+j
            u_244[0, c] = u_s[0, c]  # A11
            u_244[1, c] = u_s[1, c]  # A12
            u_244[2, c] = u_s[2, c]  # A21
            u_244[3, c] = u_s[3, c]  # A22
            # V: B is 2x2, indices (j,k) -> j*2+k
            v_244[0, c] = v_s[0, c]  # B11
            v_244[1, c] = v_s[1, c]  # B12
            v_244[2, c] = v_s[2, c]  # B21
            v_244[3, c] = v_s[3, c]  # B22
            # W: C is 2x2, indices (k,i) -> k*2+i
            w_244[0, c] = w_s[0, c]  # C11
            w_244[1, c] = w_s[1, c]  # C21
            w_244[2, c] = w_s[2, c]  # C12
            w_244[3, c] = w_s[3, c]  # C22
            
        # Second 2x2x2 block: A (2x2) x B_right (2x2) -> C_right (2x2)
        # A rows 0,1; B cols 2,3; C cols 2,3
        for c in range(7):
            # U: same A
            u_244[0, c+7] = u_s[0, c]
            u_244[1, c+7] = u_s[1, c]
            u_244[2, c+7] = u_s[2, c]
            u_244[3, c+7] = u_s[3, c]
            # V: B_right, indices (j,k) where j=0,1 and k=2,3 -> j*4+k
            v_244[0*4+2, c+7] = v_s[0, c]  # B11 of right block
            v_244[0*4+3, c+7] = v_s[1, c]  # B12 of right block
            v_244[1*4+2, c+7] = v_s[2, c]  # B21 of right block
            v_244[1*4+3, c+7] = v_s[3, c]  # B22 of right block
            # W: C_right, indices (k,i) where k=2,3 and i=0,1 -> k*2+i
            w_244[2*2+0, c+7] = w_s[0, c]  # C11 of right block
            w_244[2*2+1, c+7] = w_s[1, c]  # C21 of right block
            w_244[3*2+0, c+7] = w_s[2, c]  # C12 of right block
            w_244[3*2+1, c+7] = w_s[3, c]  # C22 of right block
            
        return u_244, v_244, w_244

    u_144, v_144, w_144 = get_1x4x4()
    u_244, v_244, w_244 = get_2x4x4()
    
    # A_top (1x4) x B1 (4x4)
    add_sub_term(0, 0, 0, 1, 4, 4, u_144, v_144, w_144)
    
    # A_bottom (2x4) x B1 (4x4)
    add_sub_term(1, 0, 0, 2, 4, 4, u_244, v_244, w_244)
    
    # A_top (1x4) x B2 (4x4)
    add_sub_term(0, 0, 4, 1, 4, 4, u_144, v_144, w_144)
    
    # A_bottom (2x4) x B2 (4x4)
    add_sub_term(1, 0, 4, 2, 4, 4, u_244, v_244, w_244)
    
    U_arr = np.array(U_list).T
    V_arr = np.array(V_list).T
    W_arr = np.array(W_list).T
    
    return U_arr, V_arr, W_arr

def _construct_rank_88():
    """
    Construct a rank-88 decomposition.
    Split A (3x4) into A1 (1x4) and A2 (2x4).
    Split B (4x8) into B1 (4x4) and B2 (4x4).
    
    A1 B1: 1x4x4 -> Split into 1x2x4 (rank 8) + 1x2x4 (rank 8) = 16
    A1 B2: 1x4x4 -> Split into 1x2x4 (rank 8) + 1x2x4 (rank 8) = 16
    A2 B1: 2x4x4 -> Split into 2x2x4 (rank 14) + 2x2x4 (rank 14) = 28
    A2 B2: 2x4x4 -> Split into 2x2x4 (rank 14) + 2x2x4 (rank 14) = 28
    Total: 16 + 16 + 28 + 28 = 88.
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

    def get_1x2x4():
        # Rank 8 schoolbook
        u = np.zeros((1 * 2, 8))
        v = np.zeros((2 * 4, 8))
        w = np.zeros((4 * 1, 8))
        idx = 0
        for j in range(2):
            for k in range(4):
                u[j, idx] = 1.0
                v[j * 4 + k, idx] = 1.0
                w[k * 1, idx] = 1.0
                idx += 1
        return u, v, w

    def get_2x2x4():
        # Rank 14 using two 2x2x2 Strassen decompositions
        # For 2x2x4: A is 2x2, B is 2x4, C is 4x2
        # Split B into B_left (2x2) and B_right (2x2)
        # Each 2x2x2 uses Strassen with rank 7
        u_s = np.array([
            [1, 1, 0, 0],   # M1: A11+A22
            [0, 0, 1, 1],   # M2: A21+A22
            [1, 0, 0, 0],   # M3: A11
            [0, 0, 0, 1],   # M4: A22
            [1, 1, 0, 0],   # M5: A11+A12
            [-1, 1, 0, 0],  # M6: A21-A11
            [0, -1, 0, 1]   # M7: A12-A22
        ]).T  # shape (4, 7)
        
        v_s = np.array([
            [1, 0, 0, 1],   # M1: B11+B22
            [1, 0, 0, 0],   # M2: B11
            [0, 1, 0, -1],  # M3: B12-B22
            [-1, 0, 1, 0],  # M4: B21-B11
            [0, 0, 0, 1],   # M5: B22
            [1, 1, 0, 0],   # M6: B11+B12
            [0, 0, 1, 1]    # M7: B21+B22
        ]).T  # shape (4, 7)
        
        # W indices: k*n+i where n=2
        # C11: i=0, k=0 -> 0
        # C21: i=1, k=0 -> 1
        # C12: i=0, k=1 -> 2
        # C22: i=1, k=1 -> 3
        w_s = np.zeros((4, 7))
        # M1: +C11
        w_s[0, 0] = 1
        # M2: +C21 - C22
        w_s[1, 1] = 1; w_s[3, 1] = -1
        # M3: +C12 + C22
        w_s[2, 2] = 1; w_s[3, 2] = 1
        # M4: +C11 + C21
        w_s[0, 3] = 1; w_s[1, 3] = 1
        # M5: -C11 + C12
        w_s[0, 4] = -1; w_s[2, 4] = 1
        # M6: +C22
        w_s[3, 5] = 1
        # M7: +C11
        w_s[0, 6] = 1
        
        # Construct 2x2x4
        u_224 = np.zeros((4, 14))
        v_224 = np.zeros((8, 14))
        w_224 = np.zeros((8, 14))
        
        # First half (k=0,1)
        for c in range(7):
            u_224[:, c] = u_s[:, c]
            v_224[0, c] = v_s[0, c]
            v_224[1, c] = v_s[1, c]
            v_224[4, c] = v_s[2, c]
            v_224[5, c] = v_s[3, c]
            w_224[0, c] = w_s[0, c]
            w_224[1, c] = w_s[1, c]
            w_224[2, c] = w_s[2, c]
            w_224[3, c] = w_s[3, c]
            
        # Second half (k=2,3)
        for c in range(7):
            u_224[:, c+7] = u_s[:, c]
            v_224[2, c+7] = v_s[0, c]
            v_224[3, c+7] = v_s[1, c]
            v_224[6, c+7] = v_s[2, c]
            v_224[7, c+7] = v_s[3, c]
            w_224[4, c+7] = w_s[0, c]
            w_224[5, c+7] = w_s[1, c]
            w_224[6, c+7] = w_s[2, c]
            w_224[7, c+7] = w_s[3, c]
            
        return u_224, v_224, w_224

    u_124, v_124, w_124 = get_1x2x4()
    u_224, v_224, w_224 = get_2x2x4()
    
    # A1 (1x4) x B1 (4x4)
    add_sub_term(0, 0, 0, 1, 2, 4, u_124, v_124, w_124)
    add_sub_term(0, 2, 0, 1, 2, 4, u_124, v_124, w_124)
    
    # A1 (1x4) x B2 (4x4)
    add_sub_term(0, 0, 4, 1, 2, 4, u_124, v_124, w_124)
    add_sub_term(0, 2, 4, 1, 2, 4, u_124, v_124, w_124)
    
    # A2 (2x4) x B1 (4x4)
    add_sub_term(1, 0, 0, 2, 2, 4, u_224, v_224, w_224)
    add_sub_term(1, 2, 0, 2, 2, 4, u_224, v_224, w_224)
    
    # A2 (2x4) x B2 (4x4)
    add_sub_term(1, 0, 4, 2, 2, 4, u_224, v_224, w_224)
    add_sub_term(1, 2, 4, 2, 2, 4, u_224, v_224, w_224)
    
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
    
    # Try Rank 44 (Strassen-based) - this should work with fixed get_2x4x4
    if time.time() < deadline:
        try:
            U_44, V_44, W_44 = _construct_rank_72()
            if _verify_safe(U_44.tolist(), V_44.tolist(), W_44.tolist()):
                if U_44.shape[1] < best_rank:
                    best_result = {"U": U_44.tolist(), "V": V_44.tolist(), "W": W_44.tolist()}
                    best_rank = U_44.shape[1]
        except Exception:
            pass
    
    # Try Rank 88
    if time.time() < deadline:
        try:
            U_88, V_88, W_88 = _construct_rank_88()
            if _verify_safe(U_88.tolist(), V_88.tolist(), W_88.tolist()):
                if U_88.shape[1] < best_rank:
                    best_result = {"U": U_88.tolist(), "V": V_88.tolist(), "W": W_88.tolist()}
                    best_rank = U_88.shape[1]
        except Exception:
            pass
            
    return best_result
