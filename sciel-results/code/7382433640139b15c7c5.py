import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 115  # safety margin before 150s kill
    
    def schoolbook():
        n, m, p = 4, 4, 4
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
    
    # 2x2 Strassen decomposition
    # Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # T[i,j,k] = sum_c U[i*2+j,c] * V[j*2+k,c] * W[k*2+i,c] = 1
    
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    
    # Verify 2x2 Strassen
    valid2 = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = np.dot(U2[i*2+j], V2[j*2+k] * W2[k*2+i])
                if abs(s - 1) > 1e-9:
                    valid2 = False
                    break
            if not valid2:
                break
        if not valid2:
            break
    
    if not valid2:
        return schoolbook()
    
    # Build 4x4 via Kronecker with proper permutation
    # np.kron(A, B) where A is 4x7, B is 4x7 gives 16x49
    # Row R in kron = (row_A * 4 + row_B)
    # For U2: row_A = i_a*2+j_a (outer 2x2 index), row_B = i_b*2+j_b (inner 2x2 index)
    # So R = (i_a*2+j_a)*4 + (i_b*2+j_b) = i_a*8 + j_a*4 + i_b*2 + j_b
    # Target S = i*4 + j where i = i_a*2+i_b, j = j_a*2+j_b
    # S = (i_a*2+i_b)*4 + (j_a*2+j_b) = i_a*8 + i_b*4 + j_a*2 + j_b
    
    U4_kron = np.kron(U2, U2)
    V4_kron = np.kron(V2, V2)
    W4_kron = np.kron(W2, W2)
    
    perm_U = np.zeros(16, dtype=int)
    perm_V = np.zeros(16, dtype=int)
    perm_W = np.zeros(16, dtype=int)
    
    for S in range(16):
        i = S // 4
        j = S % 4
        i_a, i_b = i // 2, i % 2
        j_a, j_b = j // 2, j % 2
        # U: S = i*4+j, R = (i_a*2+j_a)*4 + (i_b*2+j_b)
        perm_U[S] = (i_a*2 + j_a) * 4 + (i_b*2 + j_b)
    
    for S in range(16):
        j = S // 4
        k = S % 4
        j_a, j_b = j // 2, j % 2
        k_a, k_b = k // 2, k % 2
        # V: S = j*4+k, R = (j_a*2+k_a)*4 + (j_b*2+k_b)
        perm_V[S] = (j_a*2 + k_a) * 4 + (j_b*2 + k_b)
    
    for S in range(16):
        k = S // 4
        i = S % 4
        k_a, k_b = k // 2, k % 2
        i_a, i_b = i // 2, i % 2
        # W: S = k*4+i, R = (k_a*2+i_a)*4 + (k_b*2+i_b)
        perm_W[S] = (k_a*2 + i_a) * 4 + (k_b*2 + i_b)
    
    U4 = U4_kron[perm_U, :]
    V4 = V4_kron[perm_V, :]
    W4 = W4_kron[perm_W, :]
    
    # Verify 4x4
    valid4 = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = np.dot(U4[i*4+j], V4[j*4+k] * W4[k*4+i])
                if abs(s - 1) > 1e-9:
                    valid4 = False
                    break
            if not valid4:
                break
        if not valid4:
            break
    
    if not valid4:
        return schoolbook()
    
    # Current best: rank 49
    best_rank = 49
    best_U, best_V, best_W = U4.copy(), V4.copy(), W4.copy()
    
    # Try to find lower rank by zeroing out columns
    # For each column, check if it's redundant (can be expressed as combination of others)
    # This is a heuristic approach
    
    if time.time() - start_time < DEADLINE:
        # Try to remove redundant columns
        # A column c is redundant if for all (i,j,k):
        # U[i*4+j,c] * V[j*4+k,c] * W[k*4+i,c] can be compensated by other columns
        # Simple heuristic: try removing columns one at a time and check validity
        
        def is_valid(U_mat, V_mat, W_mat):
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        s = np.dot(U_mat[i*4+j], V_mat[j*4+k] * W_mat[k*4+i])
                        if abs(s - 1) > 1e-6:
                            return False
            return True
        
        # Try greedy column elimination
        U_cur = U4.copy()
        V_cur = V4.copy()
        W_cur = W4.copy()
        cols_to_remove = []
        
        for c in range(49):
            if time.time() - start_time > DEADLINE - 5:
                break
            # Try removing column c
            mask = [x for x in range(49) if x != c and x not in cols_to_remove]
            if len(mask) < 49:
                U_test = U_cur[:, mask]
                V_test = V_cur[:, mask]
                W_test = W_cur[:, mask]
                if is_valid(U_test, V_test, W_test):
                    cols_to_remove.append(c)
        
        if cols_to_remove:
            mask = [x for x in range(49) if x not in cols_to_remove]
            U_final = U4[:, mask]
            V_final = V4[:, mask]
            W_final = W4[:, mask]
            if is_valid(U_final, V_final, W_final):
                best_rank = len(mask)
                best_U, best_V, best_W = U_final, V_final, W_final
        
        # Convert to list with integer/half-integer entries
        def to_list(mat):
            result = []
            for row in mat:
                r = []
                for val in row:
                    # Check if it's close to integer or half-integer
                    if abs(val - round(val)) < 1e-9:
                        r.append(int(round(val)))
                    elif abs(val - (round(val*2)/2)) < 1e-9:
                        r.append(round(val*2)/2)
                    else:
                        r.append(float(val))
                result.append(r)
            return result
        
        if time.time() - start_time > DEADLINE:
            return schoolbook()
        
        return {
            "U": to_list(best_U),
            "V": to_list(best_V),
            "W": to_list(best_W)
        }
    
    return schoolbook()
