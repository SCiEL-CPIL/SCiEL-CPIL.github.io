"""Matrix multiplication 3x4 by 4x6: Strassen-based rank-66 with aggressive rank reduction."""
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 145.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    n, m, p = 3, 4, 6
    
    # Build the known rank-66 solution
    rank = 66
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_schoolbook(i, j, k):
        nonlocal col
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        U[u_idx, col] = 1
        V[v_idx, col] = 1
        W[w_idx, col] = 1
        col += 1

    def add_strassen(i_base, j_base, k_base):
        nonlocal col
        a_idx = i_base * m + j_base
        b_idx = i_base * m + (j_base + 1)
        c_idx = (i_base + 1) * m + j_base
        d_idx = (i_base + 1) * m + (j_base + 1)
        e_idx = j_base * p + k_base
        f_idx = j_base * p + (k_base + 1)
        g_idx = (j_base + 1) * p + k_base
        h_idx = (j_base + 1) * p + (k_base + 1)
        p_idx = k_base * n + i_base
        q_idx = (k_base + 1) * n + i_base
        r_idx = k_base * n + (i_base + 1)
        s_idx = (k_base + 1) * n + (i_base + 1)
        
        # P1 = (a+d)(e+h)
        U[a_idx, col] = 1; U[d_idx, col] = 1
        V[e_idx, col] = 1; V[h_idx, col] = 1
        W[p_idx, col] = 1; W[s_idx, col] = 1
        col += 1
        
        # P2 = (c+d)(e)
        U[c_idx, col] = 1; U[d_idx, col] = 1
        V[e_idx, col] = 1
        W[r_idx, col] = 1; W[s_idx, col] = -1
        col += 1
        
        # P3 = (a)(f-h)
        U[a_idx, col] = 1
        V[f_idx, col] = 1; V[h_idx, col] = -1
        W[q_idx, col] = 1; W[s_idx, col] = 1
        col += 1
        
        # P4 = (d)(g-e)
        U[d_idx, col] = 1
        V[g_idx, col] = 1; V[e_idx, col] = -1
        W[p_idx, col] = 1; W[r_idx, col] = 1
        col += 1
        
        # P5 = (a+b)(h)
        U[a_idx, col] = 1; U[b_idx, col] = 1
        V[h_idx, col] = 1
        W[p_idx, col] = -1; W[q_idx, col] = 1
        col += 1
        
        # P6 = (c-a)(e+f)
        U[c_idx, col] = 1; U[a_idx, col] = -1
        V[e_idx, col] = 1; V[f_idx, col] = 1
        W[s_idx, col] = 1
        col += 1
        
        # P7 = (b-d)(g+h)
        U[b_idx, col] = 1; U[d_idx, col] = -1
        V[g_idx, col] = 1; V[h_idx, col] = 1
        W[p_idx, col] = 1
        col += 1

    # Row 0: schoolbook (3 rows of 4x6 = 24 terms)
    for j in range(4):
        for k in range(6):
            add_schoolbook(0, j, k)
    
    # Rows 1-2: Strassen on 2x2x2 blocks
    for j_block in range(2):
        for k_block in range(3):
            i_base = 1
            j_base = j_block * 2
            k_base = k_block * 2
            add_strassen(i_base, j_base, k_base)
    
    def verify(U_mat, V_mat, W_mat, r):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W, rank):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    current_rank = rank
    
    # Aggressive column removal with multiple passes
    max_passes = 200
    for _ in range(max_passes):
        if _time_left() < 15:
            break
        removed_any = False
        for c in range(current_rank):
            if _time_left() < 10:
                break
            U_test = np.delete(current_U, c, axis=1)
            V_test = np.delete(current_V, c, axis=1)
            W_test = np.delete(current_W, c, axis=1)
            if verify(U_test, V_test, W_test, current_rank - 1):
                current_U = U_test
                current_V = V_test
                current_W = W_test
                current_rank -= 1
                best_rank = current_rank
                best_U = current_U.copy()
                best_V = current_V.copy()
                best_W = current_W.copy()
                removed_any = True
                break
        if not removed_any:
            break
    
    # Try linear combination removal: for each pair of columns, try to merge
    if _time_left() > 20:
        for c1 in range(current_rank):
            if _time_left() < 15:
                break
            for c2 in range(c1 + 1, current_rank):
                if _time_left() < 10:
                    break
                # Try removing c1 if it's a linear combination involving c2
                # Check if column c1 can be expressed via others
                # Simple check: set c1 to zero, see if we can adjust c2
                U_test = current_U.copy()
                V_test = current_V.copy()
                W_test = current_W.copy()
                # Try: new_c2 = old_c2 + alpha * old_c1, then remove c1
                # For the tensor equation to hold, we need the contribution to cancel
                # This is complex; skip for now
                pass
    
    if not verify(best_U, best_V, best_W, best_rank):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
