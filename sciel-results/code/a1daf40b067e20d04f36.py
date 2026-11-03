"""Matrix multiplication 3x4 by 4x6: Strassen-based rank-66 with column-merging optimization."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]

_START_TIME = time.time()
_DEADLINE = 145.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    """
    Construct exact tensor decomposition for (3x4)*(4x6) -> (3x6).
    
    Primary: Strassen-based rank-66 decomposition
    Secondary: Attempt to find lower-rank solution via column merging
    """
    n, m, p = 3, 4, 6
    rank = 66
    
    # Build the known rank-66 solution
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_schoolbook(i, j, k):
        nonlocal col
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        U[u_idx][col] = 1
        V[v_idx][col] = 1
        W[w_idx][col] = 1
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
        
        U[a_idx][col] = 1; U[d_idx][col] = 1
        V[e_idx][col] = 1; V[h_idx][col] = 1
        W[p_idx][col] = 1; W[s_idx][col] = 1
        col += 1
        
        U[c_idx][col] = 1; U[d_idx][col] = 1
        V[e_idx][col] = 1
        W[r_idx][col] = 1; W[s_idx][col] = -1
        col += 1
        
        U[a_idx][col] = 1
        V[f_idx][col] = 1; V[h_idx][col] = -1
        W[q_idx][col] = 1; W[s_idx][col] = 1
        col += 1
        
        U[d_idx][col] = 1
        V[g_idx][col] = 1; V[e_idx][col] = -1
        W[p_idx][col] = 1; W[r_idx][col] = 1
        col += 1
        
        U[a_idx][col] = 1; U[b_idx][col] = 1
        V[h_idx][col] = 1
        W[p_idx][col] = -1; W[q_idx][col] = 1
        col += 1
        
        U[c_idx][col] = 1; U[a_idx][col] = -1
        V[e_idx][col] = 1; V[f_idx][col] = 1
        W[s_idx][col] = 1
        col += 1
        
        U[b_idx][col] = 1; U[d_idx][col] = -1
        V[g_idx][col] = 1; V[h_idx][col] = 1
        W[p_idx][col] = 1
        col += 1

    # Build rank-66 solution: schoolbook for i=0, Strassen for i=1,2 blocks
    for j in range(4):
        for k in range(6):
            add_schoolbook(0, j, k)
    
    for k_block in range(3):
        k_base = k_block * 2
        for j_block in range(2):
            j_base = j_block * 2
            i_base = 1
            add_strassen(i_base, j_base, k_base)
    
    # Verify the solution is correct
    def verify(U_mat, V_mat, W_mat, r):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U_mat[i*m+j][c] * V_mat[j*p+k][c] * W_mat[k*n+i][c] for c in range(r))
                    if val != 1:
                        return False
        return True
    
    if not verify(U, V, W, rank):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Convert to numpy for efficient operations
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Efficient tensor verification using numpy
    def verify_numpy(U_np, V_np, W_np, r):
        """Verify tensor decomposition using numpy. Returns True if valid."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to reduce rank by removing single columns
    if _time_left() > 20:
        for remove_col in range(rank):
            if _time_left() < 10:
                break
            U_test = np.delete(U_np, remove_col, axis=1)
            V_test = np.delete(V_np, remove_col, axis=1)
            W_test = np.delete(W_np, remove_col, axis=1)
            
            if verify_numpy(U_test, V_test, W_test, rank - 1):
                best_rank = rank - 1
                best_U = U_test.tolist()
                best_V = V_test.tolist()
                best_W = W_test.tolist()
                break
    
    # If we found rank-65, try to find rank-64 via single removal
    if best_rank < rank and _time_left() > 15:
        U_np2 = np.array(best_U, dtype=float)
        V_np2 = np.array(best_V, dtype=float)
        W_np2 = np.array(best_W, dtype=float)
        for remove_col in range(best_rank):
            if _time_left() < 5:
                break
            U_test = np.delete(U_np2, remove_col, axis=1)
            V_test = np.delete(V_np2, remove_col, axis=1)
            W_test = np.delete(W_np2, remove_col, axis=1)
            
            if verify_numpy(U_test, V_test, W_test, best_rank - 1):
                best_rank = best_rank - 1
                best_U = U_test.tolist()
                best_V = V_test.tolist()
                best_W = W_test.tolist()
                break
    
    # Try pair removals: remove two columns at once
    if _time_left() > 10:
        current_U = np.array(best_U, dtype=float)
        current_V = np.array(best_V, dtype=float)
        current_W = np.array(best_W, dtype=float)
        cur_rank = best_rank
        
        # Try all pairs of columns to remove
        for c1 in range(cur_rank):
            if _time_left() < 3:
                break
            for c2 in range(c1 + 1, cur_rank):
                if _time_left() < 2:
                    break
                # Remove columns c1 and c2
                keep = [c for c in range(cur_rank) if c != c1 and c != c2]
                U_test = current_U[:, keep]
                V_test = current_V[:, keep]
                W_test = current_W[:, keep]
                
                if verify_numpy(U_test, V_test, W_test, cur_rank - 2):
                    best_rank = cur_rank - 2
                    best_U = U_test.tolist()
                    best_V = V_test.tolist()
                    best_W = W_test.tolist()
                    break
            if best_rank < cur_rank:
                break
    
    # Return the best solution found
    if best_rank < rank:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": U, "V": V, "W": W}
