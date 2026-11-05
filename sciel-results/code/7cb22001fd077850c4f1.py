import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Return the classic Strassen 2x2x2 decomposition with 7 terms.
    U, V, W are 4x7 matrices (row-major indexing).
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Term 1: M1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 2: M2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # Term 3: M3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # Term 4: M4 = d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1
    
    # Term 5: M5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # Term 6: M6 = (c-d)(f+g)
    U[2][5] = 1; U[3][5] = -1
    V[1][5] = 1; V[2][5] = 1
    W[3][5] = 1
    
    # Term 7: M7 = (b-d)(e+f)
    U[1][6] = 1; U[3][6] = -1
    V[0][6] = 1; V[1][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _verify_fast(n, m, p, U, V, W):
    """
    Vectorized verification of tensor decomposition.
    For each (i,j,k), check if sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Build arrays indexed by (i,j,k) -> value
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    rank = U_arr.shape[1]
    
    # For efficiency, we check all (i,j,k) at once using broadcasting
    # But for 5x5x5=125 elements, a loop is fine and fast enough
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_arr[i * m + j]
                v_row = V_arr[j * p + k]
                w_row = W_arr[k * n + i]
                val = np.dot(u_row, v_row * w_row)
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _construct_5x5x5():
    """
    Construct a rank-117 decomposition for 5x5x5 matrix multiplication.
    
    Strategy: Partition indices into {0,1}, {2,3}, {4}.
    For each of the 27 blocks (i_set, j_set, k_set) where each set is one of the 3 groups:
    - If the block is 2x2x2 (all sets have size 2): use Strassen (rank 7)
    - Otherwise: use schoolbook (rank = |i_set| * |j_set| * |k_set|)
    
    There are 8 pure 2x2x2 blocks (each saves 1 rank: 8->7)
    Total: 125 - 8 = 117
    """
    n, m, p = 5, 5, 5
    nm, mp, pn = n * m, m * p, p * n
    
    # Define the partition groups
    groups = [(0, 1), (2, 3), (4,)]
    
    # Calculate total rank
    total_rank = 0
    for gi in range(3):
        for gj in range(3):
            for gk in range(3):
                size_i = len(groups[gi])
                size_j = len(groups[gj])
                size_k = len(groups[gk])
                if size_i == 2 and size_j == 2 and size_k == 2:
                    total_rank += 7  # Strassen
                else:
                    total_rank += size_i * size_j * size_k  # Schoolbook
    
    U = [[0] * total_rank for _ in range(nm)]
    V = [[0] * total_rank for _ in range(mp)]
    W = [[0] * total_rank for _ in range(pn)]
    
    col = 0
    
    for gi in range(3):
        for gj in range(3):
            for gk in range(3):
                i_set = groups[gi]
                j_set = groups[gj]
                k_set = groups[gk]
                
                size_i = len(i_set)
                size_j = len(j_set)
                size_k = len(k_set)
                
                if size_i == 2 and size_j == 2 and size_k == 2:
                    # Apply Strassen to this 2x2x2 block
                    U_s, V_s, W_s = _strassen_2x2()
                    for c in range(7):
                        for r in range(4):
                            i_local = r // 2
                            j_local = r % 2
                            i = i_set[i_local]
                            j = j_set[j_local]
                            U[i * m + j][col + c] = U_s[r][c]
                            
                            # V index: j*p + k
                            # In Strassen, V is indexed by (j_local, k_local)
                            j_for_v = j_set[j_local]
                            k_local = r % 2  # Wait, this is wrong. Let me re-examine.
                            
                    # Actually, let me be more careful with indexing.
                    # Strassen U_s[r][c] where r = i_local*2 + j_local
                    # Strassen V_s[r][c] where r = j_local*2 + k_local
                    # Strassen W_s[r][c] where r = k_local*2 + i_local
                    
                    for c in range(7):
                        for r in range(4):
                            # U: index (i_local, j_local) -> r = i_local*2 + j_local
                            i_local = r // 2
                            j_local = r % 2
                            i = i_set[i_local]
                            j = j_set[j_local]
                            U[i * m + j][col + c] = U_s[r][c]
                            
                            # V: index (j_local, k_local) -> r = j_local*2 + k_local
                            j_local_v = r // 2
                            k_local_v = r % 2
                            j_v = j_set[j_local_v]
                            k_v = k_set[k_local_v]
                            V[j_v * p + k_v][col + c] = V_s[r][c]
                            
                            # W: index (k_local, i_local) -> r = k_local*2 + i_local
                            k_local_w = r // 2
                            i_local_w = r % 2
                            k_w = k_set[k_local_w]
                            i_w = i_set[i_local_w]
                            W[k_w * n + i_w][col + c] = W_s[r][c]
                    
                    col += 7
                else:
                    # Schoolbook for this block
                    for i in i_set:
                        for j in j_set:
                            for k in k_set:
                                U[i * m + j][col] = 1
                                V[j * p + k][col] = 1
                                W[k * n + i][col] = 1
                                col += 1
    
    assert col == total_rank, f"Column count mismatch: {col} vs {total_rank}"
    
    return {"U": U, "V": V, "W": W}, total_rank

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin for 150s limit
    
    result, rank = _construct_5x5x5()
    
    # Verify the decomposition
    if _verify_fast(n, m, p, result["U"], result["V"], result["W"]):
        return result
    
    # Fallback: Pure schoolbook (rank 125)
    U = [[0] * 125 for _ in range(25)]
    V = [[0] * 125 for _ in range(25)]
    W = [[0] * 125 for _ in range(25)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
