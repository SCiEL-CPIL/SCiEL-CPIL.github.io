# EVOLVE-BLOCK-START

import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def _verify_fast(U, V, W, n, m, p):
    """Faster verification using numpy."""
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    total = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        A = U[:, c].reshape(n, m)
        B = V[:, c].reshape(m, p)
        C = W[:, c].reshape(p, n)
        # Use einsum for faster contraction
        # total[i,j,k] += U[i,j,c] * V[j,k,c] * W[k,i,c]
        # This is a 3-way contraction.
        # We can do it in steps.
        # Temp[i,k] = sum_j U[i,j,c] * V[j,k,c]
        # Then total[i,k] += Temp[i,k] * W[k,i,c]
        
        # Efficient way:
        # AV = np.einsum('ij,jk->ik', A, B)
        # total += AV * W.T (since W is k,i, we need i,k)
        AV = np.einsum('ij,jk->ik', A, B)
        # W is (p, n) i.e., (k, i). We need to multiply AV (i, k) with W (k, i) element-wise?
        # No, total[i, j, k] is not right.
        # The tensor is T[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c]
        # My loop:
        # A is U[:, c] reshaped to (n, m) -> U[i, j]
        # B is V[:, c] reshaped to (m, p) -> V[j, k]
        # C is W[:, c] reshaped to (p, n) -> W[k, i]
        # s = sum_c U[i, j, c] V[j, k, c] W[k, i, c]
        
        # So for each c:
        # term[i, j, k] = U[i, j, c] V[j, k, c] W[k, i, c]
        # This is outer product of U_c (n,m), V_c (m,p), W_c (p,n) with specific index matching.
        
        # Let's just use the simple loop if it's fast enough for rank 44.
        # 44 iterations, 3x4x4 = 48 elements. 44*48 = 2112 ops. Very fast.
        
        for i in range(n):
            for j in range(m):
                if A[i, j] != 0:
                    total[i, j, :] += A[i, j] * B[j, :] * C[:, i]
    
    return np.allclose(total, 1.0, atol=1e-9)

def _prune_rank(U, V, W):
    """Remove zero columns and linearly dependent columns."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    rank = U.shape[1]
    keep = []
    for c in range(rank):
        # Check if column is zero
        if np.any(U[:, c] != 0) or np.any(V[:, c] != 0) or np.any(W[:, c] != 0):
            # Check linear dependence with previously kept columns
            # Simple check: if it's a scalar multiple of a previous one?
            # Or just keep it for now, rank minimization is hard.
            # For now, just remove zeros.
            keep.append(c)
            
    if len(keep) < rank:
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]
        
    return U.tolist(), V.tolist(), W.tolist()

def schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def strassen_2x2():
    """
    Return Strassen decomposition for 2x2 * 2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (A00+A11)*(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (A10+A11)*B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3: A00*(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4: A11*(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5: (A00+A10)*B11
    U[0][4] = 1; U[2][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6: (A01-A10)*(B00+B10)
    U[1][5] = 1; U[2][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[3][5] = 1
    
    # M7: (A01-A11)*(B01+B11)
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def schoolbook_sub(na, mb, pb):
    """Schoolbook decomposition for na x mb * mb x pb."""
    rank = na * mb * pb
    U = [[0] * rank for _ in range(na * mb)]
    V = [[0] * rank for _ in range(mb * pb)]
    W = [[0] * rank for _ in range(pb * na)]
    col = 0
    for i in range(na):
        for j in range(mb):
            for k in range(pb):
                U[i*mb+j][col] = 1
                V[j*pb+k][col] = 1
                W[k*na+i][col] = 1
                col += 1
    return U, V, W

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4 * 4x4 matrix multiplication.
    Uses Strassen for 2x2*2x2 sub-blocks to reduce rank from 48 to 44.
    """
    n, m, p = DIMENSIONS  # 3, 4, 4
    deadline = time.time() + 140  # Safety margin
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Strassen-based decomposition
    rank_total = 44
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    col = 0
    
    def add_sub(U_sub, V_sub, W_sub, i_off, j_off, k_off, na, mb, pb, col_start):
        nonlocal col
        r = len(U_sub[0])
        for c in range(r):
            for i_prime in range(na):
                for j_prime in range(mb):
                    u_val = U_sub[i_prime*mb + j_prime][c]
                    if u_val != 0:
                        U[(i_off + i_prime)*m + (j_off + j_prime)][col_start + c] += u_val
            for j_prime in range(mb):
                for k_prime in range(pb):
                    v_val = V_sub[j_prime*pb + k_prime][c]
                    if v_val != 0:
                        V[(j_off + j_prime)*p + (k_off + k_prime)][col_start + c] += v_val
            for k_prime in range(pb):
                for i_prime in range(na):
                    w_val = W_sub[k_prime*na + i_prime][c]
                    if w_val != 0:
                        W[(k_off + k_prime)*n + (i_off + i_prime)][col_start + c] += w_val
        col = col_start + r

    # C[0:2, 0:2]
    U_str, V_str, W_str = strassen_2x2()
    add_sub(U_str, V_str, W_str, 0, 0, 0, 2, 2, 2, col)
    
    # C[0:2, 2:4] part 1
    U_str, V_str, W_str = strassen_2x2()
    add_sub(U_str, V_str, W_str, 0, 2, 0, 2, 2, 2, col)
    
    # C[0:2, 2:4] part 2
    U_str, V_str, W_str = strassen_2x2()
    add_sub(U_str, V_str, W_str, 0, 0, 2, 2, 2, 2, col)
    
    # C[0:2, 2:4] part 3
    U_str, V_str, W_str = strassen_2x2()
    add_sub(U_str, V_str, W_str, 0, 2, 2, 2, 2, 2, col)
    
    # C[2, 0:2] part 1
    U_sub, V_sub, W_sub = schoolbook_sub(1, 2, 2)
    add_sub(U_sub, V_sub, W_sub, 2, 0, 0, 1, 2, 2, col)
    
    # C[2, 0:2] part 2
    U_sub, V_sub, W_sub = schoolbook_sub(1, 2, 2)
    add_sub(U_sub, V_sub, W_sub, 2, 2, 0, 1, 2, 2, col)
    
    # C[2, 2:4] part 1
    U_sub, V_sub, W_sub = schoolbook_sub(1, 2, 2)
    add_sub(U_sub, V_sub, W_sub, 2, 0, 2, 1, 2, 2, col)
    
    # C[2, 2:4] part 2
    U_sub, V_sub, W_sub = schoolbook_sub(1, 2, 2)
    add_sub(U_sub, V_sub, W_sub, 2, 2, 2, 1, 2, 2, col)
    
    # Prune and verify
    U_pruned, V_pruned, W_pruned = _prune_rank(U, V, W)
    
    if _verify_fast(U_pruned, V_pruned, W_pruned, n, m, p):
        return {"U": U_pruned, "V": V_pruned, "W": W_pruned}
    
    return best_result

# EVOLVE-BLOCK-END
