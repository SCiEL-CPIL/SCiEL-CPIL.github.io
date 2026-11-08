import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Function to verify a decomposition
    def verify(U, V, W):
        # U: (nm, rank), V: (mp, rank), W: (pn, rank)
        # Check T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
        # T[i,j,k] should be 1 if A[i] and B[j] contribute to C[k]
        # A: n x m, B: m x p, C: n x p
        # i = r_a * m + c_a
        # j = r_b * m + c_b  (Wait, B is m x p, so j = r_b * p + c_b)
        # k = r_c * p + c_c
        
        rank = U.shape[1]
        # Construct T from U, V, W
        # T = U @ V.T @ W.T ? No.
        # T[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
        # We can compute this for all i, j, k
        
        # Efficient check:
        # M1 = U @ V.T  (nm x mp)
        # M2 = M1 @ W.T ? No.
        # T is a 3D tensor.
        # We can check a few random points or all points.
        # nm=8, mp=16, pn=8. Total 1024 points. Rank 28.
        # 1024 * 28 operations is small.
        
        T_calc = np.zeros((nm, mp, pn))
        for r in range(rank):
            T_calc += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            
        T_true = np.zeros((nm, mp, pn))
        for r_a in range(n):
            for c_a in range(m):
                i = r_a * m + c_a
                for r_b in range(m):
                    for c_b in range(p):
                        j = r_b * p + c_b
                        if c_a == r_b:
                            k = r_a * p + c_b
                            T_true[i, j, k] = 1.0
        
        if not np.allclose(T_calc, T_true):
            return False
        return True

    # Attempt Rank 28 (Strassen based)
    rank = 28
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    def add_strassen_block(a_idx, b_idx, c_idx, col_offset):
        x00, x01, x10, x11 = a_idx
        y00, y01, y10, y11 = b_idx
        z00, z01, z10, z11 = c_idx
        
        # M1
        U[x00, col_offset] += 1
        U[x11, col_offset] += 1
        V[y00, col_offset] += 1
        V[y11, col_offset] += 1
        W[z00, col_offset] += 1
        W[z11, col_offset] += 1
        
        # M2
        U[x10, col_offset+1] += 1
        U[x11, col_offset+1] += 1
        V[y00, col_offset+1] += 1
        W[z10, col_offset+1] += 1
        
        # M3
        U[x00, col_offset+2] += 1
        V[y01, col_offset+2] += 1
        V[y11, col_offset+2] -= 1
        W[z01, col_offset+2] += 1
        W[z11, col_offset+2] += 1
        
        # M4
        U[x11, col_offset+3] += 1
        V[y10, col_offset+3] += 1
        V[y00, col_offset+3] -= 1
        W[z00, col_offset+3] += 1
        W[z10, col_offset+3] += 1
        
        # M5
        U[x00, col_offset+4] += 1
        U[x01, col_offset+4] += 1
        V[y11, col_offset+4] += 1
        W[z00, col_offset+4] -= 1
        W[z01, col_offset+4] += 1
        
        # M6
        U[x10, col_offset+5] += 1
        U[x00, col_offset+5] -= 1
        V[y00, col_offset+5] += 1
        V[y01, col_offset+5] += 1
        W[z11, col_offset+5] += 1
        
        # M7
        U[x01, col_offset+6] += 1
        U[x11, col_offset+6] -= 1
        V[y10, col_offset+6] += 1
        V[y11, col_offset+6] += 1
        W[z00, col_offset+6] += 1
        W[z11, col_offset+6] -= 1

    # Blocks
    add_strassen_block([0, 1, 4, 5], [0, 1, 4, 5], [0, 1, 4, 5], 0)
    add_strassen_block([2, 3, 6, 7], [8, 9, 12, 13], [0, 1, 4, 5], 7)
    add_strassen_block([0, 1, 4, 5], [2, 3, 6, 7], [2, 3, 6, 7], 14)
    add_strassen_block([2, 3, 6, 7], [10, 11, 14, 15], [2, 3, 6, 7], 21)
    
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback: Rank 32 Naive
    rank_naive = nm * mp # No, rank is at most min(nm*mp, ...) 
    # Naive rank is n*m*p = 2*4*4 = 32
    rank_naive = 32
    U = np.zeros((nm, rank_naive))
    V = np.zeros((mp, rank_naive))
    W = np.zeros((pn, rank_naive))
    
    idx = 0
    for r_a in range(n):
        for c_a in range(m):
            i = r_a * m + c_a
            for r_b in range(m):
                if c_a == r_b:
                    for c_b in range(p):
                        j = r_b * p + c_b
                        k = r_a * p + c_b
                        U[i, idx] = 1
                        V[j, idx] = 1
                        W[k, idx] = 1
                        idx += 1
                        
    # idx should be 32
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
