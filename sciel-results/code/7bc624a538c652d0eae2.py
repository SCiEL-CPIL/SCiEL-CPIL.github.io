"""Matrix multiplication 3x5x5 rank-minimal decomposition search."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = 3, 5, 5
    deadline = time.time() + 120

    def schoolbook_decomposition(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    U[i * m + j, idx] = 1.0
                    V[j * p + k, idx] = 1.0
                    W[k * n + i, idx] = 1.0
        return U, V, W

    def verify(U, V, W):
        # W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] should give (A@B)[i,k]
        # The tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # should equal 1 for all (i,j,k)
        U_t = U.reshape(n, m, -1)
        V_t = V.reshape(m, p, -1)
        W_t = W.reshape(p, n, -1)
        # T[i,j,k] = sum_c U_t[i,j,c] * V_t[j,k,c] * W_t[k,i,c]
        # This is a contraction over c of U[i,j,c]*V[j,k,c]*W[k,i,c]
        # Use einsum: 'ijc,jkc,kic->ijk'
        T = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
        return np.allclose(T, 1.0, atol=1e-9)

    def strassen_2x2x2_block(U, V, W, base_rank,
                            i0, i1, j0, j1, k0, k1):
        # 2x2x2 Strassen: 7 multiplications
        # A = U[i0:i1, j0:j1] rows (2x2 block of A indices)
        # B = V[j0:j1, k0:k1] cols (2x2 block of B indices)
        # C = W[k0:k1, i0:i1] (2x2 block of C indices)
        # The sub-tensor is the 2x2x2 multiplication tensor
        # We need to add 7 rank-1 terms
        
        # Indices in the full matrices:
        # A row index for (i,j) is i*m + j
        # B row index for (j,k) is j*p + k  
        # C row index for (k,i) is k*n + i
        
        a_rows = [i0 * m + j0, i0 * m + j1, i1 * m + j0, i1 * m + j1]
        b_rows = [j0 * p + k0, j0 * p + k1, j1 * p + k0, j1 * p + k1]
        c_rows = [k0 * n + i0, k0 * n + i1, k1 * n + i0, k1 * n + i1]
        
        # Strassen for 2x2:
        # M1 = (a+d)(e+h), M2 = (c+d)e, M3 = a(g-h), M4 = d(f-e)
        # M5 = (a+b)h, M6 = (c-a)(e+g), M7 = (b-d)(f+h)
        # C00 = M1-M5+M7+M4, C01 = M5+M3
        # C10 = M2+M4, C11 = M1-M2+M6+M3
        
        # For each M_s, we have a rank-1 decomposition of the 2x2x2 block
        # where the coefficient patterns are:
        # Each M_s corresponds to a specific (a_vec, b_vec, c_vec) triple
        
        # Let me just use the standard Strassen rank-1 terms directly.
        # For the 2x2x2 multiplication tensor, the 7 rank-1 decompositions are:
        
        # M1: a=(1,0,1,0), b=(1,1,0,0), c=(1,0,0,1)  -> (a11+a22)(b11+b22) -> c11+c22
        # Actually let me be more careful.
        
        # A = [[a11, a12], [a21, a22]]  -> rows: [a11, a12, a21, a22]
        # B = [[b11, b12], [b21, b22]]  -> rows: [b11, b12, b21, b22]  
        # C = [[c11, c12], [c21, c22]]  -> rows: [c11, c12, c21, c22]
        
        # M1 = (a11+a22)(b11+b22): A_vec=(1,0,0,1), B_vec=(1,1,0,0), C contributes to c11 and c22
        #   C11 += M1, C22 += M1 => C_vec=(1,0,0,1)
        # M2 = (a21+a22)b11: A_vec=(0,0,1,1), B_vec=(1,0,0,0), C21 += M2 => C_vec=(0,0,1,0)
        # M3 = a11(b21-b22): A_vec=(1,0,0,0), B_vec=(0,0,1,-1), C12 += M3 => C_vec=(0,1,0,0)
        # M4 = a22(b12-b11): A_vec=(0,0,0,1), B_vec=(-1,1,0,0), C11 += M4, C21 += M4 => C_vec=(1,0,1,0)
        # M5 = (a11+a12)b22: A_vec=(1,1,0,0), B_vec=(0,0,0,1), C11 -= M5, C12 += M5 => C_vec=(-1,1,0,0)
        # M6 = (a21-a11)(b11+b12): A_vec=(-1,0,1,0), B_vec=(1,1,0,0), C22 += M6 => C_vec=(0,0,0,1)
        # M7 = (a12-a22)(b21+b22): A_vec=(0,1,0,-1), B_vec=(0,0,1,1), C11 += M7, C22 -= M7 => C_vec=(1,0,0,-1)
        
        strassen_terms = [
            # (A_vec, B_vec, C_vec)
            (np.array([1,0,0,1], dtype=np.float64), np.array([1,1,0,0], dtype=np.float64), np.array([1,0,0,1], dtype=np.float64)),   # M1
            (np.array([0,0,1,1], dtype=np.float64), np.array([1,0,0,0], dtype=np.float64), np.array([0,0,1,0], dtype=np.float64)),   # M2
            (np.array([1,0,0,0], dtype=np.float64), np.array([0,0,1,-1], dtype=np.float64), np.array([0,1,0,0], dtype=np.float64)),   # M3
            (np.array([0,0,0,1], dtype=np.float64), np.array([-1,1,0,0], dtype=np.float64), np.array([1,0,1,0], dtype=np.float64)),   # M4
            (np.array([1,1,0,0], dtype=np.float64), np.array([0,0,0,1], dtype=np.float64), np.array([-1,1,0,0], dtype=np.float64)),   # M5
            (np.array([-1,0,1,0], dtype=np.float64), np.array([1,1,0,0], dtype=np.float64), np.array([0,0,0,1], dtype=np.float64)),   # M6
            (np.array([0,1,0,-1], dtype=np.float64), np.array([0,0,1,1], dtype=np.float64), np.array([1,0,0,-1], dtype=np.float64)),   # M7
        ]
        
        new_rank = base_rank + 7
        U_new = np.zeros((U.shape[0], new_rank), dtype=np.float64)
        V_new = np.zeros((V.shape[0], new_rank), dtype=np.float64)
        W_new = np.zeros((W.shape[0], new_rank), dtype=np.float64)
        U_new[:, :base_rank] = U
        V_new[:, :base_rank] = V
        W_new[:, :base_rank] = W
        
        for s, (a_vec, b_vec, c_vec) in enumerate(strassen_terms):
            col = base_rank + s
            U_new[a_rows, col] = a_vec
            V_new[b_rows, col] = b_vec
            W_new[c_rows, col] = c_vec
        
        return U_new, V_new, W_new

    # Start with schoolbook
    U, V, W = schoolbook_decomposition(n, m, p)
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try replacing 2x2x2 blocks with Strassen
    # Block decomposition: 3=2+1, 5=2+2+1, 5=2+2+1
    # Possible 2x2x2 blocks:
    # i in {0,1}, j in {0,1}, k in {0,1}
    # i in {0,1}, j in {2,3}, k in {0,1}
    # i in {0,1}, j in {0,1}, k in {2,3}
    # i in {0,1}, j in {2,3}, k in {2,3}
    # etc.
    
    # For each 2x2x2 block, if we apply Strassen, we replace 8 terms with 7
    # But the blocks must be "independent" - they should not overlap in a way that breaks the decomposition
    
    # Let's try all combinations of non-overlapping 2x2x2 blocks
    # i-blocks: [0,1] or [1,2] (but [1,2] is only 2 rows, so i in {1,2} is valid for 3 rows)
    # Actually for n=3, possible i-pairs: (0,1) and (1,2)
    # For m=5, possible j-pairs: (0,1), (1,2), (2,3), (3,4)
    # For p=5, possible k-pairs: (0,1), (1,2), (2,3), (3,4)
    
    # But Strassen requires the block to be exactly 2x2x2, so we need consecutive pairs
    i_pairs = [(0,1), (1,2)]
    j_pairs = [(0,1), (1,2), (2,3), (3,4)]
    k_pairs = [(0,1), (1,2), (2,3), (3,4)]
    
    # Try single blocks first
    for i0, i1 in i_pairs:
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                U2, V2, W2 = strassen_2x2x2_block(U, V, W, 0, i0, i1, j0, j1, k0, k1)
                # Wait, this is wrong. I need to start from scratch and build up.
                # Let me restructure.
                pass
    
    # Better approach: build the decomposition from scratch using blocks
    # Divide into blocks and for each 2x2x2 block use Strassen, for 1x1x1 use schoolbook
    
    def build_from_blocks(n, m, p):
        """Build decomposition by partitioning into blocks."""
        # For simplicity, use a fixed block structure
        # 3 = 2+1, 5 = 2+2+1
        # This gives blocks:
        # (2,2,2), (2,2,1), (2,1,2), (2,1,1), (1,2,2), (1,2,1), (1,1,2), (1,1,1)
        # For each 2x2x2 block, use Strassen (rank 7)
        # For each other block, use schoolbook (rank = i_size * j_size * k_size)
        
        total_rank = 0
        U_list, V_list, W_list = [], [], []
        
        i_sizes = [2, 1]
        j_sizes = [2, 2, 1]
        k_sizes = [2, 2, 1]
        
        i_start = 0
        for i_sz in i_sizes:
            j_start = 0
            for j_sz in j_sizes:
                k_start = 0
                for k_sz in k_sizes:
                    block_rank = i_sz * j_sz * k_sz
                    if i_sz == 2 and j_sz == 2 and k_sz == 2:
                        block_rank = 7  # Strassen
                    
                    # Create the block decomposition
                    if i_sz == 2 and j_sz == 2 and k_sz == 2:
                        # Strassen
                        a_rows = [i_start * m + j_start, i_start * m + j_start + 1,
                                  (i_start + 1) * m + j_start, (i_start + 1) * m + j_start + 1]
                        b_rows = [j_start * p + k_start, j_start * p + k_start + 1,
                                  (j_start + 1) * p + k_start, (j_start + 1) * p + k_start + 1]
                        c_rows = [k_start * n + i_start, k_start * n + i_start + 1,
                                  (k_start + 1) * n + i_start, (k_start + 1) * n + i_start + 1]
                        
                        strassen_terms = [
                            (np.array([1,0,0,1]), np.array([1,1,0,0]), np.array([1,0,0,1])),
                            (np.array([0,0,1,1]), np.array([1,0,0,0]), np.array([0,0,1,0])),
                            (np.array([1,0,0,0]), np.array([0,0,1,-1]), np.array([0,1,0,0])),
                            (np.array([0,0,0,1]), np.array([-1,1,0,0]), np.array([1,0,1,0])),
                            (np.array([1,1,0,0]), np.array([0,0,0,1]), np.array([-1,1,0,0])),
                            (np.array([-1,0,1,0]), np.array([1,1,0,0]), np.array([0,0,0,1])),
                            (np.array([0,1,0,-1]), np.array([0,0,1,1]), np.array([1,0,0,-1])),
                        ]
                        
                        for a_vec, b_vec, c_vec in strassen_terms:
                            U_blk = np.zeros((n*m, 1), dtype=np.float64)
                            V_blk = np.zeros((m*p, 1), dtype=np.float64)
                            W_blk = np.zeros((p*n, 1), dtype=np.float64)
                            U_blk[a_rows, 0] = a_vec
                            V_blk[b_rows, 0] = b_vec
                            W_blk[c_rows, 0] = c_vec
                            U_list.append(U_blk)
                            V_list.append(V_blk)
                            W_list.append(W_blk)
                            total_rank += 1
                    else:
                        # Schoolbook for this block
                        for ii in range(i_sz):
                            for jj in range(j_sz):
                                for kk in range(k_sz):
                                    U_blk = np.zeros((n*m, 1), dtype=np.float64)
                                    V_blk = np.zeros((m*p, 1), dtype=np.float64)
                                    W_blk = np.zeros((p*n, 1), dtype=np.float64)
                                    U_blk[(i_start + ii) * m + (j_start + jj), 0] = 1.0
                                    V_blk[(j_start + jj) * p + (k_start + kk), 0] = 1.0
                                    W_blk[(k_start + kk) * n + (i_start + ii), 0] = 1.0
                                    U_list.append(U_blk)
                                    V_list.append(V_blk)
                                    W_list.append(W_blk)
                                    total_rank += 1
                    
                    k_start += k_sz
                j_start += j_sz
            i_start += i_sz
        
        U = np.hstack(U_list)
        V = np.hstack(V_list)
        W = np.hstack(W_list)
        return U, V, W
    
    U, V, W = build_from_blocks(n, m, p)
    rank = U.shape[1]
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Verify
    if verify(U, V, W):
        pass
    else:
        # Fall back to schoolbook
        U, V, W = schoolbook_decomposition(n, m, p)
        best_rank = U.shape[1]
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try different block partitions
    # For n=3, m=5, p=5, try various partitions
    partitions_n = [[3], [2,1], [1,2], [1,1,1]]
    partitions_m = [[5], [2,2,1], [2,1,2], [1,2,2], [1,1,3], [3,2], [2,3], [1,4], [4,1], [1,1,1,2], [1,1,2,1], [1,2,1,1], [2,1,1,1]]
    partitions_p = partitions_m  # same structure
    
    best_rank = best_U.shape[1]
    
    for pn in partitions_n:
        for pm in partitions_m:
            for pp in partitions_p:
                if sum(pn) != n or sum(pm) != m or sum(pp) != p:
                    continue
                U2, V2, W2 = build_from_blocks_with_partitions(n, m, p, pn, pm, pp) if 'build_from_blocks_with_partitions' in dir() else (None, None, None)
                if U2 is None:
                    continue
                if U2.shape[1] < best_rank and verify(U2, V2, W2):
                    best_rank = U2.shape[1]
                    best_U, best_V, best_W = U2.copy(), V2.copy(), W2.copy()
                
                if time.time() > deadline - 5:
                    break
            if time.time() > deadline - 5:
                break
        if time.time() > deadline - 5:
            break
    
    # Convert to lists for output
    def to_list(mat):
        return [[float(v) for v in row] for row in mat]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
