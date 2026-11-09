<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    # Block structure: 5 = 2 + 2 + 1
    block_sizes = [2, 2, 1]
    num_blocks = 3
    offsets = [0]
    for bs in block_sizes:
        offsets.append(offsets[-1] + bs)
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    for bi in range(num_blocks):
        for bj in range(num_blocks):
            for bk in range(num_blocks):
                si = block_sizes[bi]
                sj = block_sizes[bj]
                sk = block_sizes[bk]
                
                a_row_start = offsets[bi]
                a_col_start = offsets[bj]
                b_row_start = offsets[bj]
                b_col_start = offsets[bk]
                c_row_start = offsets[bi]
                c_col_start = offsets[bk]
                
                if si == 2 and sj == 2 and sk == 2:
                    # Strassen: 7 rank-1 terms
                    # M1 = (A00+A11)(B00+B11)
                    # M2 = (A10+A11)(B00)
                    # M3 = (A00)(B01-B11)
                    # M4 = (A11)(B10-B00)
                    # M5 = (A00+A01)(B11)
                    # M6 = (A10-A00)(B00+B01)
                    # M7 = (A01-A11)(B10+B11)
                    # C00 = M1+M4-M5+M7
                    # C01 = M3+M5
                    # C10 = M2+M4
                    # C11 = M1-M2+M3+M6
                    
                    # Each term is a rank-1 contribution
                    # For each multiplication Mk, we have linear combos in U, V, W
                    # U has entries at (a_row_start+ii, a_col_start+jj)
                    # V has entries at (b_row_start+jj, b_col_start+kk)
                    # W has entries at (c_row_start+ii, c_col_start+kk)
                    
                    # M1: U: (0,0)=1, (1,1)=1; V: (0,0)=1, (0,1)=1; W: (0,0)=1, (1,1)=1
                    U_terms.append([(a_row_start+0, a_col_start+0, 1), (a_row_start+1, a_col_start+1, 1)])
                    V_terms.append([(b_row_start+0, b_col_start+0, 1), (b_row_start+0, b_col_start+1, 1)])
                    W_terms.append([(c_row_start+0, c_col_start+0, 1), (c_row_start+1, c_col_start+1, 1)])
                    
                    # M2: U: (1,0)=1, (1,1)=1; V: (0,0)=1; W: (1,0)=1, (1,1)=-1
                    U_terms.append([(a_row_start+1, a_col_start+0, 1), (a_row_start+1, a_col_start+1, 1)])
                    V_terms.append([(b_row_start+0, b_col_start+0, 1)])
                    W_terms.append([(c_row_start+1, c_col_start+0, 1), (c_row_start+1, c_col_start+1, -1)])
                    
                    # M3: U: (0,0)=1; V: (0,1)=1, (1,1)=-1; W: (0,1)=1, (1,1)=1
                    U_terms.append([(a_row_start+0, a_col_start+0, 1)])
                    V_terms.append([(b_row_start+0, b_col_start+1, 1), (b_row_start+1, b_col_start+1, -1)])
                    W_terms.append([(c_row_start+0, c_col_start+1, 1), (c_row_start+1, c_col_start+1, 1)])
                    
                    # M4: U: (1,1)=1; V: (1,0)=1, (0,0)=-1; W: (0,0)=1, (1,0)=1
                    U_terms.append([(a_row_start+1, a_col_start+1, 1)])
                    V_terms.append([(b_row_start+1, b_col_start+0, 1), (b_row_start+0, b_col_start+0, -1)])
                    W_terms.append([(c_row_start+0, c_col_start+0, 1), (c_row_start+1, c_col_start+0, 1)])
                    
                    # M5: U: (0,0)=1, (0,1)=1; V: (1,1)=1; W: (0,0)=-1, (0,1)=1
                    U_terms.append([(a_row_start+0, a_col_start+0, 1), (a_row_start+0, a_col_start+1, 1)])
                    V_terms.append([(b_row_start+1, b_col_start+1, 1)])
                    W_terms.append([(c_row_start+0, c_col_start+0, -1), (c_row_start+0, c_col_start+1, 1)])
                    
                    # M6: U: (1,0)=1, (0,0)=-1; V: (0,0)=1, (0,1)=1; W: (1,1)=1
                    U_terms.append([(a_row_start+1, a_col_start+0, 1), (a_row_start+0, a_col_start+0, -1)])
                    V_terms.append([(b_row_start+0, b_col_start+0, 1), (b_row_start+0, b_col_start+1, 1)])
                    W_terms.append([(c_row_start+1, c_col_start+1, 1)])
                    
                    # M7: U: (0,1)=1, (1,1)=-1; V: (1,0)=1, (1,1)=1; W: (0,0)=1
                    U_terms.append([(a_row_start+0, a_col_start+1, 1), (a_row_start+1, a_col_start+1, -1)])
                    V_terms.append([(b_row_start+1, b_col_start+0, 1), (b_row_start+1, b_col_start+1, 1)])
                    W_terms.append([(c_row_start+0, c_col_start+0, 1)])
                    
                else:
                    # Schoolbook for this block
                    for ii in range(si):
                        for jj in range(sj):
                            for kk in range(sk):
                                U_terms.append([(a_row_start+ii, a_col_start+jj, 1)])
                                V_terms.append([(b_row_start+jj, b_col_start+kk, 1)])
                                W_terms.append([(c_row_start+ii, c_col_start+kk, 1)])
    
    rank = len(U_terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col in range(rank):
        for (row_idx, val) in [(t[0], t[2]) for t in U_terms[col]]:
            U[row_idx][col] = val
        for (row_idx, val) in [(t[0], t[2]) for t in V_terms[col]]:
            V[row_idx][col] = val
        for (row_idx, val) in [(t[0], t[2]) for t in W_terms[col]]:
            W[row_idx][col] = val
    
    # Verify correctness
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check: sum_k U[i*m+j,k] * V[j*p+k,k] * W[k*n+i,k] should equal delta_{i,k} * delta_{j,k}... 
    # Actually the tensor T[i,j,k] = 1 if this is a valid multiplication term
    # The decomposition says: T[i,j,k] = sum_rank U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For matrix mult, T[i,j,k] = 1 for all i,j,k (it's the full multiplication tensor)
    # Wait no. The multiplication tensor T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # Let's verify a few entries
    for i in range(min(n, 3)):
        for j in range(min(m, 3)):
            for k in range(min(p, 3)):
                lhs = 0.0
                for r in range(rank):
                    lhs += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                if abs(lhs - 1.0) > 1e-9:
                    # Fallback to schoolbook
                    rank_fb = n * m * p
                    U_fb = [[0] * rank_fb for _ in range(n * m)]
                    V_fb = [[0] * rank_fb for _ in range(m * p)]
                    W_fb = [[0] * rank_fb for _ in range(p * n)]
                    column = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U_fb[ii * m + jj][column] = 1
                                V_fb[jj * p + kk][column] = 1
                                W_fb[kk * n + ii][column] = 1
                                column += 1
                    return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    # Use a 4x4x4 Strassen-based decomposition (rank 16) for the first 4x4x4 sub-block,
    # plus schoolbook for the remaining parts involving the 5th dimension.
    # Block structure: 5 = 4 + 1
    block_sizes = [4, 1]
    num_blocks = 2
    offsets = [0]
    for bs in block_sizes:
        offsets.append(offsets[-1] + bs)
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    # Strassen decomposition for 2x2 blocks (rank 7)
    # We'll use this for the 4x4x4 block by applying it twice (4 = 2+2)
    # First, define the Strassen terms for a 2x2 block
    def strassen_terms_2x2(row_offset, col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        """Return U, V, W term lists for a 2x2x2 Strassen block."""
        terms = []
        
        # M1 = (A00+A11)(B00+B11)
        terms.append([(row_offset+0, col_offset+0, 1), (row_offset+1, col_offset+1, 1)],
                     [(b_row_offset+0, b_col_offset+0, 1), (b_row_offset+0, b_col_offset+1, 1)],
                     [(c_row_offset+0, c_col_offset+0, 1), (c_row_offset+1, c_col_offset+1, 1)])
        
        # M2 = (A10+A11)(B00)
        terms.append([(row_offset+1, col_offset+0, 1), (row_offset+1, col_offset+1, 1)],
                     [(b_row_offset+0, b_col_offset+0, 1)],
                     [(c_row_offset+1, c_col_offset+0, 1), (c_row_offset+1, c_col_offset+1, -1)])
        
        # M3 = (A00)(B01-B11)
        terms.append([(row_offset+0, col_offset+0, 1)],
                     [(b_row_offset+0, b_col_offset+1, 1), (b_row_offset+1, b_col_offset+1, -1)],
                     [(c_row_offset+0, c_col_offset+1, 1), (c_row_offset+1, c_col_offset+1, 1)])
        
        # M4 = (A11)(B10-B00)
        terms.append([(row_offset+1, col_offset+1, 1)],
                     [(b_row_offset+1, b_col_offset+0, 1), (b_row_offset+0, b_col_offset+0, -1)],
                     [(c_row_offset+0, c_col_offset+0, 1), (c_row_offset+1, c_col_offset+0, 1)])
        
        # M5 = (A00+A01)(B11)
        terms.append([(row_offset+0, col_offset+0, 1), (row_offset+0, col_offset+1, 1)],
                     [(b_row_offset+1, b_col_offset+1, 1)],
                     [(c_row_offset+0, c_col_offset+0, -1), (c_row_offset+0, c_col_offset+1, 1)])
        
        # M6 = (A10-A00)(B00+B01)
        terms.append([(row_offset+1, col_offset+0, 1), (row_offset+0, col_offset+0, -1)],
                     [(b_row_offset+0, b_col_offset+0, 1), (b_row_offset+0, b_col_offset+1, 1)],
                     [(c_row_offset+1, c_col_offset+1, 1)])
        
        # M7 = (A01-A11)(B10+B11)
        terms.append([(row_offset+0, col_offset+1, 1), (row_offset+1, col_offset+1, -1)],
                     [(b_row_offset+1, b_col_offset+0, 1), (b_row_offset+1, b_col_offset+1, 1)],
                     [(c_row_offset+0, c_col_offset+0, 1)])
        
        return terms
    
    # For the 4x4x4 block, we decompose it into four 2x2x2 blocks
    # But we can do better: use Strassen recursively on 4x4 = 2+2
    # This gives us 4 * 7 = 28 terms, which is worse than schoolbook (64)
    # Actually, let's just use schoolbook for 4x4x4 to keep it simple and correct
    # Rank for 4x4x4 schoolbook = 64
    
    # Actually, let's use the original approach with 2+2+1 blocks
    # Block structure: 5 = 2 + 2 + 1
    block_sizes = [2, 2, 1]
    num_blocks = 3
    offsets = [0]
    for bs in block_sizes:
        offsets.append(offsets[-1] + bs)
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    for bi in range(num_blocks):
        for bj in range(num_blocks):
            for bk in range(num_blocks):
                si = block_sizes[bi]
                sj = block_sizes[bj]
                sk = block_sizes[bk]
                
                a_row_start = offsets[bi]
                a_col_start = offsets[bj]
                b_row_start = offsets[bj]
                b_col_start = offsets[bk]
                c_row_start = offsets[bi]
                c_col_start = offsets[bk]
                
                if si == 2 and sj == 2 and sk == 2:
                    # Strassen: 7 rank-1 terms
                    strassen = strassen_terms_2x2(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start)
                    for u, v, w in strassen:
                        U_terms.append(u)
                        V_terms.append(v)
                        W_terms.append(w)
                else:
                    # Schoolbook for this block
                    for ii in range(si):
                        for jj in range(sj):
                            for kk in range(sk):
                                U_terms.append([(a_row_start+ii, a_col_start+jj, 1)])
                                V_terms.append([(b_row_start+jj, b_col_start+kk, 1)])
                                W_terms.append([(c_row_start+ii, c_col_start+kk, 1)])
    
    rank = len(U_terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col in range(rank):
        for (row_idx, val) in [(t[0], t[2]) for t in U_terms[col]]:
            U[row_idx][col] = val
        for (row_idx, val) in [(t[0], t[2]) for t in V_terms[col]]:
            V[row_idx][col] = val
        for (row_idx, val) in [(t[0], t[2]) for t in W_terms[col]]:
            W[row_idx][col] = val
    
    # Verify correctness
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check all entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = 0.0
                for r in range(rank):
                    lhs += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                if abs(lhs - 1.0) > 1e-9:
                    # Fallback to schoolbook
                    rank_fb = n * m * p
                    U_fb = [[0] * rank_fb for _ in range(n * m)]
                    V_fb = [[0] * rank_fb for _ in range(m * p)]
                    W_fb = [[0] * rank_fb for _ in range(p * n)]
                    column = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U_fb[ii * m + jj][column] = 1
                                V_fb[jj * p + kk][column] = 1
                                W_fb[kk * n + ii][column] = 1
                                column += 1
                    return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
