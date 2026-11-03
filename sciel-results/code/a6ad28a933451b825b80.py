import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition 5 = 2 + 2 + 1
    # Block sizes: [2, 2, 1]
    # Offsets: [0, 2, 4]
    
    block_sizes = [2, 2, 1]
    block_offsets = [0, 2, 4]
    num_blocks = 3
    
    # Total rank calculation:
    # (2,2,2) -> 7 (Strassen)
    # (1,1,1) -> 1 (Schoolbook)
    # (2,1,2) -> 4 (Schoolbook)
    # (1,2,2) -> 4 (Schoolbook)
    
    # We need to sum up the ranks for all valid (bi, bj, bk)
    # bi, bj, bk in {0, 1, 2}
    
    # Let's define the sub-decompositions
    
    def get_strassen_2x2x2():
        """
        Returns U, V, W for 2x2x2 Strassen decomposition.
        Local indices 0,1.
        U shape (4, 7), V shape (4, 7), W shape (4, 7).
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # m1 = (A00 + A11)(B00 + B11)
        # A indices: 00->0, 01->1, 10->2, 11->3
        # B indices: 00->0, 01->1, 10->2, 11->3
        # C indices: 00->0, 01->1, 10->2, 11->3 (trace convention W[k*2+i])
        
        # Term 1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1 # C00, C11
        
        # m2 = (A10 + A11) B00
        # A10->2, A11->3
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1 # C10 (k=0, i=1 -> idx 1? No, k*2+i. k=0, i=1 -> 1. Wait. C10 is row 1 col 0. i=1, k=0. idx 0*2+1=1. 
        # My W index mapping: W[k*2+i]. 
        # C00: i=0, k=0 -> 0.
        # C01: i=1, k=0 -> 1.
        # C10: i=0, k=1 -> 2.
        # C11: i=1, k=1 -> 3.
        
        # Let's re-verify Strassen formulas with indices.
        # C00 = m1 + m4 - m5 + m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m3 + m6
        
        # Term 2: m2 = (A10 + A11) B00
        # A10 is i=1, j=0. idx 1*2+0 = 2.
        # A11 is i=1, j=1. idx 1*2+1 = 3.
        # B00 is j=0, k=0. idx 0*2+0 = 0.
        # Contributes to C10 (i=0, k=1). idx 1*2+0 = 2.
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # m3 = A00 (B01 - B11)
        # A00 idx 0.
        # B01 idx 1. B11 idx 3.
        # Contributes to C01 (i=1, k=0). idx 0*2+1 = 1.
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # m4 = A11 (B10 - B00)
        # A11 idx 3.
        # B10 idx 2. B00 idx 0.
        # Contributes to C00 (i=0, k=0) and C10 (i=0, k=1)?
        # Formula: C00 has +m4. C10 has +m4.
        # C00 idx 0. C10 idx 2.
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # m5 = (A00 + A01) B11
        # A00 idx 0. A01 idx 1.
        # B11 idx 3.
        # Contributes to C00 (-m5) and C01 (+m5).
        # C00 idx 0. C01 idx 1.
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # m6 = (A10 - A00) (B00 + B01)
        # A10 idx 2. A00 idx 0.
        # B00 idx 0. B01 idx 1.
        # Contributes to C11 (+m6).
        # C11 idx 3.
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # m7 = (A01 - A11) (B10 + B11)
        # A01 idx 1. A11 idx 3.
        # B10 idx 2. B11 idx 3.
        # Contributes to C00 (+m7).
        # C00 idx 0.
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    def get_schoolbook(si, sj, sk):
        """
        Returns U, V, W for schoolbook decomposition of si x sj x sk.
        """
        rank = si * sj * sk
        U = np.zeros((si * sj, rank))
        V = np.zeros((sj * sk, rank))
        W = np.zeros((sk * si, rank))
        
        col = 0
        for i in range(si):
            for j in range(sj):
                for k in range(sk):
                    U[i * sj + j, col] = 1
                    V[j * sk + k, col] = 1
                    W[k * si + i, col] = 1
                    col += 1
        return U, V, W

    # Precompute sub-decompositions
    # Types:
    # 0: 2x2x2 (Strassen)
    # 1: 1x1x1 (Schoolbook)
    # 2: 2x1x2 (Schoolbook)
    # 3: 1x2x2 (Schoolbook)
    
    decomp_cache = {}
    
    def get_decomp(si, sj, sk):
        key = (si, sj, sk)
        if key not in decomp_cache:
            if si == 2 and sj == 2 and sk == 2:
                decomp_cache[key] = get_strassen_2x2x2()
            else:
                decomp_cache[key] = get_schoolbook(si, sj, sk)
        return decomp_cache[key]

    # Initialize global matrices
    # We don't know the total rank yet, so we can collect blocks first
    # or estimate.
    # Max rank per block is 7. 9 blocks. Max 63.
    # Let's just accumulate into lists and stack.
    
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    for bi in range(num_blocks):
        for bj in range(num_blocks):
            for bk in range(num_blocks):
                si = block_sizes[bi]
                sj = block_sizes[bj]
                sk = block_sizes[bk]
                
                U_sub, V_sub, W_sub = get_decomp(si, sj, sk)
                rank_sub = U_sub.shape[1]
                
                # Create global-sized zero matrices for this block
                U_glob = np.zeros((n * m, rank_sub))
                V_glob = np.zeros((m * p, rank_sub))
                W_glob = np.zeros((p * n, rank_sub))
                
                off_i = block_offsets[bi]
                off_j = block_offsets[bj]
                off_k = block_offsets[bk]
                
                for r in range(rank_sub):
                    # Map U_sub indices to global
                    for li in range(si):
                        for lj in range(sj):
                            val = U_sub[li * sj + lj, r]
                            if val != 0:
                                gi = off_i + li
                                gj = off_j + lj
                                U_glob[gi * m + gj, r] = val
                            
                    # Map V_sub indices to global
                    for lj in range(sj):
                        for lk in range(sk):
                            val = V_sub[lj * sk + lk, r]
                            if val != 0:
                                gj = off_j + lj
                                gk = off_k + lk
                                V_glob[gj * p + gk, r] = val
                                
                    # Map W_sub indices to global
                    for lk in range(sk):
                        for li in range(si):
                            val = W_sub[lk * si + li, r]
                            if val != 0:
                                gk = off_k + lk
                                gi = off_i + li
                                W_glob[gk * n + gi, r] = val

                U_blocks.append(U_glob)
                V_blocks.append(V_glob)
                W_blocks.append(W_glob)

    U = np.hstack(U_blocks)
    V = np.hstack(V_blocks)
    W = np.hstack(W_blocks)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
