"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 if i in [0,n), j in [0,m), k in [0,p).
    The tensor has dimensions (n*m, m*p, p*n) indexed by (i*m+j, j*p+k, k*n+i)."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T

def _check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor.
    U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    The decomposition satisfies: T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    """
    T = _build_tensor(n, m, p)
    # Compute the reconstructed tensor
    # T_recon[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    # This is a tensor contraction
    rank = U.shape[1]
    T_recon = np.einsum('ic,jc,kc->ijk', U, V, W)
    return np.allclose(T_recon, T, atol=tol)

def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2x2.
    U: (4,7), V: (4,7), W: (4,7)
    T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    where i = i1*2+i2, j = j1*2+j2, k = k1*2+k2
    """
    # Strassen's 7 products
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = A*B = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-d)(f+h)
    # M7 = (b-d)(e+g)
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # Indexing: i = i1*2+i2, j = j1*2+j2, k = k1*2+k2
    # A[i1,i2], B[j1,j2], C[k1,k2]
    # U[i, c] where i = i1*2+i2
    # V[j, c] where j = j1*2+j2
    # W[k, c] where k = k1*2+k2
    
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # c=0: M1 = (a+d)(e+h)
    # U: a=1 (i1=0,i2=0), d=1 (i1=1,i2=1)
    U[0, 0] = 1; U[3, 0] = 1
    # V: e=1 (j1=0,j2=0), h=1 (j1=1,j2=1)
    V[0, 0] = 1; V[3, 0] = 1
    # W: contributes to C[0,0] (+1), C[1,1] (+1)
    W[0, 0] = 1; W[3, 0] = 1
    
    # c=1: M2 = (c+d)e
    # U: c=1 (i1=1,i2=0), d=1 (i1=1,i2=1)
    U[2, 1] = 1; U[3, 1] = 1
    # V: e=1 (j1=0,j2=0)
    V[0, 1] = 1
    # W: contributes to C[1,0] (+1), C[1,1] (-1)
    W[2, 1] = 1; W[3, 1] = -1
    
    # c=2: M3 = a(f-h)
    # U: a=1 (i1=0,i2=0)
    U[0, 2] = 1
    # V: f=1 (j1=0,j2=1), h=-1 (j1=1,j2=1)
    V[1, 2] = 1; V[3, 2] = -1
    # W: contributes to C[0,1] (+1), C[1,1] (+1)
    W[1, 2] = 1; W[3, 2] = 1
    
    # c=3: M4 = d(g-e)
    # U: d=1 (i1=1,i2=1)
    U[3, 3] = 1
    # V: g=1 (j1=1,j2=0), e=-1 (j1=0,j2=0)
    V[2, 3] = 1; V[0, 3] = -1
    # W: contributes to C[0,0] (+1), C[1,0] (+1)
    W[0, 3] = 1; W[2, 3] = 1
    
    # c=4: M5 = (a+b)h
    # U: a=1 (i1=0,i2=0), b=1 (i1=0,i2=1)
    U[0, 4] = 1; U[1, 4] = 1
    # V: h=1 (j1=1,j2=1)
    V[3, 4] = 1
    # W: contributes to C[0,0] (-1), C[0,1] (+1)
    W[0, 4] = -1; W[1, 4] = 1
    
    # c=5: M6 = (c-d)(f+h)
    # U: c=1 (i1=1,i2=0), d=-1 (i1=1,i2=1)
    U[2, 5] = 1; U[3, 5] = -1
    # V: f=1 (j1=0,j2=1), h=1 (j1=1,j2=1)
    V[1, 5] = 1; V[3, 5] = 1
    # W: contributes to C[1,1] (+1)
    W[3, 5] = 1
    
    # c=6: M7 = (b-d)(e+g)
    # U: b=1 (i1=0,i2=1), d=-1 (i1=1,i2=1)
    U[1, 6] = 1; U[3, 6] = -1
    # V: e=1 (j1=0,j2=0), g=1 (j1=1,j2=0)
    V[0, 6] = 1; V[2, 6] = 1
    # W: contributes to C[0,0] (+1)
    W[0, 6] = 1
    
    return U, V, W

def _pad_strassen(n, m, p):
    """Pad Strassen decomposition to dimensions n x m x p.
    We use a block approach: decompose into 2x2x2 Strassen blocks where possible,
    and use schoolbook for the remaining parts.
    
    Strategy: 
    - For the 3x4x7 case, we can try various block decompositions.
    - One approach: pad to 4x4x8 (multiple of 2) and use 2x2x2 Strassen blocks.
      But 4x4x8 = 2x2x4 blocks, each 2x2x2 Strassen.
      Actually, let's think more carefully.
    
    Better approach: Use a mixed decomposition.
    - Decompose the 3x4x7 tensor into overlapping 2x2x2 sub-tensors.
    - Each 2x2x2 sub-tensor can use Strassen (rank 7) instead of schoolbook (rank 8).
    
    Let's use a systematic approach:
    - For each combination of (i_block, j_block, k_block) where i_block in {0,1} (since 3=2+1),
      j_block in {0,1,2} (since 4=2+2), k_block in {0,1,2,3} (since 7=2+2+2+1),
      we have a 2x2x2 or smaller block.
    
    Actually, a cleaner approach:
    - Pad n=3 to 4, m=4 stays, p=7 to 8.
    - Use 2x2x2 Strassen blocks on the 4x4x8 padded tensor.
    - Number of 2x2x2 blocks: (4/2)*(4/2)*(8/2) = 2*2*4 = 16 blocks.
    - Each block uses rank 7, so total rank = 16*7 = 112.
    - But we need to handle the padding: the padded parts will have zero contributions.
    
    Wait, that gives rank 112 which is worse than 84. Let me reconsider.
    
    Actually, for 3x4x7:
    - If we pad to 4x4x8: 16 blocks * 7 = 112. Worse.
    - If we use a smarter decomposition...
    
    Let's try a different approach: direct optimization.
    """
    pass

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    # Start with schoolbook decomposition
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook), dtype=float)
    V_sb = np.zeros((m * p, rank_schoolbook), dtype=float)
    W_sb = np.zeros((p * n, rank_schoolbook), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_schoolbook
    
    # Try Strassen-based block decomposition
    # For 3x4x7, let's try to use 2x2x2 Strassen blocks strategically.
    # 
    # Idea: Partition the indices into blocks and use Strassen where we have 2x2x2 blocks.
    # n=3: indices 0,1,2. We can have blocks [0,1] and [2].
    # m=4: indices 0,1,2,3. Blocks [0,1] and [2,3].
    # p=7: indices 0,1,2,3,4,5,6. Blocks [0,1], [2,3], [4,5], [6].
    #
    # For each combination of (n_block, m_block, p_block):
    # - If n_block size 2, m_block size 2, p_block size 2: use Strassen (rank 7)
    # - Otherwise: use schoolbook for that sub-block
    #
    # Let's enumerate:
    # n_blocks: [(0,1), (2,)]  -> sizes [2, 1]
    # m_blocks: [(0,1), (2,3)] -> sizes [2, 2]
    # p_blocks: [(0,1), (2,3), (4,5), (6,)] -> sizes [2, 2, 2, 1]
    #
    # For each triple (nb, mb, pb):
    # - If all sizes are 2: Strassen, rank 7
    # - Else: schoolbook, rank = size_n * size_m * size_p
    
    n_blocks = [(0, 1), (2,)]
    m_blocks = [(0, 1), (2, 3)]
    p_blocks = [(0, 1), (2, 3), (4, 5), (6,)]
    
    # Count the rank
    total_rank = 0
    for nb in n_blocks:
        for mb in m_blocks:
            for pb in p_blocks:
                sn, sm, sp = len(nb), len(mb), len(pb)
                if sn == 2 and sm == 2 and sp == 2:
                    total_rank += 7
                else:
                    total_rank += sn * sm * sp
    
    # Let's compute: 
    # (0,1)x(0,1)x(0,1): 2x2x2 -> 7
    # (0,1)x(0,1)x(2,3): 2x2x2 -> 7
    # (0,1)x(0,1)x(4,5): 2x2x2 -> 7
    # (0,1)x(0,1)x(6,): 2x2x1 -> 4
    # (0,1)x(2,3)x(0,1): 2x2x2 -> 7
    # (0,1)x(2,3)x(2,3): 2x2x2 -> 7
    # (0,1)x(2,3)x(4,5): 2x2x2 -> 7
    # (0,1)x(2,3)x(6,): 2x2x1 -> 4
    # (2,)x(0,1)x(0,1): 1x2x2 -> 4
    # (2,)x(0,1)x(2,3): 1x2x2 -> 4
    # (2,)x(0,1)x(4,5): 1x2x2 -> 4
    # (2,)x(0,1)x(6,): 1x2x1 -> 2
    # (2,)x(2,3)x(0,1): 1x2x2 -> 4
    # (2,)x(2,3)x(2,3): 1x2x2 -> 4
    # (2,)x(2,3)x(4,5): 1x2x2 -> 4
    # (2,)x(2,3)x(6,): 1x2x1 -> 2
    # Sum: 7+7+7+4+7+7+7+4+4+4+4+2+4+4+4+2 = 80
    
    if total_rank < best_rank:
        # Build this decomposition
        U = np.zeros((n * m, total_rank), dtype=float)
        V = np.zeros((m * p, total_rank), dtype=float)
        W = np.zeros((p * n, total_rank), dtype=float)
        
        col = 0
        for nb in n_blocks:
            for mb in m_blocks:
                for pb in p_blocks:
                    sn, sm, sp = len(nb), len(mb), len(pb)
                    if sn == 2 and sm == 2 and sp == 2:
                        # Use Strassen
                        U_s, V_s, W_s = _strassen_2x2x2()
                        for ci in range(sn):
                            for cj in range(sm):
                                for ck in range(sp):
                                    i = nb[ci]
                                    j = mb[cj]
                                    k = pb[ck]
                                    for c in range(7):
                                        U[i * m + j, col + c] = U_s[ci * 2 + cj, c]
                                        V[j * p + k, col + c] = V_s[cj * 2 + ck, c]
                                        W[k * n + i, col + c] = W_s[ck * 2 + ci, c]
                        col += 7
                    else:
                        # Schoolbook
                        for ci in range(sn):
                            for cj in range(sm):
                                for ck in range(sp):
                                    i = nb[ci]
                                    j = mb[cj]
                                    k = pb[ck]
                                    U[i * m + j, col] = 1
                                    V[j * p + k, col] = 1
                                    W[k * n + i, col] = 1
                                    col += 1
        
        # Verify
        if _check_decomposition(U, V, W, n, m, p):
            best_U, best_V, best_W = U, V, W
            best_rank = total_rank
    
    # Try to further optimize with random perturbations and local search
    # This is a heuristic approach to see if we can find a lower rank decomposition
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 20:
        # Return best found so far
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try a more aggressive approach: use the fact that we can mix Strassen blocks
    # with different groupings
    
    # Alternative: try all possible 2x2x2 sub-block placements
    # For each set of 2 consecutive i's, 2 consecutive j's, 2 consecutive k's,
    # we can apply Strassen. But these blocks overlap, so we need to be careful.
    
    # Let's try a different partitioning:
    # n=3: try [0,1,2] as a whole with some 2x2 sub-blocks
    # Actually, let's try padding n to 4 and using 2x2x2 Strassen on 4x4x8,
    # but only for the non-zero parts.
    
    # Pad to 4x4x8
    n_pad, m_pad, p_pad = 4, 4, 8
    # Use 2x2x2 Strassen blocks: 2*2*4 = 16 blocks, each rank 7, total 112
    # This is worse than 80, so skip.
    
    # Try another approach: use the known fact that for 3x4x7,
    # the minimum rank might be lower than 80.
    # Let's try a random search for a better decomposition.
    
    # Actually, let's just return the best we have (80 if it worked, else 84)
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try to verify and return
    if _check_decomposition(best_U, best_V, best_W, n, m, p):
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Fallback to schoolbook
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
