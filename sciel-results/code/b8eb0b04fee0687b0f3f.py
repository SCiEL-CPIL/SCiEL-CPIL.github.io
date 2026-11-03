"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _strassen_2x2_decomp():
    """Return (U, V, W) as lists of lists for the rank-7 Strassen decomposition of 2x2x2.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c] for i,j,k in {0,1}.
    """
    # Strassen's 7 products for 2x2 matrix multiplication:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # U rows: (i,j) -> i*2+j; V rows: (j,k) -> j*2+k; W rows: (k,i) -> k*2+i
    # U[(i,j), c] * V[(j,k), c] * W[(k,i), c] = C[i,k] contribution
    #
    # Let's define each term:
    # Term 1: M1 = (A11+A22)(B11+B22)
    #   A part: A11+A22 -> U[(0,0)]=1, U[(1,1)]=1
    #   B part: B11+B22 -> V[(0,0)]=1, V[(1,1)]=1
    #   C part: contributes to C11, C22
    #   C11: +1 -> W[(0,0)] += 1 (k=0,i=0 -> row 0)
    #   C22: -1 -> W[(1,1)] += -1 (k=1,i=1 -> row 3)
    #
    # Let me build this carefully.
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # Term 0: M1 = (A11+A22)(B11+B22)
    # A: (0,0)=1, (1,1)=1
    # B: (0,0)=1, (1,1)=1
    # C: C11=+1, C22=-1
    U[0][0] += 1  # (0,0)
    U[3][0] += 1  # (1,1)
    V[0][0] += 1  # (0,0)
    V[3][0] += 1  # (1,1)
    W[0][0] += 1  # (k=0,i=0) -> C11
    W[3][0] += -1 # (k=1,i=1) -> C22
    
    # Term 1: M2 = (A21+A22)B11
    # A: (1,0)=1, (1,1)=1
    # B: (0,0)=1
    # C: C21=+1, C22=-1
    U[2][1] += 1  # (1,0)
    U[3][1] += 1  # (1,1)
    V[0][1] += 1  # (0,0)
    W[1][1] += 1  # (k=0,i=1) -> C21
    W[3][1] += -1 # (k=1,i=1) -> C22
    
    # Term 2: M3 = A11(B12-B22)
    # A: (0,0)=1
    # B: (0,1)=1, (1,1)=-1
    # C: C12=+1, C22=+1
    U[0][2] += 1  # (0,0)
    V[1][2] += 1  # (0,1)
    V[3][2] += -1 # (1,1)
    W[0][2] += 1  # (k=0,i=0)... wait
    # C12: i=0,k=1 -> W[(1,0)] = W[1]
    # C22: i=1,k=1 -> W[(1,1)] = W[3]
    W[1][2] += 1  # (k=1,i=0) -> C12
    W[3][2] += 1  # (k=1,i=1) -> C22
    
    # Term 3: M4 = A22(B21-B11)
    # A: (1,1)=1
    # B: (1,0)=1, (0,0)=-1
    # C: C11=+1, C21=+1
    U[3][3] += 1  # (1,1)
    V[2][3] += 1  # (1,0)
    V[0][3] += -1 # (0,0)
    W[0][3] += 1  # (k=0,i=0) -> C11
    W[1][3] += 1  # (k=0,i=1) -> C21
    
    # Term 4: M5 = (A11+A12)B22
    # A: (0,0)=1, (0,1)=1
    # B: (1,1)=1
    # C: C11=-1, C12=+1
    U[0][4] += 1  # (0,0)
    U[1][4] += 1  # (0,1)
    V[3][4] += 1  # (1,1)
    W[0][4] += -1 # (k=0,i=0) -> C11
    W[1][4] += 1  # (k=1,i=0) -> C12
    
    # Term 5: M6 = (A21-A11)(B11+B12)
    # A: (1,0)=1, (0,0)=-1
    # B: (0,0)=1, (0,1)=1
    # C: C22=+1
    U[2][5] += 1  # (1,0)
    U[0][5] += -1 # (0,0)
    V[0][5] += 1  # (0,0)
    V[1][5] += 1  # (0,1)
    W[3][5] += 1  # (k=1,i=1) -> C22
    
    # Term 6: M7 = (A12-A22)(B21+B22)
    # A: (0,1)=1, (1,1)=-1
    # B: (1,0)=1, (1,1)=1
    # C: C11=+1
    U[1][6] += 1  # (0,1)
    U[3][6] += -1 # (1,1)
    V[2][6] += 1  # (1,0)
    V[3][6] += 1  # (1,1)
    W[0][6] += 1  # (k=0,i=0) -> C11
    
    return U, V, W


def _verify_decomp(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # And check T[i,j,k] == 1 if this is a valid entry of the matmul tensor
    # The matmul tensor is: T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k])
    # Actually the tensor is: sum_j A[i,j] B[j,k] = C[i,k]
    # So the decomposition should satisfy:
    # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for all i,j,k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-9:
                    return False, (i, j, k, val)
    return True, None


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 4
    
    # Strategy: Use the known rank-28 decomposition for 4x4x4.
    # The 4x4x4 matmul tensor can be decomposed using 4 independent 2x2x2 Strassen
    # decompositions on the block diagonal, plus cross terms handled by additional
    # rank-1 terms, OR we can use the known optimal construction.
    #
    # Actually, a simpler and verified approach: use the rank-28 decomposition.
    # One known construction: treat 4x4 as 2x2 blocks of 2x2 matrices.
    # The 4x4 matmul tensor = (2x2 block matmul) ⊗ (2x2 scalar matmul)
    # But this gives rank 7*8 = 56, not optimal.
    #
    # Better: use the direct approach with the known rank-28.
    # The rank-28 decomposition for M_{4,4,4} is known. Let me construct it
    # using the "Laderman" style approach or a known explicit construction.
    #
    # Actually, let me use a different approach: the rank of M_{n,m,p} <= n*m*p
    # but for 4x4x4, the best known upper bound is 49 (from 2x2 Strassen on blocks)
    # or lower. Let me just use the schoolbook rank 64 as a safe fallback, but
    # try to do better.
    #
    # Wait, the inspirations say score 1.3061 was achieved. Score = seed/value.
    # Seed = 64 (schoolbook rank). Score 1.3061 means value = 64/1.3061 ≈ 49.
    # So rank 49 was achieved. That's 4 * 7^2 / 4... no, 49 = 7*7.
    # This suggests using the 2x2 Strassen on the block structure.
    #
    # For 4x4 = 2x2 blocks of 2x2:
    # C = A * B where A, B, C are 4x4, viewed as 2x2 blocks of 2x2 matrices.
    # C_{pq} = sum_r A_{pr} B_{rq} where each block is 2x2.
    # This is a 2x2 matrix multiplication over the ring of 2x2 matrices.
    # If we use Strassen over this ring, we need 7 block multiplications,
    # each of which is a 2x2 x 2x2 = 4x4... no, each block product is 2x2 matmul.
    # So each block product needs rank 8 (schoolbook) or rank 7 (Strassen) for 2x2.
    # Total: 7 * 8 = 56 or 7 * 7 = 49.
    #
    # So rank 49 = use Strassen at both levels!
    # The 2x2 block matmul uses Strassen (7 terms), and each 2x2 block matmul
    # also uses Strassen (7 terms). Total rank = 7 * 7 = 49.
    
    # Let me construct this carefully.
    #
    # Level 1 (block level): 2x2 matmul of blocks using Strassen.
    # Let A blocks be A00, A01, A10, A11 (2x2 matrices)
    # Let B blocks be B00, B01, B10, B11
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # where M1..M7 are the Strassen products (each is a 2x2 matrix product)
    #
    # Level 2 (within block): each M_t is a 2x2 x 2x2 matmul, decomposed via Strassen
    #
    # So total rank = 7 (block level) * 7 (within-block level) = 49
    
    # Now I need to map this to the U, V, W format.
    # U has shape (n*m, rank) = (16, 49)
    # V has shape (m*p, rank) = (16, 49)  
    # W has shape (p*n, rank) = (16, 49)
    #
    # U[i*m+j, c] where i in 0..3, j in 0..3
    # V[j*p+k, c] where j in 0..3, k in 0..3
    # W[k*n+i, c] where k in 0..3, i in 0..3
    #
    # Block indices: i = i_block*2 + i_local, j = j_block*2 + j_local, k = k_block*2 + k_local
    #
    # For each block-level Strassen term t (0..6) and within-block Strassen term s (0..6):
    # c = t*7 + s
    #
    # The block-level Strassen gives coefficients for how blocks combine.
    # The within-block Strassen gives coefficients for how local entries combine.
    #
    # Let me define the block-level Strassen coefficients:
    # For term t, we have:
    #   A_block combination: which A_blocks and with what coefficients
    #   B_block combination: which B_blocks and with what coefficients
    #   C_block combination: which C_blocks and with what coefficients
    #
    # Block-level Strassen:
    # t=0 (M1): A: {A00:+1, A11:+1}, B: {B00:+1, B11:+1}, C: {C00:+1, C11:-1}
    # t=1 (M2): A: {A10:+1, A11:+1}, B: {B00:+1}, C: {C10:+1, C11:-1}
    # t=2 (M3): A: {A00:+1}, B: {B01:+1, B11:-1}, C: {C01:+1, C11:+1}
    # t=3 (M4): A: {A11:+1}, B: {B10:+1, B00:-1}, C: {C00:+1, C10:+1}
    # t=4 (M5): A: {A00:+1, A01:+1}, B: {B11:+1}, C: {C00:-1, C01:+1}
    # t=5 (M6): A: {A10:+1, A00:-1}, B: {B00:+1, B01:+1}, C: {C11:+1}
    # t=6 (M7): A: {A01:+1, A11:-1}, B: {B10:+1, B11:+1}, C: {C00:+1}
    
    # Within-block Strassen (from _strassen_2x2_decomp):
    sU, sV, sW = _strassen_2x2_decomp()
    # sU[(i_local*2+j_local), s], sV[(j_local*2+k_local), s], sW[(k_local*2+i_local), s]
    
    # Block-level coefficients
    # A_block_coefs[t][block_idx] where block_idx is 0=A00, 1=A01, 2=A10, 3=A11
    A_block_coefs = [
        {0: 1, 3: 1},      # t=0
        {2: 1, 3: 1},      # t=1
        {0: 1},             # t=2
        {3: 1},             # t=3
        {0: 1, 1: 1},      # t=4
        {2: 1, 0: -1},     # t=5
        {1: 1, 3: -1},     # t=6
    ]
    # B_block_coefs[t][block_idx] where block_idx is 0=B00, 1=B01, 2=B10, 3=B11
    B_block_coefs = [
        {0: 1, 3: 1},      # t=0
        {0: 1},             # t=1
        {1: 1, 3: -1},     # t=2
        {2: 1, 0: -1},     # t=3
        {3: 1},             # t=4
        {0: 1, 1: 1},      # t=5
        {2: 1, 3: 1},      # t=6
    ]
    # C_block_coefs[t][block_idx] where block_idx is 0=C00, 1=C01, 2=C10, 3=C11
    C_block_coefs = [
        {0: 1, 3: -1},     # t=0
        {2: 1, 3: -1},     # t=1
        {1: 1, 3: 1},      # t=2
        {0: 1, 2: 1},      # t=3
        {0: -1, 1: 1},     # t=4
        {3: 1},             # t=5
        {0: 1},             # t=6
    ]
    
    rank = 49
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for t in range(7):
        for s in range(7):
            c = t * 7 + s
            # For each local index combination
            for i_local in range(2):
                for j_local in range(2):
                    for k_local in range(2):
                        # Within-block coefficients
                        u_val = sU[i_local * 2 + j_local][s]
                        v_val = sV[j_local * 2 + k_local][s]
                        w_val = sW[k_local * 2 + i_local][s]
                        
                        if u_val == 0 and v_val == 0 and w_val == 0:
                            continue
                        
                        # Block-level: for A, we need A_block_coefs[t] to tell us
                        # which blocks contribute. But the within-block part already
                        # handles the local indices. We need to combine them.
                        #
                        # The full index: i = i_block*2 + i_local, j = j_block*2 + j_local
                        # U[i*m+j, c] gets contributions from all (i_block, j_block)
                        # where A_block_coefs[t] has a nonzero entry for the block
                        # containing (i_local, j_local) in A, and similarly for V, W.
                        #
                        # Actually, the structure is:
                        # For a given (i_local, j_local, k_local, s, t):
                        # - U gets contributions for each A_block a in A_block_coefs[t]:
                        #   U[(a_block_row*2+i_local)*m + (a_block_col*2+j_local), c] += coeff_a * u_val
                        #   where a_block_row, a_block_col are determined by which block
                        #
                        # Wait, I need to be more careful. In the block decomposition:
                        # A is a 2x2 block matrix. A_block(a_row, a_col) is a 2x2 matrix.
                        # The Strassen term t combines certain A_blocks with coefficients.
                        # Similarly for B and C.
                        #
                        # For the A side of term t: A_combined = sum_{a} coef_a * A_block_a
                        # Each A_block_a is a 2x2 matrix, so for local indices (i_local, j_local),
                        # A_block_a[i_local, j_local] is the local entry.
                        #
                        # The row index in U is i*m + j where i = i_block*2 + i_local, j = j_block*2 + j_local
                        # But which block? The block is determined by (i_block, j_block).
                        #
                        # Hmm, this is getting complex. Let me think differently.
                        #
                        # The key insight: the Kronecker-like product structure means:
                        # U[i*m+j, c] = sum over (a_row, a_col) of A_block_coefs[t][(a_row,a_col)] * sU[i_local*2+j_local, s]
                        # where i = a_row*2 + i_local, j = a_col*2 + j_local
                        #
                        # But wait, the block coefficients tell us which BLOCKS are combined,
                        # not which local entries. The local entries are handled by sU.
                        #
                        # So for term (t, s):
                        # For each A_block a = (ar, ac) with coefficient ca in A_block_coefs[t]:
                        #   For each local (i_l, j_l):
                        #     U[(ar*2+i_l)*m + (ac*2+j_l), c] += ca * sU[i_l*2+j_l][s]
                        #
                        # Similarly for V and W.
                        
                        for (ar, ac), ca in [( (k//2, k%2), v) for k, v in A_block_coefs[t].items()]:
                            # Block a = (ar, ac) corresponds to A[ar*2:(ar+1)*2, ac*2:(ac+1)*2]
                            for i_l in range(2):
                                for j_l in range(2):
                                    i = ar * 2 + i_l
                                    j = ac * 2 + j_l
                                    u_contrib = ca * sU[i_l * 2 + j_l][s]
                                    if u_contrib != 0:
                                        U[i * m + j][c] += u_contrib
                        
                        for (br, bc), cb in [( (k//2, k%2), v) for k, v in B_block_coefs[t].items()]:
                            for j_l in range(2):
                                for k_l in range(2):
                                    j = br * 2 + j_l
                                    k = bc * 2 + k_l
                                    v_contrib = cb * sV[j_l * 2 + k_l][s]
                                    if v_contrib != 0:
                                        V[j * p + k][c] += v_contrib
                        
                        for (cr, cc), cw in [( (k//2, k%2), v) for k, v in C_block_coefs[t].items()]:
                            for k_l in range(2):
                                for i_l in range(2):
                                    k = cr * 2 + k_l
                                    i = cc * 2 + i_l
                                    w_contrib = cw * sW[k_l * 2 + i_l][s]
                                    if w_contrib != 0:
                                        W[k * n + i][c] += w_contrib
    
    # Verify
    valid, err = _verify_decomp(U, V, W, n, m, p)
    if not valid:
        # Fallback to schoolbook if verification fails
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
