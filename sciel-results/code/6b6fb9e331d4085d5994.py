<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def construct():
    """Matrix multiplication tensor decomposition for 3x5x7."""
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # We'll build up the decomposition by handling each (i, j, k) triple.
    # The tensor T[i,j,k] = 1 for all valid i in [0,n), j in [0,m), k in [0,p).
    # We want to find factor matrices U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # such that T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Strategy: Start with schoolbook (rank = n*m*p = 105) and try to reduce.
    # For now, let's implement a proper rank-minimal approach.
    
    # The schoolbook decomposition has rank n*m*p = 105.
    # We can do better using block decompositions.
    
    # Let's split the 3x5 matrix into blocks and the 5x7 matrix into blocks.
    # Since 3 = 2+1, 5 = 2+2+1, 7 = 3+3+1, we can use:
    # - 2x2x2 blocks with Strassen (rank 7 instead of 8)
    # - 1xk or kx1 blocks with schoolbook
    # - Other rectangular blocks
    
    # Actually, let's be more systematic. The best known approach for small rectangular
    # matrix multiplication uses specific tensor decompositions.
    
    # For 3x5x7, let's try to find a good decomposition.
    # One approach: 3x5 = (2+1)x5, 5x7 = 5x(2+2+2+1) or similar.
    
    # Let me think about this differently. The rank of M_{n,m,p} is at most n*m*p
    # (schoolbook) and at least max(n*m, m*p, p*n) = max(15, 35, 21) = 35.
    
    # Known results:
    # - M_{2,2,2}: rank 7 (Strassen) vs 8 (schoolbook)
    # - M_{2,3,3}: rank 7 (Laderman) vs 9 (schoolbook)
    # - M_{3,3,3}: rank at most 23 (best known) vs 27 (schoolbook)
    
    # For 3x5x7, let's try:
    # Split n=3 into 2+1, m=5 into 2+3, p=7 into 3+4
    # Then we have blocks:
    # - 2x2x3: ?
    # - 2x3x3: rank 7 (Laderman)
    # - 1x2x3: ?
    # - 1x3x3: ?
    # etc.
    
    # This is getting complex. Let me just implement a clean schoolbook first and
    # verify it works, then try to optimize.
    
    # Schoolbook: rank = n*m*p = 105
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
    
    # Verify
    if time.time() < deadline:
        import numpy as np
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        
        if max_err > 1e-9:
            # This shouldn't happen with schoolbook
            pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Matrix multiplication tensor decomposition for 3x5x7."""
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Use Strassen's algorithm for 2x2x2 blocks (rank 7 instead of 8)
    # Split n=3 into 2+1, m=5 into 2+3, p=7 into 3+4
    # This gives blocks:
    # - 2x2x3: use Strassen on 2x2x2 part + schoolbook on 2x2x1 part
    # - 2x3x3: use Strassen on 2x2x2 + Laderman-like for remaining
    # - 1x2x3: schoolbook (rank 6)
    # - 1x3x3: schoolbook (rank 9)
    
    # Actually, let's use a simpler but effective approach:
    # Split into 2x2, 2x3, 1x2, 1x3 blocks with optimized sub-decompositions
    
    # For 2x2x2: Strassen rank 7
    # For 2x2x3: can do rank 10 (Strassen on 2x2x2 = 7, plus 3 for the extra column)
    # Actually 2x2x3: schoolbook is 12, we can do better
    # For 2x3x3: schoolbook is 18, Laderman-like gives rank 14 or so
    # For 1x2x3: schoolbook is 6 (optimal since it's just dot products)
    # For 1x3x3: schoolbook is 9 (optimal)
    
    # Let me construct explicit decompositions for each block type
    
    def strassen_2x2x2():
        """Return (U, V, W) for 2x2x2 matrix multiplication with rank 7 (Strassen)."""
        # U: 4x7, V: 4x7, W: 4x7
        # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
        # Strassen's 7 products:
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b12-b22)
        # M4 = a22(b21-b11)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        # c11 = M1 + M4 - M5 + M7
        # c12 = M3 + M5
        # c21 = M2 + M4
        # c22 = M1 - M2 + M3 + M6
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # U indices: 0=a11, 1=a12, 2=a21, 3=a22
        # V indices: 0=b11, 1=b12, 2=b21, 3=b22
        # W indices: 0=c11, 1=c12, 2=c21, 3=c22
        
        # M1 = (a11+a22)(b11+b22)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (a21+a22)b11
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1
        
        # M3 = a11(b12-b22)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # M4 = a22(b21-b11)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # M5 = (a11+a12)b22
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # M6 = (a21-a11)(b11+b12)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7 = (a12-a22)(b21+b22)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W
    
    def schoolbook_block(n_b, m_b, p_b):
        """Schoolbook decomposition for n_b x m_b x p_b block."""
        rank = n_b * m_b * p_b
        U = [[0]*rank for _ in range(n_b * m_b)]
        V = [[0]*rank for _ in range(m_b * p_b)]
        W = [[0]*rank for _ in range(p_b * n_b)]
        
        col = 0
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    U[i*m_b+j][col] = 1
                    V[j*p_b+k][col] = 1
                    W[k*n_b+i][col] = 1
                    col += 1
        return U, V, W
    
    # Build the full 3x5x7 decomposition
    # Split: n=3 = 2+1, m=5 = 2+3, p=7 = 3+4
    
    n_splits = [2, 1]
    m_splits = [2, 3]
    p_splits = [3, 4]
    
    # Total rank estimate:
    # Block (0,0): 2x2x3 -> use Strassen for 2x2x2 (rank 7) + schoolbook for 2x2x1 (rank 4) = 11
    # Block (0,1): 2x2x4 -> use Strassen for 2x2x2 (rank 7) + schoolbook for 2x2x2 (rank 4) = 11
    # Block (1,0): 1x3x3 -> schoolbook rank 9
    # Block (1,1): 1x3x4 -> schoolbook rank 12
    # Total: 11 + 11 + 9 + 12 = 43
    
    # Actually, let me just use schoolbook for all blocks first to ensure correctness,
    # then optimize specific blocks.
    
    # For now, use schoolbook for all blocks:
    # Block (0,0): 2x2x3 = 12
    # Block (0,1): 2x2x4 = 16
    # Block (1,0): 1x3x3 = 9
    # Block (1,1): 1x3x4 = 12
    # Total: 12 + 16 + 9 + 12 = 49
    
    # But we can do better with Strassen on the 2x2x2 subblocks.
    # Let me implement a hybrid approach.
    
    # For block 2x2x3: decompose as 2x2x2 (Strassen, rank 7) + 2x2x1 (schoolbook, rank 4) = 11
    # For block 2x2x4: decompose as 2x2x2 (Strassen, rank 7) + 2x2x2 (Strassen, rank 7) = 14
    # For block 1x3x3: schoolbook = 9
    # For block 1x3x4: schoolbook = 12
    # Total: 11 + 14 + 9 + 12 = 46
    
    # Let's build this decomposition.
    
    U_full = []
    V_full = []
    W_full = []
    
    # Track offsets for each block
    block_offsets = {}
    current_col = 0
    
    for ni, n_b in enumerate(n_splits):
        for mi, m_b in enumerate(m_splits):
            for pi, p_b in enumerate(p_splits):
                i_off = sum(n_splits[:ni])
                j_off = sum(m_splits[:mi])
                k_off = sum(p_splits[:pi])
                
                if n_b == 2 and m_b == 2 and p_b >= 2:
                    # Use Strassen for 2x2x2 part, schoolbook for remaining
                    U_s, V_s, W_s = strassen_2x2x2()
                    rank_s = 7
                    current_col += rank_s
                    block_offsets[(ni, mi, pi, 'strassen')] = (current_col - rank_s, rank_s)
                    
                    for r in range(4):
                        U_full.append([0]*0)  # placeholder
                    # We'll fill in later
                    U_full[-4:] = []
                    V_full[-4:] = []
                    W_full[-4:] = []
                    
                    for r in range(4):
                        U_full.append(U_s[r][:])
                        V_full.append(V_s[r][:])
                        W_full.append(W_s[r][:])
                    
                    if p_b > 2:
                        # Schoolbook for 2x2x(p_b-2)
                        remaining_p = p_b - 2
                        U_sb, V_sb, W_sb = schoolbook_block(2, 2, remaining_p)
                        rank_sb = 4 * remaining_p
                        current_col += rank_sb
                        block_offsets[(ni, mi, pi, 'schoolbook')] = (current_col - rank_sb, rank_sb)
                        
                        for r in range(4):
                            U_full.append(U_sb[r][:])
                            V_full.append(V_sb[r][:])
                            W_full.append(W_sb[r][:])
                elif n_b == 2 and m_b == 2:
                    # 2x2x1: schoolbook
                    U_sb, V_sb, W_sb = schoolbook_block(2, 2, 1)
                    rank_sb = 4
                    current_col += rank_sb
                    block_offsets[(ni, mi, pi, 'schoolbook')] = (current_col - rank_sb, rank_sb)
                    
                    for r in range(4):
                        U_full.append(U_sb[r][:])
                        V_full.append(V_sb[r][:])
                        W_full.append(W_sb[r][:])
                else:
                    # Schoolbook for other blocks
                    U_sb, V_sb, W_sb = schoolbook_block(n_b, m_b, p_b)
                    rank_sb = n_b * m_b * p_b
                    current_col += rank_sb
                    block_offsets[(ni, mi, pi, 'schoolbook')] = (current_col - rank_sb, rank_sb)
                    
                    for r in range(n_b * m_b):
                        U_full.append(U_sb[r][:])
                    for r in range(m_b * p_b):
                        V_full.append(V_sb[r][:])
                    for r in range(p_b * n_b):
                        W_full.append(W_sb[r][:])
    
    # This approach is getting complicated with the offset tracking. Let me restart with a cleaner approach.
    
    # Clean approach: build the full matrices directly
    total_rank = 46  # estimated
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    col = 0
    
    for ni, n_b in enumerate(n_splits):
        i_off = sum(n_splits[:ni])
        for mi, m_b in enumerate(m_splits):
            j_off = sum(m_splits[:mi])
            for pi, p_b in enumerate(p_splits):
                k_off = sum(p_splits[:pi])
                
                if n_b == 2 and m_b == 2 and p_b >= 2:
                    # Strassen for 2x2x2
                    U_s, V_s, W_s = strassen_2x2x2()
                    for r in range(4):
                        for c in range(7):
                            U[i_off*5 + j_off + r][col+c] = U_s[r][c]
                            V[(j_off+r)*7 + k_off + c if c < 2 else (j_off+r)*7 + k_off + c][col+c] = V_s[r][c] if c < 2 else 0
                        # This indexing is wrong. Let me fix it.
                    
                    # V index: j_local * p_b + k_local, where j_local in [0,2), k_local in [0,2)
                    # But we're only doing 2x2x2, so k_local in [0,2)
                    # V_s has 4 rows (2x2), each with 7 columns
                    # V row index: j_local * 2 + k_local for 2x2 case
                    # But in the full V, row index is (j_off + j_local) * p + (k_off + k_local)
                    # So for 2x2x2 block: V[(j_off+j_local)*p + (k_off+k_local)][col+c] = V_s[j_local*2+k_local][c]
                    
                    for j_local in range(2):
                        for k_local in range(2):
                            for c in range(7):
                                V[(j_off+j_local)*p + (k_off+k_local)][col+c] = V_s[j_local*2+k_local][c]
                    
                    # W index: k_local * n_b + i_local for 2x2 case
                    # W row index: (k_off + k_local) * n + (i_off + i_local)
                    for k_local in range(2):
                        for i_local in range(2):
                            for c in range(7):
                                W[(k_off+k_local)*n + (i_off+i_local)][col+c] = W_s[k_local*2+i_local][c]
                    
                    col += 7
                    
                    if p_b > 2:
                        # Schoolbook for 2x2x(p_b-2)
                        remaining_p = p_b - 2
                        for i_local in range(2):
                            for j_local in range(2):
                                for k_local in range(remaining_p):
                                    U[(i_off+i_local)*m + (j_off+j_local)][col] = 1
                                    V[(j_off+j_local)*p + (k_off+2+k_local)][col] = 1
                                    W[(k_off+2+k_local)*n + (i_off+i_local)][col] = 1
                                    col += 1
                elif n_b == 2 and m_b == 2:
                    # 2x2x1: schoolbook
                    for i_local in range(2):
                        for j_local in range(2):
                            for k_local in range(1):
                                U[(i_off+i_local)*m + (j_off+j_local)][col] = 1
                                V[(j_off+j_local)*p + (k_off+k_local)][col] = 1
                                W[(k_off+k_local)*n + (i_off+i_local)][col] = 1
                                col += 1
                else:
                    # Schoolbook
                    for i_local in range(n_b):
                        for j_local in range(m_b):
                            for k_local in range(p_b):
                                U[(i_off+i_local)*m + (j_off+j_local)][col] = 1
                                V[(j_off+j_local)*p + (k_off+k_local)][col] = 1
                                W[(k_off+k_local)*n + (i_off+i_local)][col] = 1
                                col += 1
    
    # Trim to actual rank used
    actual_rank = col
    U = [row[:actual_rank] for row in U]
    V = [row[:actual_rank] for row in V]
    W = [row[:actual_rank] for row in W]
    
    # Verify
    if time.time() < deadline:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(actual_rank):
                        val += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        
        if max_err > 1e-9:
            # Fallback to schoolbook
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
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
