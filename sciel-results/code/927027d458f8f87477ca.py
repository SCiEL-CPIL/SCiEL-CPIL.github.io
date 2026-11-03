"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that sum_c U[i',j,c]*V[j,k',c]*W[k,i,c] == delta[i',i]*delta[k',k]."""
    rank = len(U[0])
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for ip in range(n):
                    for kp in range(p):
                        s = 0
                        for c in range(rank):
                            s += U[ip * m + j][c] * V[j * p + kp][c] * W[k * n + i][c]
                        expected = (1 if ip == i and kp == k else 0)
                        if s != expected:
                            return False
    return True


def _build_strassen_2x2x2_terms():
    """
    Return list of (u_vec, v_vec, w_vec) for Strassen 2x2x2.
    Each vec is length 4, indexed as: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    # Strassen 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    terms = []
    # M1: (A00+A11)(B00+B11)
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))  # C00 gets +1, C11 gets +1
    # Actually W is indexed by (k,i) -> k*2+i
    # C00: i=0,k=0 -> W[0*2+0]=W[0]
    # C01: i=0,k=1 -> W[1*2+0]=W[2]
    # C10: i=1,k=0 -> W[0*2+1]=W[1]
    # C11: i=1,k=1 -> W[1*2+1]=W[3]
    
    # Let me redo with correct W indexing: W[k*2+i]
    # M1 contributes to C00 (+1) and C11 (+1)
    terms = []
    # M1 = (A00+A11)(B00+B11) -> C00+=M1, C11+=M1
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    # M2 = (A10+A11)B00 -> C10+=M2, C11-=M2
    terms.append(([0,0,1,1], [1,0,0,0], [0,1,0,-1]))
    # M3 = A00(B10-B11) -> C01+=M3, C11+=M3
    terms.append(([1,0,0,0], [0,0,1,-1], [0,0,1,1]))
    # M4 = A11(B01-B10) -> C00+=M4, C10+=M4
    terms.append(([0,0,0,1], [0,1,-1,0], [1,1,0,0]))
    # M5 = (A00+A01)B11 -> C00-=M5, C01+=M5
    terms.append(([1,1,0,0], [0,0,0,1], [-1,0,1,0]))
    # M6 = (A10-A00)(B00+B10) -> (not used in standard Strassen for C, it's an alternative)
    # Actually standard Strassen doesn't use M6,M7 for the C formula above.
    # Let me use the standard 7:
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    # M6 = (A10-A00)(B00+B10) is not in this version
    # M7 = (A01-A11)(B01+B11)
    terms.append(([0,1,-1,0], [0,1,0,1], [0,-1,0,1]))  # M7: C00-=M7, C11+=M7
    
    return terms


def _build_2x2x1_terms():
    """
    Return list of (u_vec, v_vec, w_vec) for 2x2x1 schoolbook.
    U indexed (i,j) -> i*2+j, V indexed (j,k) -> j*2+k (k only 0), W indexed (k,i) -> k*2+i (k only 0)
    So U: length 4, V: length 2 (j=0,1 with k=0), W: length 2 (k=0, i=0,1)
    """
    terms = []
    for i in range(2):
        for j in range(2):
            u = [0]*4
            v = [0]*2
            w = [0]*2
            u[i*2+j] = 1
            v[j] = 1  # j*2+0 = j
            w[i] = 1  # 0*2+i = i
            terms.append((u, v, w))
    return terms


def _build_2x1x9_terms():
    """
    Return list of (u_vec, v_vec, w_vec) for 2x1x9 schoolbook.
    U indexed (i,j) -> i*2+j, j only 0, so length 2
    V indexed (j,k) -> j*9+k, j only 0, so length 9
    W indexed (k,i) -> k*2+i, length 18
    """
    terms = []
    for i in range(2):
        for k in range(9):
            u = [0]*2
            v = [0]*9
            w = [0]*18
            u[i] = 1
            v[k] = 1
            w[k*2+i] = 1
            terms.append((u, v, w))
    return terms


def _assemble_full_decomp(n, m, p):
    """
    Assemble full 2x3x9 decomposition using block structure.
    
    Split: A = [A_left(2x2), A_right(2x1)], B = [B_top(2x9); B_bottom(1x9)]
    
    Part 1: A_left @ B_top (2x2 x 2x9)
      Split B_top into 3 blocks of 2x3: columns 0-2, 3-5, 6-8
      Each 2x2 x 2x3 = Strassen(2x2x2) + schoolbook(2x2x1) = 7 + 4 = 11 terms
      Total: 33 terms
    
    Part 2: A_right @ B_bottom (2x1 x 1x9)
      Schoolbook: 18 terms
    
    Total: 51 terms
    """
    n, m, p = 2, 3, 9
    
    # Full matrix dimensions for U, V, W
    U_dim = n * m  # 6
    V_dim = m * p  # 27
    W_dim = p * n  # 18
    
    # Collect all terms as (u_full, v_full, w_full)
    all_terms = []
    
    # --- Part 1: A_left (cols 0,1) @ B_top (rows 0,1) ---
    # For each of 3 column blocks of B_top (3 cols each)
    for block in range(3):
        k_offset = block * 3  # starting column in B_top
        
        # Sub-part 1a: 2x2 x 2x2 (columns k_offset, k_offset+1)
        strassen_terms = _build_strassen_2x2x2_terms()
        for (u_s, v_s, w_s) in strassen_terms:
            u_full = [0] * U_dim
            v_full = [0] * V_dim
            w_full = [0] * W_dim
            # u_s indexed by (i,j) with i in {0,1}, j in {0,1} -> global (i, j) since j_offset=0
            for idx, val in enumerate(u_s):
                i, j = idx // 2, idx % 2
                u_full[i * m + j] = val
            # v_s indexed by (j,k) with j in {0,1}, k in {0,1} -> global (j, k_offset+k)
            for idx, val in enumerate(v_s):
                j, k = idx // 2, idx % 2
                v_full[j * p + (k_offset + k)] = val
            # w_s indexed by (k,i) with k in {0,1}, i in {0,1} -> global (k_offset+k, i)
            for idx, val in enumerate(w_s):
                k, i = idx // 2, idx % 2
                w_full[(k_offset + k) * n + i] = val
            all_terms.append((u_full, v_full, w_full))
        
        # Sub-part 1b: 2x2 x 2x1 (column k_offset+2)
        sb_terms = _build_2x2x1_terms()
        for (u_sb, v_sb, w_sb) in sb_terms:
            u_full = [0] * U_dim
            v_full = [0] * V_dim
            w_full = [0] * W_dim
            # u_sb indexed by (i,j) with i in {0,1}, j in {0,1}
            for idx, val in enumerate(u_sb):
                i, j = idx // 2, idx % 2
                u_full[i * m + j] = val
            # v_sb indexed by (j,0) -> global (j, k_offset+2)
            for idx, val in enumerate(v_sb):
                j = idx
                v_full[j * p + (k_offset + 2)] = val
            # w_sb indexed by (0,i) -> global (k_offset+2, i)
            for idx, val in enumerate(w_sb):
                i = idx
                w_full[(k_offset + 2) * n + i] = val
            all_terms.append((u_full, v_full, w_full))
    
    # --- Part 2: A_right (col 2) @ B_bottom (row 2) ---
    outer_terms = _build_2x1x9_terms()
    for (u_o, v_o, w_o) in outer_terms:
        u_full = [0] * U_dim
        v_full = [0] * V_dim
        w_full = [0] * W_dim
        # u_o indexed by i in {0,1}, j=0 -> global (i, 2)
        for idx, val in enumerate(u_o):
            i = idx
            u_full[i * m + 2] = val
        # v_o indexed by j=0, k in {0..8} -> global (2, k)
        for idx, val in enumerate(v_o):
            k = idx
            v_full[2 * p + k] = val
        # w_o indexed by (k,i) -> global (k, i)
        for idx, val in enumerate(w_o):
            k, i = idx // 2, idx % 2
            w_full[k * n + i] = val
        all_terms.append((u_full, v_full, w_full))
    
    # Convert to matrices
    rank = len(all_terms)
    U = [[0] * rank for _ in range(U_dim)]
    V = [[0] * rank for _ in range(V_dim)]
    W = [[0] * rank for _ in range(W_dim)]
    
    for c, (u, v, w) in enumerate(all_terms):
        for r in range(U_dim):
            U[r][c] = u[r]
        for r in range(V_dim):
            V[r][c] = v[r]
        for r in range(W_dim):
            W[r][c] = w[r]
    
    return U, V, W


def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Build the block decomposition (rank 51)
    U, V, W = _assemble_full_decomp(n, m, p)
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
