"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Check if U, V, W form an exact decomposition of the matrix multiplication tensor."""
    try:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = U_np.shape[1]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U_np[i*m + j, c] * V_np[j*p + k, c] * W_np[k*n + i, c]
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True
    except Exception:
        return False

def _schoolbook_decomp(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def _block_decomp_2x3x3():
    """
    Rank-9 decomposition for 2x3 @ 3x3.
    
    Group A entries: a0 = A[0,0]+A[0,2], a1 = A[1,0]+A[1,2], a2 = A[0,1], a3 = A[1,1]
    Wait, that's 4 groups. Let me use 3 groups instead:
    a0 = A[0,0], a1 = A[0,2], a2 = A[1,0]+A[1,2]  -- no, this doesn't work symmetrically.
    
    Better: use 3 A-groups and 3 B-groups.
    A-groups: g0 = A[i,0], g1 = A[i,1], g2 = A[i,2]  (just the original, 3 per row)
    But that gives 2*3*3 = 18.
    
    To get rank 9, I need 3 A-combinations and 3 B-combinations.
    A-combinations (per row i): 
      A0_i = A[i,0]
      A1_i = A[i,1]  
      A2_i = A[i,2]
    That's 3 per row, 6 total.
    
    B-combinations (per col k):
      B0_k = B[0,k]
      B1_k = B[1,k]
      B2_k = B[2,k]
    That's 3 per col, 9 total.
    
    Total rank = 6 * 9 / 3 = 18? No, that's not right.
    
    Actually, the rank-1 term for (A-group a, B-group b) contributes:
    sum_{i,j,k} A_group[a][i,j] * B_group[b][j,k] * W[k,i]
    
    If A_group[a] has non-zero entries in rows i and columns j, and B_group[b] has non-zero in rows j and cols k, then the product is non-zero only for (i,j,k) where both are non-zero.
    
    For rank 9, I need 9 rank-1 terms. Let me try:
    A-groups (3 total, each is a 2x3 matrix):
      G0 = [[1,0,0],[0,1,0]]  -> A[0,0] and A[1,1]
      G1 = [[0,1,0],[0,0,1]]  -> A[0,1] and A[1,2]
      G2 = [[0,0,1],[1,0,0]]  -> A[0,2] and A[1,0]
    
    B-groups (3 total, each is a 3x3 matrix):
      H0 = [[1,0,0],[0,1,0],[0,0,1]]  -> B[0,0],B[1,1],B[2,2]
      H1 = [[0,1,0],[0,0,1],[1,0,0]]  -> B[0,1],B[1,2],B[2,0]
      H2 = [[0,0,1],[1,0,0],[0,1,0]]  -> B[0,2],B[1,0],B[2,1]
    
    Then the 9 rank-1 terms are (Ga, Hb) for a,b in {0,1,2}.
    Each term contributes to (i,j,k) if Ga[i,j] != 0 and Hb[j,k] != 0.
    
    Let me check: does every (i,j,k) appear in exactly one term?
    (i,j,k) appears in term (a,b) if Ga[i,j]=1 and Hb[j,k]=1.
    
    For fixed (j,k), Hb[j,k]=1 for exactly one b (since each Hb has exactly 3 ones, and they cover all 9 positions).
    For fixed (i,j), Ga[i,j]=1 for exactly one a (since each Ga has exactly 2 ones, and together they cover all 6 positions).
    
    So each (i,j,k) appears in exactly one term, with coefficient 1*1*W[k,i].
    We need W[k,i] = 1 for all (k,i).
    
    So W is all ones: W[k*n+i] = 1 for all k,i.
    """
    n, m, p = 2, 3, 3
    rank = 9
    
    # A-groups: 3 groups, each is a 2x3 pattern
    # G0: (0,0) and (1,1)
    # G1: (0,1) and (1,2)
    # G2: (0,2) and (1,0)
    U = [[0] * rank for _ in range(n * m)]
    
    for a in range(3):
        for b in range(3):
            col = a * 3 + b
            # Set U for the two non-zero positions in Ga
            if a == 0:
                U[0 * m + 0][col] = 1  # A[0,0]
                U[1 * m + 1][col] = 1  # A[1,1]
            elif a == 1:
                U[0 * m + 1][col] = 1  # A[0,1]
                U[1 * m + 2][col] = 1  # A[1,2]
            else:  # a == 2
                U[0 * m + 2][col] = 1  # A[0,2]
                U[1 * m + 0][col] = 1  # A[1,0]
    
    # B-groups: 3 groups, each is a 3x3 pattern
    # H0: (0,0),(1,1),(2,2)
    # H1: (0,1),(1,2),(2,0)
    # H2: (0,2),(1,0),(2,1)
    V = [[0] * rank for _ in range(m * p)]
    
    for a in range(3):
        for b in range(3):
            col = a * 3 + b
            if b == 0:
                V[0 * p + 0][col] = 1  # B[0,0]
                V[1 * p + 1][col] = 1  # B[1,1]
                V[2 * p + 2][col] = 1  # B[2,2]
            elif b == 1:
                V[0 * p + 1][col] = 1  # B[0,1]
                V[1 * p + 2][col] = 1  # B[1,2]
                V[2 * p + 0][col] = 1  # B[2,0]
            else:  # b == 2
                V[0 * p + 2][col] = 1  # B[0,2]
                V[1 * p + 0][col] = 1  # B[1,0]
                V[2 * p + 1][col] = 1  # B[2,1]
    
    # W: all ones
    W = [[1] * rank for _ in range(p * n)]
    
    return U, V, W

def _block_decomp_2x3x4():
    """
    Rank-12 decomposition for 2x3 @ 3x4.
    
    Use 3 A-groups and 4 B-groups.
    A-groups (same as before):
      G0: (0,0) and (1,1)
      G1: (0,1) and (1,2)
      G2: (0,2) and (1,0)
    
    B-groups: 4 groups, each is a 3x4 pattern.
    Need each (j,k) to appear in exactly one group.
    3*4 = 12 positions, 4 groups of 3 each.
    
    H0: (0,0),(1,1),(2,2)
    H1: (0,1),(1,2),(2,3)
    H2: (0,2),(1,3),(2,0)
    H3: (0,3),(1,0),(2,1)
    
    Check coverage:
    Row 0: k=0(H0), k=1(H1), k=2(H2), k=3(H3) ✓
    Row 1: k=0(H3), k=1(H0), k=2(H1), k=3(H2) ✓
    Row 2: k=0(H2), k=1(H3), k=2(H0), k=3(H1) ✓
    """
    n, m, p = 2, 3, 4
    rank = 12
    
    U = [[0] * rank for _ in range(n * m)]
    
    for a in range(3):
        for b in range(4):
            col = a * 4 + b
            if a == 0:
                U[0 * m + 0][col] = 1
                U[1 * m + 1][col] = 1
            elif a == 1:
                U[0 * m + 1][col] = 1
                U[1 * m + 2][col] = 1
            else:
                U[0 * m + 2][col] = 1
                U[1 * m + 0][col] = 1
    
    V = [[0] * rank for _ in range(m * p)]
    
    for a in range(3):
        for b in range(4):
            col = a * 4 + b
            if b == 0:
                V[0 * p + 0][col] = 1
                V[1 * p + 1][col] = 1
                V[2 * p + 2][col] = 1
            elif b == 1:
                V[0 * p + 1][col] = 1
                V[1 * p + 2][col] = 1
                V[2 * p + 3][col] = 1
            elif b == 2:
                V[0 * p + 2][col] = 1
                V[1 * p + 3][col] = 1
                V[2 * p + 0][col] = 1
            else:  # b == 3
                V[0 * p + 3][col] = 1
                V[1 * p + 0][col] = 1
                V[2 * p + 1][col] = 1
    
    W = [[1] * rank for _ in range(p * n)]
    
    return U, V, W

def _combine_block_decomps(decomp1, decomp2, n, m, p1, p2):
    """
    Combine two decompositions for 2x3@3xp1 and 2x3@3xp2 into one for 2x3@3x(p1+p2).
    """
    U1, V1, W1 = decomp1
    U2, V2, W2 = decomp2
    rank1 = len(U1[0]) if U1 else 0
    rank2 = len(U2[0]) if U2 else 0
    total_rank = rank1 + rank2
    p = p1 + p2
    
    U = []
    for i in range(n * m):
        row = U1[i] + U2[i]
        U.append(row)
    
    V = []
    for j in range(m * p):
        row = [0] * total_rank
        if j < m * p1:
            j1 = j
            for c in range(rank1):
                row[c] = V1[j1][c]
        else:
            j2 = j - m * p1
            for c in range(rank2):
                row[rank1 + c] = V2[j2][c]
        V.append(row)
    
    W = []
    for k in range(p * n):
        row = [0] * total_rank
        if k < p1 * n:
            k1 = k
            for c in range(rank1):
                row[c] = W1[k1][c]
        else:
            k2 = k - p1 * n
            for c in range(rank2):
                row[rank1 + c] = W2[k2][c]
        W.append(row)
    
    return U, V, W

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x7 matrix.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 145
    
    # Start with schoolbook decomposition (guaranteed valid, rank 42)
    U_sb, V_sb, W_sb = _schoolbook_decomp(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p  # 42
    
    start_time = time.time()
    
    # Strategy: split p=7 into p1=3 and p2=4
    p1, p2 = 3, 4
    
    # Get the block decompositions
    decomp1 = _block_decomp_2x3x3()
    decomp2 = _block_decomp_2x3x4()
    
    # Verify the individual blocks
    if not _verify_decomposition(decomp1[0], decomp1[1], decomp1[2], n, m, p1):
        decomp1 = _schoolbook_decomp(n, m, p1)
    if not _verify_decomposition(decomp2[0], decomp2[1], decomp2[2], n, m, p2):
        decomp2 = _schoolbook_decomp(n, m, p2)
    
    # Combine the two decompositions
    U_comb, V_comb, W_comb = _combine_block_decomps(decomp1, decomp2, n, m, p1, p2)
    
    # Verify the combined decomposition
    if _verify_decomposition(U_comb, V_comb, W_comb, n, m, p):
        comb_rank = len(U_comb[0]) if U_comb else 0
        if comb_rank < best_rank:
            best_U, best_V, best_W = U_comb, V_comb, W_comb
            best_rank = comb_rank
    
    # Try to improve further with random search
    for target_rank in range(best_rank - 1, max(n*m, m*p, p*n) - 1, -1):
        if time.time() - start_time > deadline - 15:
            break
        
        found = False
        for attempt in range(100):
            if time.time() - start_time > deadline - 15:
                break
            
            rng = np.random.RandomState(hash((target_rank, attempt)) % (2**32))
            
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            for c in range(target_rank):
                i = rng.randint(0, n)
                j = rng.randint(0, m)
                k = rng.randint(0, p)
                U[i * m + j, c] = rng.choice([-1.0, 1.0])
                V[j * p + k, c] = rng.choice([-1.0, 1.0])
                W[k * n + i, c] = rng.choice([-1.0, 1.0])
            
            if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                best_U, best_V, best_W = U.tolist(), V.tolist(), W.tolist()
                best_rank = target_rank
                found = True
                break
        
        if found:
            continue
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
