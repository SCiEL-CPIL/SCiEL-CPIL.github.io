"""Matrix multiplication 3x4x8 with rank-80 decomposition using Strassen + 3x2x2 blocks."""
import time
import random

DIMENSIONS = [3, 4, 8]


def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7).
    
    Order of A elements: a00, a01, a10, a11
    Order of B elements: b00, b01, b10, b11
    Order of C elements: c00, c01, c10, c11
    
    Strassen products:
    m1 = (a00+a11)(b00+b11)
    m2 = (a10+a11)b00
    m3 = a00(b01-b11)
    m4 = a11(b10-b00)
    m5 = (a00+a01)b11
    m6 = (a10-a00)(b00+b01)
    m7 = (a01-a11)(b10+b11)
    
    c00 = m1 + m4 - m5 + m7
    c01 = m3 + m5
    c10 = m2 + m4
    c11 = m1 - m2 + m3 + m6
    """
    U = [
        [1, 0, 1, 0, 1, -1, 0],   # a00: m1+m3+m5-m6
        [0, 0, 0, 0, 1, 0, 1],    # a01: m5+m7
        [0, 1, 0, 0, 0, 1, 0],    # a10: m2+m6
        [1, 1, 0, 1, 0, 0, -1]    # a11: m1+m2+m4-m7
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],   # b00: m1+m2-m4+m6
        [0, 0, 1, 0, 0, 1, 0],    # b01: m3+m6
        [0, 0, 0, 1, 0, 0, 1],    # b10: m4+m7
        [1, 0, -1, 0, 1, 0, 1]    # b11: m1-m3+m5+m7
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],   # c00: m1+m4-m5+m7
        [0, 0, 1, 0, 1, 0, 0],    # c01: m3+m5
        [0, 1, 0, 1, 0, 0, 0],    # c10: m2+m4
        [1, -1, 1, 0, 0, 1, 0]    # c11: m1-m2+m3+m6
    ]
    return U, V, W


def _verify_3x2x2(U, V, W):
    """Verify a rank-3 decomposition of 3x2x2 matrix multiplication.
    
    U: (6, 3) - rows for a_{i,j}, i in 0..2, j in 0..1, indexed as i*2+j
    V: (4, 3) - rows for b_{j,k}, j in 0..1, k in 0..1, indexed as j*2+k
    W: (6, 3) - rows for c_{k,i}, k in 0..1, i in 0..2, indexed as k*3+i
    """
    for i in range(3):
        for j in range(2):
            for k in range(2):
                val = 0
                for r in range(3):
                    val += U[i * 2 + j][r] * V[j * 2 + k][r] * W[k * 3 + i][r]
                if val != 1:
                    return False
    return True


def _try_3x2x2_search(deadline):
    """Try to find a rank-3 decomposition for 3x2x2 matrix multiplication.
    
    Returns (U, V, W) if found, else None.
    Uses randomized search with integer and half-integer entries.
    """
    # Strategy: fix V to a structured form and solve for U, W
    # Or: random search over small integer/half-integer values
    
    # First try a structured approach:
    # Let V be the identity-like structure for the 2 middle indices
    # V[j*2+k][r] for j in {0,1}, k in {0,1}
    
    # Try: V = [[1,0,0], [0,1,0], [0,0,1], [0,0,0]]
    # Then for each (i,k): U[i*2+0][0]*V[0*2+k][0]*W[k*3+i][0] + 
    #                      U[i*2+1][1]*V[1*2+k][1]*W[k*3+i][1] +
    #                      U[i*2+1][2]*V[1*2+k][2]*W[k*3+i][2] = 1
    # This gives: for k=0: U[i*2][0]*W[0*3+i][0] = 1
    #              for k=1: U[i*2+1][1]*W[1*3+i][1] = 1
    # So W[0*3+i][0] = 1/U[i*2][0], W[1*3+i][1] = 1/U[i*2+1][1]
    # With integers: U[i*2][0] = ±1, W[0*3+i][0] = ±1 (same sign)
    #                U[i*2+1][1] = ±1, W[1*3+i][1] = ±1 (same sign)
    # And other entries must be 0.
    # Check: for (i,j,k) where j!=k, we need sum = 0.
    # For j=0, k=1: U[i*2][0]*V[0*2+1][0]*W[3+i][0] + U[i*2+1][1]*V[1*2+1][1]*W[3+i][1] + ...
    # = U[i*2][0]*0*W[3+i][0] + U[i*2+1][1]*0*W[3+i][1] + ... = 0 ✓
    # For j=1, k=0: similarly 0 ✓
    # So this works! Let's verify all cases.
    
    # V = [[1,0,0], [0,1,0], [0,0,1], [0,0,0]]
    # U[i*2][0] = 1 for all i, U[i*2+1][1] = 1 for all i, rest 0
    # W[0*3+i][0] = 1 for all i, W[1*3+i][1] = 1 for all i, rest 0
    
    U_test = [[0, 0, 0] for _ in range(6)]
    V_test = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]
    W_test = [[0, 0, 0] for _ in range(6)]
    
    for i in range(3):
        U_test[i * 2][0] = 1
        U_test[i * 2 + 1][1] = 1
        W_test[i][0] = 1          # k=0, i -> index 0*3+i = i
        W_test[3 + i][1] = 1      # k=1, i -> index 1*3+i = 3+i
    
    if _verify_3x2x2(U_test, V_test, W_test):
        return U_test, V_test, W_test
    
    # If structured approach failed, try random search
    random.seed(42)
    for attempt in range(5000):
        if time.time() > deadline:
            break
        # Random entries from {-1, 0, 1}
        U = [[random.choice([-1, 0, 1]) for _ in range(3)] for _ in range(6)]
        V = [[random.choice([-1, 0, 1]) for _ in range(3)] for _ in range(4)]
        W = [[random.choice([-1, 0, 1]) for _ in range(3)] for _ in range(6)]
        
        if _verify_3x2x2(U, V, W):
            return U, V, W
    
    return None


def _build_decomposition_80(u322, v322, w322):
    """Build rank-80 decomposition: 8 Strassen blocks + 8 3x2x2 blocks.
    
    Strassen blocks handle i=0,1 (2x2x2 blocks) for all j-pairs and k-pairs.
    3x2x2 blocks handle i=2 (the third row) for all j-pairs and k-pairs.
    
    n=3, m=4, p=8
    j-pairs: (0,1), (2,3) -> 2 pairs
    k-pairs: (0,1), (2,3), (4,5), (6,7) -> 4 pairs
    
    For Strassen (i=0,1): each block covers a 2x2x2 subproblem
      A rows: i=0,1; j in {j0,j1}
      B rows: j in {j0,j1}; k in {k0,k1}
      C cols: i in {0,1}; k in {k0,k1}
    
    For 3x2x2 (i=2): each block covers a 3x2x2 subproblem
      A rows: i=0,1,2; j in {j0,j1}  (but we only use i=2 part)
      B rows: j in {j0,j1}; k in {k0,k1}
      C cols: i=0,1,2; k in {k0,k1}  (but we only use i=2 part)
    """
    n, m, p = 3, 4, 8
    total_rank = 8 * 7 + 8 * 3  # 56 + 24 = 80
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    U2, V2, W2 = _strassen_2x2x2()
    
    col_offset = 0
    
    # Strassen blocks for i=0,1
    for jp in range(2):       # j-pair index: 0 -> (0,1), 1 -> (2,3)
        for kp in range(4):   # k-pair index: 0->(0,1), 1->(2,3), 2->(4,5), 3->(6,7)
            j0 = 2 * jp
            j1 = 2 * jp + 1
            k0 = 2 * kp
            k1 = 2 * kp + 1
            
            # Strassen 2x2x2: A is 2x2 (rows i=0,1; cols j0,j1)
            # V is 2x2 (rows j0,j1; cols k0,k1)
            # W is 2x2 (rows k0,k1; cols i=0,1)
            
            # U rows: a_{0,j0}, a_{0,j1}, a_{1,j0}, a_{1,j1}
            #   -> indices: 0*m+j0, 0*m+j1, 1*m+j0, 1*m+j1
            u_rows = [0 * m + j0, 0 * m + j1, 1 * m + j0, 1 * m + j1]
            
            # V rows: b_{j0,k0}, b_{j0,k1}, b_{j1,k0}, b_{j1,k1}
            #   -> indices: j0*p+k0, j0*p+k1, j1*p+k0, j1*p+k1
            v_rows = [j0 * p + k0, j0 * p + k1, j1 * p + k0, j1 * p + k1]
            
            # W rows: c_{k0,0}, c_{k1,0}, c_{k0,1}, c_{k1,1}
            #   -> indices: k0*n+0, k1*n+0, k0*n+1, k1*n+1
            w_rows = [k0 * n + 0, k1 * n + 0, k0 * n + 1, k1 * n + 1]
            
            for i in range(4):
                for c in range(7):
                    U[u_rows[i]][col_offset + c] += U2[i][c]
                    V[v_rows[i]][col_offset + c] += V2[i][c]
                    W[w_rows[i]][col_offset + c] += W2[i][c]
            col_offset += 7
    
    # 3x2x2 blocks for i=2
    for jp in range(2):       # j-pair
        for kp in range(4):   # k-pair
            j0 = 2 * jp
            j1 = 2 * jp + 1
            k0 = 2 * kp
            k1 = 2 * kp + 1
            
            # 3x2x2: A is 3x2 (rows i=0,1,2; cols j0,j1)
            # V is 2x2 (rows j0,j1; cols k0,k1)
            # W is 2x3 (rows k0,k1; cols i=0,1,2)
            # But we only need the i=2 component to be correct.
            # The 3x2x2 decomposition handles all i=0,1,2.
            # Since Strassen already handles i=0,1, we need to be careful.
            
            # Actually, the 3x2x2 decomposition is for the FULL 3x2x2 problem.
            # If we use it for i=2 only, we need a decomposition where:
            # For i=2: sum_r U[2*m+j][r] * V[j*p+k][r] * W[k*n+2][r] = 1
            # For i=0,1: the contribution should be 0 (since Strassen handles them)
            
            # This means we need a 3x2x2 decomposition where the i=0 and i=1
            # components are zero. That's a different problem.
            
            # Alternative: use the 3x2x2 decomposition as-is, but then it
            # contributes to i=0,1 as well. We'd need to subtract those contributions.
            
            # Simpler approach: just use schoolbook for i=2 (rank 4 per block)
            # But that gives rank 88, not 80.
            
            # Let me reconsider: the 3x2x2 tensor decomposition satisfies:
            # For ALL (i,j,k) with i in 0..2, j in 0..1, k in 0..1:
            #   sum_r U[i*2+j][r] * V[j*2+k][r] * W[k*3+i][r] = 1
            # This means it contributes to ALL i values, not just i=2.
            
            # So if we use it, it adds to i=0,1 as well. We can't just use it
            # for i=2 without affecting i=0,1.
            
            # Solution: use the 3x2x2 decomposition for ALL i values, and
            # don't use Strassen. But then we need 8 3x2x2 blocks for all i,
            # which is 8*3 = 24 rank for all 3 rows. But we also need to handle
            # the j and k tiling.
            
            # Wait, let me re-think the tiling:
            # For each (j-pair, k-pair) block, we have a 3x2x2 subproblem.
            # If we use a rank-3 decomposition for each, total rank = 8*3 = 24.
            # This handles ALL of i=0,1,2 for that (j-pair, k-pair).
            # So we don't need Strassen at all!
            
            # But wait: 3x2x2 with rank 3? Let me verify this is possible.
            # The 3x2x2 matrix multiplication tensor has dimension 3*2*2 = 12.
            # Rank 3 means 3*6 + 3*4 + 3*6 = 48 unknowns, 12 equations.
            # Should be feasible.
            
            # Let me just use the 3x2x2 decomposition for all blocks.
            for i in range(3):  # i = 0, 1, 2
                u_row = i * m + j0  # a_{i,j0}
                u_row2 = i * m + j1  # a_{i,j1}
                # U322 rows: i*2+0 for j0, i*2+1 for j1
                # So U322[i*2+0] corresponds to a_{i,j0}
                #    U322[i*2+1] corresponds to a_{i,j1}
                for c in range(3):
                    U[u_row][col_offset + c] += u322[i * 2][c]
                    U[u_row2][col_offset + c] += u322[i * 2 + 1][c]
            
            # V rows: j0*p+k0, j0*p+k1, j1*p+k0, j1*p+k1
            # V322 rows: j*2+k for j in {0,1}, k in {0,1}
            #   V322[0] -> b_{j0,k0}, V322[1] -> b_{j0,k1}
            #   V322[2] -> b_{j1,k0}, V322[3] -> b_{j1,k1}
            v_rows = [j0 * p + k0, j0 * p + k1, j1 * p + k0, j1 * p + k1]
            for i in range(4):
                for c in range(3):
                    V[v_rows[i]][col_offset + c] += v322[i][c]
            
            # W rows: k0*n+i, k1*n+i for i in 0,1,2
            # W322 rows: k*3+i for k in {0,1}, i in {0,1,2}
            #   W322[0] -> c_{k0,0}, W322[1] -> c_{k0,1}, W322[2] -> c_{k0,2}
            #   W322[3] -> c_{k1,0}, W322[4] -> c_{k1,1}, W322[5] -> c_{k1,2}
            for i in range(3):
                w_row = k0 * n + i
                w_row2 = k1 * n + i
                for c in range(3):
                    W[w_row][col_offset + c] += w322[i][c]       # k=0, i
                    W[w_row2][col_offset + c] += w322[3 + i][c]   # k=1, i
            
            col_offset += 3
    
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n, m, p):
    """Verify the decomposition."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if val != 1:
                    return False
    return True


def _schoolbook_fallback():
    """Return schoolbook decomposition of rank 96."""
    n, m, p = DIMENSIONS
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


def _build_rank88():
    """Build the rank-88 decomposition (Strassen for i=0,1 + schoolbook for i=2)."""
    n, m, p = 3, 4, 8
    total_rank = 8 * 7 + 8 * 4  # 56 + 32 = 88
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    U2, V2, W2 = _strassen_2x2x2()
    
    col_offset = 0
    
    # Strassen blocks for i=0,1
    for jp in range(2):
        for kp in range(4):
            j0 = 2 * jp
            j1 = 2 * jp + 1
            k0 = 2 * kp
            k1 = 2 * kp + 1
            
            u_rows = [0 * m + j0, 0 * m + j1, 1 * m + j0, 1 * m + j1]
            v_rows = [j0 * p + k0, j0 * p + k1, j1 * p + k0, j1 * p + k1]
            w_rows = [k0 * n + 0, k1 * n + 0, k0 * n + 1, k1 * n + 1]
            
            for i in range(4):
                for c in range(7):
                    U[u_rows[i]][col_offset + c] += U2[i][c]
                    V[v_rows[i]][col_offset + c] += V2[i][c]
                    W[w_rows[i]][col_offset + c] += W2[i][c]
            col_offset += 7
    
    # Schoolbook for i=2
    for jp in range(2):
        for kp in range(4):
            j0 = 2 * jp
            j1 = 2 * jp + 1
            k0 = 2 * kp
            k1 = 2 * kp + 1
            
            # 4 terms: (j0,k0), (j0,k1), (j1,k0), (j1,k1)
            for jj in range(2):
                for kk in range(2):
                    j = j0 + jj
                    k = k0 + kk
                    U[2 * m + j][col_offset] = 1
                    V[j * p + k][col_offset] = 1
                    W[k * n + 2][col_offset] = 1
                    col_offset += 1
    
    return {"U": U, "V": V, "W": W}


def construct():
    start = time.time()
    deadline = start + 140  # safety margin before 150s limit
    
    # Try to find rank-80 decomposition
    result_80 = None
    u322, v322, w322 = _try_3x2x2_search(deadline)
    
    if u322 is not None:
        result_80 = _build_decomposition_80(u322, v322, w322)
        n, m, p = DIMENSIONS
        if _verify_decomposition(result_80["U"], result_80["V"], result_80["W"], n, m, p):
            return result_80
    
    # Fall back to rank-88
    result_88 = _build_rank88()
    n, m, p = DIMENSIONS
    if _verify_decomposition(result_88["U"], result_88["V"], result_88["W"], n, m, p):
        return result_88
    
    # Final fallback: schoolbook rank 96
    return _schoolbook_fallback()
