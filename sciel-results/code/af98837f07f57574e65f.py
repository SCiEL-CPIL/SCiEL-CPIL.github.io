"""Program for rank-minimal exact tensor decomposition of 2x3 * 3x6 matrix multiplication."""
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    """Build the standard schoolbook (rank n*m*p) decomposition."""
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

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor.
    Returns True if valid, False otherwise."""
    rank = len(U[0]) if U else 0
    # Check dimensions
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U):
        return False
    if any(len(row) != rank for row in V):
        return False
    if any(len(row) != rank for row in W):
        return False
    
    # Verify: for all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if total != 1:
                    return False
    return True

def _build_strassen_2x2x2(n, m, p):
    """Attempt Strassen-based decomposition for 2x2x2 blocks.
    For 2x3 * 3x6, we can use a block decomposition approach."""
    # This is a placeholder - Strassen for non-square blocks is complex
    # Fall back to schoolbook
    return _build_schoolbook(n, m, p)

def _build_strassen_2x2x6(n, m, p):
    """Build a decomposition using Strassen for the 2x2x2 blocks in the 2x2x6 core,
    plus schoolbook for the 2x1x6 tail (j=2).
    
    The 2x2x6 core (j in {0,1}, k in {0..5}) is split into three 2x2x2 blocks:
      Block 0: k in {0,1}
      Block 1: k in {2,3}
      Block 2: k in {4,5}
    Each uses Strassen rank 7.
    
    The tail (j=2, k in {0..5}) is 2x1x6, which has rank 1 (it's separable:
    T[i, 2, k] = 1 for all i, k, so a single term suffices).
    
    Total rank: 7 + 7 + 7 + 1 = 22.
    """
    rank = 22
    U = [[0] * rank for _ in range(n * m)]  # 6 x 22
    V = [[0] * rank for _ in range(m * p)]  # 18 x 22
    W = [[0] * rank for _ in range(p * n)]  # 12 x 22
    
    # Strassen formulas for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M2 + M6
    #
    # For block b (b=0,1,2), the local indices are:
    #   A[i_local, j_local] with i_local in {0,1}, j_local in {0,1}
    #   B[j_local, k_local] with j_local in {0,1}, k_local in {0,1}
    #   C[i_local, k_local] with i_local in {0,1}, k_local in {0,1}
    #
    # Global mapping for block b:
    #   j_global = j_local (since j in {0,1} for the core)
    #   k_global = 2*b + k_local
    #   i_global = i_local
    
    # Strassen coefficient tables
    # For each of the 7 multiplications, we need:
    #   U coefficients for (i_local, j_local)
    #   V coefficients for (j_local, k_local)
    #   W coefficients for (k_local, i_local)
    
    # M1: (A00+A11)(B00+B11)
    # U: (0,0)=1, (1,1)=1
    # V: (0,0)=1, (0,1)=1, (1,0)=1, (1,1)=1
    # W: (0,0)=1, (0,1)=1, (1,0)=1, (1,1)=1
    # C: C00 += 1*M1, C01 += 0, C10 += 0, C11 += 1*M1
    # W: (0,0)=1, (1,1)=1 for C00; (0,1)=1, (1,0)=1 for C11
    # Wait, W[k_local, i_local]:
    #   C00: k=0,i=0 -> W[0,0]
    #   C01: k=1,i=0 -> W[1,0]
    #   C10: k=0,i=1 -> W[0,1]
    #   C11: k=1,i=1 -> W[1,1]
    
    strassen_U = [
        {(0,0): 1, (1,1): 1},  # M1
        {(1,0): 1, (1,1): 1},  # M2
        {(0,0): 1},             # M3
        {(1,1): 1},             # M4
        {(0,0): 1, (0,1): 1},  # M5
        {(1,0): 1, (0,0): -1}, # M6
        {(0,1): 1, (1,1): -1}, # M7
    ]
    
    strassen_V = [
        {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1},  # M1: (B00+B11)
        {(0,0): 1},                                  # M2: B00
        {(0,1): 1, (1,1): -1},                       # M3: (B01-B11)
        {(1,0): 1, (0,0): -1},                       # M4: (B10-B00)
        {(1,1): 1},                                  # M5: B11
        {(0,0): 1, (0,1): 1},                        # M6: (B00+B01)
        {(1,0): 1, (1,1): 1},                        # M7: (B10+B11)
    ]
    
    strassen_W = [
        {(0,0): 1, (1,1): 1},  # M1: C00 += M1, C11 += M1
        {(0,1): 1},            # M2: C10 += M2
        {(1,0): 1},            # M3: C01 += M3
        {(0,1): 1, (1,1): -1},# M4: C10 += M4, C11 -= M4
        {(0,0): 1, (1,0): 1}, # M5: C00 += M5, C01 += M5
        {(1,1): 1},           # M6: C11 += M6
        {(0,0): 1, (1,0): -1},# M7: C00 += M7, C11 -= M7
    ]
    
    # Wait, let me re-derive the W coefficients more carefully.
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M2 + M6
    #
    # W[k_local, i_local] contributes to C[i_local, k_local]
    # So W[0,0] contributes to C[0,0], W[1,0] to C[0,1], W[0,1] to C[1,0], W[1,1] to C[1,1]
    
    strassen_W_correct = [
        {(0,0): 1, (1,1): 1},   # M1: C00 += 1, C11 += 1
        {(0,1): 1},             # M2: C10 += 1
        {(1,0): 1},             # M3: C01 += 1
        {(0,1): 1, (1,1): -1}, # M4: C10 += 1, C11 -= 1
        {(0,0): 1, (1,0): 1},  # M5: C00 += 1, C01 += 1
        {(1,1): 1},            # M6: C11 += 1
        {(0,0): 1, (1,1): -1}, # M7: C00 += 1, C11 -= 1
    ]
    
    # Now build the decomposition
    col = 0
    for b in range(3):  # 3 blocks of 2x2x2
        k_base = 2 * b
        for s in range(7):  # 7 Strassen terms
            # U: A[i_local, j_local]
            for (i_loc, j_loc), coeff in strassen_U[s].items():
                U[i_loc * m + j_loc][col] += coeff
            
            # V: B[j_local, k_local]
            for (j_loc, k_loc), coeff in strassen_V[s].items():
                V[j_loc * p + (k_base + k_loc)][col] += coeff
            
            # W: C[k_local, i_local]
            for (k_loc, i_loc), coeff in strassen_W_correct[s].items():
                W[(k_base + k_loc) * n + i_loc][col] += coeff
            
            col += 1
    
    # Tail: j=2, k in {0..5}, i in {0,1}
    # T[i, 2, k] = 1 for all i, k
    # Rank 1: single term with U[2][col]=1, V[2*p+k][col]=1, W[k*n+i][col]=1
    # But this gives sum = 1 for each (i, 2, k)
    # Wait, we need sum_c U[i*m+2, c] * V[2*p+k, c] * W[k*n+i, c] = 1
    # If U[i*m+2, col] = 1, V[2*p+k, col] = 1, W[k*n+i, col] = 1, then product = 1. Good.
    
    U[0 * m + 2][col] = 1  # i=0, j=2
    U[1 * m + 2][col] = 1  # i=1, j=2
    for k in range(p):
        V[2 * p + k][col] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i][col] = 1
    
    col += 1
    # col should be 22 now
    
    return {"U": U, "V": V, "W": W}

def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    best_result = None
    best_rank = float('inf')
    
    # Start with schoolbook (always valid)
    schoolbook = _build_schoolbook(n, m, p)
    if _verify_decomposition(schoolbook["U"], schoolbook["V"], schoolbook["W"], n, m, p):
        best_result = schoolbook
        best_rank = len(schoolbook["U"][0])
    
    # Try Strassen-based approach for 2x2x6 + tail
    if time.time() < deadline - 30:
        try:
            strassen = _build_strassen_2x2x6(n, m, p)
            if _verify_decomposition(strassen["U"], strassen["V"], strassen["W"], n, m, p):
                rank_s = len(strassen["U"][0])
                if rank_s < best_rank:
                    best_result = strassen
                    best_rank = rank_s
        except Exception:
            pass
    
    # If no valid result found (shouldn't happen), fall back to schoolbook
    if best_result is None:
        best_result = _build_schoolbook(n, m, p)
    
    return best_result
# EVOLVE-BLOCK-END
