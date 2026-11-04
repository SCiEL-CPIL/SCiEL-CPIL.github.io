"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]
import numpy as np
import time
import itertools
import random

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly decomposes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    U_arr = U.T  # (rank, n*m)
    V_arr = V.T  # (rank, m*p)
    W_arr = W.T  # (rank, p*n)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.sum(U_arr[:, a] * V_arr[:, b] * W_arr[:, c])
                if abs(s - 1.0) > tol:
                    return False
    
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                if j_a == j_b and i_a == i_c and k_b == k_c:
                    continue
                s = np.sum(U_arr[:, a] * V_arr[:, b] * W_arr[:, c])
                if abs(s) > tol:
                    return False
    return True

def _make_schoolbook(n, m, p):
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

def _make_strassen_2x5x5(n, m, p):
    """
    Construct a rank-46 decomposition for 2x5 * 5x5 using Strassen's algorithm.
    
    We partition the 5x5 B matrix into 2+2+1 blocks:
    B = [ B11 B12 B13 ]
        [ B21 B22 B23 ]
        [ B31 B32 B33 ]
    where B11, B22 are 2x2, B13, B23, B31, B32 are 2x1 or 1x2, and B33 is 1x1.
    
    The 2x2 blocks (B11, B22) are multiplied using Strassen's 7-multiplication algorithm.
    The 1x1 block (B33) uses schoolbook (1 multiplication).
    The mixed blocks (B13, B23, B31, B32) use schoolbook.
    
    For each of the 2 rows of A (i=0, 1):
    Row i of A is [ A_i1 A_i2 A_i3 ] where A_i1, A_i2 are 1x2 and A_i3 is 1x1.
    
    The product C_i = A_i * B has blocks:
    C_i1 = A_i1*B11 + A_i2*B21 + A_i3*B31  (1x2)
    C_i2 = A_i1*B12 + A_i2*B22 + A_i3*B32  (1x2)
    C_i3 = A_i1*B13 + A_i2*B23 + A_i3*B33  (1x1)
    
    Using Strassen for the 2x2 parts:
    For C_i1 and C_i2, we need A_i1*B11 and A_i2*B22 (and cross terms).
    Actually, let's think more carefully.
    
    For a 1x2 row vector times a 2x2 matrix, we can use a specialized Strassen-like approach.
    But it's simpler to just use the standard Strassen for 2x2 * 2x2 where applicable.
    
    Let's use a cleaner construction:
    For each row i of A (i=0,1), we compute C_i = A_i * B.
    
    A_i = [a1, a2, a3] where a1, a2 are scalars? No, A is 2x5, so each row has 5 elements.
    A_i = [A_i1, A_i2, A_i3, A_i4, A_i5]
    B is 5x5.
    
    Let's partition indices 0..4 into {0,1}, {2,3}, {4}.
    A_i = [A_i,01, A_i,23, A_i,4] where A_i,01 is 1x2, A_i,23 is 1x2, A_i,4 is scalar.
    B = [ B,01,01  B,01,23  B,01,4 ]
        [ B,23,01  B,23,23  B,23,4 ]
        [ B,4,01   B,4,23   B,4,4   ]
    
    C_i,01 = A_i,01*B,01,01 + A_i,23*B,23,01 + A_i,4*B,4,01
    C_i,23 = A_i,01*B,01,23 + A_i,23*B,23,23 + A_i,4*B,4,23
    C_i,4  = A_i,01*B,01,4 + A_i,23*B,23,4 + A_i,4*B,4,4
    
    For the 2x2 * 2x2 parts (A_i,01 * B,01,01 and A_i,23 * B,23,23), we can use Strassen.
    But A_i,01 is 1x2, not 2x2. So we need a 1x2 * 2x2 algorithm.
    
    Actually, let's use a simpler but still effective approach:
    Use Strassen for the 2x2 sub-blocks of B, treating the A row as having 2x2 blocks too.
    
    Let A be 2x5. Pad to 2x6? No, let's just use the 2+2+1 partition.
    
    For each row i of A:
    A_i,01 = [A[i,0], A[i,1]]  (1x2)
    A_i,23 = [A[i,2], A[i,3]]  (1x2)
    A_i,4  = A[i,4]            (scalar)
    
    B,01,01 = B[0:2, 0:2]  (2x2)
    B,23,23 = B[2:4, 2:4]  (2x2)
    B,4,4   = B[4,4]       (scalar)
    
    Cross terms:
    B,01,23 = B[0:2, 2:4]  (2x2)
    B,23,01 = B[2:4, 0:2]  (2x2)
    B,01,4  = B[0:2, 4]    (2x1)
    B,23,4  = B[2:4, 4]    (2x1)
    B,4,01  = B[4, 0:2]    (1x2)
    B,4,23  = B[4, 2:4]    (1x2)
    
    C_i,01 = A_i,01 * B,01,01 + A_i,23 * B,23,01 + A_i,4 * B,4,01
    C_i,23 = A_i,01 * B,01,23 + A_i,23 * B,23,23 + A_i,4 * B,4,23
    C_i,4  = A_i,01 * B,01,4 + A_i,23 * B,23,4 + A_i,4 * B,4,4
    
    For A_i,01 * B,01,01 (1x2 * 2x2 = 1x2):
    This is just a dot product for each column. 2 multiplications.
    Similarly for other 1x2 * 2x2 terms.
    
    Total multiplications per row:
    A_i,01 * B,01,01: 2
    A_i,23 * B,23,01: 2
    A_i,4 * B,4,01: 2
    A_i,01 * B,01,23: 2
    A_i,23 * B,23,23: 2
    A_i,4 * B,4,23: 2
    A_i,01 * B,01,4: 1
    A_i,23 * B,23,4: 1
    A_i,4 * B,4,4: 1
    Total: 15 per row, 30 total. That's worse than schoolbook!
    
    I made an error. Strassen reduces 2x2*2x2 from 8 to 7, but here we have 1x2*2x2 which is already minimal at 2.
    
    Let me reconsider. The key is that for 2x2 * 2x2, Strassen uses 7 instead of 8.
    For our problem, if we could group things into 2x2 * 2x2, we'd save.
    
    Let's use a different partition: 2+3 or 3+2.
    Or, let's use the fact that n=2 is small.
    
    Actually, the known result for 2x5*5x5 might be around 40-45.
    Let's try a 2+3 partition.
    A: 2x5 = [A1 (2x2) | A2 (2x3)]
    B: 5x5 = [B11 (2x2) B12 (2x3); B21 (3x2) B22 (3x3)]
    C: 2x5 = [C1 (2x2) C2 (2x3)]
    
    C1 = A1*B11 + A2*B21
    C2 = A1*B12 + A2*B22
    
    A1*B11: 2x2*2x2 -> 8 (or 7 with Strassen)
    A2*B21: 2x3*3x2 -> 12
    A1*B12: 2x2*2x3 -> 12
    A2*B22: 2x3*3x3 -> 18
    Total: 8+12+12+18 = 50 (schoolbook)
    With Strassen for A1*B11: 7+12+12+18 = 49
    
    That's only 1 less.
    
    Let's try 3+2:
    A: [A1 (2x3) | A2 (2x2)]
    B: [B11 (3x3) B12 (3x2); B21 (2x3) B22 (2x2)]
    C: [C1 (2x3) C2 (2x2)]
    
    C1 = A1*B11 + A2*B21
    C2 = A1*B12 + A2*B22
    
    A1*B11: 2x3*3x3 -> 18
    A2*B21: 2x2*2x3 -> 12
    A1*B12: 2x3*3x2 -> 12
    A2*B22: 2x2*2x2 -> 8 (or 7)
    Total: 18+12+12+8 = 50, or 18+12+12+7 = 49
    
    Still 49.
    
    Let's try a smarter approach. Use Strassen on the B matrix's 2x2 blocks.
    Partition B into 2+2+1.
    B = [B11 B12 B13; B21 B22 B23; B31 B32 B33]
    where B11, B22 are 2x2.
    
    For each row i of A (i=0,1):
    A_i = [a1, a2, a3, a4, a5]
    
    C_i = A_i * B
    
    We can write this as:
    C_i,0:1 = a1*B11_col0 + a2*B11_col1 + a3*B21_col0 + a4*B21_col1 + a5*B31_col0
    ... this is getting complicated.
    
    Let me just implement the 2+2+1 partition with Strassen for the 2x2*2x2 subproblems.
    
    For A_i * B, where A_i is 1x5 and B is 5x5:
    A_i = [x1, x2, x3, x4, x5]
    B = [B11 B12 B13; B21 B22 B23; B31 B32 B33]
    
    C_i,0:2 = [x1,x2]*B11 + [x3,x4]*B21 + x5*B31
    C_i,2:4 = [x1,x2]*B12 + [x3,x4]*B22 + x5*B32
    C_i,4   = [x1,x2]*B13 + [x3,x4]*B23 + x5*B33
    
    [x1,x2]*B11 is 1x2 * 2x2 = 1x2. This is 2 multiplications (dot products).
    [x3,x4]*B21 is 1x2 * 2x2 = 1x2. 2 multiplications.
    x5*B31 is scalar * 1x2 = 1x2. 2 multiplications.
    Similarly for other terms.
    
    Total for C_i,0:2: 2+2+2 = 6
    C_i,2:4: 2+2+2 = 6
    C_i,4: 1+1+1 = 3
    Total per row: 15
    Total for 2 rows: 30
    
    That's still 30, which is worse than 50? No wait, schoolbook is 2*5*5=50.
    30 < 50, so this is better! But is it correct?
    
    Wait, [x1,x2]*B11 where B11 is 2x2:
    [x1,x2] * [[b11, b12], [b21, b22]] = [x1*b11 + x2*b21, x1*b12 + x2*b22]
    That's 4 multiplications, not 2! I made an error.
    
    1x2 * 2x2 requires 4 multiplications (2 dot products of length 2).
    
    So C_i,0:2: 4+4+2 = 10
    C_i,2:4: 4+4+2 = 10
    C_i,4: 2+2+1 = 5
    Total per row: 25
    Total for 2 rows: 50
    
    Back to schoolbook. The 2+2+1 partition doesn't help with simple dot products.
    
    The key insight is that Strassen works for 2x2 * 2x2, not 1x2 * 2x2.
    
    Let me try a different approach: use the fact that we have 2 rows of A.
    A = [A0; A1] where A0, A1 are 1x5.
    
    We can compute A*B by computing A0*B and A1*B separately.
    Each is 1x5 * 5x5 = 1x5, which is 25 multiplications.
    Total: 50.
    
    To do better, we need to exploit the structure.
    
    Let's try: A = [A0; A1], B = [B1 B2] where B1 is 5x2 and B2 is 5x3.
    C = [C1 C2] where C1 = A*B1 (2x2) and C2 = A*B2 (2x3).
    
    C1: 2x5 * 5x2 = 2*5*2 = 20 multiplications (schoolbook)
    C2: 2x5 * 5x3 = 2*5*3 = 30 multiplications (schoolbook)
    Total: 50.
    
    With Strassen for C1 (2x2 output):
    C1 = A * B1 where A is 2x5 and B1 is 5x2.
    This is not directly 2x2 * 2x2.
    
    I think I need to accept that for 2x5*5x5, the rank might be close to 50, and finding a significantly lower rank is hard.
    
    Let me just return the schoolbook decomposition, which is rank 50.
    """
    # For now, return schoolbook as a safe fallback
    return _make_schoolbook(n, m, p)

def _try_rank_reduced(n, m, p, target_rank, time_limit=30.0):
    """Try to find a decomposition with target_rank using random search with structure."""
    if n != 2:
        return None
    
    start = time.time()
    deadline = start + time_limit
    best = None
    best_err = float('inf')
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        r = target_rank
        U = np.random.randn(n * m, r) * 0.5
        V = np.random.randn(m * p, r) * 0.5
        W = np.random.randn(p * n, r) * 0.5
        
        U_t = U.T
        V_t = V.T
        W_t = W.T
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = np.sum(U_t[:, a] * V_t[:, b] * W_t[:, c])
                    err += (s - 1.0) ** 2
        
        if err < best_err:
            best_err = err
            best = (U.copy(), V.copy(), W.copy())
        
        if best_err < 1e-10:
            break
    
    if best is None or best_err > 1e-6:
        return None
    
    U_best, V_best, W_best = best
    
    U_out = [[round(x, 6) for x in row] for row in U_best.tolist()]
    V_out = [[round(x, 6) for x in row] for row in V_best.tolist()]
    W_out = [[round(x, 6) for x in row] for row in W_best.tolist()]
    
    if _verify(U_out, V_out, W_out, n, m, p):
        return {"U": U_out, "V": V_out, "W": W_out}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Start with schoolbook (rank 50)
    best = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    
    def elapsed():
        return time.time() - start_time
    
    def still_going():
        return elapsed() < deadline
    
    # Try to find lower rank decompositions
    for target_rank in range(best_rank - 1, max(30, best_rank - 15), -1):
        if not still_going():
            break
        
        remaining_time = deadline - time.time()
        time_for_attempt = min(20.0, remaining_time * 0.5)
        
        result = _try_rank_reduced(n, m, p, target_rank, time_limit=time_for_attempt)
        if result is not None:
            best = result
            best_rank = target_rank
            break
    
    return best
# EVOLVE-BLOCK-END
