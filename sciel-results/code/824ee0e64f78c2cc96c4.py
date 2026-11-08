"""Rank-minimal tensor decomposition for 4x5 * 5x6 matrix multiplication.

Strategy: 
1. Build a block-Kronecker decomposition combining Strassen 2x2 blocks with schoolbook.
2. Use numpy-vectorized verification for fast column-removal search.
3. Greedy column removal to reduce rank.
"""
import time
import random

try:
    import numpy as np
except Exception:
    np = None

try:
    import sympy
except Exception:
    sympy = None

DIMENSIONS = [4, 5, 6]
SEED = 42

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """Return the 7 rank-1 terms for 2x2 matrix multiplication.
    Convention: for each rank-1 term t, we have vectors
    u_t in R^{2*2}, v_t in R^{2*2}, w_t in R^{2*2} such that
    sum_t w_t[i,k] * u_t[i,j] * v_t[j,k] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    Actually the standard form for matrix multiplication tensor:
    T[i,j,k] = delta_{i,k} (where i indexes row of A/col of C, j indexes col of A/row of B, k indexes col of B/row of C)
    Wait, let me be more careful.
    
    For multiplying A (n x m) by B (m x p) to get C (n x p):
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor is T[i,j,k] = delta_{i,k} ... no that's not right either.
    Actually T[i,j,k] is the coefficient such that the multiplication tensor is:
    sum_{i,j,k} T[i,j,k] * e_i^A * e_j^B * e_k^C where T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    
    So T[i,j,k] = 1 for all valid (i,j,k) in the schoolbook, and the rank-1 decomposition is:
    T = sum_t u_t * v_t * w_t where (u_t)_i = 1 if i is in the "A part", etc.
    
    For Strassen 2x2: n=2, m=2, p=2
    The 8 entries of the tensor are all 1 (since C[i,k] = sum_j A[i,j] B[j,k] and each term A[i,j]B[j,k] appears).
    Wait no - the tensor T has T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,1} because each product A[i,j]*B[j,k] contributes to C[i,k].
    
    Actually I think the convention here is different. Let me re-read the task.
    
    "The output factor follows the official trace convention W[k*n+i,r]"
    
    So W is indexed as W[k*n+i, r] where k is the column of C (and B), i is the row of C (and A), r is the rank index.
    U is indexed as U[i*m+j, r] where i is row of A, j is col of A (and row of B).
    V is indexed as V[j*p+k, r] where j is row of B, k is col of B (and col of C).
    
    The condition is: for all i, k:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{j,j'} for all j, j'
    Wait, that doesn't make sense dimensionally.
    
    Let me think again. The matrix multiplication tensor M has entries M[i,j,k] = 1 for all i,j,k (in the appropriate ranges), representing that C[i,k] = sum_j A[i,j] B[j,k].
    
    A rank-1 decomposition is: M = sum_r u_r * v_r * w_r where (u_r)_{i,j} is a matrix, (v_r)_{j,k} is a matrix, (w_r)_{i,k} is a matrix, and the outer product means:
    (u_r * v_r * w_r)[i,j,k] = u_r[i,j] * v_r[j,k] * w_r[i,k]
    
    So the condition is: sum_r u_r[i,j] * v_r[j,k] * w_r[i,k] = 1 for all i,j,k.
    
    In the flattened convention:
    U has shape (n*m, rank), with U[i*m+j, r] = u_r[i,j]
    V has shape (m*p, rank), with V[j*p+k, r] = v_r[j,k]  
    W has shape (p*n, rank), with W[k*n+i, r] = w_r[i,k]
    
    And the condition is: for all i, j, k:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    For Strassen 2x2 (n=2, m=2, p=2):
    We need 7 terms such that for all i,j,k in {0,1}:
    sum_{t=0}^{6} U[i*2+j, t] * V[j*2+k, t] * W[k*2+i, t] = 1
    
    Let me verify the standard Strassen decomposition with this convention.
    """
    # Standard Strassen for 2x2:
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]], C = [[C00, C01], [C10, C11]]
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # where:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # For each M_t, we need u_t, v_t, w_t such that:
    # u_t[i,j] * v_t[j,k] = the coefficient of A[i,j]*B[j,k] in M_t
    # w_t[i,k] = the coefficient of M_t in C[i,k]
    
    # M1 = (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # u_1: A00=1, A11=1 -> [1,0,0,1] (flattened as [A00,A01,A10,A11])
    # v_1: B00=1, B11=1 -> [1,0,0,1] (flattened as [B00,B01,B10,B11])
    # w_1: C00=1, C11=1 -> [1,0,0,1] (flattened as [C00,C01,C10,C11])
    
    # M2 = (A10+A11)B00 = A10*B00 + A11*B00
    # u_2: A10=1, A11=1 -> [0,0,1,1]
    # v_2: B00=1 -> [1,0,0,0]
    # w_2: C10=1 -> [0,0,1,0]
    
    # M3 = A00(B01-B11) = A00*B01 - A00*B11
    # u_3: A00=1 -> [1,0,0,0]
    # v_3: B01=1, B11=-1 -> [0,1,0,-1]
    # w_3: C01=1 -> [0,1,0,0]
    
    # M4 = A11(B10-B00) = A11*B10 - A11*B00
    # u_4: A11=1 -> [0,0,0,1]
    # v_4: B10=1, B00=-1 -> [-1,0,1,0]
    # w_4: C00=1, C10=1 -> [1,0,1,0]
    
    # M5 = (A00+A01)B11 = A00*B11 + A01*B11
    # u_5: A00=1, A01=1 -> [1,1,0,0]
    # v_5: B11=1 -> [0,0,0,1]
    # w_5: C00=-1, C01=1 -> [-1,1,0,0]
    
    # M6 = (A10-A00)(B00+B01) = A10*B00 + A10*B01 - A00*B00 - A00*B01
    # u_6: A10=1, A00=-1 -> [-1,0,1,0]
    # v_6: B00=1, B01=1 -> [1,1,0,0]
    # w_6: C11=1 -> [0,0,0,1]
    
    # M7 = (A01-A11)(B10+B11) = A01*B10 + A01*B11 - A11*B10 - A11*B11
    # u_7: A01=1, A11=-1 -> [0,1,0,-1]
    # v_7: B10=1, B11=1 -> [0,0,1,1]
    # w_7: C00=1 -> [1,0,0,0]
    
    terms = []
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))   # M1
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))   # M2
    terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,0]))  # M3
    terms.append(([0,0,0,1], [-1,0,1,0], [1,0,1,0]))  # M4
    terms.append(([1,1,0,0], [0,0,0,1], [-1,1,0,0]))  # M5
    terms.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))  # M6
    terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))  # M7
    
    return terms


def _build_block_decomposition(n=4, m=5, p=6):
    """Build a block decomposition for n x m * m x p.
    
    Partition into blocks:
    - 4x4x4 block (i=0..3, j=0..3, k=0..3): 8 Strassen 2x2x2 blocks, rank 56
    - 4x1x6 block (i=0..3, j=4, k=0..5): schoolbook, rank 24
    - 4x4x2 block (i=0..3, j=0..3, k=4..5): schoolbook, rank 32
    - 4x1x2 block (i=0..3, j=4, k=4..5): schoolbook, rank 8
    
    Total: 56 + 24 + 32 + 8 = 120
    """
    R = 56 + 24 + 32 + 8  # = 120
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    strassen = _strassen_2x2()
    
    col = 0
    
    # --- Block 1: 4x4x4 as 8 Strassen 2x2x2 blocks, cols 0..55
    for bi in range(2):  # i block: 0 or 1
        for bj in range(2):  # j block: 0 or 1
            for bk in range(2):  # k block: 0 or 1
                for t in range(7):
                    u_vec, v_vec, w_vec = strassen[t]
                    for di in range(2):
                        for dj in range(2):
                            i = 2 * bi + di
                            j = 2 * bj + dj
                            U[i * m + j][col] += u_vec[di * 2 + dj]
                    for dj in range(2):
                        for dk in range(2):
                            j = 2 * bj + dj
                            k = 2 * bk + dk
                            V[j * p + k][col] += v_vec[dj * 2 + dk]
                    for dk in range(2):
                        for di in range(2):
                            k = 2 * bk + dk
                            i = 2 * bi + di
                            W[k * n + i][col] += w_vec[dk * 2 + di]
                    col += 1
    
    # --- Block 2: 4x1x6 (i=0..3, j=4, k=0..5), cols 56..79
    # Schoolbook: rank = 4*1*6 = 24
    for i in range(4):
        for k in range(6):
            U[i * m + 4][col] = 1
            V[4 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # --- Block 3: 4x4x2 (i=0..3, j=0..3, k=4..5), cols 80..111
    # Schoolbook: rank = 4*4*2 = 32
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # --- Block 4: 4x1x2 (i=0..3, j=4, k=4..5), cols 112..119
    # Schoolbook: rank = 4*1*2 = 8
    for i in range(4):
        for k in range(4, 6):
            U[i * m + 4][col] = 1
            V[4 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    return U, V, W, R


def _verify_numpy(U, V, W, n, m, p):
    """Verify the decomposition using numpy."""
    if np is None:
        return False
    R = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    for i in range(n):
        for k in range(p):
            M = np.zeros((m, m), dtype=np.float64)
            for c in range(R):
                wc = Wa[k * n + i, c]
                if wc == 0:
                    continue
                uc = Ua[i * m: i * m + m, c]
                vc = Va[k * p: k * p + m, c]
                M += np.outer(uc, vc) * wc
            target = np.eye(m, dtype=np.float64)
            if not np.allclose(M, target, atol=1e-9):
                return False
    return True


def _verify_sympy(U, V, W, n, m, p):
    """Verify the decomposition using sympy for exactness."""
    if sympy is None:
        return False
    R = len(U[0])
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for j2 in range(m):
                    s = sympy.Integer(0)
                    for c in range(R):
                        wv = sympy.Integer(W[k * n + i][c])
                        uu = sympy.Integer(U[i * m + j][c])
                        vv = sympy.Integer(V[j2 * p + k][c])
                        if wv == 0 or uu == 0 or vv == 0:
                            continue
                        s += wv * uu * vv
                    target = sympy.Integer(1) if j == j2 else sympy.Integer(0)
                    if s != target:
                        return False
    return True


def _try_reduce_rank(U, V, W, n, m, p, deadline):
    """Try to reduce rank by removing columns that are redundant."""
    R = len(U[0])
    changed = True
    while changed and time.time() < deadline - 20.0:
        changed = False
        R = len(U[0])
        for c in range(R):
            if time.time() > deadline - 20.0:
                break
            # Try removing column c
            keep = [x for x in range(R) if x != c]
            U2 = [[U[r][x] for x in keep] for r in range(n * m)]
            V2 = [[V[r][x] for x in keep] for r in range(m * p)]
            W2 = [[W[r][x] for x in keep] for r in range(p * n)]
            if _verify_numpy(U2, V2, W2, n, m, p):
                U, V, W = U2, V2, W2
                changed = True
                break
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    n, m, p = DIMENSIONS
    
    try:
        # Build initial block decomposition
        U, V, W, R = _build_block_decomposition(n, m, p)
        
        # Verify
        if sympy is not None and R <= 130:
            ok = _verify_sympy(U, V, W, n, m, p)
        else:
            ok = _verify_numpy(U, V, W, n, m, p)
        
        if not ok:
            # Fallback: pure schoolbook
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
        
        # Try to reduce rank
        U, V, W = _try_reduce_rank(U, V, W, n, m, p, deadline)
        
        return {"U": U, "V": V, "W": W}
    
    except Exception:
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
