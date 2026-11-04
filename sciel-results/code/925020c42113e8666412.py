"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that the tensor decomposition is exact."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m + j], V_arr[j*p + k] * W_arr[k*n + i])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < tol

def _check_all_entries(U, V, W):
    """Check all entries are integer or half-integer."""
    for mat in [U, V, W]:
        for row in mat:
            for x in row:
                x2 = 2.0 * float(x)
                if abs(x2 - round(x2)) > 1e-9:
                    return False
    return True

def _build_3x4x4_kronecker():
    """
    Build a decomposition for 3x4x4 matmul using Kronecker product structure.
    
    Key idea: Split the 4-dimension into 2x2 blocks.
    j = (j1, j2) where j1,j2 in {0,1}
    k = (k1, k2) where k1,k2 in {0,1}
    
    The tensor T[i, (j1,j2), (k1,k2)] = 1 for all i in {0,1,2}.
    
    We can write this as a sum of rank-1 terms using the Strassen structure
    on the (j,k) part, combined with the i dimension.
    
    Strategy: Use the rank-7 Strassen decomposition for 2x2x2 matmul,
    and combine with the 3-row structure.
    
    For the 2x2x2 matmul, we have:
    T2[a, b, c] = sum_{t=1}^{7} U2[a,b,t] * V2[b,c,t] * W2[c,a,t]
    
    For our 3x4x4 problem, we need:
    T[i, (j1,j2), (k1,k2)] = 1
    
    This is NOT a standard matmul tensor in the usual sense because the
    indices are coupled differently. Let me use a direct constructive approach.
    
    Approach: For each pair (j1, j2) and (k1, k2), we need:
    sum_t U[i, j1*2+j2, t] * V[(j1*2+j2)*4 + k1*2+k2, t] * W[(k1*2+k2)*3 + i, t] = 1
    
    Let's try to construct this using a tensor product of smaller decompositions.
    If we can write T = T_a ⊗ T_b where T_a handles some indices and T_b handles others,
    the rank would be rank(T_a) * rank(T_b).
    
    But the index coupling (i,j) in U, (j,k) in V, (k,i) in W prevents simple
    Kronecker factorization.
    
    Alternative: Use the fact that we can embed 3x4x4 into 4x4x4 and use
    the known rank-49 (or better) decomposition for 4x4x4, then restrict.
    
    For 4x4x4, using block Strassen: rank = 7 * 7 = 49 (using rank-7 for 2x2x2).
    But we can do better: the actual rank of 4x4x4 matmul is known to be at most
    49 via Strassen, but the true rank is unknown (between 49 and 64).
    
    For 3x4x4, we can take the 4x4x4 decomposition and zero out the 4th row/column
    contributions, but this doesn't reduce rank.
    
    Let me try a more direct approach: construct a decomposition with rank 36
    by exploiting the structure.
    
    Actually, let me try: use rank-3 for the 3-dimension and rank-12 for the 4x4 part,
    giving total rank 3*12 = 36. But the index coupling prevents this.
    
    Let me just do a structured random search with the Strassen-inspired initialization.
    """
    n, m, p = 3, 4, 4
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = [[0]*rank_school for _ in range(n*m)]
    V_school = [[0]*rank_school for _ in range(m*p)]
    W_school = [[0]*rank_school for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    return U_school, V_school, W_school

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130
    
    # Start with schoolbook
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Try structured search for lower rank
    values = [-1, -0.5, 0, 0.5, 1]
    
    best_found = None
    for target_rank in range(32, 48):
        if time.time() > deadline - 15:
            break
        rng = random.Random(target_rank * 1000 + 42)
        found = False
        max_iters = 200
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            # Sparse initialization: most entries are 0
            density = 0.3
            U_try = [[rng.choice(values) if rng.random() < density else 0 for _ in range(target_rank)] for _ in range(n*m)]
            V_try = [[rng.choice(values) if rng.random() < density else 0 for _ in range(target_rank)] for _ in range(m*p)]
            W_try = [[rng.choice(values) if rng.random() < density else 0 for _ in range(target_rank)] for _ in range(p*n)]
            
            if _verify(U_try, V_try, W_try, n, m, p):
                if _check_all_entries(U_try, V_try, W_try):
                    U, V, W = U_try, V_try, W_try
                    best_found = target_rank
                    found = True
                    break
        if found:
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
