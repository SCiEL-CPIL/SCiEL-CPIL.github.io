import numpy as np
import random
import time

DIMENSIONS = [2, 3, 10]


def _verify_fast(U, V, W, n, m, p, trials=100):
    """
    Fast verification by checking random linear combinations.
    Instead of reconstructing the full tensor, we check:
    sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == 1 for (i,j,k) in support
    and == 0 otherwise.
    """
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)

    # Precompute U, V, W as matrices for faster einsum if needed, 
    # but for small rank, loops are fine.
    
    # Check 100 random non-zero entries
    for _ in range(trials):
        i = random.randint(0, n - 1)
        j = random.randint(0, m - 1)
        k = random.randint(0, p - 1)
        
        a = i * m + j
        b = j * p + k
        c = k * n + i
        
        val = 0.0
        for r in range(rank):
            val += U[a, r] * V[b, r] * W[c, r]
            
        if abs(val - 1.0) > 1e-9:
            return False

    # Check 100 random zero entries
    # Generate random (a,b,c) not in the support
    for _ in range(trials):
        a = random.randint(0, n * m - 1)
        b = random.randint(0, m * p - 1)
        c = random.randint(0, p * n - 1)
        
        # Check if (a,b,c) is in the support
        # a = i*m + j, b = j*p + k, c = k*n + i
        # j = a % m
        # k = b % p
        # i = a // m
        # Check if c == k*n + i
        j = a % m
        i = a // m
        k = b % p
        
        if c == k * n + i:
            continue # This is a non-zero entry, skip for zero check
        
        val = 0.0
        for r in range(rank):
            val += U[a, r] * V[b, r] * W[c, r]
            
        if abs(val) > 1e-9:
            return False

    return True


def _is_valid_entries(U, V, W):
    """Check that all entries are integers or half-integers."""
    for matrix in [U, V, W]:
        for row in matrix:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    return True


def _build_strassen55(n, m, p):
    """
    Build the rank-55 decomposition using Strassen on the 2x2 sub-block.
    """
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part A: j=0 (first column of A, first row of B)
    # A[i,0] * B[0,k] -> C[i,k]
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1

    # Part B: j=1,2 (last 2 cols of A, last 2 rows of B)
    # A2 is 2x2, B2 is 2x10. Split B2 into 5 blocks of 2x2.
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1

        u_11 = 0 * m + 1
        u_12 = 0 * m + 2
        u_21 = 1 * m + 1
        u_22 = 1 * m + 2

        v_11 = 1 * p + k0
        v_12 = 1 * p + k1
        v_21 = 2 * p + k0
        v_22 = 2 * p + k1

        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1

        # M1 = (a11 + a22)(b11 + b22)
        U[u_11][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        W[w_11][col] += 1
        col += 1

        # M2 = (a21 + a22)b11
        U[u_21][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        W[w_10][col] += 1
        W[w_11][col] += -1
        col += 1

        # M3 = a11(b12 - b22)
        U[u_11][col] += 1
        V[v_12][col] += 1
        V[v_22][col] += -1
        W[w_01][col] += 1
        W[w_11][col] += 1
        col += 1

        # M4 = a22(b21 - b11)
        U[u_22][col] += 1
        V[v_21][col] += 1
        V[v_11][col] += -1
        W[w_00][col] += 1
        W[w_10][col] += 1
        col += 1

        # M5 = (a11 + a12)b22
        U[u_11][col] += 1
        U[u_12][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += -1
        W[w_01][col] += 1
        col += 1

        # M6 = (a21 - a11)(b11 + b12)
        U[u_21][col] += 1
        U[u_11][col] += -1
        V[v_11][col] += 1
        V[v_12][col] += 1
        W[w_11][col] += 1
        col += 1

        # M7 = (a12 - a22)(b21 + b22)
        U[u_12][col] += 1
        U[u_22][col] += -1
        V[v_21][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        col += 1

    assert col == rank, f"Expected {rank} columns, got {col}"
    return U, V, W


def _build_schoolbook(n, m, p):
    """Build the schoolbook (rank n*m*p) decomposition."""
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


def construct():
    n, m, p = DIMENSIONS
    # Strict deadline guard: 110s to ensure we finish well before 120s/150s
    deadline = time.time() + 110 

    # Start with the known rank-55 Strassen-based decomposition
    U_best, V_best, W_best = _build_strassen55(n, m, p)
    best_rank = 55

    # Verify it quickly
    if not (_verify_fast(U_best, V_best, W_best, n, m, p, trials=200) and _is_valid_entries(U_best, V_best, W_best)):
        # Fallback to schoolbook
        U_best, V_best, W_best = _build_schoolbook(n, m, p)
        best_rank = n * m * p

    # Try to find a lower rank via random search
    # Theoretical lower bound: 53
    # We try 54 and 53.
    # We limit the time spent on search to 5 seconds total to be safe.
    search_deadline = time.time() + 5
    values = [0, 0.5, 1, -0.5, -1]

    for r in [54, 53]:
        if time.time() > search_deadline:
            break
        
        # Limit trials to prevent long loops
        trials = 200 
        start_t = time.time()
        for _ in range(trials):
            if time.time() > search_deadline:
                break
            U = [[random.choice(values) for _ in range(r)] for _ in range(n * m)]
            V = [[random.choice(values) for _ in range(r)] for _ in range(m * p)]
            W = [[random.choice(values) for _ in range(r)] for _ in range(p * n)]
            if _verify_fast(U, V, W, n, m, p, trials=50) and _is_valid_entries(U, V, W):
                U_best, V_best, W_best = U, V, W
                best_rank = r
                break

    return {"U": U_best, "V": V_best, "W": W_best}
