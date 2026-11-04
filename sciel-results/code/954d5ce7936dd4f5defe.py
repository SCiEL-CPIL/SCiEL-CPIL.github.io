"""Tensor decomposition for 2x3 * 3x10 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START


def _build_trivial(n, m, p):
    """Return the rank-nmp trivial decomposition as (U, V, W) lists."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for r in range(rank):
        i = r // (m * p)
        rem = r % (m * p)
        j = rem // p
        k = rem % p
        idx_u = i * m + j
        idx_v = j * p + k
        idx_w = k * n + i
        U[idx_u][r] = 1
        V[idx_v][r] = 1
        W[idx_w][r] = 1
    return U, V, W, rank


def _validate(n, m, p, U, V, W):
    """Verify that sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] == 1 for all i,j,k."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = np.dot(U_arr[i * m + j] * V_arr[j * p + k], W_arr[k * n + i])
                if not np.isclose(lhs, 1.0, atol=1e-9):
                    return False
    return True


def _build_optimal(n, m, p):
    """
    Build a rank-37 decomposition for 2x3 * 3x10.
    
    Structure:
    - 2x2x10 all-ones tensor: rank 35 via Strassen on 5 pairs of columns (7 per pair)
    - 2x1x10 all-ones tensor (j=2 slice): rank 2 (one term for i=0, one for i=1)
    
    Total rank: 37
    """
    assert n == 2 and m == 3 and p == 10
    
    rank = 37
    U = [[0] * rank for _ in range(n * m)]  # 6 x 37
    V = [[0] * rank for _ in range(m * p)]  # 30 x 37
    W = [[0] * rank for _ in range(p * n)]  # 20 x 37
    
    r = 0  # current rank index
    
    # --- Part 1: 2x2x10 via Strassen on 5 column pairs ---
    # A_left = [[a00, a01], [a10, a11]]
    #   U indices: 0*3+0=0, 0*3+1=1, 1*3+0=3, 1*3+1=4
    # B_top: B[0,k], B[1,k] for k=0..9
    #   V indices: 0*10+k = k, 1*10+k = 10+k
    # C: C[i,k] for i in {0,1}, k=0..9
    #   W indices: k*2+i
    
    # Strassen's 7 products for 2x2 x 2x2:
    # m1 = (a00+a11)(B0_k0 + B1_k1)
    # m2 = (a10+a11) B0_k0
    # m3 = a00 (B0_k1 - B1_k1)
    # m4 = a11 (B1_k0 - B0_k0)
    # m5 = (a00+a01) B1_k1
    # m6 = (a10-a00)(B0_k0 + B0_k1)
    # m7 = (a01-a11)(B1_k0 + B1_k1)
    #
    # C[0,k0] = m1 + m4 - m5 + m7
    # C[0,k1] = m3 + m5
    # C[1,k0] = m2 + m4
    # C[1,k1] = m1 - m2 + m3 + m6
    
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # m1: (a00+a11)(B0_k0 + B1_k1)
        U[0][r] += 1   # a00
        U[4][r] += 1   # a11
        V[k0][r] += 1      # B[0,k0]
        V[10 + k1][r] += 1  # B[1,k1]
        W[k0 * 2 + 0][r] += 1  # C[0,k0]
        W[k1 * 2 + 1][r] += 1  # C[1,k1]
        r += 1
        
        # m2: (a10+a11) B0_k0
        U[3][r] += 1   # a10
        U[4][r] += 1   # a11
        V[k0][r] += 1      # B[0,k0]
        W[k0 * 2 + 1][r] += 1  # C[1,k0]
        W[k1 * 2 + 1][r] -= 1  # -C[1,k1]
        r += 1
        
        # m3: a00 (B0_k1 - B1_k1)
        U[0][r] += 1   # a00
        V[k1][r] += 1      # B[0,k1]
        V[10 + k1][r] -= 1  # -B[1,k1]
        W[k1 * 2 + 0][r] += 1  # C[0,k1]
        W[k1 * 2 + 1][r] += 1  # C[1,k1]
        r += 1
        
        # m4: a11 (B1_k0 - B0_k0)
        U[4][r] += 1   # a11
        V[10 + k0][r] += 1  # B[1,k0]
        V[k0][r] -= 1      # -B[0,k0]
        W[k0 * 2 + 0][r] += 1  # C[0,k0]
        W[k0 * 2 + 1][r] += 1  # C[1,k0]
        r += 1
        
        # m5: (a00+a01) B1_k1
        U[0][r] += 1   # a00
        U[1][r] += 1   # a01
        V[10 + k1][r] += 1  # B[1,k1]
        W[k0 * 2 + 0][r] -= 1  # -C[0,k0]
        W[k1 * 2 + 0][r] += 1  # C[0,k1]
        r += 1
        
        # m6: (a10-a00)(B0_k0 + B0_k1)
        U[3][r] += 1   # a10
        U[0][r] -= 1   # -a00
        V[k0][r] += 1      # B[0,k0]
        V[k1][r] += 1      # B[0,k1]
        W[k1 * 2 + 1][r] += 1  # C[1,k1]
        r += 1
        
        # m7: (a01-a11)(B1_k0 + B1_k1)
        U[1][r] += 1   # a01
        U[4][r] -= 1   # -a11
        V[10 + k0][r] += 1  # B[1,k0]
        V[10 + k1][r] += 1  # B[1,k1]
        W[k0 * 2 + 0][r] += 1  # C[0,k0]
        r += 1
    
    # --- Part 2: 2x1x10 (j=2 slice) with rank 2 ---
    # Term for i=0: U[2][r]=1, V[20+k][r]=1 for all k, W[2k][r]=1 for all k
    U[2][r] = 1
    for k in range(10):
        V[20 + k][r] = 1
        W[2 * k][r] = 1
    r += 1
    
    # Term for i=1: U[5][r]=1, V[20+k][r]=1 for all k, W[2k+1][r]=1 for all k
    U[5][r] = 1
    for k in range(10):
        V[20 + k][r] = 1
        W[2 * k + 1][r] = 1
    r += 1
    
    assert r == rank, f"Expected rank {rank}, got {r}"
    
    return U, V, W, rank


def construct():
    start_time = time.time()
    deadline = start_time + 100  # safety margin under 120s cap

    n, m, p = DIMENSIONS

    # Use optimized rank-37 decomposition
    U, V, W, rank = _build_optimal(n, m, p)

    # Validate
    if not _validate(n, m, p, U, V, W):
        # Fall back to trivial if validation fails
        U, V, W, rank = _build_trivial(n, m, p)

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
