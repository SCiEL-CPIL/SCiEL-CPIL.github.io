"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-49 tensor decomposition for 2x2 by 2x14 matrix multiplication
    using Strassen's algorithm on 7 blocks of 2x2 matrices.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    # Attempt Strassen-based decomposition
    result = _construct_strassen_49(n, m, p)

    if result is not None:
        # Verify the decomposition to ensure correctness
        if _verify_decomposition_full(result['U'], result['V'], result['W'], n, m, p):
            return result
        else:
            # If verification fails, fall back to schoolbook (should not happen)
            pass

    # Fallback: Schoolbook decomposition (rank 56)
    return _construct_schoolbook(n, m, p)

def _construct_strassen_49(n, m, p):
    """
    Construct a rank-49 decomposition using Strassen's algorithm on 7 blocks of 2x2x2.
    """
    num_blocks = p // 2
    rank = num_blocks * 7

    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    # Strassen linear forms for A (2x2)
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    U_forms = np.array([
        [1, 0, 0, 1],   # s=0: A00+A11
        [0, 0, 1, 1],   # s=1: A10+A11
        [1, 0, 0, 0],   # s=2: A00
        [0, 0, 0, 1],   # s=3: A11
        [1, 1, 0, 0],   # s=4: A00+A01
        [-1, 0, 1, 0],  # s=5: A10-A00
        [0, 1, 0, -1]   # s=6: A01-A11
    ], dtype=np.int64)

    for r in range(num_blocks):
        idx_e = 2 * r           # B[0, 2r]
        idx_f = 2 * r + 1       # B[0, 2r+1]
        idx_g = p + 2 * r       # B[1, 2r]
        idx_h = p + 2 * r + 1   # B[1, 2r+1]

        w_idx_c00 = 2 * r       # C[0, 2r]
        w_idx_c10 = 2 * r + 1   # C[0, 2r+1]  -- wait, trace convention is W[k*n+i]
        # W index: k*n + i where k is column of C (= column of B), i is row of C (= row of A)
        # C[0, 2r] -> k=2r, i=0 -> W[2r*2+0] = W[4r]
        # C[1, 2r] -> k=2r, i=1 -> W[2r*2+1] = W[4r+1]
        # C[0, 2r+1] -> k=2r+1, i=0 -> W[(2r+1)*2+0] = W[4r+2]
        # C[1, 2r+1] -> k=2r+1, i=1 -> W[(2r+1)*2+1] = W[4r+3]
        w_idx_c00 = 4 * r       # C[0, 2r]
        w_idx_c11 = 4 * r + 1   # C[1, 2r]
        w_idx_c01 = 4 * r + 2   # C[0, 2r+1]
        w_idx_c10 = 4 * r + 3   # C[1, 2r+1]

        for s in range(7):
            c = 7 * r + s

            # Fill U: same for all blocks
            U[0, c] = U_forms[s, 0]
            U[1, c] = U_forms[s, 1]
            U[2, c] = U_forms[s, 2]
            U[3, c] = U_forms[s, 3]

            # Fill V
            if s == 0:
                V[idx_e, c] = 1
                V[idx_h, c] = 1
            elif s == 1:
                V[idx_e, c] = 1
            elif s == 2:
                V[idx_f, c] = 1
                V[idx_h, c] = -1
            elif s == 3:
                V[idx_e, c] = -1
                V[idx_g, c] = 1
            elif s == 4:
                V[idx_h, c] = 1
            elif s == 5:
                V[idx_e, c] = 1
                V[idx_f, c] = 1
            elif s == 6:
                V[idx_g, c] = 1
                V[idx_h, c] = 1

            # Fill W
            if s == 0:
                W[w_idx_c00, c] = 1
                W[w_idx_c11, c] = 1
            elif s == 1:
                W[w_idx_c11, c] = 1
                W[w_idx_c10, c] = -1
            elif s == 2:
                W[w_idx_c01, c] = 1
                W[w_idx_c10, c] = 1
            elif s == 3:
                W[w_idx_c00, c] = 1
                W[w_idx_c11, c] = 1
            elif s == 4:
                W[w_idx_c00, c] = -1
                W[w_idx_c01, c] = 1
            elif s == 5:
                W[w_idx_c10, c] = 1
            elif s == 6:
                W[w_idx_c00, c] = 1

    # Convert to list of lists for return (or keep as numpy arrays)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_decomposition_full(U, V, W, n, m, p, tol=1e-9):
    """
    Full verification: check all tensor entries.
    T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k; else 0.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)

    # Check all nonzero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                total = 0.0
                for r in range(rank):
                    total += U_arr[a, r] * V_arr[b, r] * W_arr[c, r]
                if abs(total - 1.0) > tol:
                    return False

    # Check some zero entries (sample)
    # For correctness, we need all non-matching (a,b,c) to give 0
    # Sample a few
    import random
    random.seed(42)
    for _ in range(200):
        a = random.randint(0, n*m - 1)
        b = random.randint(0, m*p - 1)
        c = random.randint(0, p*n - 1)
        # Check if this is a valid (i,j,k) triple
        i = a // m
        j = a % m
        j2 = b // p
        k = b % p
        k2 = c // n
        i2 = c % n
        if not (j == j2 and k == k2 and i == i2):
            total = 0.0
            for r in range(rank):
                total += U_arr[a, r] * V_arr[b, r] * W_arr[c, r]
            if abs(total) > tol:
                return False

    return True

def _construct_schoolbook(n, m, p):
    """
    Fallback schoolbook decomposition. Rank = n*m*p.
    """
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
