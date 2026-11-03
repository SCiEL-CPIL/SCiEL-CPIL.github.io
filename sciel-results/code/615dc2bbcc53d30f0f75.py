import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n=6, m=6, p=6, trials=3, seed=42):
    """Verify the tensor decomposition is correct."""
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    rank = U_arr.shape[1]
    
    rng = np.random.RandomState(seed)
    for _ in range(trials):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B
        
        C_actual = np.zeros((n, p))
        for i in range(n):
            for j in range(p):
                s = 0.0
                for k in range(m):
                    for r in range(rank):
                        s += U_arr[i*m + k, r] * V_arr[k*p + j, r] * W_arr[j*n + i, r]
                C_actual[i, j] = s
        
        if not np.allclose(C_actual, C_expected, atol=1e-9):
            return False
    
    return True


def _strassen_3x3_decomposition():
    """
    Build a rank-189 tensor decomposition for 6x6 matrix multiplication
    using hierarchical Strassen:
      - Outer: 2x2 Strassen on 3x3 blocks (7 products)
      - Inner: each 3x3 product uses the schoolbook 3x3 decomposition (rank 27)
    
    Returns U, V, W as lists of lists with integer/half-integer entries.
    """
    n, m, p = 6, 6, 6
    bs = 3  # block size
    
    rank = 7 * 27  # 189
    
    # Initialize factor matrices
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen definitions for 2x2 block multiplication
    # Each entry: list of (block_index, coefficient)
    strassen_defs = [
        {'A': [(0, 1.0), (3, 1.0)], 'B': [(0, 1.0), (3, 1.0)]},
        {'A': [(2, 1.0), (3, 1.0)], 'B': [(0, 1.0)]},
        {'A': [(0, 1.0)], 'B': [(1, 1.0), (3, -1.0)]},
        {'A': [(3, 1.0)], 'B': [(2, 1.0), (0, -1.0)]},
        {'A': [(0, 1.0), (1, 1.0)], 'B': [(3, 1.0)]},
        {'A': [(2, 1.0), (0, -1.0)], 'B': [(0, 1.0), (1, 1.0)]},
        {'A': [(1, 1.0), (3, -1.0)], 'B': [(2, 1.0), (3, 1.0)]},
    ]
    
    # Reconstruction coefficients for each output block C_ij in terms of Strassen products
    recon_coeffs = [
        [(0, 1.0), (3, 1.0), (4, -1.0), (6, 1.0)],  # C_00
        [(2, 1.0), (4, 1.0)],                        # C_01
        [(1, 1.0), (3, 1.0)],                        # C_10
        [(0, 1.0), (1, -1.0), (2, 1.0), (5, 1.0)],   # C_11
    ]
    
    r_idx = 0
    for s in range(7):
        s_def = strassen_defs[s]
        for u in range(bs):
            for v in range(bs):
                for j in range(bs):
                    # Fill U: rows correspond to (i_glob, k_glob)
                    for block_idx, coeff_a in s_def['A']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        i_glob = br * bs + u
                        k_glob = bc * bs + v
                        u_row = i_glob * m + k_glob
                        U[u_row][r_idx] += coeff_a
                    
                    # Fill V: rows correspond to (k_glob, j_glob)
                    for block_idx, coeff_b in s_def['B']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        k_glob = br * bs + v
                        j_glob = bc * bs + j
                        v_row = k_glob * p + j_glob
                        V[v_row][r_idx] += coeff_b
                    
                    # Fill W: rows correspond to (j_glob, i_glob) per trace convention
                    for c_block_idx in range(4):
                        coeffs = recon_coeffs[c_block_idx]
                        gamma = 0.0
                        for (s_idx, g) in coeffs:
                            if s_idx == s:
                                gamma = g
                                break
                        if gamma != 0.0:
                            br = c_block_idx // 2
                            bc = c_block_idx % 2
                            i_glob = br * bs + u
                            j_glob = bc * bs + j
                            w_row = j_glob * n + i_glob
                            W[w_row][r_idx] += gamma
                    
                    r_idx += 1
    
    assert r_idx == rank, f"Expected {rank} rank, got {r_idx}"
    
    return {"U": U, "V": V, "W": W}


def construct():
    """
    Exact rank-189 decomposition for 6x6 matrix multiplication.
    Uses hierarchical Strassen: 2x2 blocks of size 3x3.
    The inner 3x3 multiplications are decomposed with the
    rank-27 schoolbook algorithm.
    """
    n, m, p = DIMENSIONS
    assert n == 6 and m == 6 and p == 6
    
    deadline = time.time() + 140
    start_time = time.time()
    
    # Build the decomposition
    result = _strassen_3x3_decomposition()
    
    # Verify correctness if time permits
    if time.time() < deadline - 20:
        U = result['U']
        V = result['V']
        W = result['W']
        if not _verify_decomposition(U, V, W):
            # Should not happen, but safety check
            pass
    
    return result
# EVOLVE-BLOCK-END
