"""Matrix multiplication 2,4,6 - rank-42 tensor decomposition via 2x2 Strassen blocks."""
import numpy as np
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)."""
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    T = np.zeros((U_dim, V_dim, W_dim), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify(U, V, W, T, atol=1e-9):
    """Verify that the decomposition exactly reproduces T."""
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    T_check = np.zeros_like(T)
    for r in range(U.shape[1]):
        T_check += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.allclose(T_check, T, atol=atol)

def _construct_rank42(n, m, p):
    """
    Construct a rank-42 decomposition using Strassen on 2x2 blocks.
    A (2x4) = [A1 | A2] where A1, A2 are 2x2.
    B (4x6) partitioned into 2x2 blocks: B = [[B11,B12,B13],[B21,B22,B23]]
    C = A1*B1t + A2*B2t for each column-block t.
    Total: 6 blocks * 7 Strassen = 42.
    """
    # Strassen coefficients for 2x2 * 2x2
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    U_coeffs = [
        {(0, 0): 1, (1, 1): 1},  # P1
        {(1, 0): 1, (1, 1): 1},  # P2
        {(0, 0): 1},             # P3
        {(1, 1): 1},             # P4
        {(0, 0): 1, (0, 1): 1},  # P5
        {(1, 0): 1, (0, 0): -1}, # P6
        {(0, 1): 1, (1, 1): -1}  # P7
    ]
    V_coeffs = [
        {(0, 0): 1, (1, 1): 1},  # P1
        {(0, 0): 1},             # P2
        {(0, 1): 1, (1, 1): -1}, # P3
        {(1, 0): 1, (0, 0): -1}, # P4
        {(1, 1): 1},             # P5
        {(0, 0): 1, (0, 1): 1},  # P6
        {(1, 0): 1, (1, 1): 1}   # P7
    ]
    W_contrib = [
        {(0, 0): 1, (1, 1): 1},  # P1
        {(1, 0): 1, (1, 1): -1}, # P2
        {(0, 1): 1, (1, 1): 1},  # P3
        {(0, 0): 1, (1, 0): 1},  # P4
        {(0, 0): -1, (0, 1): 1}, # P5
        {(1, 1): 1},             # P6
        {(0, 0): 1}              # P7
    ]

    rank = 42
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    col = 0
    for a_block in range(2):       # A1 (cols 0-1) or A2 (cols 2-3)
        for b_col_block in range(3):  # 3 column blocks of width 2
            for s in range(7): # Strassen term
                # Update U
                for (i_loc, j_loc), coeff in U_coeffs[s].items():
                    i = i_loc
                    j = a_block * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx, col] += coeff
                
                # Update V
                for (j_loc, k_loc), coeff in V_coeffs[s].items():
                    j = a_block * 2 + j_loc
                    k = b_col_block * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx, col] += coeff
                
                # Update W
                for (i_loc, k_loc), coeff in W_contrib[s].items():
                    i = i_loc
                    k = b_col_block * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx, col] += coeff
                
                col += 1

    return U, V, W, rank

def _construct_schoolbook(n, m, p):
    """Construct the schoolbook rank-nmp decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    start_time = time.time()
    deadline = start_time + 110

    T = _build_tensor(n, m, p)

    # Primary: Rank-42 Strassen
    if time.time() < deadline:
        U, V, W, rank = _construct_rank42(n, m, p)
        if _verify(U, V, W, T):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: Schoolbook Rank-48
    U, V, W, rank = _construct_schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
