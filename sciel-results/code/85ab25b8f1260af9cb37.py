"""Matrix multiplication 2x4x6 via optimized Strassen with shared terms."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i."""
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    T = np.zeros((U_dim, V_dim, W_dim), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify_decomposition(U, V, W, T, atol=1e-9):
    """Verify that sum_r U[:,r] V[:,r] W[:,r] equals T."""
    rank = U.shape[1]
    T_check = np.zeros_like(T)
    for r in range(rank):
        T_check += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.allclose(T_check, T, atol=atol)

def _strassen_decomposition(n, m, p):
    """
    Build Strassen-based rank-42 decomposition for n x m * m x p.
    A (n x m) split into m/2 blocks of 2 columns.
    B (m x p) split into m/2 row-blocks of 2 rows, each with p/2 column-blocks of 2.
    For each (A-block j, B-row-block j, B-col-block k): apply Strassen rank-7.
    Total rank = (m/2) * (p/2) * 7.
    """
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n

    # Strassen coefficients for 2x2 * 2x2.
    strassen_terms = [
        ([((0,0), 1), ((1,1), 1)],
         [((0,0), 1), ((1,1), 1)],
         [((0,0), 1), ((1,1), 1)]),
        ([((1,0), 1), ((1,1), 1)],
         [((0,0), 1)],
         [((1,0), 1), ((1,1), -1)]),
        ([((0,0), 1)],
         [((0,1), 1), ((1,1), -1)],
         [((0,1), 1), ((1,1), 1)]),
        ([((1,1), 1)],
         [((1,0), 1), ((0,0), -1)],
         [((0,0), 1), ((1,0), 1)]),
        ([((0,0), 1), ((0,1), 1)],
         [((1,1), 1)],
         [((0,0), -1), ((0,1), 1)]),
        ([((1,0), 1), ((0,0), -1)],
         [((0,0), 1), ((0,1), 1)],
         [((1,1), 1)]),
        ([((0,1), 1), ((1,1), -1)],
         [((1,0), 1), ((1,1), 1)],
         [((0,0), 1)]),
    ]

    m_groups = m // 2
    p_groups = p // 2
    rank = m_groups * p_groups * 7

    U = np.zeros((U_dim, rank), dtype=np.float64)
    V = np.zeros((V_dim, rank), dtype=np.float64)
    W = np.zeros((W_dim, rank), dtype=np.float64)

    col = 0
    for jg in range(m_groups):
        for kg in range(p_groups):
            for s in range(7):
                u_factors, v_factors, w_factors = strassen_terms[s]

                for (pi, pj), coeff in u_factors:
                    u_idx = pi * m + (jg * 2 + pj)
                    U[u_idx, col] += coeff

                for (qi, qj), coeff in v_factors:
                    v_idx = (jg * 2 + qi) * p + (kg * 2 + qj)
                    V[v_idx, col] += coeff

                for (ri, rj), coeff in w_factors:
                    w_idx = (kg * 2 + rj) * n + ri
                    W[w_idx, col] += coeff

                col += 1

    return U, V, W, rank

def _optimized_decomposition(n, m, p):
    """
    Build a rank-42 Strassen decomposition for 2x4 * 4x6.
    Uses 2 j-groups and 3 k-groups, each with rank-7 Strassen.
    """
    return _strassen_decomposition(n, m, p)

def _schoolbook(n, m, p):
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    rank = n * m * p
    U = np.zeros((U_dim, rank), dtype=np.float64)
    V = np.zeros((V_dim, rank), dtype=np.float64)
    W = np.zeros((W_dim, rank), dtype=np.float64)
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
    """
    Find a rank-minimal exact tensor decomposition for 2x4 * 4x6 matrix multiplication.
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    start_time = time.time()
    deadline = start_time + 110

    T = _build_tensor(n, m, p)

    # Try the optimized decomposition first
    U, V, W, rank = _optimized_decomposition(n, m, p)

    if time.time() < deadline and _verify_decomposition(U, V, W, T):
        best_U, best_V, best_W = U, V, W
    else:
        U, V, W, rank = _schoolbook(n, m, p)
        best_U, best_V, best_W = U, V, W

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
