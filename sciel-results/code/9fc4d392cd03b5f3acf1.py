"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

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


def _strassen_2x2x2_terms():
    """
    Return the 7 Strassen terms for 2x2 * 2x2 matrix multiplication.
    Indexing (row-major): A: 0=(0,0),1=(0,1),2=(1,0),3=(1,1); B,C same.
    Each term is a dict with 'u', 'v', 'w' keys, each a list of (index, coeff).
    """
    return [
        {'u': [(0, 1), (3, 1)], 'v': [(0, 1), (3, 1)], 'w': [(0, 1), (3, 1)]},
        {'u': [(2, 1), (3, 1)], 'v': [(0, 1)], 'w': [(2, 1), (3, -1)]},
        {'u': [(0, 1)], 'v': [(1, 1), (3, -1)], 'w': [(1, 1), (3, 1)]},
        {'u': [(3, 1)], 'v': [(2, 1), (0, -1)], 'w': [(0, 1), (2, 1)]},
        {'u': [(0, 1), (1, 1)], 'v': [(3, 1)], 'w': [(0, -1), (1, 1)]},
        {'u': [(2, 1), (0, -1)], 'v': [(0, 1), (1, 1)], 'w': [(3, 1)]},
        {'u': [(1, 1), (3, -1)], 'v': [(2, 1), (3, 1)], 'w': [(0, 1)]},
    ]


def _construct_strassen_2x2x4(n, m, p):
    """
    Construct a rank-42 decomposition using Strassen on 2x2x2 blocks.
    For 2x4 * 4x6: split j into groups of 2, k into groups of 2.
    2 j-groups x 3 k-groups x 7 Strassen terms = 42 terms.
    """
    strassen_terms = _strassen_2x2x2_terms()
    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    j_groups = [0, 2]
    k_groups = [0, 2, 4]

    col = 0
    for j_base in j_groups:
        for k_base in k_groups:
            for s in range(7):
                for (idx, coeff) in strassen_terms[s]['u']:
                    i, j_loc = divmod(idx, 2)
                    j = j_base + j_loc
                    U[i * m + j, col] += coeff
                for (idx, coeff) in strassen_terms[s]['v']:
                    j_loc, k_loc = divmod(idx, 2)
                    j = j_base + j_loc
                    k = k_base + k_loc
                    V[j * p + k, col] += coeff
                for (idx, coeff) in strassen_terms[s]['w']:
                    i_loc, k_loc = divmod(idx, 2)
                    k = k_base + k_loc
                    i = i_loc
                    W[k * n + i, col] += coeff
                col += 1

    return U, V, W


def _construct_rank24(n, m, p):
    """
    Construct a rank-24 decomposition exploiting n=2 structure.
    For each (j, k) pair, use one term that computes contributions to
    both output rows C[0,k] and C[1,k].
    
    U: 1 at positions i*m+j for i=0,1 (both rows of A, column j)
    V: 1 at position j*p+k (row j of B, column k)
    W: 1 at positions k*n+i for i=0,1 (both rows of C, column k)
    
    Rank = m * p = 4 * 6 = 24.
    """
    rank = m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0
    for j in range(m):
        for k in range(p):
            for i in range(n):
                U[i * m + j, col] = 1.0
            V[j * p + k, col] = 1.0
            for i in range(n):
                W[k * n + i, col] = 1.0
            col += 1

    return U, V, W


def _try_reduce_rank(U, V, W, T, max_rank, deadline):
    """
    Attempt to find a lower-rank decomposition by checking for linear
    dependencies in the current decomposition.
    """
    R0 = U.shape[1]
    if R0 <= max_rank:
        return U, V, W

    n, m, p = DIMENSIONS
    U_dim, V_dim, W_dim = n * m, m * p, p * n

    T_flat = T.reshape(-1)
    term_matrix = np.zeros((T_flat.size, R0))
    for r in range(R0):
        term_matrix[:, r] = np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r]).reshape(-1)

    try:
        if time.time() < deadline - 20:
            sv = np.linalg.svd(term_matrix, compute_uv=False)
            tol = 1e-10
            effective_rank = int(np.sum(sv > tol))
            
            if effective_rank < R0:
                U_red = U[:, :effective_rank]
                V_red = V[:, :effective_rank]
                W_red = W[:, :effective_rank]
                if _verify(U_red, V_red, W_red, T):
                    return U_red, V_red, W_red
    except:
        pass

    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    start_time = time.time()
    deadline = start_time + 140.0

    T = _build_tensor(n, m, p)

    # Build the rank-24 decomposition exploiting n=2 structure.
    U, V, W = _construct_rank24(n, m, p)

    # Verify correctness.
    if not _verify(U, V, W, T):
        # Fallback to full schoolbook (rank n*m*p)
        rank_fb = n * m * p
        U = np.zeros((n * m, rank_fb))
        V = np.zeros((m * p, rank_fb))
        W = np.zeros((p * n, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1

    # Attempt rank reduction.
    U, V, W = _try_reduce_rank(U, V, W, T, 23, deadline)

    def to_list(arr):
        return [[(int(x) if float(x).is_integer() else float(x)) for x in row]
                for row in arr.tolist()]

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

# EVOLVE-BLOCK-END
