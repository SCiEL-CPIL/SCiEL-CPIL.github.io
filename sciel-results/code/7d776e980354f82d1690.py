"""
Program to find a rank-minimal exact tensor decomposition for matrix multiplication 2x5 by 5x6.
The output factor follows the official trace convention W[k*n+i,r].
"""

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # Safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return _TIME_LIMIT - _elapsed()

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition is exact.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid (i,j,k)
    And 0 for invalid triples.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    # Check valid entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    
    # Check invalid entries (exhaustive is fast enough for these dims)
    # Total entries: (p*n) * (n*m) * (m*p) = 12 * 10 * 30 = 3600
    for x in range(p*n):
        i = x % n
        k = x // n
        for y in range(n*m):
            j = y % m
            i2 = y // m
            for z in range(m*p):
                j2 = z // p
                k2 = z % p
                # Check consistency
                if i == i2 and j == j2 and k == k2:
                    # Valid case, already checked
                    continue
                else:
                    s = 0.0
                    for r in range(rank):
                        s += U[y, r] * V[z, r] * W[x, r]
                    if abs(s) > tol:
                        return False
    return True

def _build_54_rank(n, m, p):
    """
    Build 2x5x6 decomposition using block Strassen. Rank 54.
    Split j into {0,1}, {2,3}, {4}. Split k into {0,1}, {2,3}, {4,5}.
    """
    rank = 54
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Pairs of j indices for Strassen blocks
    j_pairs = [(0, 1), (2, 3)]
    # Pairs of k indices for Strassen blocks
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            # 7 products for Strassen 2x2x2 block
            # Mapping local indices to global:
            # A: (i, j) with i in {0,1}, j in {j0, j1}
            # B: (j, k) with j in {j0, j1}, k in {k0, k1}
            # C: (i, k) with i in {0,1}, k in {k0, k1}
            
            # Standard Strassen formulas adapted for indices
            # M1 = (A_0,j0 + A_1,j1)(B_j0,k0 + B_j1,k1)
            U[0*m + j0, col] += 1
            U[1*m + j1, col] += 1
            V[j0*p + k0, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            
            # M2 = (A_1,j0 + A_1,j1)B_j0,k0
            U[1*m + j0, col] += 1
            U[1*m + j1, col] += 1
            V[j0*p + k0, col] += 1
            W[k0*n + 1, col] += 1
            W[k1*n + 1, col] -= 1
            col += 1
            
            # M3 = A_0,j0(B_j0,k1 - B_j1,k1)
            U[0*m + j0, col] += 1
            V[j0*p + k1, col] += 1
            V[j1*p + k1, col] -= 1
            W[k1*n + 0, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            
            # M4 = A_1,j1(B_j1,k0 - B_j0,k0)
            U[1*m + j1, col] += 1
            V[j1*p + k0, col] += 1
            V[j0*p + k0, col] -= 1
            W[k0*n + 0, col] += 1
            W[k0*n + 1, col] += 1
            col += 1
            
            # M5 = (A_0,j0 + A_0,j1)B_j1,k1
            U[0*m + j0, col] += 1
            U[0*m + j1, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] -= 1
            W[k1*n + 0, col] += 1
            col += 1
            
            # M6 = (A_1,j0 - A_0,j0)(B_j0,k0 + B_j0,k1)
            U[1*m + j0, col] += 1
            U[0*m + j0, col] -= 1
            V[j0*p + k0, col] += 1
            V[j0*p + k1, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            
            # M7 = (A_0,j1 - A_1,j1)(B_j1,k0 + B_j1,k1)
            U[0*m + j1, col] += 1
            U[1*m + j1, col] -= 1
            V[j1*p + k0, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] += 1
            col += 1
            
    # Remaining j=4 (Schoolbook for 2x1x6)
    for k in range(p):
        # C_0,k = A_0,4 B_4,k
        U[0*m + 4, col] += 1
        V[4*p + k, col] += 1
        W[k*n + 0, col] += 1
        col += 1
        
        # C_1,k = A_1,4 B_4,k
        U[1*m + 4, col] += 1
        V[4*p + k, col] += 1
        W[k*n + 1, col] += 1
        col += 1
        
    assert col == rank, f"Expected col={rank}, got {col}"
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p, time_limit=10):
    """
    Attempt to reduce rank by finding linear dependencies among the tensor terms.
    We flatten each term r into a vector v_r of length (n*m)*(m*p)*(p*n).
    We check if the set of vectors is linearly dependent.
    If so, we can potentially remove a column.
    """
    start_time = time.time()
    current_rank = U.shape[1]
    
    if current_rank == 0:
        return U, V, W
    
    # Flatten tensor space size
    size_u = n * m
    size_v = m * p
    size_w = p * n
    total_size = size_u * size_v * size_w
    
    # Construct the matrix of flattened terms
    # M will be total_size x current_rank
    # This might be large: 3600 x 54 = 194400 floats. ~1.5 MB. Very fast.
    
    M = np.zeros((total_size, current_rank))
    
    for r in range(current_rank):
        # Compute outer product U[:,r] (V[:,r] (W[:,r]))
        # Flatten into M[:,r]
        term = np.outer(U[:, r], np.outer(V[:, r], W[:, r]))
        M[:, r] = term.flatten()
        
        if time.time() - start_time > time_limit:
            break
    
    # Check rank of M
    # Use SVD to find rank
    try:
        _, s, vh = np.linalg.svd(M, full_matrices=False)
        # Rank is number of significant singular values
        tol = 1e-9
        rank_M = np.sum(s > tol)
    except Exception:
        return U, V, W
    
    if rank_M < current_rank:
        # There are dependencies.
        # We can try to remove columns corresponding to the null space.
        # For simplicity and safety, we just return the original if we don't have a robust
        # column-removal algorithm that guarantees correctness in all cases.
        # A proper removal requires re-solving or finding a specific null vector.
        # Given the disjoint support analysis, this is unlikely to succeed for block Strassen.
        pass
        
    return U, V, W

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    n, m, p = DIMENSIONS
    
    # Start with the known 54-rank decomposition
    U, V, W = _build_54_rank(n, m, p)
    best_U, best_V, best_W = U, V, W
    best_rank = 54
    
    # Verify the initial decomposition
    if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        # Try to reduce rank if time permits
        if _time_left() > 20:
            U_red, V_red, W_red = _try_reduce_rank(U, V, W, n, m, p, time_limit=10)
            if U_red.shape[1] < best_rank:
                if _verify_decomposition(U_red.tolist(), V_red.tolist(), W_red.tolist(), n, m, p):
                    best_U, best_V, best_W = U_red, V_red, W_red
                    best_rank = U_red.shape[1]
    else:
        # Fallback to schoolbook if construction failed (should not happen)
        U_s, V_s, W_s = _build_schoolbook(n, m, p)
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
        
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    return U, V, W
# EVOLVE-BLOCK-END
