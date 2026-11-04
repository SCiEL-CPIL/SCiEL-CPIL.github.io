import numpy as np
from fractions import Fraction
import itertools
import time

DIMENSIONS = [2, 3, 5]

def _to_fraction(x):
    if isinstance(x, Fraction):
        return x
    if isinstance(x, (int, float)):
        return Fraction(x)
    return Fraction(x)

def _is_half_integer(frac):
    return (frac * 2).denominator == 1

def _verify_decomposition(U, V, W, n, m, p):
    rank = len(U[0])
    U_arr = np.array([[float(x) for x in row] for row in U], dtype=float)
    V_arr = np.array([[float(x) for x in row] for row in V], dtype=float)
    W_arr = np.array([[float(x) for x in row] for row in W], dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                s = 0.0
                for r in range(rank):
                    s += U_arr[idx_u, r] * V_arr[idx_v, r] * W_arr[idx_w, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_rank28():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    # Part 1: j=2, k=0..4
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
    # Part 2: j=0,1, k=4
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
    # Part 3: j=0,1, k=0,1 Strassen
    U[0][r]=1; U[4][r]=1; V[0][r]=1; V[6][r]=1; W[0][r]=1; W[3][r]=1; r+=1
    U[3][r]=1; U[4][r]=1; V[0][r]=1; W[1][r]=1; W[4][r]=-1; r+=1
    U[0][r]=1; V[1][r]=1; V[7][r]=-1; W[2][r]=1; W[5][r]=1; r+=1
    U[4][r]=1; V[6][r]=1; V[0][r]=-1; W[0][r]=1; W[1][r]=1; r+=1
    U[0][r]=1; U[1][r]=1; V[7][r]=1; W[0][r]=-1; W[3][r]=1; r+=1
    U[3][r]=1; U[0][r]=-1; V[0][r]=1; V[1][r]=1; W[4][r]=1; r+=1
    U[1][r]=1; U[4][r]=-1; V[6][r]=1; V[7][r]=1; W[0][r]=1; r+=1
    # Part 4: j=0,1, k=2,3 Strassen
    U[0][r]=1; U[4][r]=1; V[2][r]=1; V[8][r]=1; W[4][r]=1; W[7][r]=1; r+=1
    U[3][r]=1; U[4][r]=1; V[2][r]=1; W[5][r]=1; W[8][r]=-1; r+=1
    U[0][r]=1; V[3][r]=1; V[9][r]=-1; W[6][r]=1; W[9][r]=1; r+=1
    U[4][r]=1; V[8][r]=1; V[2][r]=-1; W[4][r]=1; W[5][r]=1; r+=1
    U[0][r]=1; U[1][r]=1; V[9][r]=1; W[4][r]=-1; W[7][r]=1; r+=1
    U[3][r]=1; U[0][r]=-1; V[2][r]=1; V[3][r]=1; W[8][r]=1; r+=1
    U[1][r]=1; U[4][r]=-1; V[8][r]=1; V[9][r]=1; W[4][r]=1; r+=1
    
    assert r == 28, f"Expected 28, got {r}"
    return U, V, W

def _try_rank27(U28, V28, W28):
    """Try to find a rank-27 solution by eliminating one term."""
    n, m, p = DIMENSIONS
    rank28 = 28
    
    # For each term to eliminate, try to solve for adjusted coefficients
    # This is a nonlinear problem, so we use a heuristic:
    # Check if the term's contribution is linearly dependent on others
    
    U28_arr = np.array([[float(x) for x in row] for row in U28], dtype=float)
    V28_arr = np.array([[float(x) for x in row] for row in V28], dtype=float)
    W28_arr = np.array([[float(x) for x in row] for row in W28], dtype=float)
    
    # Build the tensor T as a 3D array
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank28):
        for i in range(n*m):
            for j in range(m*p):
                for k in range(p*n):
                    T[i,j,k] += U28_arr[i,r] * V28_arr[j,r] * W28_arr[k,r]
    
    # T should be 1 where (i,j,k) corresponds to valid (i,j,k) in matmul
    # i = i_a * m + j_a, j = j_a * p + k_a, k = k_a * n + i_a
    # Valid when i_a == i_a and j_a == j_a and k_a == k_a (always true for the structure)
    # Actually T[i,j,k] = 1 iff the indices are consistent
    
    # Check which entries should be 1
    target = np.zeros((n*m, m*p, p*n))
    for i_a in range(n):
        for j_a in range(m):
            for k_a in range(p):
                idx_u = i_a * m + j_a
                idx_v = j_a * p + k_a
                idx_w = k_a * n + i_a
                target[idx_u, idx_v, idx_w] = 1
    
    # Verify T matches target
    diff = T - target
    if np.max(np.abs(diff)) > 1e-9:
        return None
    
    # Try to find a rank-1 tensor that can be subtracted from one term
    # and the residual is still rank <= 27
    # Heuristic: for each term r, check if its rank-1 contribution is "small"
    # in some sense, or if removing it creates a low-rank residual
    
    # More practical: try to find linear dependencies among the 28 rank-1 tensors
    # Flatten each rank-1 tensor and check for linear dependence
    flat_terms = []
    for r in range(rank28):
        term = np.outer(np.outer(U28_arr[:,r], V28_arr[:,r]), W28_arr[:,r]).flatten()
        flat_terms.append(term)
    
    flat_matrix = np.array(flat_terms).T  # shape (n*m*m*p*p*n, 28)
    
    # Check rank of the matrix of flattened terms
    # If rank < 28, there's a linear dependence
    from numpy.linalg import matrix_rank
    mr = matrix_rank(flat_matrix, tol=1e-10)
    
    if mr < 28:
        # There's a linear dependence, meaning we might be able to reduce rank
        # Find the null space vector
        from numpy.linalg import svd
        U_svd, s_vals, Vt = svd(flat_matrix, full_matrices=False)
        # The last column of Vt gives the null space vector
        null_vec = Vt[-1, :]
        # Check if all coefficients in null_vec have the same sign or can be adjusted
        # If the null space vector has entries that allow us to zero out one coefficient
        # while keeping others valid, we can reduce rank
        
        # For now, just return None as a placeholder
        pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    U, V, W = _build_rank28()
    
    # Try to reduce to rank 27
    if time.time() < deadline - 5:
        result = _try_rank27(U, V, W)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Fallback to rank 28
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Decomposition verification failed")
    
    return {"U": U, "V": V, "W": W}
