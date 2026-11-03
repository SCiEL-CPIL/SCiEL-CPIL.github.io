"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _strassen_block(j0, j1, k0, k1, U, V, W, col, n, m, p):
    """Add 7 Strassen products for a 2x2x2 block and return new col."""
    # P1 = (A[0,j0]+A[1,j1]) * (B[j0,k0]+B[j1,k1])
    U[0*m + j0][col] += 1
    U[1*m + j1][col] += 1
    V[j0*p + k0][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    # P2 = (A[1,j0]+A[1,j1]) * B[j0,k0]
    U[1*m + j0][col] += 1
    U[1*m + j1][col] += 1
    V[j0*p + k0][col] += 1
    W[k0*n + 1][col] += 1
    W[k1*n + 1][col] -= 1
    col += 1
    
    # P3 = A[0,j0] * (B[j0,k1]-B[j1,k1])
    U[0*m + j0][col] += 1
    V[j0*p + k1][col] += 1
    V[j1*p + k1][col] -= 1
    W[k1*n + 0][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    # P4 = A[1,j1] * (B[j1,k0]-B[j0,k0])
    U[1*m + j1][col] += 1
    V[j1*p + k0][col] += 1
    V[j0*p + k0][col] -= 1
    W[k0*n + 0][col] += 1
    W[k0*n + 1][col] += 1
    col += 1
    
    # P5 = (A[0,j0]+A[0,j1]) * B[j1,k1]
    U[0*m + j0][col] += 1
    U[0*m + j1][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] -= 1
    W[k1*n + 0][col] += 1
    col += 1
    
    # P6 = (A[1,j0]-A[0,j0]) * (B[j0,k0]+B[j0,k1])
    U[1*m + j0][col] += 1
    U[0*m + j0][col] -= 1
    V[j0*p + k0][col] += 1
    V[j0*p + k1][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    # P7 = (A[0,j1]-A[1,j1]) * (B[j1,k0]+B[j1,k1])
    U[0*m + j1][col] += 1
    U[1*m + j1][col] -= 1
    V[j1*p + k0][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] += 1
    col += 1
    
    return col

def _verify_decomp(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct using numpy."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for l in range(m):
            for k in range(p):
                s = np.dot(U_arr[i*m+l], V_arr[l*p+k] * W_arr[k*n+i])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_constraint_matrix(U, V, W, n, m, p, rank):
    """Build the constraint matrix for finding linear dependencies.
    
    We want to find alpha such that sum_r alpha_r * U(i,l,r) * V(l,k,r) * W(k,i,r) = 0
    for all (i,l,k). This is a system of n*m*p equations in rank unknowns.
    """
    num_eqs = n * m * p
    A = np.zeros((num_eqs, rank))
    
    eq_idx = 0
    for i in range(n):
        for l in range(m):
            for k in range(p):
                for r in range(rank):
                    A[eq_idx, r] = U[i*m+l][r] * V[l*p+k][r] * W[k*n+i][r]
                eq_idx += 1
    
    return A

def _find_linear_dependency(U, V, W, n, m, p, rank):
    """Find a non-trivial linear dependency among the rank-1 terms.
    
    Returns a vector alpha of length rank such that
    sum_r alpha[r] * U(i,l,r) * V(l,k,r) * W(k,i,r) = 0 for all (i,l,k),
    or None if no such dependency exists.
    """
    A = _build_constraint_matrix(U, V, W, n, m, p, rank)
    
    # Find null space of A
    # Use SVD to find the null space
    try:
        U_svd, s, Vt = np.linalg.svd(A, full_matrices=True)
    except:
        return None
    
    # The null space is spanned by the last (rank - rank_A) rows of Vt
    # where rank_A is the numerical rank of A
    tol = 1e-10 * max(s) if len(s) > 0 else 1e-10
    rank_A = np.sum(s > tol)
    
    if rank_A < rank:
        # There is a null space
        # The null space vectors are the last (rank - rank_A) rows of Vt
        null_vecs = Vt[rank_A:, :].T  # shape (rank - rank_A, rank)
        
        # Check if any null vector is non-trivial
        for nv in null_vecs:
            if np.max(np.abs(nv)) > 1e-10:
                return nv
    
    return None

def _reduce_rank_by_dependency(U, V, W, n, m, p, rank, alpha):
    """Given a linear dependency alpha, reduce the rank by 1.
    
    We find a rank-1 term that can be eliminated by expressing it as a
    combination of the others using the dependency.
    """
    # Find the index with the largest |alpha|
    max_idx = np.argmax(np.abs(alpha))
    
    # We want to eliminate term max_idx.
    # From the dependency: sum_r alpha[r] * T_r = 0
    # So: alpha[max_idx] * T_max_idx = -sum_{r != max_idx} alpha[r] * T_r
    # T_max_idx = -sum_{r != max_idx} (alpha[r]/alpha[max_idx]) * T_r
    
    if abs(alpha[max_idx]) < 1e-10:
        return None
    
    new_rank = rank - 1
    keep = [r for r in range(rank) if r != max_idx]
    
    new_U = [[0] * new_rank for _ in range(n * m)]
    new_V = [[0] * new_rank for _ in range(m * p)]
    new_W = [[0] * new_rank for _ in range(p * n)]
    
    for new_idx, r in enumerate(keep):
        scale = 1.0 - alpha[r] / alpha[max_idx]
        for i in range(n):
            for l in range(m):
                new_U[i*m+l][new_idx] = U[i*m+l][r] * scale
        for l in range(m):
            for k in range(p):
                new_V[l*p+k][new_idx] = V[l*p+k][r]
        for k in range(p):
            for i in range(n):
                new_W[k*n+i][new_idx] = W[k*n+i][r]
    
    if not _verify_decomp(new_U, new_V, new_W, n, m, p, new_rank):
        return None
    
    return (new_U, new_V, new_W, new_rank)

def construct():
    n, m, p = DIMENSIONS
    _start_time = time.time()
    _deadline = _start_time + 140
    
    # Build the rank-54 decomposition
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            col = _strassen_block(j0, j1, k0, k1, U, V, W, col, n, m, p)
    
    # Remaining j=4
    for k in range(p):
        U[0*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 0][col] += 1
        col += 1
        
        U[1*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 1][col] += 1
        col += 1
    
    if not _verify_decomp(U, V, W, n, m, p, rank):
        pass
    
    best_rank = rank
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    
    # Try to reduce rank by finding linear dependencies
    if time.time() < _deadline - 60:
        current_rank = best_rank
        cur_U = [row[:] for row in best_U]
        cur_V = [row[:] for row in best_V]
        cur_W = [row[:] for row in best_W]
        
        # Iteratively try to find and exploit linear dependencies
        for iteration in range(10):
            if time.time() > _deadline - 30:
                break
            
            alpha = _find_linear_dependency(cur_U, cur_V, cur_W, n, m, p, current_rank)
            if alpha is None:
                break
            
            result = _reduce_rank_by_dependency(cur_U, cur_V, cur_W, n, m, p, current_rank, alpha)
            if result is None:
                break
            
            cur_U, cur_V, cur_W, current_rank = result
            best_rank = current_rank
            best_U = [row[:] for row in cur_U]
            best_V = [row[:] for row in cur_V]
            best_W = [row[:] for row in cur_W]
        
        # If we made progress, try again with a different approach
        if best_rank < rank and time.time() < _deadline - 30:
            # Try to find more dependencies in the reduced decomposition
            for iteration in range(5):
                if time.time() > _deadline - 30:
                    break
                
                alpha = _find_linear_dependency(best_U, best_V, best_W, n, m, p, best_rank)
                if alpha is None:
                    break
                
                result = _reduce_rank_by_dependency(best_U, best_V, best_W, n, m, p, best_rank, alpha)
                if result is None:
                    break
                
                best_U, best_V, best_W, best_rank = result
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
