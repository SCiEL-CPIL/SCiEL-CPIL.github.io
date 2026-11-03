"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

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
    """Build the constraint matrix for finding linear dependencies."""
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

def _find_dependency_and_reduce(U, V, W, n, m, p, rank):
    """Find a linear dependency and use it to reduce rank by 1."""
    A = _build_constraint_matrix(U, V, W, n, m, p, rank)
    
    try:
        U_svd, s, Vt = np.linalg.svd(A, full_matrices=True)
    except:
        return None
    
    tol = 1e-10 * max(s) if len(s) > 0 else 1e-10
    rank_A = np.sum(s > tol)
    
    if rank_A < rank:
        null_vecs = Vt[rank_A:, :].T
        for nv in null_vecs:
            if np.max(np.abs(nv)) > 1e-10:
                max_idx = int(np.argmax(np.abs(nv)))
                if abs(nv[max_idx]) < 1e-10:
                    continue
                new_rank = rank - 1
                keep = [r for r in range(rank) if r != max_idx]
                
                new_U = [[0] * new_rank for _ in range(n * m)]
                new_V = [[0] * new_rank for _ in range(m * p)]
                new_W = [[0] * new_rank for _ in range(p * n)]
                
                for new_idx, r in enumerate(keep):
                    scale = 1.0 - nv[r] / nv[max_idx]
                    for i in range(n):
                        for l in range(m):
                            new_U[i*m+l][new_idx] = U[i*m+l][r] * scale
                    for l in range(m):
                        for k in range(p):
                            new_V[l*p+k][new_idx] = V[l*p+k][r]
                    for k in range(p):
                        for i in range(n):
                            new_W[k*n+i][new_idx] = W[k*n+i][r]
                
                if _verify_decomp(new_U, new_V, new_W, n, m, p, new_rank):
                    return (new_U, new_V, new_W, new_rank)
    
    return None

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
    cur_U = [row[:] for row in best_U]
    cur_V = [row[:] for row in best_V]
    cur_W = [row[:] for row in best_W]
    current_rank = best_rank
    
    for iteration in range(20):
        if time.time() > _deadline - 20:
            break
        
        result = _find_dependency_and_reduce(cur_U, cur_V, cur_W, n, m, p, current_rank)
        if result is None:
            break
        
        cur_U, cur_V, cur_W, current_rank = result
        best_rank = current_rank
        best_U = [row[:] for row in cur_U]
        best_V = [row[:] for row in cur_V]
        best_W = [row[:] for row in cur_W]
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
