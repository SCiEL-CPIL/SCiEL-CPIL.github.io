"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomp(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct using numpy."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each (i, l, k) in the multiplication tensor, check sum_r U[i*m+l][r] * V[l*p+k][r] * W[k*n+i][r] == 1
    for i in range(n):
        for l in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+l][r] * V[l*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_constraint_matrix(U, V, W, n, m, p, rank):
    """Build the constraint matrix A where A[e, r] = U[i*m+l][r] * V[l*p+k][r] * W[k*n+i][r]"""
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

def _find_and_reduce(U, V, W, n, m, p, rank, max_iterations=30):
    """Iteratively find linear dependencies and reduce rank."""
    cur_U = [row[:] for row in U]
    cur_V = [row[:] for row in V]
    cur_W = [row[:] for row in W]
    cur_rank = rank
    
    for iteration in range(max_iterations):
        A = _build_constraint_matrix(cur_U, cur_V, cur_W, n, m, p, cur_rank)
        
        try:
            U_svd, s, Vt = np.linalg.svd(A, full_matrices=True)
        except:
            break
        
        tol = 1e-10 * max(s) if len(s) > 0 else 1e-10
        rank_A = np.sum(s > tol)
        
        if rank_A >= cur_rank:
            break
        
        # Get null space vectors
        null_vecs = Vt[rank_A:, :].T
        
        reduced = False
        for nv in null_vecs:
            if np.max(np.abs(nv)) < 1e-10:
                continue
            
            max_idx = int(np.argmax(np.abs(nv)))
            if abs(nv[max_idx]) < 1e-10:
                continue
            
            new_rank = cur_rank - 1
            keep = [r for r in range(cur_rank) if r != max_idx]
            
            new_U = [[0] * new_rank for _ in range(n * m)]
            new_V = [[0] * new_rank for _ in range(m * p)]
            new_W = [[0] * new_rank for _ in range(p * n)]
            
            for new_idx, r in enumerate(keep):
                scale = 1.0 - nv[r] / nv[max_idx]
                for i in range(n):
                    for l in range(m):
                        new_U[i*m+l][new_idx] = cur_U[i*m+l][r] * scale
                for l in range(m):
                    for k in range(p):
                        new_V[l*p+k][new_idx] = cur_V[l*p+k][r]
                for k in range(p):
                    for i in range(n):
                        new_W[k*n+i][new_idx] = cur_W[k*n+i][r]
            
            if _verify_decomp(new_U, new_V, new_W, n, m, p, new_rank):
                cur_U, cur_V, cur_W, cur_rank = new_U, new_V, new_W, new_rank
                reduced = True
                break
        
        if not reduced:
            break
    
    return cur_U, cur_V, cur_W, cur_rank

def _strassen_2x2(j0, j1, k0, k1, U, V, W, col, n, m, p):
    """Add 7 Strassen products for a 2x2x2 block."""
    # P1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
    U[0*m + j0][col] += 1
    U[1*m + j1][col] += 1
    V[j0*p + k0][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    # P2 = (A[i1,j0]+A[i1,j1]) * B[j0,k0]
    U[1*m + j0][col] += 1
    U[1*m + j1][col] += 1
    V[j0*p + k0][col] += 1
    W[k0*n + 1][col] += 1
    W[k1*n + 1][col] -= 1
    col += 1
    
    # P3 = A[i0,j0] * (B[j0,k1]-B[j1,k1])
    U[0*m + j0][col] += 1
    V[j0*p + k1][col] += 1
    V[j1*p + k1][col] -= 1
    W[k1*n + 0][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    # P4 = A[i1,j1] * (B[j1,k0]-B[j0,k0])
    U[1*m + j1][col] += 1
    V[j1*p + k0][col] += 1
    V[j0*p + k0][col] -= 1
    W[k0*n + 0][col] += 1
    W[k0*n + 1][col] += 1
    col += 1
    
    # P5 = (A[i0,j0]+A[i0,j1]) * B[j1,k1]
    U[0*m + j0][col] += 1
    U[0*m + j1][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] -= 1
    W[k1*n + 0][col] += 1
    col += 1
    
    # P6 = (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1])
    U[1*m + j0][col] += 1
    U[0*m + j0][col] -= 1
    V[j0*p + k0][col] += 1
    V[j0*p + k1][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    # P7 = (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1])
    U[0*m + j1][col] += 1
    U[1*m + j1][col] -= 1
    V[j1*p + k0][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] += 1
    col += 1
    
    return col

def construct():
    n, m, p = DIMENSIONS
    _start_time = time.time()
    _deadline = _start_time + 130
    
    # Build initial decomposition using Strassen blocks
    # Strategy: Use 2x2 Strassen for j pairs (0,1), (2,3) and handle j=4 separately
    # For each j-pair, use 2x2 Strassen for k pairs (0,1), (2,3), (4,5)
    
    rank = 60  # Start with naive, will reduce
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            col = _strassen_2x2(j0, j1, k0, k1, U, V, W, col, n, m, p)
    
    # Handle j=4 with naive products
    for k in range(p):
        U[0*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 0][col] += 1
        col += 1
        
        U[1*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 1][col] += 1
        col += 1
    
    actual_rank = col
    if not _verify_decomp(U, V, W, n, m, p, actual_rank):
        # Fallback to naive if Strassen verification fails
        U = [[0] * 60 for _ in range(n * m)]
        V = [[0] * 60 for _ in range(m * p)]
        W = [[0] * 60 for _ in range(p * n)]
        col = 0
        for i in range(n):
            for l in range(m):
                for k in range(p):
                    U[i*m+l][col] = 1
                    V[l*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        actual_rank = 60
    
    best_rank = actual_rank
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    
    # Try to reduce rank by finding linear dependencies
    cur_U = [row[:] for row in best_U]
    cur_V = [row[:] for row in best_V]
    cur_W = [row[:] for row in best_W]
    current_rank = best_rank
    
    for iteration in range(40):
        if time.time() > _deadline - 15:
            break
        
        result_U, result_V, result_W, result_rank = _find_and_reduce(
            cur_U, cur_V, cur_W, n, m, p, current_rank, max_iterations=10
        )
        
        if result_rank < current_rank:
            cur_U, cur_V, cur_W, current_rank = result_U, result_V, result_W, result_rank
            best_rank = current_rank
            best_U = [row[:] for row in cur_U]
            best_V = [row[:] for row in cur_V]
            best_W = [row[:] for row in cur_W]
        else:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
