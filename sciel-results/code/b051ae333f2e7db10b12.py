"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from collections import defaultdict

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def _validate_decomposition(U, V, W, n, m, p):
    """Validate that the decomposition exactly decomposes the matrix multiplication tensor."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    # Check dimensions
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for row in U:
        if len(row) != rank:
            return False
    for row in V:
        if len(row) != rank:
            return False
    for row in W:
        if len(row) != rank:
            return False
    
    # Check that the decomposition is exact
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(total - 1) > 1e-10:
                    return False
    return True

def _try_single_removal(U, V, W, r_remove, n, m, p):
    """Try removing a single term and check if decomposition is still valid."""
    rank = len(U[0])
    U_test = [row[:] for row in U]
    V_test = [row[:] for row in V]
    W_test = [row[:] for row in W]
    
    # Zero out the column for r_remove
    for i in range(len(U_test)):
        U_test[i][r_remove] = 0
    for i in range(len(V_test)):
        V_test[i][r_remove] = 0
    for i in range(len(W_test)):
        W_test[i][r_remove] = 0
    
    # Check if still valid
    if _validate_decomposition(U_test, V_test, W_test, n, m, p):
        # Remove the column
        U_test = [row[:r_remove] + row[r_remove+1:] for row in U_test]
        V_test = [row[:r_remove] + row[r_remove+1:] for row in V_test]
        W_test = [row[:r_remove] + row[r_remove+1:] for row in W_test]
        return True, U_test, V_test, W_test
    return False, None, None, None

def _try_double_removal(U, V, W, r1, r2, n, m, p):
    """Try removing two terms and check if decomposition is still valid."""
    rank = len(U[0])
    U_test = [row[:] for row in U]
    V_test = [row[:] for row in V]
    W_test = [row[:] for row in W]
    
    # Zero out the columns for r1 and r2
    for i in range(len(U_test)):
        U_test[i][r1] = 0
        U_test[i][r2] = 0
    for i in range(len(V_test)):
        V_test[i][r1] = 0
        V_test[i][r2] = 0
    for i in range(len(W_test)):
        W_test[i][r1] = 0
        W_test[i][r2] = 0
    
    # Check if still valid
    if _validate_decomposition(U_test, V_test, W_test, n, m, p):
        # Remove the columns (handle ordering)
        if r1 < r2:
            U_test = [row[:r1] + row[r1+1:r2] + row[r2+1:] for row in U_test]
            V_test = [row[:r1] + row[r1+1:r2] + row[r2+1:] for row in V_test]
            W_test = [row[:r1] + row[r1+1:r2] + row[r2+1:] for row in W_test]
        else:
            U_test = [row[:r2] + row[r2+1:r1] + row[r1+1:] for row in U_test]
            V_test = [row[:r2] + row[r2+1:r1] + row[r1+1:] for row in V_test]
            W_test = [row[:r2] + row[r2+1:r1] + row[r1+1:] for row in W_test]
        return True, U_test, V_test, W_test
    return False, None, None, None

def _direct_sum_decomposition(n, m, p):
    """Build a rank n*m*p decomposition via direct sum of 1x m x p blocks.
    For each i in range(n), the block multiplies A[i,:] (row vector of length m) by B (m x p).
    This gives rank m*p per i, total n*m*p.
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each i, use m*p terms to compute C[i,:] = A[i,:] * B
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                # U: row index i*m + j, col r -> 1
                U[i * m + j][r] = 1
                # V: row index j*p + k, col r -> 1
                V[j * p + k][r] = 1
                # W: row index k*n + i, col r -> 1
                W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}

def _try_pattern_decomposition(n, m, p, rank, deadline):
    """Try to find a rank-minimal decomposition using pattern-based approach."""
    start_time = time.time()
    
    # Start with the direct sum decomposition (rank n*m*p)
    result = _direct_sum_decomposition(n, m, p)
    U = result["U"]
    V = result["V"]
    W = result["W"]
    
    current_rank = rank
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    
    # Phase 1: Try removing single terms repeatedly
    progress = True
    while progress:
        if time.time() > deadline - 30:
            break
        progress = False
        for r_remove in range(current_rank):
            if time.time() > deadline - 30:
                break
            success, U_new, V_new, W_new = _try_single_removal(U, V, W, r_remove, n, m, p)
            if success:
                U, V, W = U_new, V_new, W_new
                current_rank -= 1
                best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
                progress = True
                break
    
    # Phase 2: Try removing pairs of terms
    if time.time() < deadline - 15:
        progress = True
        while progress:
            if time.time() > deadline - 15:
                break
            progress = False
            found = False
            for r1 in range(current_rank):
                if time.time() > deadline - 15:
                    break
                for r2 in range(r1 + 1, current_rank):
                    if time.time() > deadline - 15:
                        break
                    success, U_new, V_new, W_new = _try_double_removal(U, V, W, r1, r2, n, m, p)
                    if success:
                        U, V, W = U_new, V_new, W_new
                        current_rank -= 2
                        best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
                        progress = True
                        found = True
                        break
                if found:
                    break
    
    return {"U": best_U, "V": best_V, "W": best_W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Start with the direct sum decomposition (rank n*m*p = 50 for 2x5x5)
    rank = n * m * p
    
    # Try to find a better decomposition
    result = _try_pattern_decomposition(n, m, p, rank, deadline)
    
    # Validate the result
    if _validate_decomposition(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # Fallback to direct sum
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
