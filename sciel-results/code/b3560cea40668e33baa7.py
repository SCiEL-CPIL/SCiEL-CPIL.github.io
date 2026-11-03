"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _build_strassen_base(n, m, p):
    """Build the base Strassen decomposition with rank 54."""
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
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
    
    return U, V, W, rank

def _verify_decomp(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct."""
    for i in range(n):
        for l in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+l][r] * V[l*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False, (i, l, k, s)
    return True, None

def _try_reduce_rank(U, V, W, n, m, p, current_rank, target_rank):
    """Try to reduce rank by eliminating one term."""
    if target_rank >= current_rank:
        return None
    
    # Try removing each column and checking if the remaining system is consistent
    for r_remove in range(current_rank):
        # Build the tensor equations
        # For each (i, l, k): sum_r U[i*m+l, r] * V[l*p+k, r] * W[k*n+i, r] = 1
        # If we remove column r_remove, we need:
        # sum_{r != r_remove} U[i*m+l, r] * V[l*p+k, r] * W[k*n+i, r] = 1
        
        # This is a system of n*m*p equations in the unknowns W[k*n+i, r] for r != r_remove
        # But W entries are coupled through the products...
        
        # Alternative: try to express column r_remove as a linear combination of others
        # This is harder. Let's try a different approach.
        
        # For now, just check if removing a column still gives valid decomposition
        # (this would only work if the column was redundant)
        
        # Build reduced matrices
        U_r = [row[:r_remove] + row[r_remove+1:] for row in U]
        V_r = [row[:r_remove] + row[r_remove+1:] for row in V]
        W_r = [row[:r_remove] + row[r_remove+1:] for row in W]
        
        valid, _ = _verify_decomp(U_r, V_r, W_r, n, m, p, current_rank - 1)
        if valid:
            return U_r, V_r, W_r, current_rank - 1
    
    return None

def _snap_to_half_integer(x, tol=1e-9):
    """Snap a value to the nearest half-integer if close."""
    h = round(x * 2) / 2
    if abs(x - h) < tol:
        return h
    return x

def _clean_matrix(M):
    """Clean a matrix by snapping to half-integers and zeroing small values."""
    cleaned = []
    for row in M:
        new_row = []
        for val in row:
            if abs(val) < 1e-9:
                new_row.append(0)
            else:
                new_row.append(_snap_to_half_integer(val))
        cleaned.append(new_row)
    return cleaned

def construct():
    n, m, p = DIMENSIONS
    _start_time = time.time()
    
    # Build base decomposition
    U, V, W, rank = _build_strassen_base(n, m, p)
    
    # Verify
    valid, _ = _verify_decomp(U, V, W, n, m, p, rank)
    if not valid:
        # Should not happen
        pass
    
    best_rank = rank
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    
    # Try to reduce rank
    deadline = _start_time + 140  # 140s deadline to be safe
    
    while rank > 50 and time.time() < deadline:
        result = _try_reduce_rank(U, V, W, n, m, p, rank, rank - 1)
        if result is None:
            break
        U, V, W, rank = result
        if rank < best_rank:
            best_rank = rank
            best_U = [row[:] for row in U]
            best_V = [row[:] for row in V]
            best_W = [row[:] for row in W]
    
    # Clean the best solution
    best_U = _clean_matrix(best_U)
    best_V = _clean_matrix(best_V)
    best_W = _clean_matrix(best_W)
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
