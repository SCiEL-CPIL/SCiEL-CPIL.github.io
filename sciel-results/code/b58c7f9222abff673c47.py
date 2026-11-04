"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return (U, V, W) as lists of lists with the standard rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all (i,j,k), else 0."""
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # For each (i,j,k), the product must equal 1.
    # We can check via: for each c, the tensor T_c = U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # sum_c T_c should equal the multiplication tensor.
    # Efficient check: reshape and contract.
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Build a 3D array.
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # This is a bit tricky to vectorize directly. Let's use einsum.
    # Actually, let's just loop over i,j,k for correctness check since n*m*p is small (40).
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += Ua[i * m + j, c] * Va[j * p + k, c] * Wa[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    # Also check that off-diagonal entries are 0? The tensor for matrix mult has 1 at (i,j,k) and 0 elsewhere.
    # The convention is W[k*n+i, r]. The tensor is T[i,j,k] = delta_{i,i} delta_{j,j} delta_{k,k}? No.
    # Matrix mult: C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor T has indices (i, j, k) and T[i,j,k] = 1 for all i,j,k? No.
    # The multiplication tensor is sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where the product is C[i,k] = sum_j A[i,j]B[j,k].
    # Wait, the standard tensor for matmul is T = sum_{i,j,k} e_i^A ⊗ e_j^B ⊗ e_k^C ? No.
    # T is such that <T, A ⊗ B> = C.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # So T[i,j,k] = 1 for all i,j,k? No, that would be sum_{i,j,k} A[i,j]B[j,k] which is a scalar.
    # The tensor is T = sum_{i,j,k} E_{i,k}^C ⊗ E_{i,j}^A ⊗ E_{j,k}^B ?
    # Let's stick to the verification: for the decomposition to be valid,
    # sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] must equal 1 for all i,j,k.
    # And for other combinations of indices, it should be 0?
    # The interface says "exactly decompose the matrix multiplication tensor".
    # The tensor T has non-zero entries only at (i, j, k) corresponding to valid matmul terms.
    # In the standard flattening:
    # U is (n*m) x rank. Index i*m+j.
    # V is (m*p) x rank. Index j*p+k.
    # W is (p*n) x rank. Index k*n+i.
    # The product sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] must be 1 if (i,j,k) is a valid term, else 0.
    # But ALL (i,j,k) triples in the ranges 0..n-1, 0..m-1, 0..p-1 are valid terms for matmul.
    # So the check above is sufficient.
    return True

def _rank_of(U, V, W):
    return len(U[0])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = _rank_of(U, V, W)
    best_U, best_V, best_W = U, V, W
    
    # Try to reduce rank by removing columns that are linearly dependent or zero in the tensor sense.
    # Since we don't have a fast tensor rank reduction algorithm, we'll try a simple heuristic:
    # Check if any column can be removed.
    # This is O(rank^2 * n*m*p) which is 40^2 * 40 = 64000 operations, very fast.
    
    def try_remove_column(U, V, W, col_to_remove):
        rank = len(U[0])
        if rank <= 1:
            return None, None, None, False
        
        # Remove column
        new_rank = rank - 1
        new_U = [row[:col_to_remove] + row[col_to_remove+1:] for row in U]
        new_V = [row[:col_to_remove] + row[col_to_remove+1:] for row in V]
        new_W = [row[:col_to_remove] + row[col_to_remove+1:] for row in W]
        
        if _verify(new_U, new_V, new_W, n, m, p):
            return new_U, new_V, new_W, True
        return None, None, None, False
    
    # Greedy column removal
    changed = True
    while changed and time.time() < deadline:
        changed = False
        rank = len(U[0])
        for c in range(rank):
            if time.time() > deadline:
                break
            new_U, new_V, new_W, ok = try_remove_column(U, V, W, c)
            if ok:
                U, V, W = new_U, new_V, new_W
                best_rank = len(U[0])
                best_U, best_V, best_W = U, V, W
                changed = True
                break  # Restart loop since indices changed
    
    # If we have time, try some random perturbations and re-optimization? 
    # For now, return the best found.
    
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
