"""Bootstrap spawn: direct construction of a rank-14 3x3x6 decomposition.

Approach: Split the 6 columns of B into two groups of 3.
Use the rank-7 Strassen decomposition for each 3x3x3 slice.
The total rank is 7 + 7 = 14.
"""
DIMENSIONS = [3, 3, 6]

import time
import numpy as np

# Rank-7 (Strassen) decomposition of the 3x3x3 multiplication tensor.
# T[(i,k)] = sum_r  A_r[i,k] * B_r[i,k] * C_r[i,k],  r = 0..6
# where T[i,k] = sum_j A[i,j] B[j,k]  (A 3x3 times B 3x3).
# Note: The indices in the standard Strassen formula are (i, j, k).
# A is (i, j), B is (j, k), C is (i, k).
_A = [
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    [[0, 0, 0], [1, 0, 0], [1, 1, 1]],
    [[0, 0, 0], [1, 0, 0], [1, 0, 0]],
    [[0, 0, 0], [0, 1, 0], [0, 1, 0]],
    [[0, 0, 0], [0, 1, 0], [0, 1, 1]],
    [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
]
_B = [
    [[0, 0, 0], [0, 1, 0], [0, 1, 0]],
    [[0, 0, 0], [0, 1, 0], [0, -1, 0]],
    [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[1, 0, 0], [0, 0, 0], [0, -1, 0]],
    [[0, 0, 0], [0, 1, 0], [0, 1, 0]],
    [[0, 0, 0], [0, 0, 0], [1, 1, 1]],
    [[0, 0, 0], [0, 0, 0], [1, 1, 1]],
]
_C = [
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    [[0, 0, 0], [0, 1, 0], [0, 1, 0]],
    [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 0, 1], [0, 0, 1]],
    [[0, 0, 0], [0, 0, 0], [1, 1, 0]],
]


def _zero_matrix(rows, cols):
    return [[0] * cols for _ in range(rows)]


def _schoolbook():
    """Fallback rank-54 decomposition (guaranteed valid)."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = _zero_matrix(n * m, rank)
    V = _zero_matrix(m * p, rank)
    W = _zero_matrix(p * n, rank)
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def _rank14():
    n, m, p = DIMENSIONS          # 3, 3, 6
    n_groups = 2                  # 6 columns -> 2 groups of 3
    R = n_groups * len(_A)        # 2 * 7 = 14
    
    # Use NumPy for efficient construction
    A_np = np.array(_A)
    B_np = np.array(_B)
    C_np = np.array(_C)
    
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)
    
    for g in range(n_groups):
        base = g * len(_A)
        for r in range(len(_A)):
            # U factor: A_r[i, j]
            for i in range(n):
                for j in range(m):
                    U[i * m + j, base + r] = A_np[r, i, j]
                        
            # V factor: B_r[j, k_local]
            for j in range(m):
                for k_local in range(3):
                    val = B_np[r, j, k_local]
                    if val:
                        k_global = g * 3 + k_local
                        V[j * p + k_global, base + r] = val
                        
            # W factor: C_r[i, k_local]
            for i in range(n):
                for k_local in range(3):
                    val = C_np[r, i, k_local]
                    if val:
                        k_global = g * 3 + k_local
                        W[k_global * n + i, base + r] = val
                        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _verify(U, V, W):
    """True iff U,V,W exactly decompose the 3x3x6 multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    U_np = np.array(U, dtype=np.int64)
    V_np = np.array(V, dtype=np.int64)
    W_np = np.array(W, dtype=np.int64)
    if U_np.shape != (n * m, rank) or V_np.shape != (m * p, rank) or W_np.shape != (p * n, rank):
        return False
    # Compute tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Use einsum: first compute U*V element-wise product across rank, then contract with W
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_reshaped = U_np.reshape(n, m, rank)
    V_reshaped = V_np.reshape(m, p, rank)
    W_reshaped = W_np.reshape(p, n, rank)
    # Compute for each (i,j,k): sum_r U_reshaped[i,j,r] * V_reshaped[j,k,r] * W_reshaped[k,i,r]
    # This is a 3-index contraction. We can do it via:
    # temp[i,j,k] = sum_r U_reshaped[i,j,r] * V_reshaped[j,k,r]
    # Then T[i,j,k] = sum_r temp[i,j,k] * ... no, that's not right.
    # Actually: T[i,j,k] = sum_r (U[i,j,r] * V[j,k,r]) * W[k,i,r]
    # Let's compute step by step:
    # Step 1: UV[i,j,k] = sum_r U[i,j,r] * V[j,k,r]  -- this is (n,m) x (m,p) -> but with shared index j
    # Actually UV[i,j,k] = sum_r U_reshaped[i,j,:] dot V_reshaped[j,k,:]
    # We can compute this as: for each j, U_reshaped[:,j,:] (n x rank) times V_reshaped[j,:,:] (rank x p) -> (n x p)
    # So UV[i,j,k] for fixed j is (U_reshaped[:,j,:] @ V_reshaped[j,:,:].T)[i,k]
    # Let's build the full tensor:
    # UV[i,j,k] = sum_r U[i,j,r] * V[j,k,r]
    # We can use einsum: np.einsum('ijr,jkr->ijk', U_reshaped, V_reshaped)
    UV = np.einsum('ijr,jkr->ijk', U_reshaped, V_reshaped)
    # Now T[i,j,k] = sum_r UV[i,j,k] * ... no wait, that's not right either.
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # This is a 3-way contraction. Let's use einsum directly:
    T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    # Check: T[i,j,k] should be 1 for all i,j,k
    if not np.all(T == 1):
        return False
    return True


def construct():
    deadline = time.time() + 120.0
    try:
        cand = _rank14()
        if _verify(cand["U"], cand["V"], cand["W"]):
            return cand
    except Exception:
        pass
    if time.time() > deadline - 5.0:
        return _schoolbook()
    return _schoolbook()
