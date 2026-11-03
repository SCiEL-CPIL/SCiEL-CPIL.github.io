"""
Matrix multiplication tensor decomposition for 2x3 * 3x10.

Strategy:
  1. Build a known rank-55 decomposition (Strassen on 2x2 blocks + naive 2x1*1x10).
  2. Attempt to reduce rank to 54 by checking if one term can be expressed 
     as a linear combination of the others (with scalar coefficients).
  3. If such a linear combination exists and the resulting entries are valid 
     (integers or half-integers), return the rank-54 decomposition.
  4. Otherwise, return the rank-55 decomposition.
"""

import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 10]


def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """
    Fast verification using matrix multiplication.
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    U_mat = U.reshape(n, m, rank)
    V_mat = V.reshape(m, p, rank)
    W_mat = W.reshape(p, n, rank)
    
    for i in range(n):
        for k in range(p):
            Uw = U_mat[i] * W_mat[k, i]  # shape (m, rank)
            M_ik = Uw @ V_mat[:, k].T  # shape (m, m)
            if not np.allclose(M_ik, np.eye(m), atol=tol):
                return False
    return True


def _is_valid_entries(U, V, W):
    """Check that all entries are integers or half-integers."""
    for matrix in [U, V, W]:
        for row in matrix:
            for val in row:
                # Check if val is close to k/2 for some integer k
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    return True


def _build_strassen55(n, m, p):
    """
    Build the rank-55 decomposition using Strassen on the 2x2 sub-block.
    """
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part A: j=0 (first column of A, first row of B)
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1

    # Part B: j=1,2 (last 2 cols of A, last 2 rows of B)
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1

        u_11 = 0 * m + 1
        u_12 = 0 * m + 2
        u_21 = 1 * m + 1
        u_22 = 1 * m + 2

        v_11 = 1 * p + k0
        v_12 = 1 * p + k1
        v_21 = 2 * p + k0
        v_22 = 2 * p + k1

        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1

        # M1 = (a11 + a22)(b11 + b22)
        U[u_11][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        W[w_11][col] += 1
        col += 1

        # M2 = (a21 + a22)b11
        U[u_21][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        W[w_10][col] += 1
        W[w_11][col] += -1
        col += 1

        # M3 = a11(b12 - b22)
        U[u_11][col] += 1
        V[v_12][col] += 1
        V[v_22][col] += -1
        W[w_01][col] += 1
        W[w_11][col] += 1
        col += 1

        # M4 = a22(b21 - b11)
        U[u_22][col] += 1
        V[v_21][col] += 1
        V[v_11][col] += -1
        W[w_00][col] += 1
        W[w_10][col] += 1
        col += 1

        # M5 = (a11 + a12)b22
        U[u_11][col] += 1
        U[u_12][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += -1
        W[w_01][col] += 1
        col += 1

        # M6 = (a21 - a11)(b11 + b12)
        U[u_21][col] += 1
        U[u_11][col] += -1
        V[v_11][col] += 1
        V[v_12][col] += 1
        W[w_11][col] += 1
        col += 1

        # M7 = (a12 - a22)(b21 + b22)
        U[u_12][col] += 1
        U[u_22][col] += -1
        V[v_21][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        col += 1

    assert col == rank, f"Expected {rank} columns, got {col}"
    return U, V, W


def _try_linear_reduction(U, V, W, n, m, p):
    """
    Try to find a rank-54 decomposition by expressing one term as a linear
    combination of the others.
    """
    rank = 55
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Flatten the target tensor
    # T[i, j, k] = 1 if indices match, 0 otherwise
    # We represent the tensor as a vector of length n*m*p
    target = np.zeros(n * m * p, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = (i * m + j) * (m * p) + (j * p + k)
                # Note: The tensor is T[a, b, c] where a=i*m+j, b=j*p+k, c=k*n+i
                # We need to map this to a 1D index.
                # The verification uses:
                # sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                # This corresponds to the coefficient of the term (i,j,k) in the tensor product space.
                # Let's construct the vector of rank-1 tensors.
                
    # Construct matrix A of shape (n*m*p, rank)
    # A[t, r] = (U_r \otimes V_r \otimes W_r)_t
    A = np.zeros((n * m * p, rank), dtype=np.float64)
    
    for r in range(rank):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        
        # u has shape (n*m,), v has shape (m*p,), w has shape (p*n,)
        # We need to fill A[:, r] such that A[t, r] is the value of the tensor at index t.
        # The tensor index t corresponds to (i, j, k) -> a=i*m+j, b=j*p+k, c=k*n+i.
        # The value is u[a] * v[b] * w[c].
        
        for i in range(n):
            for j in range(m):
                a = i * m + j
                u_val = u[a]
                if u_val == 0:
                    continue
                for k in range(p):
                    b = j * p + k
                    c = k * n + i
                    v_val = v[b]
                    w_val = w[c]
                    val = u_val * v_val * w_val
                    if val != 0:
                        # Map (a, b, c) to 1D index. 
                        # The space is (n*m) x (m*p) x (p*n).
                        # Total size n*m * m*p * p*n.
                        # But we only care about the support?
                        # No, the linear system must hold for ALL entries.
                        # However, the target tensor is 0 outside the support.
                        # And the rank-1 tensors might have non-zeros outside the support.
                        # So we must include all entries.
                        
                        # Index calculation:
                        # flat_index = a * (m*p) * (p*n) + b * (p*n) + c
                        flat_idx = a * (m * p) * (p * n) + b * (p * n) + c
                        A[flat_idx, r] = val

    # Target vector
    target_vec = np.zeros(n * m * p, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                flat_idx = a * (m * p) * (p * n) + b * (p * n) + c
                target_vec[flat_idx] = 1.0

    # Try dropping each term
    for drop_idx in range(rank):
        keep_idx = [i for i in range(rank) if i != drop_idx]
        A_sub = A[:, keep_idx]
        
        # Solve A_sub * coeffs = target_vec
        try:
            coeffs, residuals, rank_A, sv = np.linalg.lstsq(A_sub, target_vec, rcond=None)
        except Exception:
            continue
            
        if len(residuals) > 0 and residuals[0] > 1e-8:
            continue
            
        # Check if coeffs are "nice"
        # We need to construct new U, V, W.
        # New term r' (from keep_idx) is scaled by coeffs[i].
        # U_new[:, r'] = coeffs[i] * U[:, r']
        # V_new[:, r'] = V[:, r']
        # W_new[:, r'] = W[:, r']
        # This is a valid decomposition if coeffs are valid.
        # But wait, if we scale U by coeffs, we are changing the rank-1 tensor.
        # Tensor = sum (coeffs[i] U_i) \otimes V_i \otimes W_i.
        # This is equivalent to sum U'_i \otimes V_i \otimes W_i.
        # So we just need U'_i entries to be valid.
        
        valid = True
        for i, ci in enumerate(coeffs):
            if abs(ci * 2 - round(ci * 2)) > 1e-6:
                valid = False
                break
                
        if not valid:
            continue
            
        # Construct new matrices
        new_rank = len(keep_idx)
        U_new = np.zeros((n * m, new_rank), dtype=np.float64)
        V_new = np.zeros((m * p, new_rank), dtype=np.float64)
        W_new = np.zeros((p * n, new_rank), dtype=np.float64)
        
        for new_i, old_i in enumerate(keep_idx):
            c = coeffs[new_i]
            U_new[:, new_i] = U[:, old_i] * c
            V_new[:, new_i] = V[:, old_i]
            W_new[:, new_i] = W[:, old_i]
            
        # Verify
        if _verify_fast(U_new.tolist(), V_new.tolist(), W_new.tolist(), n, m, p):
            if _is_valid_entries(U_new.tolist(), V_new.tolist(), W_new.tolist()):
                return U_new.tolist(), V_new.tolist(), W_new.tolist()
                
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

    # Start with the known rank-55 Strassen-based decomposition
    U_best, V_best, W_best = _build_strassen55(n, m, p)
    best_rank = 55

    if not (_verify_fast(U_best, V_best, W_best, n, m, p) and _is_valid_entries(U_best, V_best, W_best)):
        # Fallback: Schoolbook (Rank 60)
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
        U_best, V_best, W_best = U, V, W
        best_rank = rank

    # Try to find rank 54
    if time.time() < deadline - 20:
        result = _try_linear_reduction(U_best, V_best, W_best, n, m, p)
        if result is not None:
            U_best, V_best, W_best = result
            best_rank = 54
            return {"U": U_best, "V": V_best, "W": W_best}

    return {"U": U_best, "V": V_best, "W": W_best}
