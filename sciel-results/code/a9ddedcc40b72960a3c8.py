import numpy as np
import time
import random
from itertools import combinations

DIMENSIONS = [2, 3, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verifies if U, V, W form a valid tensor decomposition for matrix multiplication.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Indices:
    U[i*m + j, r]
    V[j*p + k, r]
    W[k*n + i, r]
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = np.dot(U_arr[idx_u, :], np.multiply(V_arr[idx_v, :], W_arr[idx_w, :]))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _construct_rank_28():
    """
    Constructs the known rank-28 decomposition for 2x3 * 3x5.
    """
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: A2 B2 (j=2, k=0..4) - 10 terms
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1.0
            V[2 * p + k][r] = 1.0
            W[k * n + i][r] = 1.0
            r += 1
            
    # Part 2: A1 B13 (j=0,1, k=4) - 4 terms
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1.0
            V[j * p + 4][r] = 1.0
            W[4 * n + i][r] = 1.0
            r += 1

    # Part 3: A1 B11 (j=0,1, k=0,1) using Strassen-like 2x2 - 7 terms
    U[0 * m + 0][r] = 1.0
    U[1 * m + 1][r] = 1.0
    V[0 * p + 0][r] = 1.0
    V[1 * p + 1][r] = 1.0
    W[0 * n + 0][r] = 1.0
    W[1 * n + 1][r] = 1.0
    r += 1
    
    U[1 * m + 0][r] = 1.0
    U[1 * m + 1][r] = 1.0
    V[0 * p + 0][r] = 1.0
    W[0 * n + 1][r] = 1.0
    W[1 * n + 1][r] = -1.0
    r += 1
    
    U[0 * m + 0][r] = 1.0
    V[0 * p + 1][r] = 1.0
    V[1 * p + 1][r] = -1.0
    W[1 * n + 0][r] = 1.0
    W[1 * n + 1][r] = 1.0
    r += 1
    
    U[1 * m + 1][r] = 1.0
    V[1 * p + 0][r] = 1.0
    V[0 * p + 0][r] = -1.0
    W[0 * n + 0][r] = 1.0
    W[0 * n + 1][r] = 1.0
    r += 1
    
    U[0 * m + 0][r] = 1.0
    U[0 * m + 1][r] = 1.0
    V[1 * p + 1][r] = 1.0
    W[0 * n + 0][r] = -1.0
    W[1 * n + 0][r] = 1.0
    r += 1
    
    U[1 * m + 0][r] = 1.0
    U[0 * m + 0][r] = -1.0
    V[0 * p + 0][r] = 1.0
    V[0 * p + 1][r] = 1.0
    W[1 * n + 1][r] = 1.0
    r += 1
    
    U[0 * m + 1][r] = 1.0
    U[1 * m + 1][r] = -1.0
    V[1 * p + 0][r] = 1.0
    V[1 * p + 1][r] = 1.0
    W[0 * n + 0][r] = 1.0
    r += 1

    # Part 4: A1 B12 (j=0,1, k=2,3) using Strassen-like 2x2 - 7 terms
    U[0 * m + 0][r] = 1.0
    U[1 * m + 1][r] = 1.0
    V[0 * p + 2][r] = 1.0
    V[1 * p + 3][r] = 1.0
    W[2 * n + 0][r] = 1.0
    W[3 * n + 1][r] = 1.0
    r += 1
    
    U[1 * m + 0][r] = 1.0
    U[1 * m + 1][r] = 1.0
    V[0 * p + 2][r] = 1.0
    W[2 * n + 1][r] = 1.0
    W[3 * n + 1][r] = -1.0
    r += 1
    
    U[0 * m + 0][r] = 1.0
    V[0 * p + 3][r] = 1.0
    V[1 * p + 3][r] = -1.0
    W[3 * n + 0][r] = 1.0
    W[3 * n + 1][r] = 1.0
    r += 1
    
    U[1 * m + 1][r] = 1.0
    V[1 * p + 2][r] = 1.0
    V[0 * p + 2][r] = -1.0
    W[2 * n + 0][r] = 1.0
    W[2 * n + 1][r] = 1.0
    r += 1
    
    U[0 * m + 0][r] = 1.0
    U[0 * m + 1][r] = 1.0
    V[1 * p + 3][r] = 1.0
    W[2 * n + 0][r] = -1.0
    W[3 * n + 0][r] = 1.0
    r += 1
    
    U[1 * m + 0][r] = 1.0
    U[0 * m + 0][r] = -1.0
    V[0 * p + 2][r] = 1.0
    V[0 * p + 3][r] = 1.0
    W[3 * n + 1][r] = 1.0
    r += 1
    
    U[0 * m + 1][r] = 1.0
    U[1 * m + 1][r] = -1.0
    V[1 * p + 2][r] = 1.0
    V[1 * p + 3][r] = 1.0
    W[2 * n + 0][r] = 1.0
    r += 1
    
    return {"U": U, "V": V, "W": W}

def _peel_and_test_rank_27(U28, V28, W28, n, m, p, deadline):
    """
    Peeling technique: Try removing one term from the rank-28 decomposition and
    see if the remaining 27 terms + one new rank-1 term can exactly reproduce
    the multiplication tensor.
    
    For a removed term r0, the residual tensor R = T_mult - sum_{r != r0} u_r v_r w_r
    must itself be a rank-1 tensor (i.e., R[i,j,k] = a[i] * b[j] * c[k] for some vectors a, b, c).
    """
    n, m, p = 2, 3, 5
    U_arr = np.array(U28, dtype=float)
    V_arr = np.array(V28, dtype=float)
    W_arr = np.array(W28, dtype=float)
    rank28 = U_arr.shape[1]
    
    # Precompute the full multiplication tensor
    T_mult = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_mult[i, j, k] = 1.0  # The multiplication tensor has 1s at all positions
    
    # For each term to remove, compute the residual and check if it's rank-1
    for r0 in range(rank28):
        if time.time() > deadline:
            break
        
        # Compute sum of all terms except r0
        T_partial = np.zeros((n, m, p))
        for r in range(rank28):
            if r == r0:
                continue
            # Term r contributes to T[i,j,k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_partial[i, j, k] += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
        
        # Residual: what we need from the new term
        R = T_mult - T_partial
        
        # Check if R is a rank-1 tensor: R[i,j,k] = a[i] * b[j] * c[k]
        # A tensor is rank-1 iff all 2x2 minors of any unfolding are zero
        # For a 3D tensor, we can check by flattening and checking the matrix rank
        # of various unfoldings, or more directly:
        
        # Method: Pick a non-zero entry R[i0,j0,k0] = v0
        # Then a[i] = R[i,j0,k0]/v0, b[j] = R[i0,j,k0]/v0, c[k] = R[i0,j0,k]/v0
        # Then verify R[i,j,k] == a[i]*b[j]*c[k] for all i,j,k
        
        found_nonzero = False
        i0 = j0 = k0 = -1
        v0 = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(R[i, j, k]) > 1e-12:
                        found_nonzero = True
                        i0, j0, k0 = i, j, k
                        v0 = R[i, j, k]
                        break
                if found_nonzero:
                    break
            if found_nonzero:
                break
        
        if not found_nonzero:
            # R is the zero tensor, meaning the 27 terms already work!
            # This would mean rank 27 is achievable (actually rank <= 27)
            # Build the rank-27 solution by dropping term r0
            U27 = [row[:r0] + row[r0+1:] for row in U28]
            V27 = [row[:r0] + row[r0+1:] for row in V28]
            W27 = [row[:r0] + row[r0+1:] for row in W28]
            return {"U": U27, "V": V27, "W": W27}
        
        # Extract candidate vectors
        a = np.zeros(n)
        b = np.zeros(m)
        c = np.zeros(p)
        
        for i in range(n):
            a[i] = R[i, j0, k0] / v0
        for j in range(m):
            b[j] = R[i0, j, k0] / v0
        for k in range(p):
            c[k] = R[i0, j0, k] / v0
        
        # Verify: R[i,j,k] == a[i] * b[j] * c[k] for all i,j,k
        is_rank1 = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = a[i] * b[j] * c[k]
                    if abs(R[i, j, k] - expected) > 1e-9:
                        is_rank1 = False
                        break
                if not is_rank1:
                    break
            if not is_rank1:
                break
        
        if is_rank1:
            # Found a valid rank-27 decomposition!
            # Build the 27 kept terms + the new rank-1 term
            U27 = [row[:r0] + row[r0+1:] for row in U28]
            V27 = [row[:r0] + row[r0+1:] for row in V28]
            W27 = [row[:r0] + row[r0+1:] for row in W28]
            
            # Append the new rank-1 term
            new_U = [row + [a[i]] for i, row in enumerate(range(n*m))]
            # Wait, need to be careful with indexing
            new_U = []
            for idx in range(n * m):
                i = idx // m
                j = idx % m
                new_U.append(U27[idx] + [a[i]])
            
            new_V = []
            for idx in range(m * p):
                j = idx // p
                k = idx % p
                new_V.append(V27[idx] + [b[j]])
            
            new_W = []
            for idx in range(p * n):
                k = idx // n
                i = idx % n
                new_W.append(W27[idx] + [c[i]])
            
            return {"U": new_U, "V": new_V, "W": new_W}
    
    return None

def _try_find_lower_rank(deadline):
    """
    Attempts to find a rank-27 decomposition using the peeling technique.
    """
    n, m, p = 2, 3, 5
    
    # First construct the rank-28 decomposition
    res_28 = _construct_rank_28()
    
    # Try peeling
    result = _peel_and_test_rank_27(res_28['U'], res_28['V'], res_28['W'], n, m, p, deadline)
    
    if result is not None:
        return result
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # 1. Attempt to find a lower rank (e.g., 27) using peeling
    res_lower = _try_find_lower_rank(deadline)
    if res_lower is not None:
        if _verify_decomposition(res_lower['U'], res_lower['V'], res_lower['W'], 2, 3, 5):
            return res_lower
            
    # 2. Construct the known rank-28 decomposition
    res_28 = _construct_rank_28()
    
    # 3. Verify it
    if _verify_decomposition(res_28['U'], res_28['V'], res_28['W'], 2, 3, 5):
        return res_28
    else:
        raise ValueError("Known rank-28 decomposition failed verification")
