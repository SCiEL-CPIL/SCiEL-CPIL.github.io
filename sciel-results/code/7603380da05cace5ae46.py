"""Tensor decomposition for 2x4 * 4x6 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Check if the tensor decomposition is exact using vectorized computation."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    
    # Vectorized verification:
    # T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    # Expected: T[i,j,k] = 1 iff (i%m == j//p) and (i//m == k%n) and (j%p == k//n)
    
    # Build index arrays
    i_A = np.arange(n * m) // m
    j_A = np.arange(n * m) % m
    j_B = np.arange(m * p) // p
    k_B = np.arange(m * p) % p
    k_C = np.arange(p * n) // n
    i_C = np.arange(p * n) % n
    
    # Expected tensor: T[i,j,k] = 1 iff j_A[i] == j_B[j] and i_A[i] == i_C[k] and k_B[j] == k_C[k]
    # Compute using broadcasting
    # Shape: (n*m, m*p, p*n)
    cond1 = (j_A[:, None, None] == j_B[None, :, None])  # (n*m, m*p, 1)
    cond2 = (i_A[:, None, None] == i_C[None, None, :])  # (n*m, 1, p*n)
    cond3 = (k_B[None, :, None] == k_C[None, None, :])  # (1, m*p, p*n)
    
    expected = cond1 & cond2 & cond3
    
    # Compute actual tensor: T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    # = sum_c U[i,c] * (V[j,c] * W[k,c])
    # For each c: outer product of U[:,c] and (V[:,c] * W[:,c])... no
    # T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    # This is a 3-way outer product summed over c
    
    # Efficient: T = sum_c U[:,c,None,None] * V[None,:,c,None] * W[None,None,:c]
    # But that's memory heavy. Instead, use:
    # T[i,j,k] = sum_c U[i,c] * (V[j,c] * W[k,c])
    # = sum_c U[i,c] * (V @ W^T)[j,k]... no, that's not right either
    
    # Let's do it in chunks or use einsum
    # T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    # = np.einsum('ic,jc,kc->ijk', U, V, W)
    
    actual = np.einsum('ic,jc,kc->ijk', U, V, W)
    
    max_err = np.max(np.abs(actual - expected.astype(float)))
    return max_err < 1e-9

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_rank_optimized(rank, n, m, p, time_limit=10):
    """Try to find a decomposition of given rank using alternating least squares."""
    start_time = time.time()
    
    # Precompute valid index combinations
    valid_triples = []
    for i_A in range(n):
        for j in range(m):
            for k in range(p):
                i = i_A * m + j
                j_idx = j * p + k
                k_idx = k * n + i_A
                valid_triples.append((i, j_idx, k_idx))
    
    valid_triples = np.array(valid_triples)
    n_valid = len(valid_triples)
    
    np.random.seed(42)
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    for attempt in range(10):
        if time.time() - start_time > time_limit:
            break
        
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        for iteration in range(200):
            if time.time() - start_time > time_limit:
                break
            
            # Update U: for each i, solve least squares
            for i in range(n * m):
                i_A = i // m
                j_A = i % m
                
                # Find all (j, k) where the triple is valid
                valid_jk = []
                for j in range(m * p):
                    j_B = j // p
                    k_B = j % p
                    if j_A == j_B:
                        for k in range(p * n):
                            k_C = k // n
                            i_C = k % n
                            if i_A == i_C and k_B == k_C:
                                valid_jk.append((j, k))
                
                if not valid_jk:
                    continue
                
                valid_jk = np.array(valid_jk)
                
                # Build A matrix: A[row, col] = V[j, col] * W[k, col]
                # b[row] = 1.0
                A_mat = np.zeros((len(valid_jk), rank))
                for idx, (j, k) in enumerate(valid_jk):
                    A_mat[idx] = V[j] * W[k]
                
                b_vec = np.ones(len(valid_jk))
                
                # Solve least squares: U[i,:] = A_mat^+ b_vec
                try:
                    U[i] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
            
            # Update V
            for j in range(m * p):
                j_B = j // p
                k_B = j % p
                
                valid_ik = []
                for i in range(n * m):
                    i_A = i // m
                    j_A = i % m
                    if j_A == j_B:
                        for k in range(p * n):
                            k_C = k // n
                            i_C = k % n
                            if i_A == i_C and k_B == k_C:
                                valid_ik.append((i, k))
                
                if not valid_ik:
                    continue
                
                valid_ik = np.array(valid_ik)
                
                A_mat = np.zeros((len(valid_ik), rank))
                for idx, (i, k) in enumerate(valid_ik):
                    A_mat[idx] = U[i] * W[k]
                
                b_vec = np.ones(len(valid_ik))
                
                try:
                    V[j] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
            
            # Update W
            for k in range(p * n):
                k_C = k // n
                i_C = k % n
                
                valid_ij = []
                for i in range(n * m):
                    i_A = i // m
                    j_A = i % m
                    if i_A == i_C:
                        for j in range(m * p):
                            j_B = j // p
                            k_B = j % p
                            if j_A == j_B and k_B == k_C:
                                valid_ij.append((i, j))
                
                if not valid_ij:
                    continue
                
                valid_ij = np.array(valid_ij)
                
                A_mat = np.zeros((len(valid_ij), rank))
                for idx, (i, j) in enumerate(valid_ij):
                    A_mat[idx] = U[i] * V[j]
                
                b_vec = np.ones(len(valid_ij))
                
                try:
                    W[k] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
            
            # Check error periodically
            if iteration % 10 == 0:
                actual = np.einsum('ic,jc,kc->ijk', U, V, W)
                cond1 = (j_A[:, None, None] == j_B[None, :, None])
                cond2 = (i_A[:, None, None] == i_C[None, None, :])
                cond3 = (k_B[None, :, None] == k_C[None, None, :])
                expected = (cond1 & cond2 & cond3).astype(float)
                err = np.max(np.abs(actual - expected))
                
                if err < best_err:
                    best_err = err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                
                if best_err < 1e-9:
                    break
    
    if best_err < 1e-9 and best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return None

def _try_rank_with_structure(rank, n, m, p, time_limit=10):
    """Try to find a decomposition exploiting the tensor structure."""
    start_time = time.time()
    
    # The tensor T[i,j,k] = 1 iff i%m == j//p, i//m == k%n, j%p == k//n
    # Let's think about this differently.
    # For fixed i (i_A, j_A), the valid (j, k) pairs are:
    #   j = j_A * p + k_B for k_B in [0,p)
    #   k = k_B * n + i_A for k_B in [0,p)
    # So for fixed i, there are p valid (j,k) pairs.
    
    # Similarly for fixed j (j_B, k_B), valid (i,k) pairs:
    #   i = i_A * m + j_B for i_A in [0,n)
    #   k = k_B * n + i_A for i_A in [0,n)
    # So n valid pairs.
    
    # For fixed k (k_C, i_C), valid (i,j) pairs:
    #   i = i_C * m + j_A for j_A in [0,m)
    #   j = j_A * p + k_C for j_A in [0,m)
    # So m valid pairs.
    
    # This structure suggests we can use a more efficient approach.
    # Let's try to find a decomposition where each "slice" has a simple form.
    
    np.random.seed(123)
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    for attempt in range(5):
        if time.time() - start_time > time_limit:
            break
        
        U = np.random.randn(n * m, rank) * 0.3
        V = np.random.randn(m * p, rank) * 0.3
        W = np.random.randn(p * n, rank) * 0.3
        
        for iteration in range(300):
            if time.time() - start_time > time_limit:
                break
            
            # Update U using the structure
            for i_A in range(n):
                for j_A in range(m):
                    i = i_A * m + j_A
                    
                    # Valid (j, k) pairs: j = j_A*p + k_B, k = k_B*n + i_A for k_B in [0,p)
                    A_mat = np.zeros((p, rank))
                    for k_B in range(p):
                        j = j_A * p + k_B
                        k = k_B * n + i_A
                        A_mat[k_B] = V[j] * W[k]
                    
                    b_vec = np.ones(p)
                    
                    try:
                        U[i] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                    except:
                        pass
            
            # Update V using the structure
            for j_B in range(m):
                for k_B in range(p):
                    j = j_B * p + k_B
                    
                    # Valid (i, k) pairs: i = i_A*m + j_B, k = k_B*n + i_A for i_A in [0,n)
                    A_mat = np.zeros((n, rank))
                    for i_A in range(n):
                        i = i_A * m + j_B
                        k = k_B * n + i_A
                        A_mat[i_A] = U[i] * W[k]
                    
                    b_vec = np.ones(n)
                    
                    try:
                        V[j] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                    except:
                        pass
            
            # Update W using the structure
            for k_C in range(p):
                for i_C in range(n):
                    k = k_C * n + i_C
                    
                    # Valid (i, j) pairs: i = i_C*m + j_A, j = j_A*p + k_C for j_A in [0,m)
                    A_mat = np.zeros((m, rank))
                    for j_A in range(m):
                        i = i_C * m + j_A
                        j = j_A * p + k_C
                        A_mat[j_A] = U[i] * V[j]
                    
                    b_vec = np.ones(m)
                    
                    try:
                        W[k] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                    except:
                        pass
            
            # Check error periodically
            if iteration % 20 == 0:
                actual = np.einsum('ic,jc,kc->ijk', U, V, W)
                
                i_A_arr = np.arange(n * m) // m
                j_A_arr = np.arange(n * m) % m
                j_B_arr = np.arange(m * p) // p
                k_B_arr = np.arange(m * p) % p
                k_C_arr = np.arange(p * n) // n
                i_C_arr = np.arange(p * n) % n
                
                cond1 = (j_A_arr[:, None, None] == j_B_arr[None, :, None])
                cond2 = (i_A_arr[:, None, None] == i_C_arr[None, None, :])
                cond3 = (k_B_arr[None, :, None] == k_C_arr[None, None, :])
                expected = (cond1 & cond2 & cond3).astype(float)
                
                err = np.max(np.abs(actual - expected))
                
                if err < best_err:
                    best_err = err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                
                if best_err < 1e-9:
                    break
    
    if best_err < 1e-9 and best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    schoolbook_rank = n * m * p  # 48
    
    # Try to find lower-rank decompositions
    # Start from a reasonable lower bound and go up
    # The rank must be at least max(n*m, m*p, p*n) = max(8, 24, 12) = 24
    # But likely higher due to the structure
    
    for rank in range(24, schoolbook_rank + 1):
        if time.time() > deadline - 20:
            break
        
        remaining_time = deadline - time.time() - 5
        if remaining_time < 5:
            break
        
        # Try structured approach first
        result = _try_rank_with_structure(rank, n, m, p, time_limit=min(15, remaining_time))
        if result is not None:
            if _verify(result["U"], result["V"], result["W"], n, m, p):
                return result
        
        # Try general approach if time remains
        remaining_time = deadline - time.time() - 5
        if remaining_time > 10:
            result = _try_rank_optimized(rank, n, m, p, time_limit=min(10, remaining_time))
            if result is not None:
                if _verify(result["U"], result["V"], result["W"], n, m, p):
                    return result
    
    # Fall back to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
