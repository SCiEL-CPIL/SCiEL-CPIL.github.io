"""Matrix multiplication 2x4x4 with optimized tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 4]


def _strassen_2x2():
    """
    Return the 7 Strassen multiplication patterns for 2x2 matrix multiplication.
    """
    patterns = [
        # M1 = (a11+a22)(b11+b22)
        {'a': {0: 1, 3: 1}, 'b': {0: 1, 3: 1}, 'c': {0: 1, 3: 1}},
        # M2 = (a21+a22)(b11)
        {'a': {2: 1, 3: 1}, 'b': {0: 1}, 'c': {2: 1, 3: -1}},
        # M3 = (a11)(b12-b22)
        {'a': {0: 1}, 'b': {1: 1, 3: -1}, 'c': {1: 1, 3: 1}},
        # M4 = (a22)(b21-b11)
        {'a': {3: 1}, 'b': {2: 1, 0: -1}, 'c': {0: 1, 2: 1}},
        # M5 = (a11+a12)(b22)
        {'a': {0: 1, 1: 1}, 'b': {3: 1}, 'c': {0: -1, 1: 1}},
        # M6 = (a21-a11)(b11+b12)
        {'a': {2: 1, 0: -1}, 'b': {0: 1, 1: 1}, 'c': {3: 1}},
        # M7 = (a12-a22)(b21+b22)
        {'a': {1: 1, 3: -1}, 'b': {2: 1, 3: 1}, 'c': {0: 1}},
    ]
    return patterns


def _build_strassen_decomposition(n=2, m=4, p=4):
    """
    Build a rank-28 Strassen-based decomposition for 2x4 * 4x4 matrix multiplication.
    """
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_patterns = _strassen_2x2()
    
    blocks = [
        (0, 0, 0, 0),
        (0, 0, 2, 2),
        (2, 2, 0, 0),
        (2, 2, 2, 2),
    ]
    
    col = 0
    for (a_co, b_ro, b_co, c_co) in blocks:
        for pattern in strassen_patterns:
            for local_a_idx, coeff in pattern['a'].items():
                local_i = local_a_idx // 2
                local_j = local_a_idx % 2
                global_i = local_i
                global_j = a_co + local_j
                U[global_i * m + global_j][col] = coeff
            
            for local_b_idx, coeff in pattern['b'].items():
                local_j = local_b_idx // 2
                local_k = local_b_idx % 2
                global_j = b_ro + local_j
                global_k = b_co + local_k
                V[global_j * p + global_k][col] = coeff
            
            for local_c_idx, coeff in pattern['c'].items():
                local_i = local_c_idx // 2
                local_k = local_c_idx % 2
                global_i = local_i
                global_k = c_co + local_k
                W[global_k * n + global_i][col] = coeff
            
            col += 1
    
    return U, V, W


def _verify_decomposition(U, V, W, n=2, m=4, p=4, seed=42, num_tests=10):
    """
    Verify that the decomposition correctly computes matrix multiplication.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    np.random.seed(seed)
    for _ in range(num_tests):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            a_factor = np.dot(U_arr[:, r], A.flatten())
            b_factor = np.dot(V_arr[:, r], B.flatten())
            if a_factor != 0 and b_factor != 0:
                C_computed += W_arr[:, r].reshape(p, n).T * (a_factor * b_factor)
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    
    return True


def _verify_tensor_exact(U, V, W, n=2, m=4, p=4):
    """
    Verify tensor equation exactly: sum_r U[i,r]*V[j,r]*W[k,r] = T[i,j,k]
    where T[i,j,k] = 1 if i//m * p + k%m == j//p and i%m == k//p, else 0.
    Actually: A is n x m, B is m x p, C is p x n (trace convention W[k*n+i])
    C[i,k] = sum_l A[i,l]*B[l,k]
    U index: i*m + l (row i, col l of A)
    V index: l*p + k (row l, col k of B)
    W index: k*n + i (col k, row i of C)
    T[i*m+l, l*p+k, k*n+i] = 1
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = len(U[0])
    
    max_err = 0.0
    # Check all entries that should be 1
    for i in range(n):
        for l in range(m):
            for k in range(p):
                u_idx = i * m + l
                v_idx = l * p + k
                w_idx = k * n + i
                val = 0.0
                for r in range(rank):
                    val += U_arr[u_idx, r] * V_arr[v_idx, r] * W_arr[w_idx, r]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    # Check entries that should be 0 (sample a subset for speed)
    # The tensor has n*m * m*p * p*n = 8*16*8 = 1024 entries total
    # 32 should be 1, rest should be 0
    for i in range(n):
        for l in range(m):
            for k in range(p):
                u_idx = i * m + l
                v_idx = l * p + k
                w_idx = k * n + i
                # Already checked above, skip
                continue
    
    # Check some zero entries
    for i in range(n):
        for l in range(m):
            for k in range(p):
                for k2 in range(p):
                    if k2 == k:
                        continue
                    u_idx = i * m + l
                    v_idx = l * p + k2
                    w_idx = k2 * n + i
                    val = 0.0
                    for r in range(rank):
                        val += U_arr[u_idx, r] * V_arr[v_idx, r] * W_arr[w_idx, r]
                    err = abs(val)
                    if err > max_err:
                        max_err = err
    
    return max_err


def _compute_slice_ranks(U, V, W, n=2, m=4, p=4):
    """
    Compute the slice ranks of the tensor to get lower bounds on the rank.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the full tensor T with shape (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n))
    for r in range(rank):
        T += np.einsum('i,j,k->ijk', U_arr[:, r], V_arr[:, r], W_arr[:, r])
    
    mode1_ranks = [np.linalg.matrix_rank(T[i, :, :]) for i in range(n * m)]
    mode1_max_rank = max(mode1_ranks)
    
    mode2_ranks = [np.linalg.matrix_rank(T[:, j, :]) for j in range(m * p)]
    mode2_max_rank = max(mode2_ranks)
    
    mode3_ranks = [np.linalg.matrix_rank(T[:, :, k]) for k in range(p * n)]
    mode3_max_rank = max(mode3_ranks)
    
    lower_bound = max(mode1_max_rank, mode2_max_rank, mode3_max_rank)
    
    return lower_bound, mode1_max_rank, mode2_max_rank, mode3_max_rank


def _try_rank_reduction(U, V, W, n=2, m=4, p=4, deadline=None):
    """
    Try to reduce rank by checking if any column can be eliminated
    by expressing it as a linear combination of others.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build tensor
    T = np.zeros((n * m, m * p, p * n))
    for r in range(rank):
        T += np.einsum('i,j,k->ijk', U_arr[:, r], V_arr[:, r], W_arr[:, r])
    
    # Try removing each column and see if we can still represent T
    # This is expensive, so we try a greedy approach
    best_rank = rank
    best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
    
    for r_remove in range(rank):
        if deadline and time.time() > deadline - 5:
            break
        
        # Try to remove column r_remove
        U_rem = np.delete(U_arr, r_remove, axis=1)
        V_rem = np.delete(V_arr, r_remove, axis=1)
        W_rem = np.delete(W_arr, r_remove, axis=1)
        
        # Check if T can be represented with remaining columns
        # This requires solving a linear system, which is complex
        # For now, just check if the removed column is in the span of others
        # by checking if the tensor contribution is in the span
        
        # Simple check: is the removed tensor slice a linear combination of others?
        # This is a necessary but not sufficient condition
        removed_slice = np.einsum('i,j,k->ijk', U_arr[:, r_remove], V_arr[:, r_remove], W_arr[:, r_remove])
        
        # Check if removed_slice is in the span of remaining slices
        # Flatten all tensors
        remaining_slices = []
        for r in range(rank):
            if r == r_remove:
                continue
            remaining_slices.append(np.einsum('i,j,k->ijk', U_arr[:, r], V_arr[:, r], W_arr[:, r]).flatten())
        
        if not remaining_slices:
            continue
            
        remaining_matrix = np.array(remaining_slices).T  # shape (n*m*m*p*p*n, rank-1)
        removed_flat = removed_slice.flatten()
        
        # Check if removed_flat is in the column space of remaining_matrix
        try:
            # Project removed_flat onto column space
            Q, R = np.linalg.qr(remaining_matrix, mode='reduced')
            coeffs = Q.T @ removed_flat
            reconstruction = Q @ coeffs
            residual = np.linalg.norm(removed_flat - reconstruction)
            
            if residual < 1e-10:
                # Column can be removed!
                best_rank = rank - 1
                best_U = U_rem
                best_V = V_rem
                best_W = W_rem
                break
        except np.linalg.LinAlgError:
            continue
    
    if best_rank < rank:
        # Convert back to list of lists
        best_U_list = best_U.tolist()
        best_V_list = best_V.tolist()
        best_W_list = best_W.tolist()
        return best_U_list, best_V_list, best_W_list, best_rank
    
    return U, V, W, rank


def construct():
    """
    Construct a low-rank tensor decomposition for 2x4 * 4x4 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    # Build the rank-28 Strassen decomposition
    U, V, W = _build_strassen_decomposition(n, m, p)
    
    # Verify the decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Strassen decomposition verification failed")
    
    # Try to reduce rank
    U, V, W, rank = _try_rank_reduction(U, V, W, n, m, p, deadline)
    
    # Final verification
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to original
        U, V, W = _build_strassen_decomposition(n, m, p)
    
    return {"U": U, "V": V, "W": W}
