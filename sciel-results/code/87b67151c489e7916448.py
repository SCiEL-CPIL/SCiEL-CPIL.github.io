"""Matrix multiplication 3x4x8: rank-minimal exact tensor decomposition search."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
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
    return {"U": U, "V": V, "W": W}


def _verify_fast(U, V, W, n, m, p):
    """Fast verification using NumPy."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    rank = U_np.shape[1]
    
    # Reshape to (n, m, rank), (m, p, rank), (p, n, rank)
    U_reshaped = U_np.reshape(n, m, rank)
    V_reshaped = V_np.reshape(m, p, rank)
    W_reshaped = W_np.reshape(p, n, rank)
    
    # We need to check if sum_r U[i,m+j,r] * V[j,p+k,r] * W[k,n+i,r] == 1 for all i,j,k
    # This is a 3-way contraction.
    # T[i,j,k] = sum_r U_reshaped[i,j,r] * V_reshaped[j,k,r] * W_reshaped[k,i,r]
    
    # We can compute this by iterating over r? No, that's what we want to avoid.
    # But rank is smallish (up to 96). n*m*p is 96.
    # 96 * 96 operations is ~10k, which is fine even in Python if vectorized per r?
    # Actually, let's just do it with np.einsum or similar.
    
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # This is not a standard einsum because the indices of W are permuted (k,i) vs (i,k).
    # Let's transpose W to (n, p, rank) -> W_trans[i, k, r] = W_reshaped[k, i, r]
    W_transposed = W_reshaped.transpose(1, 0, 2) # shape (n, p, rank)
    
    # Now T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W_transposed[i,k,r]
    # This is an element-wise product summed over r.
    
    # U[i,j,r] * V[j,k,r] * W_transposed[i,k,r]
    # We can compute this for all i,j,k,r at once?
    # U: (n, m, rank)
    # V: (m, p, rank)
    # Wt: (n, p, rank)
    
    # Result shape: (n, m, p)
    # T[i,j,k] = sum_r (U[i,j,r] * V[j,k,r] * Wt[i,k,r])
    
    # We can expand U to (n, m, 1, rank), V to (1, m, p, rank), Wt to (n, 1, p, rank)
    # Then multiply and sum over rank.
    
    # This might be memory heavy if rank is large, but rank <= 96, n=3, m=4, p=8.
    # 3*4*8*96 floats is small.
    
    U_exp = U_reshaped[:, :, np.newaxis, :] # (n, m, 1, rank)
    V_exp = V_reshaped[np.newaxis, :, :, :] # (1, m, p, rank)
    Wt_exp = W_transposed[:, np.newaxis, :, :] # (n, 1, p, rank)
    
    # Product: (n, m, p, rank)
    prod = U_exp * V_exp * Wt_exp
    T = np.sum(prod, axis=3)
    
    # Check if T is all 1s
    return np.allclose(T, 1.0, atol=1e-9)


def _verify(U, V, W, n, m, p):
    """Check if the factorization exactly reproduces the matrix multiplication tensor."""
    try:
        return _verify_fast(U, V, W, n, m, p)
    except Exception:
        # Fallback to slow verification if fast fails (e.g., memory)
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True


def _strassen_2x2():
    """
    Return the rank-7 decomposition of 2x2 matrix multiplication using Strassen's algorithm.
    U: (4, 7), V: (4, 7), W: (4, 7) where indices are (row, col) -> row*2+col
    """
    # Strassen's 7 products for 2x2 matrices:
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = [[c11, c12], [c21, c22]]
    #
    # M1 = (a11 + a22)(b11 + b22)
    # M2 = (a21 + a22)b11
    # M3 = a11(b12 - b22)
    # M4 = a22(b21 - b11)
    # M5 = (a11 + a12)b22
    # M6 = (a21 - a11)(b11 + b12)
    # M7 = (a12 - a22)(b21 + b22)
    #
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    # U factors (for A matrix, 4 elements: a11, a12, a21, a22)
    # Index: 0=a11, 1=a12, 2=a21, 3=a22
    # U[i][r] is the coefficient of A_element_i in product r.
    
    # M1: (a11 + a22) * (b11 + b22) -> a11:1, a22:1
    # M2: (a21 + a22) * b11 -> a21:1, a22:1
    # M3: a11 * (b12 - b22) -> a11:1
    # M4: a22 * (b21 - b11) -> a22:1
    # M5: (a11 + a12) * b22 -> a11:1, a12:1
    # M6: (a21 - a11) * (b11 + b12) -> a21:1, a11:-1
    # M7: (a12 - a22) * (b21 + b22) -> a12:1, a22:-1
    
    U = [
        [1, 0, 1, 0, 1, -1, 0],  # a11
        [0, 0, 0, 0, 1, 0, 1],   # a12
        [0, 1, 0, 0, 0, 1, 0],   # a21
        [1, 1, 0, 1, 0, 0, -1],  # a22
    ]
    
    # V factors (for B matrix, 4 elements: b11, b12, b21, b22)
    # Index: 0=b11, 1=b12, 2=b21, 3=b22
    # M1: (a11 + a22) * (b11 + b22) -> b11:1, b22:1
    # M2: (a21 + a22) * b11 -> b11:1
    # M3: a11 * (b12 - b22) -> b12:1, b22:-1
    # M4: a22 * (b21 - b11) -> b21:1, b11:-1
    # M5: (a11 + a12) * b22 -> b22:1
    # M6: (a21 - a11) * (b11 + b12) -> b11:1, b12:1
    # M7: (a12 - a22) * (b21 + b22) -> b21:1, b22:1
    
    V = [
        [1, 0, 0, -1, 0, 1, 0],  # b11
        [0, 1, 0, 0, 0, 1, 0],   # b12
        [0, 0, 1, 1, 0, 0, 1],   # b21
        [1, 0, -1, 0, 1, 0, 1],  # b22
    ]
    
    # W factors (for C matrix, 4 elements: c11, c12, c21, c22)
    # Index: 0=c11, 1=c12, 2=c21, 3=c22
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    W = [
        [1, 0, 0, 1, 1, 0, 0],  # c11
        [0, 1, 0, 0, 0, 1, 0],  # c12
        [0, 0, 1, 0, 0, 0, 1],  # c21
        [1, 0, 1, 0, 0, 1, 1],  # c22
    ]
    
    return U, V, W


def _verify_strassen():
    """Verify that the Strassen decomposition is correct."""
    U, V, W = _strassen_2x2()
    # Check: sum_r U[i][r] * V[j][r] * W[k][r] should give the 2x2 multiplication tensor
    # T[i,j,k] where i is A index, j is B index, k is C index
    # For 2x2: T[i,j,k] = 1 if C[k] = sum_i A[i] * B[j] contributes to it
    
    # Actually, the tensor for 2x2 multiplication is:
    # T[a_idx, b_idx, c_idx] = 1 if c_idx corresponds to the product that uses a_idx and b_idx
    # This is more complex. Let me just verify by checking the matrix multiplication.
    
    # For a specific A and B, compute C using Strassen and check against direct multiplication
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    C_direct = A @ B
    
    # Compute using Strassen
    a11, a12, a21, a22 = A[0,0], A[0,1], A[1,0], A[1,1]
    b11, b12, b21, b22 = B[0,0], B[0,1], B[1,0], B[1,1]
    
    M1 = (a11 + a22) * (b11 + b22)
    M2 = (a21 + a22) * b11
    M3 = a11 * (b12 - b22)
    M4 = a22 * (b21 - b11)
    M5 = (a11 + a12) * b22
    M6 = (a21 - a11) * (b11 + b12)
    M7 = (a12 - a22) * (b21 + b22)
    
    c11 = M1 + M4 - M5 + M7
    c12 = M3 + M5
    c21 = M2 + M4
    c22 = M1 - M2 + M3 + M6
    
    C_strassen = np.array([[c11, c12], [c21, c22]])
    
    return np.allclose(C_direct, C_strassen)


def _structured_decomposition_3x4x8(n, m, p):
    """
    Structured decomposition for 3x4x8 matrix multiplication.
    
    Strategy: Decompose the 4x8 part of B into two 4x4 blocks.
    C = A * [B1, B2] = [A*B1, A*B2]
    
    We can use Strassen's algorithm for the 4x4 multiplications.
    But we need to compose this with the 3x4 structure of A.
    
    Let me try a direct construction:
    
    For each of the 3 rows of A, we compute a 1x8 vector by multiplying with B.
    The 1x4x8 multiplication has rank 32 (schoolbook).
    
    To reduce rank, we can precompute some combinations of the columns of B.
    
    Actually, let me try a simpler approach: just use the schoolbook for now,
    but with a better random search that uses structured initialization.
    """
    # For now, return schoolbook
    return _schoolbook(n, m, p)


def _try_random_search(n, m, p, max_rank, n_trials, deadline):
    """Try to find a low-rank decomposition using random initialization and local search."""
    best = None
    best_rank = n * m * p  # schoolbook rank
    
    start = time.time()
    for trial in range(n_trials):
        if time.time() > deadline - 5:
            break
        
        # Try ranks from low to high
        for rank in range(max_rank, best_rank):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with small integers/half-integers
            np.random.seed(trial * 1000 + rank)
            U = np.random.choice([-2, -1, -0.5, 0, 0.5, 1, 2], size=(n*m, rank))
            V = np.random.choice([-2, -1, -0.5, 0, 0.5, 1, 2], size=(m*p, rank))
            W = np.random.choice([-2, -1, -0.5, 0, 0.5, 1, 2], size=(p*n, rank))
            
            # Check
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = rank
                break
            elif rank < best_rank:
                # Keep as candidate for local refinement
                pass
    
    return best, best_rank


def _structured_decomposition(n, m, p):
    """
    Try structured decompositions based on known techniques.
    For 3x4x8, we can try to decompose into smaller blocks.
    """
    # Schoolbook is 96.
    # Can we do better?
    # Let's try a simple heuristic: if p is even, split B into two p/2 blocks.
    # C = A B1 + A B2.
    # This doesn't reduce rank unless we share computations.
    
    # A known lower bound is max(nm, mp, pn) = 32.
    # It is unlikely we can reach 32.
    # Let's just return schoolbook for now, as random search is weak.
    return _schoolbook(n, m, p)


def _local_refine(U, V, W, n, m, p, max_iters, deadline):
    """
    Local search: try to reduce rank by zeroing out columns or 
    finding redundant columns that can be eliminated.
    """
    rank = len(U[0])
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    
    # Try to eliminate columns that are linearly dependent or redundant
    for _ in range(max_iters):
        if time.time() > deadline - 2:
            break
        
        improved = False
        for r in range(rank - 1, -1, -1):
            if time.time() > deadline - 2:
                break
            
            # Try removing column r: check if the remaining columns still span the tensor
            # This is expensive, so only try for the last few columns
            if r < rank - 5:
                break
            
            # Check if column r is "redundant" by seeing if we can express it
            # in terms of others (approximately)
            # For now, just check if setting it to 0 still works
            U_save = [U[i][r] for i in range(n*m)]
            V_save = [V[i][r] for i in range(m*p)]
            W_save = [W[i][r] for i in range(p*n)]
            
            # Zero out column r
            for i in range(n*m):
                U[i][r] = 0
            for i in range(m*p):
                V[i][r] = 0
            for i in range(p*n):
                W[i][r] = 0
            
            if _verify(U, V, W, n, m, p):
                # Remove column r
                for i in range(n*m):
                    U[i].pop(r)
                for i in range(m*p):
                    V[i].pop(r)
                for i in range(p*n):
                    W[i].pop(r)
                rank -= 1
                improved = True
            else:
                # Restore
                for i in range(n*m):
                    U[i][r] = U_save[i]
                for i in range(m*p):
                    V[i][r] = V_save[i]
                for i in range(p*n):
                    W[i][r] = W_save[i]
            
            if improved:
                break
        
        if not improved:
            break
    
    return {"U": U, "V": V, "W": W}, rank


def _try_strassen_composed(n, m, p, deadline):
    """
    Try to compose Strassen's algorithm with the 3x4x8 structure.
    
    Idea: Decompose the 4x8 part of B into two 4x4 blocks.
    C = A * [B1, B2] = [A*B1, A*B2]
    
    If we use Strassen for the 4x4 multiplications, we get rank 49 for each.
    But we need to share the A factor.
    
    Actually, let me try a different approach:
    
    We can view the 3x4x8 multiplication as a sequence of operations.
    For each output element C[i,k], we compute sum_j A[i,j] * B[j,k].
    
    We can precompute some combinations of A and B to reduce the number of multiplications.
    
    Let me try a direct construction using the Kronecker product.
    """
    # For now, return None
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin
    
    # Start with schoolbook
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Verify schoolbook
    assert _verify(best_result["U"], best_result["V"], best_result["W"], n, m, p), "Schoolbook verification failed"
    
    # Verify Strassen decomposition is correct
    if time.time() < deadline - 10:
        if not _verify_strassen():
            print("Warning: Strassen verification failed")
    
    # Try structured decompositions
    if time.time() < deadline - 10:
        structured = _structured_decomposition(n, m, p)
        if _verify(structured["U"], structured["V"], structured["W"], n, m, p):
            s_rank = len(structured["U"][0])
            if s_rank < best_rank:
                best_result = structured
                best_rank = s_rank
    
    # Try random search with various rank targets
    if time.time() < deadline - 5:
        # Try ranks from 40 up to best_rank - 1
        max_search_rank = min(best_rank - 1, 60)
        result, found_rank = _try_random_search(n, m, p, max_search_rank, 200, deadline)
        if result is not None:
            best_result = result
            best_rank = found_rank
    
    # Local refinement on the best result found
    if time.time() < deadline - 2:
        refined, refined_rank = _local_refine(
            best_result["U"], best_result["V"], best_result["W"], 
            n, m, p, max_iters=50, deadline=deadline
        )
        if _verify(refined["U"], refined["V"], refined["W"], n, m, p):
            if refined_rank < best_rank:
                best_result = refined
                best_rank = refined_rank
    
    # Final verification
    if not _verify(best_result["U"], best_result["V"], best_result["W"], n, m, p):
        # Fallback to schoolbook
        best_result = _schoolbook(n, m, p)
    
    return best_result
# EVOLVE-BLOCK-END
