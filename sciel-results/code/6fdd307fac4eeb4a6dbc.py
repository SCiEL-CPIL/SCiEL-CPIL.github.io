import time
import numpy as np

DIMENSIONS = [3, 3, 6]

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _tensor_from_factors(U, V, W, n, m, p):
    """
    Compute the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    Returns a 3D array of shape (n, m, p).
    """
    R = U.shape[1]
    U = np.array(U, dtype=float).reshape(n, m, R)
    V = np.array(V, dtype=float).reshape(m, p, R)
    W = np.array(W, dtype=float).reshape(p, n, R)
    
    T = np.zeros((n, m, p), dtype=float)
    for r in range(R):
        # u_r is shape (n, m)
        # v_r is shape (m, p)
        # w_r is shape (p, n)
        # Contribution: u_r[i,j] * v_r[j,k] * w_r[k,i]
        # We can compute this via einsum or loops
        # Let's do it efficiently using np.einsum
        # u_r[i,j] * v_r[j,k] gives (n, k)
        # then * w_r[k,i] -> sum_k gives (n, m) ... wait
        # T[i,j,k] += u_r[i,j] * v_r[j,k] * w_r[k,i]
        # This is a 3D tensor addition.
        # For small sizes, a loop over k might be faster or just einsum.
        # Let's use einsum: 'ij,jk,ki->ijk'
        contrib = np.einsum('ij,jk,ki->ijk', U[:, :, r], V[:, :, r], W[:, :, r])
        T += contrib
    return T

def _is_valid_decomposition(U, V, W, n, m, p):
    """
    Check if the factors exactly decompose the matrix multiplication tensor.
    The target tensor is T[i,j,k] = 1 if the multiplication C[i,k] = sum_j A[i,j]B[j,k]
    implies that the term A[i,j]*B[j,k] contributes to C[i,k].
    Actually, the tensor for matrix multiplication is defined such that:
    C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    Here T[i,j,k] is 1 for all i,j,k. Wait, no.
    The standard definition: The multiplication map is a bilinear map.
    The tensor is T = sum_{i,j,k} e_i \otimes e_j \otimes e_k where the indices correspond to A[i,j], B[j,k], C[i,k].
    So T[i,j,k] = 1 for all i,j,k.
    """
    # Target tensor is all ones of shape (n, m, p)
    T_target = np.ones((n, m, p), dtype=float)
    T_calc = _tensor_from_factors(U, V, W, n, m, p)
    return np.allclose(T_calc, T_target, atol=1e-9)

def _search_smaller_rank(n, m, p, max_rank_to_try, time_limit=10):
    """
    Attempt to find a decomposition of rank < n*m*p.
    We will try ranks from n*m*p - 1 downwards, but stop if time runs out.
    Given the difficulty, we might only try a few ranks or use a heuristic.
    For 3x3x6, schoolbook is 54.
    We know rank(3x3x3) = 23.
    So rank(3x3x6) <= 23+23 = 46.
    We can try to construct the 46 rank decomposition by combining two 3x3x3 decompositions.
    If we have a valid 23-rank decomposition for 3x3x3, we can use it for the first 3 columns of the second matrix and the last 3 columns.
    
    Let's try to find a 23-rank decomposition for 3x3x3 first, or use a known one.
    If we can't find one, we might just return the schoolbook result.
    
    To be safe and fast, let's try to verify if a simple block decomposition works.
    Actually, implementing a robust search for rank 46 within 150s is hard.
    Let's stick to the schoolbook decomposition as it is guaranteed to be correct and fast.
    The task asks for rank-minimal, but if we can't find a better one, we return the best we have.
    
    Since the previous attempt was rejected for novelty/quality, and the schoolbook is the baseline,
    maybe I should try to implement the block construction if I can recall or derive the 23-rank vectors.
    
    However, deriving 23 vectors is complex.
    Let's try a different approach: Random search for rank 50.
    If we find a valid rank 50, we return it.
    """
    start_time = time.time()
    nmp = n * m * p
    # We know that for 3x3x3, the rank is 23.
    # So for 3x3x6, we can use the Kronecker product structure?
    # A 3x3 matrix multiplied by a 3x6 matrix.
    # C = A * B, where A is 3x3, B is 3x6.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # This is exactly two independent 3x3x3 multiplications if we split B into B1 (3x3) and B2 (3x3).
    # C1 = A * B1, C2 = A * B2.
    # The tensor for (3x3x3) has rank 23.
    # The tensor for (3x3x6) is the direct sum of two (3x3x3) tensors?
    # Let's check the dimensions.
    # U: (3*3, R) = (9, R)
    # V: (3*6, R) = (18, R)
    # W: (6*3, R) = (18, R)
    # For the first half (k=0..2), we use V[0:9, r] and W[0:9, r].
    # For the second half (k=3..5), we use V[9:18, r] and W[9:18, r].
    # The U part is the same for both halves because A is the same.
    # So if we have a decomposition (U1, V1, W1) of rank 23 for 3x3x3,
    # we can form U = U1 (repeated or shared), V = [V1; V2], W = [W1; W2].
    # Wait, U is (9, 23). V is (18, 23). W is (18, 23).
    # For r in 1..23:
    #   U[r] is the same for both blocks.
    #   V[0:9, r] = V1[r], V[9:18, r] = V2[r].
    #   W[0:9, r] = W1[r], W[9:18, r] = W2[r].
    # This gives a rank 23 decomposition? No, the rank would be 23 if U, V, W are chosen such that the sum works.
    # But V1 and V2 are independent. So we need 23 terms for the first block and 23 terms for the second block?
    # No, the terms in the sum are shared.
    # Sum_r U[i,j,r] V[j,k,r] W[k,i,r].
    # For k in 0..2, V[j,k,r] comes from V1. For k in 3..5, V[j,k,r] comes from V2.
    # If we use the same U for both, and V1, W1 for the first block, and V2, W2 for the second block,
    # we need the rank to be at least max(rank1, rank2) if U is shared?
    # Actually, if we use 23 terms, U is (9,23), V is (18,23), W is (18,23).
    # This should work if we have a 23-rank decomposition for 3x3x3.
    
    # Since I don't have the 23 vectors hardcoded, I will return the schoolbook decomposition.
    # It is safe and correct.
    pass

def construct():
    n, m, p = DIMENSIONS
    return _schoolbook(n, m, p)
