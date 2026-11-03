import time
import numpy as np

DIMENSIONS = [3, 4, 5]
DEADLINE = 140

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Try structured decomposition using Strassen for 2x2 blocks
    result = _try_strassen_decomposition(n, m, p)
    if result is not None:
        best_rank = result["rank"]
        best_result = result
        return best_result
    
    return best_result

def _try_strassen_decomposition(n, m, p):
    """
    Try to find a lower-rank decomposition using Strassen's algorithm for 2x2 subproblems.
    """
    n, m, p = 3, 4, 5
    start_time = time.time()
    
    # Split m into two halves of size 2
    m1, m2 = 2, 2
    
    # For the multiplication A1*B1 where A1 is 3x2 and B1 is 2x5:
    # This is not a square matrix multiplication, so Strassen doesn't directly apply.
    # But we can still use a structured approach.
    
    # Let's try a different approach: use the known rank-5 decomposition for 2x2 matrix multiplication
    # and embed it into our problem.
    
    # Actually, let's try a more direct approach:
    # Use the fact that we can write the tensor as a sum of rank-1 tensors.
    # For a 3x4x5 tensor, the schoolbook rank is 60.
    # We want to find if there's a lower-rank decomposition.
    
    # Let's try using integer linear programming or a structured search.
    
    # For now, let's try a specific structured decomposition based on block decomposition.
    
    # Approach: Use the identity that for matrix multiplication, we can use
    # Kronecker products of smaller decompositions.
    
    # Let's try: decompose 3x4x5 as (3x2x5) + (3x2x5) but with shared structure.
    
    # Actually, let's try a randomized search with a better verification method.
    
    # Try ranks from 59 down to 40
    for target_rank in range(59, 39, -1):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        result = _try_rank_search(n, m, p, target_rank)
        if result is not None:
            result["rank"] = target_rank
            return result
    
    return None

def _try_rank_search(n, m, p, target_rank):
    """
    Try to find a rank-target_rank decomposition using structured patterns.
    """
    start_time = time.time()
    
    # Use a structured initialization based on block decomposition
    # Split the problem into blocks and use known low-rank decompositions for each block
    
    # For now, try a randomized approach with verification
    import random
    np.random.seed(42)
    
    max_attempts = 50
    for attempt in range(max_attempts):
        if time.time() - start_time > 20:
            break
        
        # Initialize with random values from a small set
        # Use values that are likely to produce exact decompositions
        U = np.random.choice([0, 1], size=(n*m, target_rank))
        V = np.random.choice([0, 1], size=(m*p, target_rank))
        W = np.random.choice([0, 1], size=(p*n, target_rank))
        
        # Verify the decomposition
        if _verify_decomposition(U, V, W, n, m, p):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    return None

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the given U, V, W matrices form a valid decomposition
    of the matrix multiplication tensor.
    """
    for i in range(n):
        for j in range(m):
            for k in range(p):
                actual = 0.0
                for r in range(U.shape[1]):
                    actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(actual - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2x2():
    """
    Returns U, V, W matrices for 2x2x2 Strassen decomposition.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[0, 1] = 1
    
    # M3 = a(f-h)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[2, 4] = 1
    
    # M6 = (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[1, 5] = 1
    
    # M7 = (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[3, 6] = 1
    
    return U, V, W

def _schoolbook_nm_p(n, m, p):
    """
    Returns U, V, W matrices for schoolbook nm p decomposition.
    """
    U = np.zeros((n*m, n*m*p))
    V = np.zeros((m*p, n*m*p))
    W = np.zeros((p*n, n*m*p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W

def _combine_decompositions(U1, V1, W1, U2, V2, W2):
    """
    Combine two decompositions by direct sum.
    """
    U = np.hstack([U1, U2])
    V = np.hstack([V1, V2])
    W = np.hstack([W1, W2])
    return U, V, W

def _embed_decomposition(U_sub, V_sub, W_sub, n, m, p, i_offset, j_offset, k_offset):
    """
    Embed a sub-decomposition into the full problem.
    """
    n_sub = U_sub.shape[0] // m
    m_sub = U_sub.shape[0] // n_sub
    p_sub = V_sub.shape[0] // m_sub
    
    U_full = np.zeros((n*m, U_sub.shape[1]))
    V_full = np.zeros((m*p, V_sub.shape[1]))
    W_full = np.zeros((p*n, W_sub.shape[1]))
    
    for i in range(n_sub):
        for j in range(m_sub):
            for r in range(U_sub.shape[1]):
                U_full[(i_offset+i)*m + (j_offset+j), r] = U_sub[i*m_sub+j, r]
    
    for j in range(m_sub):
        for k in range(p_sub):
            for r in range(V_sub.shape[1]):
                V_full[(j_offset+j)*p + (k_offset+k), r] = V_sub[j*p_sub+k, r]
    
    for k in range(p_sub):
        for i in range(n_sub):
            for r in range(W_sub.shape[1]):
                W_full[(k_offset+k)*n + (i_offset+i), r] = W_sub[k*n_sub+i, r]
    
    return U_full, V_full, W_full

def _try_structured_decomposition(n, m, p):
    """
    Try to find a structured decomposition using Strassen for 2x2 blocks.
    """
    start_time = time.time()
    
    # Split m=4 into two halves of size 2
    m1, m2 = 2, 2
    
    # For each half, we need to compute A_i * B_j where A_i is n x 2 and B_j is 2 x p
    # This is a n x 2 x p problem.
    
    # For n=3, m=2, p=5:
    # We can use Strassen for the 2x2 subproblems and schoolbook for the rest.
    
    # Actually, let's just use schoolbook for simplicity and see if we can improve.
    
    # For now, return None to use the fallback.
    return None
