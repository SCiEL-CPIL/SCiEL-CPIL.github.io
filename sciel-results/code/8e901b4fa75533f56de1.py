"""Program for 3x5 by 5x5 matrix multiplication tensor decomposition."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
import random
import itertools

def _random_ternary_matrix(rows, cols, rng):
    """Generate a random matrix with entries in {-1, 0, 1}."""
    return rng.randint(-1, 1, size=(rows, cols))

def _random_half_integer_matrix(rows, cols, rng, scale=2):
    """Generate a random matrix with entries in {-scale/2, ..., scale/2} divided by scale."""
    # entries are multiples of 1/scale, in range [-scale/2, scale/2]
    vals = rng.randint(-scale//2, scale//2, size=(rows, cols))
    return vals / float(scale)

def _check_decomposition(U, V, W, n, m, p):
    """
    Check if U, V, W decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For each term r, the tensor is U[:,r] * V[:,r] * W[:,r]
    The full tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    We need T[i,j,k] = 1 if (i,j,k) is a valid multiplication index, else 0.
    Wait, the matrix multiplication tensor T_{i,j,k} for C = A*B where C is n x p, A is n x m, B is m x p:
    C_{i,k} = sum_j A_{i,j} B_{j,k}
    So the tensor T_{i,j,k} = 1 for all valid i,j,k, and the structure is that the sum over r of 
    U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    Actually, the tensor for matrix multiplication has entries T_{i,j,k} = 1 for all i,j,k in valid ranges,
    because the multiplication tensor is the sum of rank-1 tensors e_i * e_j * e_k for all i,j,k.
    
    So we need: for all i in range(n), j in range(m), k in range(p):
        sum_{r=0}^{rank-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(total - 1.0) > 1e-6:
                    return False
    return True

def _check_decomposition_vectorized(U, V, W, n, m, p):
    """Vectorized check."""
    rank = U.shape[1]
    # Build full tensor check
    # For each (i,j,k), compute sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    
    # For each (i,j,k), we need sum_r U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
    # This is a bit tricky to vectorize directly. Let's do it per (i,j,k) triple using einsum
    # T[i,j,k] = sum_r U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
    # We can compute this by iterating over i and j, then for each (i,j) we need sum over k,r
    # Alternatively, use a loop over i, j, k but with numpy for the sum over r
    
    # Let's try: for each i, j, k:
    #   np.sum(U_3d[i,j,:] * V_3d[j,k,:] * W_3d[k,i,:])
    # This is O(n*m*p*rank) which for n=3,m=5,p=5 is 75*rank operations per check.
    # For rank ~ 50, that's 3750 operations per check, which is fast.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_3d[i, j, :] * V_3d[j, k, :] * W_3d[k, i, :])
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def _try_random_decomposition(n, m, p, rank, rng, max_iterations=500):
    """
    Try to find a rank-`rank` decomposition using random search.
    Returns (U, V, W) as lists of lists if found, else None.
    """
    for _ in range(max_iterations):
        U = _random_ternary_matrix(n*m, rank, rng)
        V = _random_ternary_matrix(m*p, rank, rng)
        W = _random_ternary_matrix(p*n, rank, rng)
        
        # Check
        if _check_decomposition_vectorized(U, V, W, n, m, p):
            # Convert to lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return U_list, V_list, W_list
    return None

def _try_structured_decomposition(n, m, p, rank, rng):
    """
    Try a structured approach: use the fact that the tensor is separable in certain ways.
    For matrix multiplication, we know that the rank is at most n*m*p (trivial) and at least max(n*m, m*p, p*n).
    For 3x5x5, lower bound is max(15, 25, 15) = 25.
    Upper bound known: we can do better than 75.
    
    Let's try a heuristic: for each term r, pick a random subset of (i,j,k) indices and try to set
    U, V, W such that the product matches.
    
    Alternative: use the "splitting" method. For small dimensions, we can try to find a set of
    rank-1 tensors whose sum equals the multiplication tensor.
    
    Actually, let's try a different approach: use integer programming / backtracking for small ranks.
    But that might be too slow.
    
    Let's try: for rank R, we can try to set each column of U, V, W to be a random vector from a small set,
    and check if the sum of outer products equals the target tensor.
    """
    # For now, fall back to random search
    return _try_random_decomposition(n, m, p, rank, rng, max_iterations=200)

def construct():
    n, m, p = DIMENSIONS
    rng = random.Random(42)
    
    # Start from a lower bound and go up
    # Lower bound: max(n*m, m*p, p*n) = max(15, 25, 15) = 25
    lower_bound = max(n * m, m * p, p * n)
    
    # Try ranks from lower_bound up to n*m*p
    for rank in range(lower_bound, n * m * p + 1):
        # Try random search
        result = _try_random_decomposition(n, m, p, rank, rng, max_iterations=100)
        if result is not None:
            U, V, W = result
            # Verify
            U_arr = np.array(U)
            V_arr = np.array(V)
            W_arr = np.array(W)
            if _check_decomposition_vectorized(U_arr, V_arr, W_arr, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Fallback: trivial decomposition
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

# EVOLVE-BLOCK-END
