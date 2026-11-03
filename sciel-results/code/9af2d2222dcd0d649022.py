import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for multiplying
    a 3x4 matrix by a 4x5 matrix.
    
    The schoolbook rank for M_{3,4,5} is 3*4*5 = 60.
    We attempt to find a sub-schoolbook decomposition using a structured
    bilinear algorithm construction. If no better solution is found within
    the time budget, we return the canonical schoolbook decomposition.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    # First, build the schoolbook decomposition as a fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    # Build the target tensor for verification
    # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T[a,b,c] = 1 iff j==j', i==i', k==k'
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    def verify(U, V, W, tol=1e-9):
        """Verify decomposition using numpy."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Compute the tensor decomposition
        # T_computed[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        T_computed = np.einsum('ar,br,cr->abc', U, V, W)
        
        return np.allclose(T_computed, T_target, atol=tol)

    def to_half_integer(x):
        """Convert a float to the nearest half-integer if close, else keep as is."""
        # Check if it's close to a half-integer
        x2 = round(x * 2)
        if abs(x * 2 - x2) < 1e-7:
            return x2 / 2.0
        return x

    def clean_matrix(M, tol=1e-7):
        """Clean a matrix by snapping near-zero and near-half-integer values."""
        M = np.array(M, dtype=float)
        M_clean = np.zeros_like(M)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                val = M[i, j]
                # Snap to 0 if very small
                if abs(val) < tol:
                    M_clean[i, j] = 0.0
                else:
                    # Snap to nearest half-integer
                    val2 = round(val * 2)
                    if abs(val * 2 - val2) < tol * 10:
                        M_clean[i, j] = val2 / 2.0
                    else:
                        M_clean[i, j] = val
        return M_clean

    # Try to find a sub-schoolbook decomposition using optimization
    # We'll try ranks from a reasonable estimate upward
    # Lower bound: rank >= max(n*m, m*p, p*n) = 20
    # Known: For 2x2x2, rank is 7 (Strassen)
    # For 3x4x5, let's try to find a rank < 60
    
    # Strategy: Use a random initialization and optimize
    # The objective is to minimize ||T_computed - T_target||_F^2
    # subject to U, V, W having small integer/half-integer entries
    
    # This is a non-convex problem, so we'll use multiple random restarts
    # and local optimization
    
    best_rank = rank_schoolbook
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try ranks from 40 to 59 (below schoolbook)
    # We'll use a heuristic approach: fix one matrix and solve for others
    
    # For each rank r, we have r columns in U, V, W
    # U: (12, r), V: (20, r), W: (15, r)
    # Total unknowns: r * (12 + 20 + 15) = 47r
    
    # The equations are: for each (a,b,c), sum_r U[a,r] V[b,r] W[c,r] = T_target[a,b,c]
    # There are 12*20*15 = 3600 equations, but only 60 are non-zero
    
    # Let's try a constructive approach:
    # 1. Fix W to be a sparse matrix with small entries
    # 2. For each column r, we need U[:,r] and V[:,r] such that
    #    U[:,r] * V[:,r] * W[:,r] approximates the target
    # 3. This is a rank-1 approximation problem
    
    # Alternative: Use the fact that the tensor is sparse
    # T[a,b,c] = 1 only when a%m == b//p and a//m == c%n and b%p == c//n
    
    # Let's try a greedy approach:
    # For each non-zero entry of T, try to cover it with a rank-1 term
    # and minimize the number of terms
    
    # This is equivalent to finding a set cover, which is NP-hard
    # But for small instances, we can use heuristics
    
    # Let's try a different approach: use known results
    # For matrix multiplication, the rank is at most n*m*p
    # and at least max(n*m, m*p, p*n)
    
    # For 3x4x5, let's try to see if we can do better than 60
    # by using a block decomposition
    
    # Split 3x4 into [2x2, 2x2; 1x2, 1x2]
    # Split 4x5 into [2x2, 2x2, 2x1; 2x2, 2x2, 2x1]
    # Split 3x5 into [2x2, 2x2, 2x1; 1x2, 1x2, 1x1]
    
    # The 2x2x2 blocks can be multiplied with rank 7 (Strassen)
    # The 1x1x1 blocks require rank 1
    # The mixed blocks (1x2x2, 2x1x2, etc.) require rank 2
    
    # Let's count the blocks:
    # A: 3x4 has 4 blocks: 2x2, 2x2, 1x2, 1x2
    # B: 4x5 has 6 blocks: 2x2, 2x2, 2x1, 2x2, 2x2, 2x1
    # C: 3x5 has 6 blocks: 2x2, 2x2, 2x1, 1x2, 1x2, 1x1
    
    # The multiplication involves all combinations of blocks
    # This is getting complex, so let's just try random optimization
    
    # Try a few ranks with random initialization and local search
    for rank in range(50, 60):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for trial in range(10):
            if time.time() > deadline:
                break
            
            np.random.seed(rank * 100 + trial)
            
            # Initialize with small random values
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            # Optimize using gradient descent
            # The objective is ||T_computed - T_target||_F^2
            # We'll use a simple gradient descent with learning rate scheduling
            
            lr = 0.01
            for iteration in range(1000):
                if time.time() > deadline:
                    break
                
                # Compute T_computed
                T_computed = np.einsum('ar,br,cr->abc', U, V, W)
                residual = T_computed - T_target
                
                # Compute gradients
                # d/dU = sum_{b,c} residual[a,b,c] * V[b,r] * W[c,r]
                dU = np.einsum('abc,br,cr->ar', residual, V, W)
                dV = np.einsum('abc,ar,cr->br', residual, U, W)
                dW = np.einsum('abc,ar,br->cr', residual, U, V)
                
                # Update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Decay learning rate
                lr *= 0.999
            
            # Clean the matrices to half-integers
            U_clean = clean_matrix(U)
            V_clean = clean_matrix(V)
            W_clean = clean_matrix(W)
            
            # Verify
            if verify(U_clean, V_clean, W_clean):
                # Found a valid decomposition
                best_rank = rank
                best_U = U_clean.tolist()
                best_V = V_clean.tolist()
                best_W = W_clean.tolist()
                break
        
        if best_rank < rank_schoolbook:
            break
    
    # If we found a better rank, return it
    if best_rank < rank_schoolbook:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Otherwise, return the schoolbook decomposition
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
