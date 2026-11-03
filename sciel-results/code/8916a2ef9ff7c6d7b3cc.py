"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time
    
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Start with schoolbook
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find a lower rank decomposition
    # The tensor T has T[i,j,k] = 1 for all i,j,k in valid ranges
    # We want to write T as sum of rank-1 tensors
    
    # Use a greedy approach: try to find rank-r decompositions
    # For each candidate rank r, try to see if we can express the tensor as sum of r rank-1 terms
    
    # Alternative: use the fact that we can sometimes reduce by combining
    # Let's try a randomized search for lower rank
    
    def verify(U, V, W, r):
        """Check if U,V,W of rank r correctly decompose the tensor."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Reconstruct tensor: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        T_reconstructed = np.zeros((n, m, p))
        for c in range(r):
            u = U[:, c].reshape(n, m)
            v = V[:, c].reshape(m, p)
            w = W[:, c].reshape(p, n)
            # The contribution to T[i,j,k] is u[i,j] * v[j,k] * w[k,i]
            # This is an outer product in the (i,j,k) space
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_reconstructed[i,j,k] += u[i,j] * v[j,k] * w[k,i]
        
        # The target tensor is all 1s
        target = np.ones((n, m, p))
        return np.allclose(T_reconstructed, target, atol=1e-9)
    
    # Try rank from lower values up
    # Known lower bound: rank >= max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # But actually the rank must be at least the "slice rank" which is more complex
    
    # Let's try a few candidate ranks
    # For 3x4x6, let's try ranks from 24 up to 71
    
    # Use a randomized approach: randomly initialize and try to fit
    # This is hard, so let's try a different approach
    
    # Actually, let me try to use the structure more directly.
    # One known technique: if we can find linear dependencies in the tensor
    
    # For now, let's just return the schoolbook decomposition
    # But let me try a quick check for rank 24-30 with random initialization
    
    np.random.seed(42)
    
    for r in range(24, min(40, rank_school)):
        if time.time() > deadline - 10:
            break
            
        # Try random initialization
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
                
            U_try = np.random.randn(n * m, r)
            V_try = np.random.randn(m * p, r)
            W_try = np.random.randn(p * n, r)
            
            # Check if this works
            if verify(U_try, V_try, W_try, r):
                best_rank = r
                best_U, best_V, best_W = U_try, V_try, W_try
                break
    
    # Convert to lists with integer/half-integer entries if possible
    # For now, just return as floats - the evaluator should handle it
    # But the task says "integer/half-integer entries"
    
    # Let's try to round to nearest half-integer
    def to_half_integer(arr):
        arr = np.array(arr, dtype=float)
        rounded = np.round(arr * 2) / 2
        return rounded.tolist()
    
    U_out = to_half_integer(best_U)
    V_out = to_half_integer(best_V)
    W_out = to_half_integer(best_W)
    
    # Verify the rounded version
    if not verify(U_out, V_out, W_out, best_rank):
        # Fall back to schoolbook
        U_out = U_school.tolist()
        V_out = V_school.tolist()
        W_out = W_school.tolist()
        best_rank = rank_school
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
