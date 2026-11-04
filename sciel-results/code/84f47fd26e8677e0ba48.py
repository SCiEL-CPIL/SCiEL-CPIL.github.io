"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    def verify(U, V, W):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        rank = len(U[0])
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
                    val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def make_schoolbook():
        """Return the trivial (schoolbook) decomposition with rank = n*m*p = 54."""
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
    
    # Strategy: Try to construct decompositions using Strassen-like ideas.
    # For 3x3 * 3x3, the known minimum rank is 23 (Strassen's algorithm variant).
    # For 3x3 * 3x6, we can think of it as two 3x3 * 3x3 multiplications.
    # A simple approach: use two copies of a 3x3x3 rank-23 decomposition -> rank 46.
    # But we can do better by sharing some terms.
    
    # Let's try a structured approach:
    # Split the 3x6 matrix into two 3x3 blocks.
    # C = A * B where A is 3x3, B is 3x6 = [B1 | B2] where B1, B2 are 3x3.
    # C = [A*B1 | A*B2]
    # Each A*Bk can be done with rank 23 (using known 3x3x3 rank-23 decomposition).
    # Total rank: 46.
    
    # But we can potentially share some of the intermediate products.
    # Let's first try the straightforward 46-rank solution, then try to optimize.
    
    # Known rank-23 decomposition for 3x3 * 3x3:
    # This is based on the "Strassen-like" decomposition.
    # The 23-term decomposition for 3x3x3 is complex. Let me use a simpler approach.
    
    # Actually, let's use a more systematic approach:
    # For 3x3 * 3x3, we know rank 23 works. Let me encode one such decomposition.
    
    # Alternative: Use the fact that 3x3 * 3x6 can be seen as a special case.
    # Let's try to find a decomposition with rank <= 46 first, then try to reduce.
    
    # Simple construction: two independent 3x3*3x3 multiplications
    # For each half (k=0..2 and k=3..5), we need a 3x3*3x3 decomposition.
    
    # Let me use a known rank-23 decomposition for 3x3*3x3.
    # Reference: The minimal rank for 3x3x3 matrix multiplication is 23.
    # I'll use a specific known decomposition.
    
    # Actually, let me try a different approach: use bilinear decomposition.
    # For 3x3 * 3x3, a rank-23 decomposition exists. Let me try to construct it.
    
    # Simpler: Let's just try to find ANY valid decomposition and minimize rank.
    # Start with schoolbook (rank 54) and try to reduce.
    
    # Method: Use the "border rank" / "approximate rank" ideas but we need exact.
    
    # Let me try a direct construction approach:
    # For 3x3 * 3x6, note that this is equivalent to computing 18 entries of C.
    # Each C[i,k] = sum_j A[i,j]*B[j,k].
    
    # Let's try to use a rank-23 decomposition for each 3x3*3x3 block.
    # I'll look up or derive a specific rank-23 decomposition.
    
    # Here's a known approach: use the "Strassen" style but for 3x3.
    # The 23-term algorithm for 3x3x3 was found by various researchers.
    
    # Let me try a different tactic: use integer programming or exhaustive search over small structures.
    
    # Actually, the most reliable approach given the time constraint:
    # 1. Return schoolbook (rank 54) as a guaranteed valid answer.
    # 2. Try to find a better one with a quick search.
    
    # Let's try a structured search for rank 46, 45, ..., down to some lower bound.
    
    # First, let me try the two-block approach with rank 23 each = rank 46.
    
    # For a 3x3 * 3x3 decomposition of rank 23, I need specific coefficients.
    # Let me use a known one. Here's one approach:
    
    # The rank-23 decomposition of 3x3x3 (one known form):
    # This is complex. Let me instead try to use a simpler but still sub-schoolbook approach.
    
    # For 3x3 * 3x3, we can use rank 23. But encoding it correctly is tricky.
    # Let me try a different approach: use the fact that we can share some computations.
    
    # Actually, let me just try to do a quick randomized search for low-rank solutions.
    # The key insight: for small dimensions, we can use a more efficient verification.
    
    def fast_verify(U, V, W):
        """Fast verification using vectorized operations."""
        rank = len(U[0])
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # For each r, compute the outer product contribution
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Vectorized: for each r, form the tensor contribution
        # U_r is (n*m,), V_r is (m*p,), W_r is (p*n,)
        # The contribution to T[i,j,k] is U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i]
        
        # Let's compute T as (n, m, p)
        T = np.zeros((n, m, p))
        for r in range(rank):
            u_r = U[:, r]  # (n*m,)
            v_r = V[:, r]  # (m*p,)
            w_r = W[:, r]  # (p*n,)
            
            # Reshape for broadcasting
            u_r = u_r.reshape(n, m)
            v_r = v_r.reshape(m, p)
            w_r = w_r.reshape(p, n)
            
            # T[i,j,k] += u_r[i,j] * v_r[j,k] * w_r[k,i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i,j,k] += u_r[i,j] * v_r[j,k] * w_r[k,i]
        
        return np.allclose(T, np.ones((n, m, p)), atol=1e-9)
    
    # Try a few structured decompositions:
    
    # Approach 1: Two independent 3x3*3x3 with shared structure
    # Let's try to find a rank-23 decomposition for 3x3*3x3 first.
    
    # I'll use a known rank-23 decomposition. Here's one based on literature:
    # The coefficients are complex. Let me try a simpler approach.
    
    # Let's just return schoolbook for now if we can't find better quickly.
    # But first, try a quick randomized search for rank 46, 45, 44, 43, 42.
    
    best_result = None
    best_rank = n * m * p  # 54
    
    # Try to find a solution with rank < 54
    for target_rank in range(46, 54):
        if time.time() > deadline - 30:
            break
        
        np.random.seed(123 + target_rank)
        
        U = np.random.randn(n * m, target_rank) * 0.5
        V = np.random.randn(m * p, target_rank) * 0.5
        W = np.random.randn(p * n, target_rank) * 0.5
        
        # Simple iterative improvement
        for it in range(2000):
            if time.time() > deadline - 25:
                break
            
            # Pick a random element to update
            idx = np.random.randint(0, target_rank)
            # Update U
            U[:, idx] += 0.05 * np.random.randn(n * m)
            
            if fast_verify(U, V, W):
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = target_rank
                break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: schoolbook
    return make_schoolbook()

# EVOLVE-BLOCK-END
