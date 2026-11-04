"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin from 150s limit
    
    n, m, p = DIMENSIONS
    trivial_rank = n * m * p  # 84
    
    # Build the target tensor T[a, b, c] where:
    # a = i*m + j (index into A)
    # b = j*p + k (index into B)  
    # c = k*n + i (index into C, using trace convention W[k*n+i])
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def verify_decomposition(U, V, W, rank):
        """Verify that sum_r U[:,r] outer V[:,r] outer W[:,r] == T"""
        recon = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for r in range(rank):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            recon += np.einsum('a,b,c->abc', u, v, w)
        return np.allclose(recon, T, atol=1e-10)
    
    # Trivial decomposition
    U_triv = np.zeros((n*m, trivial_rank), dtype=np.float64)
    V_triv = np.zeros((m*p, trivial_rank), dtype=np.float64)
    W_triv = np.zeros((p*n, trivial_rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_triv[a, col] = 1.0
                V_triv[b, col] = 1.0
                W_triv[c, col] = 1.0
                col += 1
    
    best_rank = trivial_rank
    best_U = U_triv.copy()
    best_V = V_triv.copy()
    best_W = W_triv.copy()
    
    # Strategy: Try to find rank 83 by searching for a linear dependency
    # If we can find coefficients c_r such that sum_r c_r * (u_r ⊗ v_r ⊗ w_r) = 0
    # and c_r != 0 for some r, then we can eliminate one term.
    
    # For rank 83: we need to find if there's a non-trivial linear combination
    # of the 84 rank-1 tensors that gives zero.
    
    # Let's try a different approach: use the structure of the problem.
    # The tensor has a specific algebraic structure related to matrix multiplication.
    
    # Try to use a known low-rank algorithm for matrix multiplication.
    # For M_{3,4,7}, we can try to use a block decomposition.
    
    # Approach: Try ranks from 83 down, using a randomized search with verification.
    # For each candidate rank, initialize with the trivial decomposition and try
    # to "merge" terms by finding linear dependencies.
    
    np.random.seed(42)
    
    # Try rank 83 first
    # The idea: if we can find a linear relation among the 84 basis tensors,
    # we can reduce the rank by 1.
    
    # Let's try to find if the 84 rank-1 tensors are linearly dependent.
    # We can do this by checking if the 84 vectors (each of dimension n*m*m*p*p*n) are linearly dependent.
    # But that's a huge dimension (84*28*21 = 49392), so we need to be clever.
    
    # Alternative: use the fact that the tensor has a specific structure.
    # Let's try to construct a rank-83 decomposition directly.
    
    # For M_{n,m,p}, there's a known bound: rank <= n*m*p - min(n,m,p) + 1
    # For n=3, m=4, p=7: rank <= 84 - 3 + 1 = 82? Let's verify this bound.
    
    # Actually, let's just try to find a valid decomposition with rank < 84.
    
    # Try a structured approach: use the fact that for each j, the slice is an all-ones
    # 3x7 matrix in the (i,k) plane. We can decompose this all-ones matrix using
    # rank-1 matrices: J_{3,7} = sum_{r=1}^{3} u_r ⊗ v_r where u_r are standard basis
    # and v_r are all-ones vectors... no, that gives rank 3, not 1.
    
    # The all-ones 3x7 matrix has rank 1: J = 1_3 ⊗ 1_7.
    # So for each j, S_j can be written as... but the issue is the coupling of indices.
    
    # Let me try a direct search for rank 83.
    # We'll try to find coefficients such that one term can be expressed as a combination
    # of the others.
    
    # For efficiency, let's try a randomized approach with multiple seeds.
    
    found_better = False
    for target_rank in range(trivial_rank - 1, max(60, trivial_rank - 10), -1):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random values
            U_cand = np.random.randn(n*m, target_rank) * 0.1
            V_cand = np.random.randn(m*p, target_rank) * 0.1
            W_cand = np.random.randn(p*n, target_rank) * 0.1
            
            # Try to refine using a simple iterative approach
            # This is a non-convex problem, so we'll use a heuristic.
            
            # For exact integer/half-integer solutions, random real-valued
            # approaches won't work well. Let's try a different strategy.
            
            pass
    
    # Fallback: Use the trivial decomposition
    U_out = best_U[:, :best_rank].tolist()
    V_out = best_V[:, :best_rank].tolist()
    W_out = best_W[:, :best_rank].tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
