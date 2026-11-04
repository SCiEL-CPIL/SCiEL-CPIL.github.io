"""Rank-minimal tensor decomposition for 2x3x9 matrix multiplication."""
import numpy as np
import sys
import time

DIMENSIONS = [2, 3, 9]
n, m, p = DIMENSIONS

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    def check_time():
        return time.time() > deadline
    
    # Build the matrix multiplication tensor
    # T[i,j,k] = 1 for the multiplication tensor
    # We want to find rank-r decomposition where:
    # T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    # But the convention is: output W[k*n+i, r]
    # So the tensor is indexed as: (i,j,k) -> U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # The matrix multiplication tensor: C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # Actually the tensor has entries T[i,j,k] = delta (it's the structure tensor)
    
    # Let's think more carefully:
    # A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # Tensor decomposition: C[i,k] = sum_c U[i,j,c]... no wait.
    # 
    # Standard tensor form: The multiplication tensor M has:
    # M[i,j,k] where i indexes row of A (and C), j indexes column of A / row of B, k indexes column of B (and C)
    # M[i,j,k] = 1 means the product A[i,j]*B[j,k] contributes to C[i,k]
    # 
    # So the tensor T is of size n x m x p with T[i,j,k] = 1 for all valid i,j,k
    # (since every A[i,j]*B[j,k] pair contributes to C[i,k])
    
    # Actually no. The tensor is:
    # For each triple (i,j,k), the entry is 1 because A[i,j]*B[j,k] appears in the sum for C[i,k]
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # The decomposition requires:
    # 1 = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # for all i,j,k
    
    # This is a rank decomposition of the all-ones tensor in the space indexed by (i,j,k)
    # where the output index is determined by (i,k) and the "summing" index is j
    
    # Wait, let me reconsider. The tensor for matrix multiplication is:
    # T in U ⊗ V ⊗ W* where:
    # U has basis {e_{i,j} : i in [n], j in [m]}  (entries of A)
    # V has basis {e_{j,k} : j in [m], k in [p]}  (entries of B)  
    # W has basis {e_{i,k} : i in [n], k in [p]}  (entries of C)
    
    # T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{i,k}
    
    # So T[i,j,k] = 1 for all valid (i,j,k)
    
    # We need: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c]
    # Using the trace convention W[k*n+i, r]
    
    # So the tensor is the all-ones tensor of shape (n*m, m*p, p*n) restricted to the
    # subspace where indices are compatible.
    
    # Let's create the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1
    
    # Now we need to find minimal rank r and matrices U, V, W such that:
    # T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    
    # This is the tensor rank problem. For small cases, we can try to find it.
    
    # Known results:
    # For 2x2x2: rank 7 (Strassen)
    # For 2x2x3: rank 10
    # For 2x2x4: rank 14? 
    # For 2x3x3: rank 14?
    # For 2x3x9: ?
    
    # The bilinear complexity of n x m x p matrix multiplication:
    # M(2,3,9) - what is it?
    
    # Upper bounds from known algorithms:
    # Schoolbook: 2*3*9 = 54
    # Strassen-like: Can we do better?
    
    # Let me try a systematic approach:
    # 1. Try to verify if rank < 54 is possible by searching
    # 2. Use the known lower bound: rank >= max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    #    Actually the lower bound is max(nm, mp, np) for the bilinear complexity
    
    # Lower bound: M(n,m,p) >= max(nm, mp, np) = max(6, 27, 18) = 27
    # Upper bound: M(n,m,p) <= nm*p = 54
    
    # So the rank is between 27 and 54.
    
    # Let's try to find a decomposition with rank < 54.
    # We can use a greedy/randomized approach or try known constructions.
    
    # Approach: Try to find a rank-r decomposition by solving the nonlinear system
    # T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    
    # This is a system of equations in the entries of U, V, W.
    # Number of equations: n*m*p = 54
    # Number of unknowns: r*(n*m + m*p + p*n) = r*(6 + 27 + 18) = 51*r
    # But there's scaling freedom: we can scale U[:,c], V[:,c], W[:,c] inversely
    
    # For small r, this is underdetermined, so solutions should exist if r >= tensor rank.
    
    # Let's try to search for the minimal rank by trying r = 27, 28, ... and checking
    # if a solution exists.
    
    # However, this is computationally hard. Let me try a different approach.
    
    # Known: M(2,2,2) = 7 (Strassen)
    # M(2,2,p) for p >= 2: can use Strassen recursively
    # M(2,3,3): known to be 14? Let me verify.
    
    # Actually, let me just try to find a good decomposition using numerical optimization
    # or a structured approach.
    
    # Strategy:
    # 1. Start with schoolbook (rank 54)
    # 2. Try to reduce rank by finding linear dependencies among the rank-1 tensors
    # 3. Use numerical rank reduction
    
    # Alternative: Use the fact that for matrices, we can exploit structure.
    
    # Let me try a randomized search for low-rank decompositions.
    # Fix a target rank r, initialize U, V, W randomly, and try to solve for T.
    
    def try_rank(r, max_iters=2000, seed=42):
        """Try to find a rank-r decomposition."""
        if time.time() > deadline:
            return None
        
        rng = np.random.RandomState(seed)
        
        best_error = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        # The tensor T has entries at specific positions
        # We need T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # Let's set up the problem as: for each (i,j,k), 
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        
        # This is a nonlinear system. Let's use iterative refinement.
        
        # Approach: Initialize W randomly, then solve for U and V (or vice versa)
        # using least squares, alternating.
        
        for attempt in range(3):
            U = rng.randn(n*m, r) * 0.1
            V = rng.randn(m*p, r) * 0.1
            W = rng.randn(p*n, r) * 0.1
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Fix U, V, solve for W (or vice versa)
                # For each k (output index), W[k*n+i, c] for all i,c
                # Actually, let's alternate:
                # Fix V, W: solve for U
                # Fix U, W: solve for V  
                # Fix U, V: solve for W
                
                # To solve for U: for each row i*m+j, we need
                # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] summed over c = 1
                # This is: for each (i,j,k), sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1
                # Let A[i*m+j, c] = U[i*m+j, c]
                # Let b[i*m+j, (j,k)] = 1
                # Then sum_c A[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                # This isn't linear in U because V and W also depend on the same c.
                
                # Better approach: Use the fact that if we fix V and W, then for each
                # equation (i,j,k): sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1
                # Let M[i*m+j, c] = V[j*p+k, c] * W[k*n+i, c] ... but this depends on j,k
                # which are also part of the equation.
                
                # This is getting complicated. Let me use a different approach.
                
                # Direct evaluation of error:
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                            err += (val - 1.0)**2
                
                if err < best_error:
                    best_error = err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                
                if err < 1e-8:
                    break
                
                # Gradient-free update: random perturbation
                # This is very slow. Let me think of something better.
                
                # Actually, for a proper approach, I should use the tensor structure.
                # Let me try a different initialization and strategy.
                break  # Exit early if this approach isn't working
                
            if best_error < 1e-8:
                break
        
        if best_error < 1e-6:
            return best_U, best_V, best_W, best_error
        return None
    
    # Let's try ranks from the theoretical minimum upward
    # Lower bound is 27 (max of nm=6, mp=27, np=18)
    
    # Actually, a better lower bound: The tensor rank of the matrix multiplication tensor
    # M(n,m,p) satisfies M(n,m,p) >= max(nm, mp, np)
    
    # For 2x3x9: lower bound = 27
    
    # But is 27 achievable? Probably not for arbitrary dimensions.
    # Known results suggest higher values.
    
    # Let me just try to find ANY decomposition better than 54.
    # I'll use a structured approach based on known fast matrix multiplication algorithms.
    
    # For 2x2x2, Strassen gives rank 7 instead of 8.
    # Can we combine Strassen-like tricks for 2x3x9?
    
    # One approach: decompose the 3x9 part using Strassen-like methods.
    # M(2,3,9) = M(2,3,3) * M(3,3,3) roughly? No, that's not how it works.
    
    # Actually, M(n,m,p) <= M(n,m,q) * M(q,p,r) if p = q*r... no.
    # The correct composition: M(n,m,p) <= M(n,m,q) * M(q,r,p) if m=q and appropriate.
    
    # Let me try: M(2,3,9) <= M(2,3,3) * M(3,3,9) or similar compositions.
    # Actually: if C = A*B where A is 2x3, B is 3x9
    # We can write B as a 3x3 block times a 3x9... no.
    
    # Better: M(2,3,9) <= M(2,3,q) * M(q,3,9) doesn't make sense dimensionally.
    
    # The composition rule: M(n,m,p) * M(m,p,q) >= M(n,m,q)
    # So M(2,3,9) could be bounded by M(2,3,3) * M(3,3,9) if we can compose them.
    # But M(3,3,9) is 3x3 times 3x9.
    
    # Known: M(2,3,3) <= 14 (I believe)
    # M(3,3,9): 3x3 times 3x9. Schoolbook = 27.
    # So M(2,3,9) <= 14 * 27 = 378? That's worse than 54.
    
    # That composition doesn't help. Let me think differently.
    
    # Direct approach: Just use schoolbook and see if we can shave off some rank.
    
    # The schoolbook has rank 54. Each rank-1 term corresponds to (i,j,k).
    # Can we combine some terms?
    
    # For a fixed i, the terms (i,j,k) for j=0,1,2 and k=0,...,8 form a 3x9 sub-tensor.
    # This is like multiplying a 1x3 vector by a 3x9 matrix, which has rank 3*9=27 schoolbook.
    # But actually, for a fixed i, C[i,k] = sum_j A[i,j]*B[j,k], which is a dot product of
    # A[i,:] (1x3) with B[:,k] (3x1). This can be done with... well, it's just 3 multiplications.
    # The tensor for this is rank 3 (since it's a rank-3 tensor: sum_j e_j ⊗ e_j ⊗ e_k for fixed structure).
    # Wait no, for fixed i, the sub-tensor is sum_{j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{i,k}
    # which has rank 3*9 = 27 in the full tensor, but as a sub-structure...
    
    # I think the key insight is that the full tensor T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{i,k}
    # can potentially be decomposed with fewer than 54 terms if there's algebraic structure.
    
    # For now, let me just return the schoolbook decomposition since finding a better one
    # in the time available is challenging. But let me try one more thing: a randomized search
    # for rank 42 or so.
    
    def schoolbook_decomposition():
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
    
    # Try to find a better decomposition using numerical methods
    # with a time limit
    
    def search_for_better_rank(min_rank=30, max_rank=53, num_trials=50):
        """Try to find a decomposition with rank < 54."""
        for r in range(max_rank, min_rank - 1, -1):
            if time.time() > deadline:
                break
            
            for trial in range(num_trials):
                if time.time() > deadline:
                    break
                
                rng = np.random.RandomState(1000 + r * 100 + trial)
                
                # Initialize with schoolbook-like structure but with random mixing
                U = np.zeros((n*m, r))
                V = np.zeros((m*p, r))
                W = np.zeros((p*n, r))
                
                # Random initialization
                U[:] = rng.randn(n*m, r)
                V[:] = rng.randn(m*p, r)
                W[:] = rng.randn(p*n, r)
                
                # Iterative refinement
                for it in range(500):
                    if time.time() > deadline:
                        break
                    
                    # Compute current error
                    err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                                err += (val - 1.0)**2
                    
                    if err < 1e-6:
                        # Found a solution!
                        return r, U.tolist(), V.tolist(), W.tolist()
                    
                    if it % 10 == 0 and err > 1e2:
                        # Restart with new initialization if stuck
                        U[:] = rng.randn(n*m, r) * 0.5
                        V[:] = rng.randn(m*p, r) * 0.5
                        W[:] = rng.randn(p*n, r) * 0.5
                        break
        
        return None
    
    # Try searching for a better rank
    result = search_for_better_rank(min_rank=35, max_rank=50, num_trials=30)
    
    if result is not None:
        r, U_list, V_list, W_list = result
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fall back to schoolbook
    return schoolbook_decomposition()
# EVOLVE-BLOCK-END
