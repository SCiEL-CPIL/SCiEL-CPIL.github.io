import numpy as np
import time

def construct():
    n, m, p = 3, 3, 7
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # 1. Baseline: Schoolbook decomposition (rank 63)
    # This is always valid and fast to construct.
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook), dtype=np.float64)
    V_sb = np.zeros((m * p, rank_schoolbook), dtype=np.float64)
    W_sb = np.zeros((p * n, rank_schoolbook), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
                
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_schoolbook

    # 2. Structural Change: Try to find a lower rank decomposition.
    # The current program (score 1.0) just returns schoolbook.
    # The inspirations suggest random search is inefficient.
    # Let's try a structured search for rank < 63.
    
    # Lower bound is max(n*m, m*p, p*n) = max(9, 21, 21) = 21.
    # Known best for 3x3*3x3 is 23 (or 22).
    # For 3x3*3x7, we can view it as 7 columns of 3x3*3x1.
    # Or simply as a tensor of size 3x3x7.
    
    # Strategy: 
    # If we can find a rank R decomposition, the score improves.
    # Since random search over {-1,0,1} is unlikely to hit the exact integer/rational solution 
    # for non-trivial ranks in 145s, we should focus on known algebraic structures 
    # or a very targeted local search if we had a starting point.
    
    # However, finding a rank < 63 decomposition for 3x3x7 is a hard open problem 
    # unless there's a specific known algorithm. 
    # The "Strassen-like" decomposition for 2x2x2 is rank 7.
    # For 3x3x3, best known is 23.
    # For 3x3x7, a block decomposition might work:
    # Split B into B1 (3x3) and B2 (3x4).
    # C1 = A * B1 (3x3 result)
    # C2 = A * B2 (3x4 result)
    # Rank(C1) + Rank(C2).
    # If Rank(3x3x3) = 23 and Rank(3x3x4) = ?
    # 3x3x4 is equivalent to 3x3x3 + 3x3x1.
    # Rank(3x3x1) is 9 (schoolbook for vector multiplication? No, 3x3*3x1 is just 9 mults).
    # Actually, 3x3*3x1 is just 9 scalar multiplications. So rank 9.
    # So Rank(3x3x4) <= 23 + 9 = 32?
    # Total rank <= 23 + 32 = 55.
    # This is better than 63!
    
    # Let's try to construct this explicitly.
    # We need the explicit factors for 3x3x3 rank 23 and 3x3x4 rank 32 (or better).
    # Finding the explicit factors for rank 23 3x3x3 multiplication is non-trivial 
    # and I don't have them memorized in a form I can hardcode quickly and verify 
    # within the time limit without risk of error.
    
    # Alternative: Use the fact that the evaluator accepts integer/half-integer entries.
    # Strassen for 2x2x2 uses integers.
    # Can we embed 3x3x7 into a larger problem or use a known sparse algorithm?
    
    # Let's try a different structural approach: 
    # Local search starting from a randomized low-rank initialization?
    # The cost function is sum_{i,j,k} (sum_c U_ic V_jc W_kc - 1)^2.
    # This is a quartic (if we square) or we can minimize the max error.
    # Gradient descent on rank R matrices.
    
    # Let's try a simple gradient descent for a few ranks, say 50, 55, 60.
    # If we can find a valid decomposition with rank < 63, we win.
    
    def verify(U, V, W, n, m, p, tol=1e-5):
        """Check if decomposition is valid."""
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We need T[i,j,k] == 1 for all i,j,k.
        # For efficiency, we can compute the tensor T for all i,j,k.
        # U shape (n*m, R), V shape (m*p, R), W shape (p*n, R)
        # This is equivalent to checking if the tensor product matches.
        
        # Efficient check:
        # For each c, we add outer product of U[:,c], V[:,c], W[:,c] to a tensor.
        # But R is up to 63, and tensor size is 3*3*7 = 63.
        # Total ops: R * n * m * p. 63 * 63 ~ 4000. Very fast.
        
        T = np.zeros((n, m, p), dtype=np.float64)
        for c in range(U.shape[1]):
            u = U[:, c].reshape(n, m)
            v = V[:, c].reshape(m, p)
            w = W[:, c].reshape(p, n)
            # T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
            # This is not a standard outer product because of the indices.
            # u[i,j] is n x m
            # v[j,k] is m x p
            # w[k,i] is p x n
            # We need to sum over j for fixed i,k? No, the sum is over c.
            # For fixed c, the contribution to T[i,j,k] is u[i,j] * v[j,k] * w[k,i].
            # Let's construct the tensor for this c.
            # T_c[i,j,k] = u[i,j] * v[j,k] * w[k,i]
            
            # We can do this with einsum or loops.
            # n=3, m=3, p=7. Small enough for loops.
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
                        
        return np.allclose(T, 1.0, atol=tol)

    # Try gradient descent for rank R
    def try_grad_descent(R, n, m, p, max_iters=500, lr=0.01):
        """
        Attempt to find a rank R decomposition using gradient descent.
        Returns (U, V, W) if successful, else None.
        """
        # Initialize with small random values
        np.random.seed(42) # Fixed seed for reproducibility in this attempt
        U = np.random.randn(n * m, R) * 0.1
        V = np.random.randn(m * p, R) * 0.1
        W = np.random.randn(p * n, R) * 0.1
        
        # Objective: minimize sum_{i,j,k} (sum_c U_ic V_jc W_kc - 1)^2
        # Let T_c be the tensor for column c. T = sum_c T_c.
        # Error E = sum_{i,j,k} (T[i,j,k] - 1)^2
        
        # Gradients:
        # dE/du[i,j] = 2 * sum_k (T[i,j,k] - 1) * v[j,k] * w[k,i]  (for a single c? No, sum over c)
        # Actually, T[i,j,k] = sum_c u_c[i,j] v_c[j,k] w_c[k,i]
        # dT[i,j,k]/du_c[i,j] = v_c[j,k] w_c[k,i]
        # dE/du_c[i,j] = 2 * sum_{i',j',k'} (T[i',j',k'] - 1) * dT[i',j',k']/du_c[i,j]
        #                = 2 * sum_{i',j',k'} (T[i',j',k'] - 1) * delta_{i',i} delta_{j',j} v_c[j,k'] w_c[k',i]
        #                = 2 * sum_{k'} (T[i,j,k'] - 1) v_c[j,k'] w_c[k',i]
        
        # This is O(n*m*p*R) per gradient step.
        # n*m*p = 63, R ~ 50. 63*50 = 3150 ops per matrix update.
        # 3 matrices * 3150 * 500 iters = 4.7M ops. Fast enough.
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                return None
            
            # Compute T
            T = np.zeros((n, m, p), dtype=np.float64)
            for c in range(R):
                u = U[:, c].reshape(n, m)
                v = V[:, c].reshape(m, p)
                w = W[:, c].reshape(p, n)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
            
            # Check convergence
            err = np.max(np.abs(T - 1.0))
            if err < 1e-6:
                return U, V, W
            
            # Compute Gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            # dU_c[i,j] = 2 * sum_k (T[i,j,k] - 1) V_c[j,k] W_c[k,i]
            # dV_c[j,k] = 2 * sum_i (T[i,j,k] - 1) U_c[i,j] W_c[k,i]
            # dW_c[k,i] = 2 * sum_j (T[i,j,k] - 1) U_c[i,j] V_c[j,k]
            
            # We can compute these efficiently.
            # Let E[i,j,k] = 2 * (T[i,j,k] - 1)
            
            E = 2.0 * (T - 1.0)
            
            for c in range(R):
                u = U[:, c].reshape(n, m)
                v = V[:, c].reshape(m, p)
                w = W[:, c].reshape(p, n)
                
                # dU_c[i,j] = sum_k E[i,j,k] v[j,k] w[k,i]
                # dV_c[j,k] = sum_i E[i,j,k] u[i,j] w[k,i]
                # dW_c[k,i] = sum_j E[i,j,k] u[i,j] v[j,k]
                
                for i in range(n):
                    for j in range(m):
                        # dU
                        s = 0.0
                        for k in range(p):
                            s += E[i, j, k] * v[j, k] * w[k, i]
                        dU[i * m + j, c] = s
                        
                        for k in range(p):
                            # dV
                            s_v = 0.0
                            for i2 in range(n):
                                s_v += E[i2, j, k] * u[i2, j] * w[k, i2]
                            dV[j * p + k, c] = s_v
                            
                            # dW
                            s_w = 0.0
                            for j2 in range(m):
                                s_w += E[i, j2, k] * u[i, j2] * v[j2, k]
                            dW[k * n + i, c] = s_w

            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
        return None

    # Try a few ranks with gradient descent
    # Start from ranks likely to be feasible if a better decomposition exists.
    # 63 is schoolbook. Let's try 60, 55, 50.
    # If we find a valid one, we return it.
    
    ranks_to_try = [62, 61, 60, 55, 50]
    
    for R in ranks_to_try:
        if time.time() > deadline - 20:
            break
            
        # Try a few different seeds for gradient descent
        for seed in range(5):
            np.random.seed(seed)
            result = try_grad_descent(R, n, m, p, max_iters=200, lr=0.005)
            if result is not None:
                U, V, W = result
                # Verify with strict tolerance
                if verify(U, V, W, n, m, p, tol=1e-5):
                    best_U, best_V, best_W = U, V, W
                    best_rank = R
                    # Break out of loops to return immediately
                    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
            
            if time.time() > deadline - 10:
                break

    # If no better rank found, return schoolbook
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
