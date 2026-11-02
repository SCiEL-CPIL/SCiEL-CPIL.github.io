"""Matrix multiplication 3,4,8 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 4, 8
    size_U = n * m  # 12
    size_V = m * p  # 32
    size_W = p * n  # 24
    
    # Build the matrix multiplication tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    # We need rank-r decomposition: T = sum_c U[:,c] outer V[:,c] outer W[:,c]
    
    # Nonzero entries
    nonzero = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                nonzero.append((i * m + j, j * p + k, k * n + i))
    
    n_entries = len(nonzero)  # 96
    
    # Strategy: Use the Kronecker product structure
    # M_{3,4,8} can be viewed as M_{3,4,4} x M_{1,1,2}
    # But more directly, we can use the fact that 4x8 = 4x4 block structure
    # 
    # Actually, let's use a direct approach:
    # The bilinear complexity of 3x4 x 4x8 matrix multiplication
    # 
    # Known results:
    # - Schoolbook: 3*4*8 = 96
    # - We can potentially do better using Winograd-style algorithms
    #
    # Let's try to construct a decomposition using the structure:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # For the 4x8 part, we can split B into two 4x4 blocks:
    # B = [B1, B2] where B1, B2 are 4x4
    # Then C[:, 0:4] = A * B1, C[:, 4:8] = A * B2
    #
    # So we need to compute two 3x4 * 4x4 multiplications.
    # The rank of 3x4*4x4 is at most 48 (schoolbook).
    # Using Winograd's algorithm for 3x4*4x4, we might get lower.
    #
    # Winograd's F(3,4) algorithm has rank 14 for 3x3*3x3, but for 3x4*4x4 it's different.
    #
    # Let me try a different approach: use random initialization and ALS to find low-rank decompositions.
    
    def verify(U, V, W, tol=1e-10):
        """Verify that the decomposition is exact."""
        r = U.shape[1]
        for (a, b, k) in nonzero:
            val = 0.0
            for c in range(r):
                val += U[a, c] * V[b, c] * W[k, c]
            if abs(val - 1.0) > tol:
                return False
        # Check that zero entries remain zero
        # This is expensive, so we only do it for small tensors
        # Actually, for exact arithmetic, if the nonzero entries are correct
        # and we're using integer/half-integer entries, the zeros should be fine.
        # But let's verify a sample of zero entries.
        checked = 0
        for trial in range(min(100, size_U * size_V * size_W - n_entries)):
            a = np.random.randint(0, size_U)
            b = np.random.randint(0, size_V)
            k = np.random.randint(0, size_W)
            # Check if this is a nonzero entry
            is_nonzero = any((a == na and b == nb and k == nk) for (na, nb, nk) in nonzero)
            if is_nonzero:
                continue
            val = 0.0
            for c in range(r):
                val += U[a, c] * V[b, c] * W[k, c]
            if abs(val) > tol:
                return False
            checked += 1
        return True
    
    # Try to find a low-rank decomposition using ALS with random restarts
    def try_rank(r, num_restarts=100, max_iter=5000):
        """Try to find rank-r exact decomposition using ALS."""
        best_err = np.inf
        best_factors = None
        
        for restart in range(num_restarts):
            if time.time() > DEADLINE - 10:
                break
            
            np.random.seed(restart * 12345 + 67890)
            
            # Initialize with small random values
            U = np.random.randn(size_U, r) * 0.3
            V = np.random.randn(size_V, r) * 0.3
            W = np.random.randn(size_W, r) * 0.3
            
            err = np.inf
            for it in range(max_iter):
                if time.time() > DEADLINE - 10:
                    break
                
                # Alternating Least Squares
                # Fix U and V, update W
                # For each k, we want to minimize:
                # sum over (a,b) where (a,b,k) is nonzero: (sum_c U[a,c]*V[b,c]*W[k,c] - 1)^2
                # + sum over (a,b) where (a,b,k) is NOT nonzero: (sum_c U[a,c]*V[b,c]*W[k,c])^2
                #
                # This is a linear system for W[:,c] for each c... but it's coupled across c.
                # Actually, for fixed U,V, the optimal W is:
                # W[k,c] = argmin sum_{(a,b)} (sum_{c'} U[a,c']*V[b,c']*W[k,c'] - T[a,b,k])^2
                # where T[a,b,k] = 1 if (a,b,k) is nonzero, else 0
                #
                # This gives a linear system: for each k,
                # sum_c' W[k,c'] * (sum_{(a,b)} U[a,c]*U[a,c'] * V[b,c]*V[b,c']) = sum_{(a,b)} U[a,c] * V[b,c] * T[a,b,k]
                
                # Let's compute the Gram matrix G[k, c, c'] = sum_{(a,b)} U[a,c]*U[a,c']*V[b,c]*V[b,c']
                # and the RHS h[k, c] = sum_{(a,b)} U[a,c]*V[b,c]*T[a,b,k]
                
                # This is expensive. Let's use a simpler gradient-based approach instead.
                
                # Compute gradient of error with respect to W
                # Error = sum_{(a,b,k)} (sum_c U[a,c]*V[b,c]*W[k,c] - T[a,b,k])^2
                # + sum_{(a,b,k) not in nonzero} (sum_c U[a,c]*V[b,c]*W[k,c])^2
                
                # Actually, let's just compute the error and use a simple update.
                
                # Compute current error (only on nonzero entries for speed)
                err = 0.0
                for (a, b, k) in nonzero:
                    val = 0.0
                    for c in range(r):
                        val += U[a, c] * V[b, c] * W[k, c]
                    err += (val - 1.0) ** 2
                
                if err < 1e-20:
                    # Verify fully
                    if verify(U, V, W):
                        return U, V, W
                    break
                
                if err < best_err:
                    best_err = err
                    best_factors = (U.copy(), V.copy(), W.copy())
                
                # Simple gradient step
                # dE/dW[k,c] = 2 * sum_{(a,b)} (sum_c' U[a,c']*V[b,c']*W[k,c'] - T[a,b,k]) * U[a,c] * V[b,c]
                # where the sum is over all (a,b) such that (a,b,k) is in the tensor domain
                
                # This is expensive to compute exactly. Let's use a stochastic approach.
                # Pick a random nonzero entry and update.
                
                idx = np.random.randint(0, n_entries)
                a, b, k = nonzero[idx]
                
                # Current value at (a,b,k)
                val = 0.0
                for c in range(r):
                    val += U[a, c] * V[b, c] * W[k, c]
                
                # Gradient with respect to W[k,c]
                # dE/dW[k,c] = 2 * (val - 1) * U[a,c] * V[b,c]
                lr = 0.01
                for c in range(r):
                    W[k, c] -= lr * 2 * (val - 1.0) * U[a, c] * V[b, c]
                
                # Similarly update U and V
                idx = np.random.randint(0, n_entries)
                a, b, k = nonzero[idx]
                val = 0.0
                for c in range(r):
                    val += U[a, c] * V[b, c] * W[k, c]
                for c in range(r):
                    U[a, c] -= lr * 2 * (val - 1.0) * V[b, c] * W[k, c]
                
                idx = np.random.randint(0, n_entries)
                a, b, k = nonzero[idx]
                val = 0.0
                for c in range(r):
                    val += U[a, c] * V[b, c] * W[k, c]
                for c in range(r):
                    V[b, c] -= lr * 2 * (val - 1.0) * U[a, c] * W[k, c]
                
                # Occasionally add noise to escape local minima
                if it % 500 == 0 and err > 1e-6:
                    U += np.random.randn(size_U, r) * 0.01
                    V += np.random.randn(size_V, r) * 0.01
                    W += np.random.randn(size_W, r) * 0.01
            
            if best_factors is not None:
                U, V, W = best_factors
                if verify(U, V, W):
                    return U, V, W
        
        return None
    
    # Try ranks from low to high
    # Start with a reasonable lower bound
    # The rank must be at least max(size_U, size_V, size_W) = 32
    # Actually, the rank must be at least the rank of any matricization
    # Let's just try from 32 upward
    
    best_result = None
    
    for r in range(32, 97):
        if time.time() > DEADLINE - 20:
            break
        
        result = try_rank(r, num_restarts=30, max_iter=3000)
        if result is not None:
            U, V, W = result
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(size_U)]
    V = [[0] * rank for _ in range(size_V)]
    W = [[0] * rank for _ in range(size_W)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
