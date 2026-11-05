import numpy as np
import time
import random

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Target rank for 1 x 3 x 7 is 10, so total rank is 20.
    # If 10 fails, try 11 (total 22).
    # If all fail, return schoolbook.
    
    def solve_bls(n, m, p, rank, iterations=500, time_limit=100):
        """
        Solve for decomposition of M_{n,m,p} with given rank.
        Returns U, V, W or None.
        """
        # Tensor T is all ones for M_{n,m,p} in this context?
        # No, T_{i,j,k} = 1 for all i,j,k.
        # We want sum_t U[i,t] V[j,t] W[k,t] = 1 for all i,j,k.
        
        # Initialize U, V, W
        U = np.random.randn(n, rank)
        V = np.random.randn(m, rank)
        W = np.random.randn(p, rank)
        
        # We want U V W^T to be close to 1s tensor.
        # Objective: minimize || U V W^T - 1 ||^2
        # This is non-convex.
        
        start_time = time.time()
        best_err = np.inf
        best_U, best_V, best_W = U, V, W
        
        for it in range(iterations):
            if time.time() - start_time > time_limit:
                break
            
            # Compute current tensor
            # T_approx[i,j,k] = sum_t U[i,t] V[j,t] W[k,t]
            # This is expensive to compute directly for large rank.
            # But n,m,p are small.
            
            # Gradient descent on U, V, W
            # Loss = sum_{i,j,k} (sum_t U[i,t] V[j,t] W[k,t] - 1)^2
            
            # Instead of full tensor, we can use structure.
            # But for small dims, just compute.
            
            # Vectorize
            # T_approx = np.einsum('it,jt,kt->ijk', U, V, W)
            # This is O(n*m*p*rank)
            
            # Gradient with respect to U:
            # dL/dU[i,t] = 2 * (T_approx[i,:,:] - 1) * V[j,t] W[k,t] summed over j,k
            # = 2 * (T_approx[i,:,:] - 1) * (V W^T)[j,k]? No.
            # Let M_VW[j,k] = sum_t V[j,t] W[k,t]
            # Then T_approx[i,j,k] = sum_t U[i,t] V[j,t] W[k,t] is NOT U[i,t] * M_VW[j,k].
            # It is sum_t U[i,t] (V[j,t] W[k,t]).
            # So let Q_t[j,k] = V[j,t] W[k,t].
            # T_approx[i,j,k] = sum_t U[i,t] Q_t[j,k].
            
            # This is a linear combination of rank-1 matrices Q_t.
            
            # We can update U, V, W simultaneously.
            
            # Simple update:
            # T_approx = np.einsum('it,jt,kt->ijk', U, V, W)
            # Error = T_approx - 1
            # Grad_U[i,t] = 2 * sum_{j,k} Error[i,j,k] V[j,t] W[k,t]
            # Grad_V[j,t] = 2 * sum_{i,k} Error[i,j,k] U[i,t] W[k,t]
            # Grad_W[k,t] = 2 * sum_{i,j} Error[i,j,k] U[i,t] V[j,t]
            
            T_approx = np.einsum('it,jt,kt->ijk', U, V, W)
            Error = T_approx - 1.0
            err = np.linalg.norm(Error)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if err < 1e-6:
                break
                
            # Gradients
            # Sum over j,k for U
            # Sum_{j,k} Error[i,j,k] V[j,t] W[k,t]
            # = sum_t' U[i,t'] ( sum_{j,k} Error[i,j,k] V[j,t] W[k,t] )? No.
            
            # Let's use einsum for gradients
            # Grad_U[i,t] = 2 * np.einsum('ijk,jt,kt->it', Error, V, W)
            # Grad_V[j,t] = 2 * np.einsum('ijk,it,kt->jt', Error, U, W)
            # Grad_W[k,t] = 2 * np.einsum('ijk,it,jt->kt', Error, U, V)
            
            Grad_U = 2 * np.einsum('ijk,jt,kt->it', Error, V, W)
            Grad_V = 2 * np.einsum('ijk,it,kt->jt', Error, U, W)
            Grad_W = 2 * np.einsum('ijk,it,jt->kt', Error, U, V)
            
            # Learning rate
            lr = 0.01
            U -= lr * Grad_U
            V -= lr * Grad_V
            W -= lr * Grad_W
            
            # Normalize to prevent explosion
            U /= np.linalg.norm(U) + 1e-9
            V /= np.linalg.norm(V) + 1e-9
            W /= np.linalg.norm(W) + 1e-9
            
        if best_err < 1e-4:
            return best_U, best_V, best_W
        return None

    # Try to find rank 10 decomposition for 1x3x7
    U1, V1, W1 = solve_bls(1, 3, 7, 10, iterations=1000, time_limit=60)
    if U1 is not None:
        # Construct full 2x3x7 decomposition
        # Rank 20
        rank = 20
        U = np.zeros((2, rank))
        V = np.zeros((3, rank))
        W = np.zeros((7, rank))
        
        # First 10 cols for row 0
        U[0, 0:10] = U1[0, :]
        V[0:3, 0:10] = V1[:, :]
        W[0:7, 0:10] = W1[:, :]
        
        # Next 10 cols for row 1
        U[1, 10:20] = U1[0, :]
        V[0:3, 10:20] = V1[:, :]
        W[0:7, 10:20] = W1[:, :]
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback to schoolbook
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
