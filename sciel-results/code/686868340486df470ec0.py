"""Rewritten program for matrix multiplication 3x3 by 3x5 rank minimization."""
import numpy as np
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_sb = n * m * p  # 45
    
    # We'll try to find a lower rank decomposition.
    # The minimum rank for matrix multiplication is known to be at least
    # max(n*m, m*p, p*n) = max(9, 15, 15) = 15, but for small matrices
    # the exact minimum is often higher.
    
    # For 3x3 * 3x5, let's try to find a decomposition with rank < 45.
    # We'll use a randomized approach with ALS.
    
    deadline = time.time() + 120
    
    # Try various ranks from low to high
    # Known results: For 3x3*3x3, min rank is 23 (Strassen-like)
    # For 3x3*3x5, we need to figure out the minimum.
    
    # Let's try ranks from 15 upward, but focus on likely candidates
    # The bilinear complexity of multiplying 3x3 by 3x5...
    
    # Actually, let's use a more systematic approach.
    # We'll try to verify if a given rank decomposition works using
    # the tensor equation.
    
    # The tensor T[i,j,k] = 1 if the product C[i,k] = sum_j A[i,j]*B[j,k]
    # involves this term. The decomposition is:
    # sum_r U[i,j,r] * V[j,k,r] * W[r] should equal T[i,j,k]
    # where W[r] is the coefficient for the output.
    
    # More precisely, the multiplication tensor has entries T_{i,j,k} = 1
    # for all valid i,j,k (since each product term appears).
    # The decomposition satisfies:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Let's try a grid search over possible ranks with random initializations
    # and ALS iterations.
    
    best_rank = rank_sb
    best_decomp = None
    
    # Precompute the target tensor
    # T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # We'll try ranks from a reasonable lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = 15
    # But realistically, for these small dimensions, the minimum might be
    # in the range 20-30.
    
    # Let's try a few candidate ranks
    candidate_ranks = []
    for r in range(15, min(rank_sb + 1, 46)):
        candidate_ranks.append(r)
    
    # Sort by likelihood - try middle ranges first
    # Actually, let's just try from low to high and stop when we find one
    # But we need to be careful about time.
    
    # Let's try a smarter approach: use known results.
    # For 3x3 matrix multiplication, the minimum rank is 23.
    # For 3x3 * 3x5, it's likely higher.
    
    # Let's try to use a numerical approach with ALS.
    
    def try_rank(r, num_restarts=5, max_iter=100):
        """Try to find a rank-r decomposition using ALS."""
        if time.time() > deadline - 10:
            return None
        
        np.random.seed(42)  # Fixed seed for reproducibility
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                return None
            
            # Random initialization
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # Normalize
            U /= np.linalg.norm(U, axis=0, keepdims=True) + 1e-10
            V /= np.linalg.norm(V, axis=0, keepdims=True) + 1e-10
            W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-10
            
            for it in range(max_iter):
                if time.time() > deadline - 10:
                    return None
                
                # Update U: for each column r, solve
                # U[:, r] to minimize sum_{i,j,k} (T[i,j,k] - sum_r' U[i*m+j, r'] * V[j*p+k, r'] * W[k*n+i, r'])^2
                # This is a linear system in U[:, r]
                
                # The gradient with respect to U[i*m+j, r] is:
                # -2 * sum_k T[i,j,k] * V[j*p+k, r] * W[k*n+i, r]
                # + 2 * sum_k (sum_r' U[i*m+j, r'] * V[j*p+k, r'] * W[k*n+i, r']) * V[j*p+k, r] * W[k*n+i, r]
                
                # This is complex. Let's use a different approach.
                # We can view this as a linear least squares problem.
                
                # For fixed V and W, the problem is linear in U.
                # Let's form the system.
                
                # Actually, let's use a simpler approach: gradient descent
                # or use the fact that this is a bilinear problem.
                
                # Let's use a different strategy: we'll use the fact that
                # the tensor can be viewed as a matrix and use SVD.
                
                # Reshape the problem: for each r, the contribution is
                # outer product structure.
                
                # Let's just do a simple verification approach instead.
                # Check if the current decomposition is valid.
                
                # Compute the residual
                # residual[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] - 1
                # for all i,j,k
                
                # This is O(n*m*p*r) which is fine for small n,m,p.
                
                residual = np.zeros((n, m, p))
                for r in range(r):
                    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    # This is a 3D tensor outer product
                    u_col = U[:, r].reshape(n, m)
                    v_col = V[:, r].reshape(m, p)
                    w_col = W[:, r].reshape(p, n)
                    
                    # For each i,j,k: u_col[i,j] * v_col[j,k] * w_col[k,i]
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                residual[i, j, k] += u_col[i, j] * v_col[j, k] * w_col[k, i]
                
                # Target is all ones
                target = np.ones((n, m, p))
                err = np.linalg.norm(residual - target)
                
                if err < 1e-6:
                    # Found a valid decomposition!
                    return U, V, W, err
                
                # If not valid, we need to update.
                # Let's try a simple gradient step.
                
                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                # dU[i*m+j, r] = -2 * sum_k (residual[i,j,k] - 1) * V[j*p+k, r] * W[k*n+i, r]
                # Actually, the loss is sum_{i,j,k} (sum_r' U[i*m+j, r'] * V[j*p+k, r'] * W[k*n+i, r'] - 1)^2
                # dL/dU[i*m+j, r] = 2 * sum_k (residual[i,j,k] - 1) * V[j*p+k, r] * W[k*n+i, r]
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            diff = residual[i, j, k] - 1
                            for r in range(r):
                                dU[i*m+j, r] += 2 * diff * V[j*p+k, r] * W[k*n+i, r]
                                dV[j*p+k, r] += 2 * diff * U[i*m+j, r] * W[k*n+i, r]
                                dW[k*n+i, r] += 2 * diff * U[i*m+j, r] * V[j*p+k, r]
                
                # Gradient descent step
                lr = 0.01
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Check convergence
                if np.linalg.norm(dU) < 1e-8:
                    break
        
        return None
    
    # Try ranks from low to high
    # But we need to be efficient. Let's try a few key ranks.
    
    # For 3x3*3x5, let's try ranks 20-35
    for r in range(20, 36):
        if time.time() > deadline - 20:
            break
        
        result = try_rank(r, num_restarts=3, max_iter=50)
        if result is not None:
            U, V, W, err = result
            best_rank = r
            best_decomp = (U, V, W)
            break
    
    # If we found a better rank, use it; otherwise use schoolbook
    if best_decomp is not None:
        U, V, W = best_decomp
        # Convert to lists of lists with integer/half-integer entries if possible
        # Round to nearest 0.5
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Use schoolbook decomposition
        rank = rank_sb
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
