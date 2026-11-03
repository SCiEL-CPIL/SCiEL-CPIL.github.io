import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120.0
    
    # Schoolbook decomposition
    rank_school = n * m * p
    
    U_school = np.zeros((n * m, rank_school), dtype=float)
    V_school = np.zeros((m * p, rank_school), dtype=float)
    W_school = np.zeros((p * n, rank_school), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_a = i * m + j
                idx_b = j * p + k
                idx_c = k * n + i
                U_school[idx_a, col] = 1
                V_school[idx_b, col] = 1
                W_school[idx_c, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Try to find lower rank decompositions using random initialization + optimization
    # For matrix multiplication, we can use the structure to guide the search
    
    # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want to find minimal rank such that this holds
    
    # Known results: 
    # For 2x2x2, rank is 7 (Strassen)
    # For 2x3x3, rank is 14
    # For 3x3x3, rank is 23 (Landsberg)
    # For 3x4x5, we need to search
    
    # Let's try a structured approach: use the fact that matrix multiplication
    # tensors have specific symmetries
    
    # Approach: Try to find a decomposition of rank r < 60
    # We'll use a greedy/iterative approach
    
    # First, let's try to verify if rank 59 or lower is possible
    # by attempting to solve the system of equations
    
    # For small ranks, we can try to use the structure of the problem
    # The key insight is that many of the schoolbook terms can be combined
    
    # Let's try a different approach: use the fact that for rectangular matrices,
    # we can sometimes achieve better than n*m*p
    
    # For 3x4x5, let's try rank values from 59 down
    
    # Actually, let's use a smarter approach: 
    # The matrix multiplication tensor M_{n,m,p} has a known structure.
    # We can try to find linear dependencies among the schoolbook terms.
    
    # Let's set up the problem as finding U, V, W of rank r such that
    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = delta[i,j,k] (for the correct indices)
    
    # This is a system of n*m*p = 60 polynomial equations in 3*r variables
    # (with the constraint that the product equals 1 for each (i,j,k))
    
    # For r = 60, we have a trivial solution.
    # For r < 60, we need to find a non-trivial solution.
    
    # Let's try a numerical optimization approach for small rank reductions
    
    # We'll parameterize U, V, W and minimize the error
    # error = sum_{i,j,k} (sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - 1)^2
    
    # But this is a non-convex problem, so we need to be careful
    
    # Let's try a few random initializations for ranks 59, 58, ...
    
    np.random.seed(42)
    
    # For efficiency, let's only try a few ranks
    for target_rank in [59, 58, 57, 56, 55]:
        if time.time() > deadline - 10:
            break
            
        # Initialize with small perturbations of the schoolbook solution
        # Take the first target_rank columns and add noise to the rest
        
        # Actually, let's use a different initialization:
        # Start with random matrices and use gradient descent
        
        U = np.random.randn(n * m, target_rank) * 0.5
        V = np.random.randn(m * p, target_rank) * 0.5
        W = np.random.randn(p * n, target_rank) * 0.5
        
        # Simple gradient descent
        lr = 0.01
        for step in range(100):
            if time.time() > deadline - 5:
                break
            
            # Compute the current tensor values
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_a = i * m + j
                        idx_b = j * p + k
                        idx_c = k * n + i
                        
                        # T[i,j,k] = sum_r U[idx_a, r] * V[idx_b, r] * W[idx_c, r]
                        T_ijk = np.sum(U[idx_a, :] * V[idx_b, :] * W[idx_c, :])
                        error = T_ijk - 1.0
                        
                        # dU[idx_a, r] += error * V[idx_b, r] * W[idx_c, r]
                        # dV[idx_b, r] += error * U[idx_a, r] * W[idx_c, r]
                        # dW[idx_c, r] += error * U[idx_a, r] * V[idx_b, r]
                        
                        dU[idx_a, :] += error * V[idx_b, :] * W[idx_c, :]
                        dV[idx_b, :] += error * U[idx_a, :] * W[idx_c, :]
                        dW[idx_c, :] += error * U[idx_a, :] * V[idx_b, :]
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Check convergence
            if step % 10 == 0:
                # Compute error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx_a = i * m + j
                            idx_b = j * p + k
                            idx_c = k * n + i
                            T_ijk = np.sum(U[idx_a, :] * V[idx_b, :] * W[idx_c, :])
                            err += (T_ijk - 1.0) ** 2
                if err < 1e-6:
                    # Found a valid decomposition
                    best_U, best_V, best_W = U, V, W
                    best_rank = target_rank
                    break
        
        if best_rank < target_rank:
            break
    
    # Convert to half-integer if possible
    def to_half_int(mat):
        # Multiply by 2 and round to nearest integer
        mat2 = mat * 2
        mat_int = np.round(mat2).astype(int)
        # Check if this is close enough
        if np.max(np.abs(mat_int - mat2)) < 1e-6:
            return mat_int / 2.0
        return mat
    
    best_U = to_half_int(best_U)
    best_V = to_half_int(best_V)
    best_W = to_half_int(best_W)
    
    # If we didn't find a better solution, use schoolbook
    if best_rank == rank_school:
        best_U, best_V, best_W = U_school, V_school, W_school
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
