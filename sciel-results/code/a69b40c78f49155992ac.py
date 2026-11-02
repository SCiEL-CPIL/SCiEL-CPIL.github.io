import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    # Strategy: Decompose 3x3x7 as direct sum of 3x3x3 and 3x3x4
    # The 3x3x3 multiplication tensor has known rank 23 (Strassen-like)
    # We'll use schoolbook for both parts initially, then try to optimize
    
    # Part 1: 3x3x3 multiplication (indices i,j,k where k=0,1,2)
    # Part 2: 3x3x4 multiplication (indices i,j,k where k=3,4,5,6)
    
    # Schoolbook for 3x3x3: rank 27
    # Schoolbook for 3x3x4: rank 36
    # Total schoolbook: 63
    
    # Try to find better decompositions using optimization
    # For 3x3x3, known best is rank 23
    # For 3x3x4, let's search for rank reduction
    
    best_rank = n * m * p  # Start with schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize with schoolbook
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    best_rank = rank_schoolbook
    
    # Try to improve using gradient-based optimization
    # We'll try different rank values and see if we can find valid decompositions
    
    def validate_decomposition(U, V, W, tol=1e-10):
        """Check if U, V, W form a valid decomposition of the matrix multiplication tensor."""
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        
        # The tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # should equal 1 if i,j,k are all in range, else 0
        # Actually, for matrix multiplication: T[i,j,k] = delta_{i,j} * delta_{j,k} doesn't apply
        # The correct condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        # Wait, that's not right either. Let me reconsider.
        
        # For matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor is T[i,j,k] = 1 (it's just the identity in the middle index)
        # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    errors.append(abs(val - 1.0))
        
        return max(errors) < tol
    
    # Try to find lower rank decompositions
    # Start with rank 50 and try to go lower
    for target_rank in range(55, 25, -1):
        if time.time() > deadline - 5:
            break
        
        # Try random initialization and gradient descent
        n_trials = 10
        found = False
        
        for trial in range(n_trials):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with small values
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Simple gradient descent
            lr = 0.01
            for step in range(500):
                if time.time() > deadline - 5:
                    break
                
                # Compute gradient
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # Loss: (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
                            val = 0.0
                            for r in range(target_rank):
                                val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                            loss = (val - 1.0) ** 2
                            grad = 2 * (val - 1.0)
                            
                            for r in range(target_rank):
                                dU[i * m + j, r] += grad * V[j * p + k, r] * W[k * n + i, r]
                                dV[j * p + k, r] += grad * U[i * m + j, r] * W[k * n + i, r]
                                dW[k * n + i, r] += grad * U[i * m + j, r] * V[j * p + k, r]
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Check convergence
                if step % 50 == 0:
                    if validate_decomposition(U, V, W, tol=1e-6):
                        # Found a valid decomposition!
                        # Try to polish with higher precision
                        lr_fine = 0.001
                        for step_fine in range(200):
                            if time.time() > deadline - 5:
                                break
                            dU = np.zeros_like(U)
                            dV = np.zeros_like(V)
                            dW = np.zeros_like(W)
                            
                            for i in range(n):
                                for j in range(m):
                                    for k in range(p):
                                        val = 0.0
                                        for r in range(target_rank):
                                            val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                                        grad = 2 * (val - 1.0)
                                        
                                        for r in range(target_rank):
                                            dU[i * m + j, r] += grad * V[j * p + k, r] * W[k * n + i, r]
                                            dV[j * p + k, r] += grad * U[i * m + j, r] * W[k * n + i, r]
                                            dW[k * n + i, r] += grad * U[i * m + j, r] * V[j * p + k, r]
                            
                            U -= lr_fine * dU
                            V -= lr_fine * dV
                            W -= lr_fine * dW
                        
                        if validate_decomposition(U, V, W, tol=1e-10):
                            best_U = U.copy()
                            best_V = V.copy()
                            best_W = W.copy()
                            best_rank = target_rank
                            found = True
                            break
                        else:
                            # Revert if polish didn't work
                            break
                    break
            
            if found:
                break
        
        if found:
            # Try to go even lower
            continue
        else:
            # Can't achieve this rank, try lower
            break
    
    # Final validation
    if best_U is not None and best_V is not None and best_W is not None:
        if not validate_decomposition(best_U, best_V, best_W, tol=1e-10):
            # Fall back to schoolbook
            best_U = U_school
            best_V = V_school
            best_W = W_school
            best_rank = rank_schoolbook
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
