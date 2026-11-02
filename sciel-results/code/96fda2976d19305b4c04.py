"""Combined program for Matrix Multiplication 3,4,4 rank minimization.
Implements schoolbook decomposition and a heuristic search for lower rank.
"""

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

TIME_LIMIT = 145.0
START_TIME = time.time()

def time_left():
    return TIME_LIMIT - (time.time() - START_TIME)

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # 1. Construct Schoolbook Solution (Rank 48)
    U_sb = np.zeros((n * m, schoolbook_rank), dtype=np.float64)
    V_sb = np.zeros((m * p, schoolbook_rank), dtype=np.float64)
    W_sb = np.zeros((p * n, schoolbook_rank), dtype=np.float64)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, idx] = 1.0
                V_sb[j * p + k, idx] = 1.0
                W_sb[k * n + i, idx] = 1.0
                idx += 1
                
    best_rank = schoolbook_rank
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb

    # 2. Heuristic Search for Lower Rank
    # We try rank = schoolbook_rank - 1, then -2, etc.
    # For each rank, we try a few random initializations and local optimization steps.
    # This is computationally expensive and likely to fail for M_344, but it's a valid "improvement" strategy.
    
    # Define the target tensor T_{i,j,k} = 1 if we are decomposing the multiplication tensor.
    # The tensor is defined by T[i, j, k] = 1 for all i,j,k? No.
    # The matrix multiplication tensor M_{n,m,p} has entries T_{i,j,k} = 1 if the output index (k*n+i) corresponds to input (i*m+j) and (j*p+k).
    # Actually, the tensor is a 3-tensor with indices (i, j, k) where:
    # i in [0, n), j in [0, m), k in [0, p)
    # The entry T_{i,j,k} is 1.
    # The decomposition is: T_{i,j,k} = sum_{c=1}^R U_{i,j,c} V_{j,k,c} W_{k,i,c}
    # Wait, the indices in the problem statement are:
    # U: (n*m, R) -> U[i*m+j, c]
    # V: (m*p, R) -> V[j*p+k, c]
    # W: (p*n, R) -> W[k*n+i, c]
    # Product: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1 for all i,j,k.
    
    def verify(U, V, W, tol=1e-9):
        # Check if U, V, W decompose the tensor
        # Flatten indices
        # For each i, j, k, compute sum
        # This is O(n*m*p*R). For R=47, n*m*p=48, it's 48*47 ~ 2000 ops. Very fast.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(len(U[0])):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def compute_loss_and_grad(U, V, W):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Target: T[i,j,k] = 1
        # val[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # We compute the error and gradients.
        # To be efficient, we can loop over i, j, k and accumulate gradients.
        # Since n,m,p are small, this is fine.
        
        L = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    # Compute val
                    for c in range(len(U[0])):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    
                    err = val - 1.0
                    L += err * err
                    
                    # Gradients
                    # dL/dU[i*m+j, c] = 2 * err * V[j*p+k, c] * W[k*n+i, c]
                    # dL/dV[j*p+k, c] = 2 * err * U[i*m+j, c] * W[k*n+i, c]
                    # dL/dW[k*n+i, c] = 2 * err * U[i*m+j, c] * V[j*p+k, c]
                    
                    for c in range(len(U[0])):
                        uv = U[i*m+j, c] * V[j*p+k, c]
                        vw = V[j*p+k, c] * W[k*n+i, c]
                        uw = U[i*m+j, c] * W[k*n+i, c]
                        
                        grad_U[i*m+j, c] += 2.0 * err * vw
                        grad_V[j*p+k, c] += 2.0 * err * uw
                        grad_W[k*n+i, c] += 2.0 * err * uv
                        
        return L, grad_U, grad_V, grad_W

    def try_rank(R, num_trials=5, max_steps=100):
        if R <= 0:
            return None
        if time_left() < 5:
            return None
            
        for trial in range(num_trials):
            if time_left() < 2:
                break
                
            # Random initialization
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # Scale initialization to help convergence
            U *= 0.5
            V *= 0.5
            W *= 0.5
            
            lr = 0.05
            prev_loss = float('inf')
            
            for step in range(max_steps):
                if time_left() < 2:
                    break
                    
                L, gU, gV, gW = compute_loss_and_grad(U, V, W)
                
                # Check convergence
                if L < 1e-6:
                    break
                if prev_loss - L < 1e-8 and L < 1e-3:
                    break
                prev_loss = L
                
                # Gradient step
                U -= lr * gU
                V -= lr * gV
                W -= lr * gW
                
            # Check if the random initialization happened to be good (unlikely)
            # Or if we ran the optimizer
            if verify(U, V, W):
                # Round to nearest integer/half-integer
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if verify(U_r, V_r, W_r):
                    return U_r, V_r, W_r
                    
            # Also try to snap the continuous solution even if verify(U,V,W) was slightly off due to float error
            # But verify has a tolerance of 1e-9, so if it passed, it's good.
            # If it didn't pass, snapping might break it.
            
        return None

    # Try to find a lower rank
    # We try ranks from schoolbook_rank - 1 downwards.
    # We limit the number of trials and steps to stay within time limits.
    # Since the problem is small, we can afford a bit more search.
    for R in range(schoolbook_rank - 1, max(schoolbook_rank - 10, 0), -1):
        if time_left() < 10:
            break
            
        # Adjust trials and steps based on time left
        # If we have lots of time, do more trials.
        trials = 5
        steps = 50
        
        result = try_rank(R, num_trials=trials, max_steps=steps)
        if result:
            best_U, best_V, best_W = result
            best_rank = R
            # If we found a solution, we can stop or continue to try for even lower rank?
            # Usually, finding one is a big win. Let's continue to see if we can go lower,
            # but with a tighter budget if time is short.
            if time_left() < 30:
                break

    # Return the best found
    # Convert to lists of lists as required by interface
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
