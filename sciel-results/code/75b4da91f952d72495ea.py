"""Combined program for Matrix Multiplication 3,4,4 rank minimization.
Implements schoolbook decomposition and a heuristic search for lower rank.
"""

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import product

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
    
    def verify(U, V, W, tol=1e-9):
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
        L = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(len(U[0])):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    
                    err = val - 1.0
                    L += err * err
                    
                    for c in range(len(U[0])):
                        vw = V[j*p+k, c] * W[k*n+i, c]
                        uw = U[i*m+j, c] * W[k*n+i, c]
                        uv = U[i*m+j, c] * V[j*p+k, c]
                        
                        grad_U[i*m+j, c] += 2.0 * err * vw
                        grad_V[j*p+k, c] += 2.0 * err * uw
                        grad_W[k*n+i, c] += 2.0 * err * uv
                        
        return L, grad_U, grad_V, grad_W

    def polish(U, V, W, max_steps=200, tol=1e-12):
        """Refine a near-solution using adaptive learning rate and momentum."""
        # Copy inputs to avoid modifying originals
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        lr = 0.1
        momentum = 0.9
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        
        best_loss = float('inf')
        best_state = (U.copy(), V.copy(), W.copy())
        
        for step in range(max_steps):
            if time_left() < 2:
                break
                
            L, gU, gV, gW = compute_loss_and_grad(U, V, W)
            
            if L < tol:
                break
                
            if L < best_loss:
                best_loss = L
                best_state = (U.copy(), V.copy(), W.copy())
            
            # Update momentum
            vU = momentum * vU + gU
            vV = momentum * vV + gV
            vW = momentum * vW + gW
            
            U -= lr * vU
            V -= lr * vV
            W -= lr * vW
            
            # Simple adaptive learning rate: decrease if loss increases
            # (Simplified: just decay slowly)
            lr *= 0.99
            
        # Return the best state found
        return best_state

    def try_rank(R, num_trials=20, max_steps=100):
        if R <= 0:
            return None
        if time_left() < 5:
            return None
            
        best_solution = None
        min_loss = float('inf')
        
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
            converged = False
            
            for step in range(max_steps):
                if time_left() < 2:
                    break
                    
                L, gU, gV, gW = compute_loss_and_grad(U, V, W)
                
                # Check convergence
                if L < 1e-6:
                    converged = True
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
            
            # If not verified, check loss and keep track of best
            L, _, _, _ = compute_loss_and_grad(U, V, W)
            if L < min_loss:
                min_loss = L
                best_solution = (U.copy(), V.copy(), W.copy())
                
            # If we have a very good solution (loss < 1e-3), try to polish it
            if L < 1e-3:
                U_pol, V_pol, W_pol = polish(U, V, W, max_steps=100)
                if verify(U_pol, V_pol, W_pol):
                    U_r = np.round(U_pol * 2) / 2
                    V_r = np.round(V_pol * 2) / 2
                    W_r = np.round(W_pol * 2) / 2
                    if verify(U_r, V_r, W_r):
                        return U_r, V_r, W_r
                    # If rounding failed, check if unrounded is valid (should be, but verify again with tighter tol)
                    if verify(U_pol, V_pol, W_pol, tol=1e-8):
                        # Accept half-integer approximation if it's close enough? 
                        # The problem requires exact integer/half-integer.
                        # So we must return integer/half-integer.
                        # If verify(U_pol) passes but verify(U_r) fails, U_pol is not a valid solution.
                        pass

        # If no exact solution found, return None
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
        trials = 20
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
