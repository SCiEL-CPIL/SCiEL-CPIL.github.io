"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
import random

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 7 matrix.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Schoolbook decomposition as fallback
    def schoolbook_decomposition():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Precompute target tensor indices
    # T[i*m+j, j*p+k, k*n+i] = 1
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_indices.append((i * m + j, j * p + k, k * n + i))
                
    def check_decomposition(U, V, W, tol=1e-6):
        """
        Check if U, V, W exactly decompose the tensor.
        U: (n*m, R), V: (m*p, R), W: (p*n, R)
        """
        R = U.shape[1]
        # Compute S[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # We only need to check if S matches T.
        # T is 1 at target_indices, 0 elsewhere.
        
        # For efficiency, we can just check the target indices and a few random zeros.
        # But for exactness, we should check all or use a robust check.
        # Given R is small, checking all 1764 entries is fast.
        
        # Initialize S to 0
        # We don't need to store full S if we just sum errors.
        error = 0.0
        
        # Check target indices (should be 1)
        for idx in target_indices:
            a, b, c = idx
            s = 0.0
            for r in range(R):
                s += U[a, r] * V[b, r] * W[c, r]
            error += (s - 1.0) ** 2
            
        # Check non-target indices (should be 0)
        # We can sample or check all. Checking all is 1764 - 42 = 1722 entries.
        # 1722 * 20 * 3 ops ~ 100k ops. Fast.
        
        # Create a set of target tuples for O(1) lookup?
        # Or just iterate all and check if it's in target set.
        target_set = set(target_indices)
        
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    if (a, b, c) in target_set:
                        continue
                    s = 0.0
                    for r in range(R):
                        s += U[a, r] * V[b, r] * W[c, r]
                    error += s ** 2
                    
        return error < tol

    def optimize_rank(R, max_restarts=10, max_iters=2000, lr=0.01):
        """
        Try to find a decomposition of rank R using gradient descent.
        """
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
                
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # Normalize initial columns
            for r in range(R):
                U[:, r] /= (np.linalg.norm(U[:, r]) + 1e-9)
                V[:, r] /= (np.linalg.norm(V[:, r]) + 1e-9)
                W[:, r] /= (np.linalg.norm(W[:, r]) + 1e-9)
            
            current_lr = lr
            best_error = float('inf')
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Compute S and error
                # S = sum_r U_r V_r W_r
                # We compute gradient directly in the loop to save memory?
                # No, S is small (1764).
                
                S = np.zeros((n * m, m * p, p * n))
                for r in range(R):
                    S += U[:, r][:, None, None] * V[:, r][None, :, None] * W[:, r][None, None, :]
                
                # Compute error
                # T is 1 at target_indices, 0 elsewhere
                # error = sum (S - T)^2
                # We can compute this by initializing error with sum(S^2)
                # and subtracting 2*S_target + 1 for target indices?
                # No, T is 1 at target, 0 else.
                # (S-T)^2 = S^2 - 2ST + T^2
                # Sum over all abc:
                # Sum S^2 - 2 * Sum_{target} S + Count_target
                
                error = np.sum(S ** 2)
                for idx in target_indices:
                    a, b, c = idx
                    error -= 2.0 * S[a, b, c]
                error += len(target_indices) # T^2 is 1 for target, 0 else
                
                if error < best_error:
                    best_error = error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if error < 1e-8:
                    return best_U, best_V, best_W
                
                # Compute Gradient
                diff = S
                # For target indices, diff = S - 1
                # For others, diff = S
                # dU[a,r] = sum_bc diff[a,b,c] V[b,r] W[c,r]
                
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                # We need to apply the "diff" correctly.
                # Let's create a full diff tensor? 1764 floats. Fine.
                diff_tensor = S.copy()
                for idx in target_indices:
                    a, b, c = idx
                    diff_tensor[a, b, c] -= 1.0
                
                for r in range(R):
                    dU[:, r] = np.einsum('abc,br,cr->a', diff_tensor, V[:, r], W[:, r])
                    dV[:, r] = np.einsum('abc,ar,cr->b', diff_tensor, U[:, r], W[:, r])
                    dW[:, r] = np.einsum('abc,ar,br->c', diff_tensor, U[:, r], V[:, r])
                    
                # Update
                U -= current_lr * dU
                V -= current_lr * dV
                W -= current_lr * dW
                
                # Renormalize
                for r in range(R):
                    U[:, r] /= (np.linalg.norm(U[:, r]) + 1e-9)
                    V[:, r] /= (np.linalg.norm(V[:, r]) + 1e-9)
                    W[:, r] /= (np.linalg.norm(W[:, r]) + 1e-9)
                    
                # Reduce learning rate if stuck
                if it % 100 == 0 and it > 0:
                    current_lr *= 0.9
                    
            # End iterations for this restart
            # Check if best is good enough
            if best_error < 1e-6:
                return best_U, best_V, best_W
                
        return None

    # Try ranks
    # Based on 2p + ceil(p/2) pattern:
    # p=7 -> 20
    # p=6 -> 17
    # p=5 -> 14
    # p=4 -> 11
    # p=3 -> 8
    
    target_ranks = [20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 35, 42]
    
    best_decomp = None
    best_rank = n * m * p
    
    for R in target_ranks:
        if time.time() > deadline:
            break
            
        result = optimize_rank(R)
        if result is not None:
            U, V, W = result
            # Final verification
            if check_decomposition(U, V, W, tol=1e-6):
                best_decomp = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = R
                break # Found the lowest rank in our list
        
    if best_decomp is None:
        best_decomp = schoolbook_decomposition()
        
    return best_decomp
# EVOLVE-BLOCK-END
