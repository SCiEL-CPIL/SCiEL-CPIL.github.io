"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Finds a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix
    by a 3 by 10 matrix.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Schoolbook Rank Construction
    def get_schoolbook():
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
        return {"U": U, "V": V, "W": W}, rank
    
    best_result, best_rank = get_schoolbook()
    
    # Try to find a lower rank decomposition using numerical optimization
    # The tensor equation: T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    # where T is the matrix multiplication tensor.
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i, j, k.
    # Otherwise T[a, b, c] = 0.
    
    # We want to find U, V, W such that:
    # sum_r U[a, r] * V[b, r] * W[c, r] = T[a, b, c]
    
    # This is a rank-r decomposition of the 3-tensor T.
    
    # Let's build the tensor T efficiently.
    # T has shape (n*m, m*p, p*n) = (6, 30, 20)
    # Total elements: 6 * 30 * 20 = 3600.
    
    # We can represent T as a sparse structure or just iterate over the non-zero entries.
    # The non-zero entries are at (i*m+j, j*p+k, k*n+i) for i in [0,2), j in [0,3), k in [0,10).
    # There are 2 * 3 * 10 = 60 non-zero entries.
    
    # To check if a candidate (U, V, W) is valid:
    # We need to verify that for all a, b, c:
    # sum_r U[a, r] * V[b, r] * W[c, r] == T[a, b, c]
    
    # We can compute the reconstructed tensor R = U @ V^T @ W^T? No, that's not right.
    # R[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    
    # This is a 3-mode contraction.
    # We can compute this using einsum: np.einsum('ar,br,cr->abc', U, V, W)
    
    # Let's try to solve for a lower rank.
    # The schoolbook rank is 60.
    # Can we do better?
    # For matrix multiplication M_{n,m,p}, the rank is denoted R(n,m,p).
    # R(2,3,10) is unknown exactly, but lower bounds exist.
    # A trivial lower bound is max(nm, mp, pn) = max(6, 30, 20) = 30.
    # Another lower bound is np + mp + mn - n - m - p + 2? No, that's for something else.
    # Let's try to find a rank < 60 decomposition.
    
    # Strategy:
    # 1. Initialize U, V, W randomly for a given rank r.
    # 2. Use gradient descent to minimize ||R - T||_F^2.
    # 3. If the loss is close to 0, we have a valid decomposition.
    # 4. Try to round the entries to integers or half-integers if possible.
    
    # However, finding an exact integer/half-integer decomposition is hard.
    # The problem statement says "integer/half-integer entries".
    # This suggests that if a decomposition exists, it might have such entries.
    # Or perhaps the evaluator accepts floating point numbers that are close to such values?
    # "exactly decompose" implies we need to be very precise.
    
    # Let's try a simple gradient descent approach for rank r < 60.
    
    def build_tensor():
        T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0
        return T
    
    T = build_tensor()
    
    def check_validity(U, V, W, tol=1e-6):
        """Check if U, V, W form a valid decomposition."""
        # R = sum_r U[:, r] V[:, r]^T W[:, r]^T? No.
        # R[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        R = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(R, T, atol=tol)
    
    def try_rank(r, num_iters=1000, lr=0.01):
        if time.time() > deadline - 5:
            return None
        
        # Random initialization
        rng = np.random.RandomState(42)
        U = rng.randn(n * m, r)
        V = rng.randn(m * p, r)
        W = rng.randn(p * n, r)
        
        best_loss = float('inf')
        
        for it in range(num_iters):
            if time.time() > deadline - 5:
                break
            
            # Compute gradient
            # Loss L = sum_{a,b,c} (R[a,b,c] - T[a,b,c])^2
            # dL/dU[a,r] = 2 * sum_{b,c} (R[a,b,c] - T[a,b,c]) * V[b,r] * W[c,r]
            # dL/dV[b,r] = 2 * sum_{a,c} (R[a,b,c] - T[a,b,c]) * U[a,r] * W[c,r]
            # dL/dW[c,r] = 2 * sum_{a,b} (R[a,b,c] - T[a,b,c]) * U[a,r] * V[b,r]
            
            R = np.einsum('ar,br,cr->abc', U, V, W)
            diff = R - T
            loss = np.sum(diff ** 2)
            
            if loss < best_loss:
                best_loss = loss
                if loss < 1e-10:
                    # Found a solution
                    return U, V, W
            
            # Compute gradients
            # grad_U[a, r] = 2 * sum_{b,c} diff[a,b,c] * V[b,r] * W[c,r]
            # This can be computed as: 2 * einsum('abc,br,cr->ar', diff, V, W)
            grad_U = 2 * np.einsum('abc,br,cr->ar', diff, V, W)
            grad_V = 2 * np.einsum('abc,ar,cr->br', diff, U, W)
            grad_W = 2 * np.einsum('abc,ar,br->cr', diff, U, V)
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Normalize to prevent divergence?
            # Not strictly necessary if lr is small.
        
        return None
    
    # Try ranks from 30 to 59
    # We will try a few ranks to see if we can find a solution.
    # Given the time constraint, we can't do extensive search.
    # Let's try rank 59, 58, ... down to 50.
    
    for r in range(59, 49, -1):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(r, num_iters=500, lr=0.005)
        if result is not None:
            U, V, W = result
            # Check if we can round to half-integers
            # If the values are close to half-integers, we can snap them.
            # For now, return the floating point result if it's valid.
            # The problem asks for integer/half-integer entries.
            # If we find a floating point solution, we might need to round it.
            # Let's check if rounding works.
            
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            if check_validity(U_rounded, V_rounded, W_rounded, tol=1e-5):
                # Convert to list of lists
                best_result = {
                    "U": U_rounded.tolist(),
                    "V": V_rounded.tolist(),
                    "W": W_rounded.tolist()
                }
                best_rank = r
                break
            else:
                # If rounding doesn't work, maybe return the float version?
                # The prompt says "integer/half-integer entries".
                # If we can't satisfy this, we should stick to the schoolbook.
                pass
    
    return best_result
# EVOLVE-BLOCK-END
