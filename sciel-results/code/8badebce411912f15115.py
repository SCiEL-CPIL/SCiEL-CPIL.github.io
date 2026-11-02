"""Matrix multiplication tensor decomposition 3x5x7 with rank search via ALS."""
import time
import numpy as np
import random

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Trivial decomposition (always valid)
    trivial_rank = n * m * p  # 105
    U_trivial = np.zeros((n*m, trivial_rank), dtype=float)
    V_trivial = np.zeros((m*p, trivial_rank), dtype=float)
    W_trivial = np.zeros((p*n, trivial_rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i*m+j, col] = 1.0
                V_trivial[j*p+k, col] = 1.0
                W_trivial[k*n+i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_trivial, V_trivial, W_trivial
    best_rank = trivial_rank
    
    def time_left():
        return deadline - time.time()
    
    def verify(U, V, W, tol=1e-6):
        """Check if the decomposition is valid."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for t in range(U.shape[1]):
                        val += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def try_rank_als(target_rank, max_iters=500, seed=42):
        """Try to find a decomposition of given rank using ALS."""
        if time_left() < 5:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Initialize
        U = rng.randn(n*m, target_rank) * 0.1
        V = rng.randn(m*p, target_rank) * 0.1
        W = rng.randn(p*n, target_rank) * 0.1
        
        # Precompute index lists
        indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        
        for iteration in range(max_iters):
            if time_left() < 10:
                break
            
            # Update U
            for t in range(target_rank):
                for (i, j, k) in indices:
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    # Solve for U[a, t]
                    # sum_t' U[a,t'] V[b,t'] W[c,t'] = 1
                    # For fixed t, U[a,t] = (1 - sum_{t' != t} U[a,t'] V[b,t'] W[c,t']) / (V[b,t] W[c,t])
                    # But this is per (i,j,k), and U[a,t] appears in multiple equations
                    # Better: use least squares
                    
            # Simpler approach: alternate solving for each matrix
            # For U: fix V, W. For each row a of U, we have equations:
            # For each (j,k) such that a = i*m+j for some i... 
            # Actually, let's do a different ALS approach
            
            # Build the system differently
            # For each t, U[:,t], V[:,t], W[:,t] are vectors
            # The constraint is: for all (i,j,k): sum_t U[i*m+j,t] V[j*p+k,t] W[k*n+i,t] = 1
            
            # Let's use a gradient-based approach instead
            pass
        
        # Try a different approach: use the structure more directly
        # For rank r, we need r terms
        # Let's try random restarts with a simpler verification
        
        return None
    
    # Try smaller ranks using random search
    # The theoretical lower bound is max(n*m, m*p, p*n) = max(15, 35, 21) = 35
    # But actual minimal rank might be higher
    
    # Try ranks from 35 upward
    for target_rank in range(35, min(105, 60)):
        if time_left() < 15:
            break
        
        # Try a few random initializations
        for attempt in range(5):
            if time_left() < 10:
                break
            
            rng = np.random.RandomState(42 + attempt)
            
            # Initialize with small random values
            U = rng.randn(n*m, target_rank) * 0.5
            V = rng.randn(m*p, target_rank) * 0.5
            W = rng.randn(p*n, target_rank) * 0.5
            
            # Simple gradient descent
            lr = 0.01
            for it in range(100):
                if time_left() < 10:
                    break
                
                # Compute gradients
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for (i, j, k) in indices:
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    # f = sum_t U[a,t] V[b,t] W[c,t]
                    # grad_U[a,t] += V[b,t] W[c,t]
                    # grad_V[b,t] += U[a,t] W[c,t]
                    # grad_W[c,t] += U[a,t] V[b,t]
                    # But f should be 1, so gradient of (f-1)^2 is 2(f-1)*...
                    f = np.sum(U[a, :] * V[b, :] * W[c, :])
                    diff = f - 1.0
                    if abs(diff) > 1e-8:
                        for t in range(target_rank):
                            grad_U[a, t] += diff * V[b, t] * W[c, t]
                            grad_V[b, t] += diff * U[a, t] * W[c, t]
                            grad_W[c, t] += diff * U[a, t] * V[b, t]
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
            
            # Check if we converged
            if verify(U, V, W, tol=1e-4):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                break
        
        if best_rank < trivial_rank:
            break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
