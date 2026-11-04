import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin (150s limit, 35s buffer)

    # Build target tensor T[a,b,c] where:
    # a = i*m + j (A index)
    # b = j*p + k (B index)  
    # c = k*n + i (W index, trace convention)
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify(U, V, W, tol=1e-9):
        """Verify that U, V, W exactly decompose T."""
        P = np.einsum('ac,bc,cc->abc', U, V, W)
        return np.max(np.abs(P - T)) < tol

    def schoolbook():
        r = n * m * p
        U = np.zeros((n*m, r), dtype=np.float64)
        V = np.zeros((m*p, r), dtype=np.float64)
        W = np.zeros((p*n, r), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W, r

    # Start with schoolbook as guaranteed fallback
    best_U, best_V, best_W, best_rank = schoolbook()
    
    # Try to find lower rank decompositions
    # Strategy: try various ranks from low to high, use randomized init + gradient descent
    
    def try_rank(r, max_restarts=10, steps_per_restart=500, lr=0.1):
        """Try to find a rank-r decomposition using gradient descent."""
        nonlocal best_U, best_V, best_W, best_rank
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                return False
            
            # Random initialization
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            W = np.random.randn(p*n, r) * 0.5
            
            for step in range(steps_per_restart):
                if time.time() > deadline - 5:
                    return False
                
                # Compute current product
                P = np.einsum('ac,bc,cc->abc', U, V, W)
                diff = P - T
                
                # Check if we've converged
                err = np.max(np.abs(diff))
                if err < 1e-9:
                    # Verify and update best
                    if verify(U, V, W):
                        best_U, best_V, best_W, best_rank = U, V, W, r
                        return True
                    break
                
                # Compute gradients
                # dL/dU[a,c] = sum_{b,c'} diff[a,b,c'] * V[b,c] * W[c',c]... 
                # Actually: P[a,b,c] = sum_k U[a,k]*V[b,k]*W[c,k]
                # dP[a,b,c]/dU[a',k] = delta(a,a') * V[b,k] * W[c,k]
                # grad_U[a,k] = sum_{b,c} diff[a,b,c] * V[b,k] * W[c,k]
                
                grad_U = np.einsum('abc,bk,ck->ak', diff, V, W)
                grad_V = np.einsum('abc,ak,ck->bk', diff, U, W)
                grad_W = np.einsum('abc,ak,bk->ck', diff, U, V)
                
                # Gradient descent step with adaptive learning rate
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Adaptive learning rate: reduce if error increases
                if step % 50 == 49:
                    err_now = np.max(np.abs(np.einsum('ac,bc,cc->abc', U, V, W) - T))
                    if err_now > err * 1.1:
                        lr *= 0.5
                    else:
                        lr = min(lr * 1.05, 0.5)
        
        return False

    # Try ranks from 20 up to 44
    # Known theoretical bounds suggest rank could be as low as ~20-30 for this size
    for target_rank in [20, 22, 25, 28, 30, 32, 35, 38, 40, 42, 44]:
        if time.time() > deadline - 10:
            break
        
        if target_rank >= best_rank:
            continue
        
        # Use more restarts for lower ranks (harder to find)
        max_restarts = 5 if target_rank <= 25 else 3
        steps = 300 if target_rank <= 25 else 200
        
        if try_rank(target_rank, max_restarts=max_restarts, steps_per_restart=steps, lr=0.1):
            break  # Found a valid decomposition at this rank
    
    # If we haven't improved, try a few more ranks with different strategies
    if best_rank == 45 and time.time() < deadline - 15:
        # Try rank 30 with more aggressive search
        for target_rank in [28, 30, 33]:
            if time.time() > deadline - 10:
                break
            if try_rank(target_rank, max_restarts=3, steps_per_restart=400, lr=0.15):
                break
    
    # Final verification
    if not verify(best_U, best_V, best_W, tol=1e-8):
        # Fall back to schoolbook if something went wrong
        best_U, best_V, best_W, best_rank = schoolbook()
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
