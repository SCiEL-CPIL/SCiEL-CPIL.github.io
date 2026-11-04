import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s out of 150s
    
    n, m, p = 4, 4, 5
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Constraint: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # The tensor T has T[i,j,k] = 1 for all i in [0,4), j in [0,4), k in [0,5)
    
    # Let's try to find a decomposition of decreasing rank
    # Start with schoolbook rank 80 and try to reduce
    
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p  # 80
    
    # First, let's get the schoolbook solution
    rank0 = n * m * p
    U0 = np.zeros((n*m, rank0))
    V0 = np.zeros((m*p, rank0))
    W0 = np.zeros((p*n, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U0, V0, W0
    best_rank = rank0
    
    # Now try to find lower rank decompositions
    # Use a random search / optimization approach
    
    def check_decomposition(U, V, W):
        """Check if U, V, W form a valid decomposition."""
        r = U.shape[1]
        if V.shape[1] != r or W.shape[1] != r:
            return False
        # For each (i,j,k), check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True
    
    def check_decomposition_fast(U, V, W):
        """Faster check using numpy."""
        r = U.shape[1]
        if V.shape[1] != r or W.shape[1] != r:
            return False
        # Create the tensor and check
        # T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # Reshape and use einsum
        # U is (n*m, r), V is (m*p, r), W is (p*n, r)
        # We need: for each c, the contribution is u_c ⊗ v_c ⊗ w_c
        # where u_c[i*m+j] = U[i*m+j,c], v_c[j*p+k] = V[j*p+k,c], w_c[k*n+i] = W[k*n+i,c]
        
        # Let's compute the full tensor
        # T = sum_c np.outer(np.outer(u_c, v_c), w_c) reshaped appropriately
        # But this is expensive. Let's use einsum.
        
        # Actually, let's just do it directly but vectorized
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    errors.append(s - 1.0)
        return max(abs(e) for e in errors) < 1e-6
    
    # Try random searches for lower rank
    # This is a hard problem, so let's try a few strategies
    
    # Strategy 1: Try to use the 4x4x4 decomposition extended
    # Known: 4x4x4 has rank 49 (Bini et al.)
    # But I don't have the explicit decomposition memorized.
    
    # Let me try a numerical optimization approach
    # Minimize the "tensor rank" by finding low-rank approximations
    
    # Actually, let me try a different approach: 
    # Use the fact that we can express the problem as a bilinear map
    # and use known results.
    
    # For now, let's try to find any decomposition with rank < 80
    # using random initialization and local optimization
    
    np.random.seed(42)
    
    for target_rank in [79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64]:
        if time.time() > deadline - 30:
            break
        
        found = False
        for attempt in range(5):
            if time.time() > deadline - 30:
                break
            
            r = target_rank
            U = np.random.randn(n*m, r) * 0.1
            V = np.random.randn(m*p, r) * 0.1
            W = np.random.randn(p*n, r) * 0.1
            
            # Simple gradient descent on the constraint violation
            # Loss = sum over (i,j,k) of (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
            
            lr = 0.01
            for step in range(200):
                if time.time() > deadline - 30:
                    break
                
                # Compute gradients
                loss = 0.0
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            err = s - 1.0
                            loss += err * err
                            # d/dU[i*m+j,c] = 2*err * V[j*p+k,c]*W[k*n+i,c]
                            grad_U[i*m+j, :] += 2 * err * V[j*p+k, :] * W[k*n+i, :]
                            grad_V[j*p+k, :] += 2 * err * U[i*m+j, :] * W[k*n+i, :]
                            grad_W[k*n+i, :] += 2 * err * U[i*m+j, :] * V[j*p+k, :]
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                if loss < 1e-10:
                    break
            
            if loss < 1e-8:
                # Check if entries are close to integer/half-integer
                # For now, accept if it's a valid decomposition
                if check_decomposition_fast(U, V, W):
                    best_U, best_V, best_W = U, V, W
                    best_rank = r
                    found = True
                    break
        
        if found:
            break
    
    # Convert to lists with rounding to nearest half-integer if close
    def round_half(x):
        return np.round(x * 2) / 2
    
    if best_U is not None:
        U_out = round_half(best_U)
        V_out = round_half(best_V)
        W_out = round_half(best_W)
        
        # Verify the rounded version
        if check_decomposition_fast(U_out, V_out, W_out):
            best_U, best_V, best_W = U_out, V_out, W_out
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
