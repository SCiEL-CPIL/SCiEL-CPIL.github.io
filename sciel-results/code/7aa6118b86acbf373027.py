<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    # Schoolbook baseline (always valid)
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    def check_decomposition(U, V, W, tol=1e-8):
        """Check if U,V,W form a valid decomposition.
        For all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should equal 1.
        """
        # Build the tensor
        # U shape: (n*m, r)
        # V shape: (m*p, r)
        # W shape: (p*n, r)
        r = U.shape[1]
        
        # For each (i,j,k), compute sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # This is equivalent to: for each c, the outer product structure
        # Let's compute it efficiently
        
        # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
        U_reshaped = U.reshape(n, m, r)
        V_reshaped = V.reshape(m, p, r)
        W_reshaped = W.reshape(p, n, r)
        
        # Compute the tensor T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # Use einsum
        # First compute A[i,j,k] = sum_c U[i,j,c] * V[j,k,c] -> shape (n, m, p)
        # Then T[i,j,k] = sum_k' ... wait no
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        
        # Let's compute step by step
        # For fixed i,j: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # = sum_c U[i,j,c] * (V[j,:] * W[:,i])_k  ... no that's not right either
        # Actually for fixed i,j,k: it's a scalar
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        
        # Using einsum:
        # T[i,j,k] = U[i,j,c] * V[j,k,c] * W[k,i,c] summed over c
        # = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        # Check if T is all ones
        return np.allclose(T, np.ones((n, m, p)), atol=tol)
    
    def try_rank_optimization(target_rank, num_trials=10, max_iter=50):
        """Try to find a decomposition of given rank using iterative optimization."""
        if time.time() > deadline:
            return None
        
        n, m, p = DIMENSIONS
        nm = n * m
        mp = m * p
        pn = p * n
        
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            # Random initialization
            U = np.random.randn(nm, target_rank) * 0.5
            V = np.random.randn(mp, target_rank) * 0.5
            W = np.random.randn(pn, target_rank) * 0.5
            
            # Iterative optimization: minimize ||T - 1||^2 where T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
            # Use gradient descent with learning rate scheduling
            
            lr = 0.01
            best_loss = float('inf')
            best_state = (U.copy(), V.copy(), W.copy())
            
            for it in range(max_iter):
                if time.time() > deadline:
                    return None
                
                # Compute current tensor
                U_r = U.reshape(n, m, target_rank)
                V_r = V.reshape(m, p, target_rank)
                W_r = W.reshape(p, n, target_rank)
                
                T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
                loss = np.sum((T - 1.0)**2)
                
                if loss < best_loss:
                    best_loss = loss
                    best_state = (U.copy(), V.copy(), W.copy())
                
                if loss < 1e-10:
                    return (U.copy(), V.copy(), W.copy())
                
                # Compute gradients
                # dL/dU[i,j,c] = 2*(T[i,j,k]-1) * V[j,k,c] * W[k,i,c] summed over k
                # dL/dV[j,k,c] = 2*(T[i,j,k]-1) * U[i,j,c] * W[k,i,c] summed over i
                # dL/dW[k,i,c] = 2*(T[i,j,k]-1) * U[i,j,c] * V[j,k,c] summed over j
                
                diff = 2.0 * (T - 1.0)
                
                # Gradient for U: shape (n, m, target_rank)
                # dU[i,j,c] = sum_k diff[i,j,k] * V[j,k,c] * W[k,i,c]
                grad_U = np.einsum('ijk,jkc,kic->ijc', diff, V_r, W_r)
                grad_V = np.einsum('ijk,ijc,kic->jkc', diff, U_r, W_r)
                grad_W = np.einsum('ijk,ijc,jkc->kic', diff, U_r, V_r)
                
                # Update
                U -= lr * grad_U.reshape(nm, target_rank)
                V -= lr * grad_V.reshape(mp, target_rank)
                W -= lr * grad_W.reshape(pn, target_rank)
                
                # Decay learning rate
                lr *= 0.99
            
            # Check best state
            if check_decomposition(best_state[0], best_state[1], best_state[2]):
                return best_state
        
        return None
    
    # Try various ranks
    for target_rank in [30, 28, 25, 22, 20, 18, 15]:
        if time.time() > deadline:
            break
        
        result = try_rank_optimization(target_rank, num_trials=3, max_iter=100)
        if result is not None:
            U, V, W = result
            if check_decomposition(U, V, W):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break  # Found a valid decomposition, try lower ranks
    
    # Final check
    if not check_decomposition(best_U, best_V, best_W):
        # Fallback to schoolbook
        best_rank = rank_sb
        best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(), 
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit
    
    # Schoolbook baseline (always valid)
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    def check_decomposition(U, V, W, tol=1e-6):
        """Check if U,V,W form a valid decomposition."""
        r = U.shape[1]
        U_reshaped = U.reshape(n, m, r)
        V_reshaped = V.reshape(m, p, r)
        W_reshaped = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        return np.allclose(T, np.ones((n, m, p)), atol=tol)
    
    def compute_loss_and_grads(U, V, W, r):
        """Compute loss and gradients for tensor decomposition."""
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        loss = np.sum((T - 1.0)**2)
        
        diff = 2.0 * (T - 1.0)
        grad_U = np.einsum('ijk,jkc,kic->ijc', diff, V_r, W_r)
        grad_V = np.einsum('ijk,ijc,kic->jkc', diff, U_r, W_r)
        grad_W = np.einsum('ijk,ijc,jkc->kic', diff, U_r, V_r)
        
        return loss, grad_U, grad_V, grad_W
    
    def polish_decomposition(U, V, W, r, max_iter=200, tol=1e-12):
        """Polish a near-solution to high precision using adaptive step size."""
        best_loss, best_U, best_V, best_W = float('inf'), U.copy(), V.copy(), W.copy()
        lr = 0.05
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            loss, gU, gV, gW = compute_loss_and_grads(U, V, W, r)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < tol:
                return U.copy(), V.copy(), W.copy()
            
            # Adaptive learning rate: reduce if loss increases
            U_new = U - lr * gU.reshape(n*m, r)
            V_new = V - lr * gV.reshape(m*p, r)
            W_new = W - lr * gW.reshape(p*n, r)
            
            loss_new, _, _, _ = compute_loss_and_grads(U_new, V_new, W_new, r)
            
            if loss_new < loss:
                U, V, W = U_new, V_new, W_new
                lr = min(lr * 1.05, 0.5)
            else:
                lr *= 0.7
        
        return best_U, best_V, best_W
    
    def try_rank_optimization(target_rank, num_trials=5, max_iter=300):
        """Try to find a decomposition of given rank using iterative optimization."""
        if time.time() > deadline:
            return None
        
        nm = n * m
        mp = m * p
        pn = p * n
        
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            # Random initialization with varied scales
            scale = np.random.uniform(0.3, 1.0)
            U = np.random.randn(nm, target_rank) * scale
            V = np.random.randn(mp, target_rank) * scale
            W = np.random.randn(pn, target_rank) * scale
            
            # Iterative optimization with adaptive learning rate
            lr = 0.03
            best_loss = float('inf')
            best_state = (U.copy(), V.copy(), W.copy())
            stagnant = 0
            
            for it in range(max_iter):
                if time.time() > deadline:
                    return None
                
                loss, gU, gV, gW = compute_loss_and_grads(U, V, W, target_rank)
                
                if loss < best_loss:
                    best_loss = loss
                    best_state = (U.copy(), V.copy(), W.copy())
                    stagnant = 0
                else:
                    stagnant += 1
                
                if loss < 1e-10:
                    # Polish to high precision
                    U_p, V_p, W_p = polish_decomposition(U, V, W, target_rank)
                    return (U_p, V_p, W_p)
                
                # Adaptive step: try the update, reject if loss increases
                U_new = U - lr * gU.reshape(nm, target_rank)
                V_new = V - lr * gV.reshape(mp, target_rank)
                W_new = W - lr * gW.reshape(pn, target_rank)
                
                loss_new, _, _, _ = compute_loss_and_grads(U_new, V_new, W_new, target_rank)
                
                if loss_new < loss:
                    U, V, W = U_new, V_new, W_new
                    lr = min(lr * 1.02, 0.3)
                else:
                    lr *= 0.8
                
                # If stagnant for too long, try a restart with perturbation
                if stagnant > 50:
                    U = best_state[0] + np.random.randn(nm, target_rank) * 0.01
                    V = best_state[1] + np.random.randn(mp, target_rank) * 0.01
                    W = best_state[2] + np.random.randn(pn, target_rank) * 0.01
                    lr = 0.03
                    stagnant = 0
            
            # Final polish on best state
            if best_loss < 0.1:
                U_p, V_p, W_p = polish_decomposition(best_state[0], best_state[1], best_state[2], target_rank)
                if check_decomposition(U_p, V_p, W_p):
                    return (U_p, V_p, W_p)
            elif check_decomposition(best_state[0], best_state[1], best_state[2]):
                return best_state
        
        return None
    
    # Try various ranks with more trials for promising ones
    rank_attempts = [
        (30, 8, 400),
        (28, 6, 350),
        (25, 6, 350),
        (22, 5, 300),
        (20, 4, 250),
        (18, 3, 200),
        (15, 3, 200),
    ]
    
    for target_rank, num_trials, max_iter in rank_attempts:
        if time.time() > deadline:
            break
        
        result = try_rank_optimization(target_rank, num_trials=num_trials, max_iter=max_iter)
        if result is not None:
            U, V, W = result
            if check_decomposition(U, V, W):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                # Continue trying lower ranks
                continue
    
    # Final check
    if not check_decomposition(best_U, best_V, best_W):
        # Fallback to schoolbook
        best_rank = rank_sb
        best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(), 
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
