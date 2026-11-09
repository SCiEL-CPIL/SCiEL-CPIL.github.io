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
        r = U.shape[1]
        
        U_reshaped = U.reshape(n, m, r)
        V_reshaped = V.reshape(m, p, r)
        W_reshaped = W.reshape(p, n, r)
        
        T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        
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
            
            # Structured initialization exploiting symmetry between the two dimension-5 indices
            # Initialize with Kronecker-like structure to respect the trace convention
            if trial % 2 == 0:
                # Symmetric initialization: use identity blocks for the 5x5 structure
                U = np.zeros((nm, target_rank))
                V = np.zeros((mp, target_rank))
                W = np.zeros((pn, target_rank))
                
                # For each of the 3 rows, initialize with identity-like patterns
                for i in range(n):
                    for j in range(m):
                        # Assign a subset of rank dimensions to each (i,j) pair
                        rank_per_ij = max(1, target_rank // (n*m))
                        start_c = (i*m + j) * rank_per_ij
                        end_c = min(start_c + rank_per_ij, target_rank)
                        for c in range(start_c, end_c):
                            U[i*m+j, c] = 1.0
                            V[j*p+k if False else j*p + (c % p), c] = 1.0  # Will be overwritten
                            W[(c % p)*n + i, c] = 1.0
                
                # Reset V properly
                V = np.zeros((mp, target_rank))
                for j in range(m):
                    for k in range(p):
                        rank_per_jk = max(1, target_rank // (m*p))
                        start_c = (j*p + k) * rank_per_jk
                        end_c = min(start_c + rank_per_jk, target_rank)
                        for c in range(start_c, end_c):
                            V[j*p+k, c] = 1.0
                
                # Add small random perturbation to break symmetry
                U += np.random.randn(nm, target_rank) * 0.1
                V += np.random.randn(mp, target_rank) * 0.1
                W += np.random.randn(pn, target_rank) * 0.1
            else:
                # Random initialization
                U = np.random.randn(nm, target_rank) * 0.5
                V = np.random.randn(mp, target_rank) * 0.5
                W = np.random.randn(pn, target_rank) * 0.5
            
            # Iterative optimization: minimize ||T - 1||^2 where T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
            # Use gradient descent with adaptive learning rate and momentum
            
            lr = 0.01
            momentum = 0.9
            vel_U = np.zeros_like(U)
            vel_V = np.zeros_like(V)
            vel_W = np.zeros_like(W)
            
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
                diff = 2.0 * (T - 1.0)
                
                grad_U = np.einsum('ijk,jkc,kic->ijc', diff, V_r, W_r)
                grad_V = np.einsum('ijk,ijc,kic->jkc', diff, U_r, W_r)
                grad_W = np.einsum('ijk,ijc,jkc->kic', diff, U_r, V_r)
                
                # Momentum update
                vel_U = momentum * vel_U - lr * grad_U.reshape(nm, target_rank)
                vel_V = momentum * vel_V - lr * grad_V.reshape(mp, target_rank)
                vel_W = momentum * vel_W - lr * grad_W.reshape(pn, target_rank)
                
                U += vel_U
                V += vel_V
                W += vel_W
                
                # Decay learning rate
                lr *= 0.995
            
            # Final high-precision polish pass if close
            if best_loss < 1e-4:
                U, V, W = best_state
                U_r = U.reshape(n, m, target_rank)
                V_r = V.reshape(m, p, target_rank)
                W_r = W.reshape(p, n, target_rank)
                
                for polish_iter in range(50):
                    if time.time() > deadline:
                        break
                    
                    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
                    loss = np.sum((T - 1.0)**2)
                    
                    if loss < 1e-12:
                        break
                    
                    diff = 2.0 * (T - 1.0)
                    grad_U = np.einsum('ijk,jkc,kic->ijc', diff, V_r, W_r)
                    grad_V = np.einsum('ijk,ijc,kic->jkc', diff, U_r, W_r)
                    grad_W = np.einsum('ijk,ijc,jkc->kic', diff, U_r, V_r)
                    
                    U -= 0.001 * grad_U.reshape(nm, target_rank)
                    V -= 0.001 * grad_V.reshape(mp, target_rank)
                    W -= 0.001 * grad_W.reshape(pn, target_rank)
                    
                    U_r = U.reshape(n, m, target_rank)
                    V_r = V.reshape(m, p, target_rank)
                    W_r = W.reshape(p, n, target_rank)
                
                if check_decomposition(U, V, W):
                    return (U.copy(), V.copy(), W.copy())
            
            # Check best state
            if check_decomposition(best_state[0], best_state[1], best_state[2]):
                return best_state
        
        return None
    
    # Try various ranks - start higher and work down, with more trials for promising ranks
    for target_rank in [35, 32, 30, 28, 25, 22, 20, 18, 15]:
        if time.time() > deadline:
            break
        
        # Allocate more trials for lower ranks (harder to find)
        num_trials = 5 if target_rank <= 25 else 3
        max_iter = 150 if target_rank <= 25 else 100
        
        result = try_rank_optimization(target_rank, num_trials=num_trials, max_iter=max_iter)
        if result is not None:
            U, V, W = result
            if check_decomposition(U, V, W):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                # Don't break immediately - try one more lower rank if time permits
                if time.time() < deadline - 30:
                    continue
                else:
                    break
    
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
