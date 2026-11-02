"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j'*p+k, k'*n+i'] = 1 if i==i', j==j', k==k' else 0
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Schoolbook decomposition: rank = n*m*p
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    best_rank = rank_sb
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    def verify(U, V, W):
        """Check if U, V, W exactly decompose T."""
        r = U.shape[1]
        T_computed = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_computed, T, atol=1e-8)
    
    def try_reduce_rank(U, V, W, target_rank, max_iter=200):
        """Try to reduce rank by removing one term and solving for the rest."""
        current_rank = U.shape[1]
        if current_rank <= target_rank:
            return None
        
        # Try removing each term and see if we can solve for the remaining
        for remove_idx in range(current_rank):
            if time.time() > deadline - 5:
                break
            
            # Remove the term
            mask = [i for i in range(current_rank) if i != remove_idx]
            U_rem = U[:, mask]
            V_rem = V[:, mask]
            W_rem = W[:, mask]
            
            # Now we have target_rank terms. We need to adjust them so that
            # sum_s U_rem[:,s] * V_rem[:,s] * W_rem[:,s] = T
            # This is still nonlinear, but we can try iterative refinement.
            
            # Use alternating minimization: fix two, solve for the third.
            U_new = U_rem.copy()
            V_new = V_rem.copy()
            W_new = W_rem.copy()
            
            converged = False
            for iteration in range(max_iter):
                if time.time() > deadline - 3:
                    break
                
                # Fix V and W, solve for U: T = U * (V_r * W_r)
                # For each column s: U[:,s] = T[:, :, s] / (V[:,s] * W[:,s])... no, this is trilinear.
                
                # Alternative: use gradient descent on the residual
                T_computed = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
                residual = T - T_computed
                
                if np.allclose(residual, 0, atol=1e-8):
                    converged = True
                    break
                
                # Compute gradients
                # dT/dU[a,r] = V[b,r] * W[c,r] for the (a,b,c) component
                # We want to minimize ||T - T_computed||^2
                # Gradient wrt U: sum_{b,c} residual[a,b,c] * V[b,r] * W[c,r]
                
                grad_U = np.einsum('abc,br,cr->ar', residual, V_new, W_new)
                grad_V = np.einsum('abc,ar,cr->br', residual, U_new, W_new)
                grad_W = np.einsum('abc,ar,br->cr', residual, U_new, V_new)
                
                # Simple gradient step
                lr = 0.01
                U_new -= lr * grad_U
                V_new -= lr * grad_V
                W_new -= lr * grad_W
                
                # Check convergence
                T_computed_check = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
                if np.allclose(T_computed_check, T, atol=1e-8):
                    converged = True
                    break
            
            if converged:
                # Snap to nearest half-integers
                U_snap = np.round(U_new * 2) / 2
                V_snap = np.round(V_new * 2) / 2
                W_snap = np.round(W_new * 2) / 2
                
                if verify(U_snap, V_snap, W_snap):
                    return (U_snap, V_snap, W_snap)
                
                # Try without snapping
                if verify(U_new, V_new, W_new):
                    return (U_new, V_new, W_new)
        
        return None
    
    # Try to reduce rank from 50 down
    for target_rank in range(rank_sb - 1, 45, -1):
        if time.time() > deadline - 15:
            break
        
        result = try_reduce_rank(best_U, best_V, best_W, target_rank)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = target_rank
    
    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
