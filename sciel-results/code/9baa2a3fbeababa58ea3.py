import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 115  # 115s out of 120s
    
    # Construct the target tensor T
    # T has shape (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Fallback to schoolbook
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, column] = 1
                V_school[j * p + k, column] = 1
                W_school[k * n + i, column] = 1
                column += 1
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = rank_schoolbook
    
    def verify_decomposition(U, V, W, tol=1e-10):
        """Verify that U, V, W exactly decompose T."""
        S = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(S, T, atol=tol)
    
    def round_to_half(U, V, W):
        """Round entries to nearest half-integer and verify."""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if verify_decomposition(U_r, V_r, W_r, tol=1e-10):
            return U_r, V_r, W_r
        return None
    
    def try_als(rank, U_init, V_init, W_init, max_iters=100, tol=1e-10):
        """Alternating Least Squares for tensor decomposition."""
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U: fix V, W
            # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
            # For fixed i: T[i,:,:] = U[i,:] @ (V*W^T)
            # M_U[r, (j,k)] = V[j,r] * W[k,r]
            M_U = np.einsum('jr,kr->rjk', V, W).reshape(rank, m * p * p * n)
            T_flat_U = T.reshape(n * m, m * p * p * n)
            try:
                U_new, _, _, _ = np.linalg.lstsq(M_U.T, T_flat_U.T, rcond=None)
                U = U_new.T
            except:
                return None
            
            # Update V: fix U, W
            # For fixed j: T[:,j,:] = V[j,:] @ (U*W^T)
            # M_V[r, (i,k)] = U[i,r] * W[k,r]
            M_V = np.einsum('ir,kr->rik', U, W).reshape(rank, n * m * p * n)
            # Need to permute T so that index j is first
            T_perm = np.transpose(T, (1, 0, 2))  # (m*p, n*m, p*n)
            T_flat_V = T_perm.reshape(m * p, n * m * p * n)
            try:
                V_new, _, _, _ = np.linalg.lstsq(M_V.T, T_flat_V.T, rcond=None)
                V = V_new.T
            except:
                return None
            
            # Update W: fix U, V
            # For fixed k: T[:,:,k] = W[k,:] @ (U*V^T)
            # M_W[r, (i,j)] = U[i,r] * V[j,r]
            M_W = np.einsum('ir,jr->rij', U, V).reshape(rank, n * m * m * p)
            T_perm_w = np.transpose(T, (2, 0, 1))  # (p*n, n*m, m*p)
            T_flat_W = T_perm_w.reshape(p * n, n * m * m * p)
            try:
                W_new, _, _, _ = np.linalg.lstsq(M_W.T, T_flat_W.T, rcond=None)
                W = W_new.T
            except:
                return None
            
            # Check convergence
            if verify_decomposition(U, V, W, tol=1e-12):
                return (U, V, W)
        
        if verify_decomposition(U, V, W, tol=1e-10):
            return (U, V, W)
        return None
    
    def try_gradient_descent(rank, U_init, V_init, W_init, max_iters=500, lr=0.01):
        """Simple gradient descent with momentum for tensor decomposition."""
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        U_mom = np.zeros_like(U)
        V_mom = np.zeros_like(V)
        W_mom = np.zeros_like(W)
        beta = 0.9
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Compute residual
            S = np.einsum('ir,jr,kr->ijk', U, V, W)
            R = T - S
            
            # Compute gradients
            # dL/dU[i,r] = sum_{j,k} R[i,j,k] * V[j,r] * W[k,r]
            grad_U = np.einsum('ijk,jr,kr->ir', R, V, W)
            grad_V = np.einsum('ijk,ir,kr->jr', R, U, W)
            grad_W = np.einsum('ijk,ir,jr->kr', R, U, V)
            
            # Momentum update
            U_mom = beta * U_mom - lr * grad_U
            V_mom = beta * V_mom - lr * grad_V
            W_mom = beta * W_mom - lr * grad_W
            
            U += U_mom
            V += V_mom
            W += W_mom
            
            # Check convergence
            if verify_decomposition(U, V, W, tol=1e-12):
                return (U, V, W)
            
            # Adaptive learning rate
            if it % 50 == 0 and it > 0:
                err = np.linalg.norm(R)
                if err < 1e-4:
                    lr *= 0.5
                elif err > 1.0:
                    lr *= 1.5
        
        if verify_decomposition(U, V, W, tol=1e-10):
            return (U, V, W)
        return None
    
    # Strategy: Try various ranks and methods
    # Based on the structure, try ranks that are divisors or near-divisors of 96
    target_ranks = [64, 56, 48, 40, 32, 24]
    
    for rank in target_ranks:
        if time.time() > deadline:
            break
        
        # Try ALS with various initializations
        for init_seed in range(20):
            if time.time() > deadline:
                break
            
            np.random.seed(init_seed)
            
            # Random initialization
            U_init = np.random.randn(n * m, rank) * 0.3
            V_init = np.random.randn(m * p, rank) * 0.3
            W_init = np.random.randn(p * n, rank) * 0.3
            
            result = try_als(rank, U_init, V_init, W_init, max_iters=50)
            if result is not None:
                U, V, W = result
                rounded = round_to_half(U, V, W)
                if rounded is not None:
                    return {
                        "U": rounded[0].tolist(),
                        "V": rounded[1].tolist(),
                        "W": rounded[2].tolist()
                    }
                # Return unrounded if it works
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
        
        # Try gradient descent with various initializations
        for init_seed in range(10):
            if time.time() > deadline:
                break
            
            np.random.seed(100 + init_seed)
            
            U_init = np.random.randn(n * m, rank) * 0.3
            V_init = np.random.randn(m * p, rank) * 0.3
            W_init = np.random.randn(p * n, rank) * 0.3
            
            result = try_gradient_descent(rank, U_init, V_init, W_init, max_iters=300, lr=0.005)
            if result is not None:
                U, V, W = result
                rounded = round_to_half(U, V, W)
                if rounded is not None:
                    return {
                        "U": rounded[0].tolist(),
                        "V": rounded[1].tolist(),
                        "W": rounded[2].tolist()
                    }
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
        
        # Try starting from schoolbook and projecting
        if rank < rank_schoolbook:
            for trial in range(5):
                if time.time() > deadline:
                    break
                
                np.random.seed(200 + trial)
                # Select random columns from schoolbook
                cols = np.random.choice(rank_schoolbook, rank, replace=False)
                U_init = U_school[:, cols]
                V_init = V_school[:, cols]
                W_init = W_school[:, cols]
                
                result = try_als(rank, U_init, V_init, W_init, max_iters=30)
                if result is not None:
                    U, V, W = result
                    rounded = round_to_half(U, V, W)
                    if rounded is not None:
                        return {
                            "U": rounded[0].tolist(),
                            "V": rounded[1].tolist(),
                            "W": rounded[2].tolist()
                        }
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
    
    return best_result
# EVOLVE-BLOCK-END
