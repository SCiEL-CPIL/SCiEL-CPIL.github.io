"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _residual(U, V, W, n, m, p):
    """Compute the residual tensor: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - delta(i,j,k in range)"""
    # Build the target tensor (identity-like for matmul)
    # The matmul tensor: T[i,j,k] = 1 if this is a valid (i,j,k) triple for (A*B)[i,k]
    # Actually, the tensor for multiplying (n x m) by (m x p) to get (n x p):
    # T[i,k,j] = 1 for all i in [0,n), k in [0,p), j in [0,m)
    # Wait, let me think about this more carefully.
    # Matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T is a 3-tensor with indices (i, j, k) where:
    # T[i,j,k] = 1 if j is a valid middle index (always for our dimensions)
    # Actually the tensor decomposition represents: C[i,k] = sum_c U[i,c]*V[k,c]... 
    # Let me re-read the contract.
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # W[k*n+i, r] follows the trace convention.
    # The decomposition: for each c, U[:,c] * V[:,c] * W[:,c] should sum to the matmul tensor.
    # The matmul tensor T has entries T[i,j,k] = 1 for all valid i,j,k (it's all ones in the relevant subspace)
    # Actually no. The tensor for matrix multiplication is:
    # T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # So T is a 3-tensor where T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # And the decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Build target: all ones of shape (n, m, p)
    target = np.ones((n, m, p))
    
    # Compute reconstructed tensor
    recon = np.zeros((n, m, p))
    Ua = np.array(U)
    Va = np.array(V)
    Wa = np.array(W)
    rank = Ua.shape[1]
    for c in range(rank):
        # U[:,c] is shape (n*m,), reshape to (n,m)
        U_c = Ua[:, c].reshape(n, m)
        V_c = Va[:, c].reshape(m, p)
        W_c = Wa[:, c].reshape(p, n)
        # For each (i,j,k): U_c[i,j] * V_c[j,k] * W_c[k,i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    recon[i, j, k] += U_c[i, j] * V_c[j, k] * W_c[k, i]
    
    diff = recon - target
    return np.linalg.norm(diff)

def _residual_fast(U, V, W, n, m, p):
    """Faster residual computation using vectorized operations."""
    Ua = np.array(U)
    Va = np.array(V)
    Wa = np.array(W)
    rank = Ua.shape[1]
    
    # For each component c, we add U_c[i,j] * V_c[j,k] * W_c[k,i] to recon[i,j,k]
    # We can do this more efficiently
    recon = np.zeros((n, m, p))
    for c in range(rank):
        U_c = Ua[:, c].reshape(n, m)
        V_c = Va[:, c].reshape(m, p)
        W_c = Wa[:, c].reshape(p, n)
        # recon[i,j,k] += U_c[i,j] * V_c[j,k] * W_c[k,i]
        # For fixed c: this is an outer-product-like operation
        # Let's compute it as: for each j, U_c[:,j] (n,) * V_c[j,:] (p,) * W_c[:,j]... no
        # W_c[k,i] is indexed by (k,i), so W_c is (p,n)
        # recon[i,j,k] = U_c[i,j] * V_c[j,k] * W_c[k,i]
        # For fixed j: sum over i,k of U_c[i,j]*V_c[j,k]*W_c[k,i]
        # Actually let's just do it with broadcasting
        # U_c[i,j] * V_c[j,k] * W_c[k,i]
        # Shape: (n,m) * (m,p) * (p,n) -> need to align indices
        # U_c[i,j] -> (n, m, 1)
        # V_c[j,k] -> (1, m, p)  
        # W_c[k,i] -> (n, 1, p)
        recon += U_c[:, :, np.newaxis] * V_c[np.newaxis, :, :] * W_c.T[np.newaxis, :, :]
    
    diff = recon - np.ones((n, m, p))
    return np.linalg.norm(diff)

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = n * m * p
    best_residual = 0.0
    
    # Try to find lower rank decompositions
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(9, 15, 15) = 15
    # Known results suggest rank might be around 15-20 for 3x3x5
    
    np.random.seed(42)
    
    # Try various ranks
    for target_rank in range(max(n*m, m*p, p*n), best_rank):
        if time.time() > deadline - 30:
            break
        
        # Multiple restarts for this rank
        max_restarts = 5
        for restart in range(max_restarts):
            if time.time() > deadline - 30:
                break
            
            # Random initialization
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Iterative refinement: minimize ||sum_c U_c ⊙ V_c ⊙ W_c - T||^2
            # Use alternating minimization or gradient descent
            lr = 0.1
            for iteration in range(200):
                if time.time() > deadline - 30:
                    break
                
                # Compute residual
                Ua = U
                Va = V
                Wa = W
                recon = np.zeros((n, m, p))
                for c in range(target_rank):
                    U_c = Ua[:, c].reshape(n, m)
                    V_c = Va[:, c].reshape(m, p)
                    W_c = Wa[:, c].reshape(p, n)
                    recon += U_c[:, :, np.newaxis] * V_c[np.newaxis, :, :] * W_c.T[np.newaxis, :, :]
                
                target = np.ones((n, m, p))
                diff = recon - target
                residual = np.linalg.norm(diff)
                
                if residual < 1e-10:
                    # Found exact solution
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    # Convert to integer/half-integer if close
                    best = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                    best_residual = residual
                    break
                
                # Update U: for each c, U_c should be updated to reduce residual
                # Gradient of ||diff||^2 w.r.t. U_c[i,j] = 2 * diff[i,j] * V_c[j,:] * W_c.T[i,:]... 
                # Actually: d/dU_c[i,j] sum_{i',j',k'} diff[i',j',k']^2
                # = 2 * sum_{i',j',k'} diff[i',j',k'] * d(recon[i',j',k'])/dU_c[i,j]
                # d(recon[i',j',k'])/dU_c[i,j] = delta(i',i)*delta(j',j)*V_c[j,k]*W_c[k,i]
                # = V_c[j,:] * W_c[:,i] (summed over k)
                # So gradient w.r.t. U_c[i,j] = 2 * sum_k diff[i,j,k] * V_c[j,k] * W_c[k,i]
                
                # Compute gradients and update
                # For U
                U_grad = np.zeros_like(U)
                V_grad = np.zeros_like(V)
                W_grad = np.zeros_like(W)
                
                for c in range(target_rank):
                    U_c = Ua[:, c].reshape(n, m)
                    V_c = Va[:, c].reshape(m, p)
                    W_c = Wa[:, c].reshape(p, n)
                    
                    # U_grad[i,j,c] = 2 * sum_k diff[i,j,k] * V_c[j,k] * W_c[k,i]
                    # diff[i,j,k] * V_c[j,k] * W_c[k,i] -> sum over k
                    # = (diff[:,:,None] * V_c[None,:,:] * W_c.T[None,:,:])... let me think
                    # For fixed i,j: sum_k diff[i,j,k] * V_c[j,k] * W_c[k,i]
                    # = diff[i,j,:] dot (V_c[j,:] * W_c[:,i])
                    # V_c[j,:] is shape (p,), W_c[:,i] is shape (p,)
                    # So: sum_k diff[i,j,k] * V_c[j,k] * W_c[k,i]
                    
                    U_grad_c = np.zeros((n, m))
                    for i in range(n):
                        for j in range(m):
                            U_grad_c[i, j] = 2.0 * np.sum(diff[i, j, :] * V_c[j, :] * W_c[:, i])
                    U_grad[:, c] = U_grad_c.flatten()
                    
                    # V_grad[j,k,c] = 2 * sum_i diff[i,j,k] * U_c[i,j] * W_c[k,i]
                    V_grad_c = np.zeros((m, p))
                    for j in range(m):
                        for k in range(p):
                            V_grad_c[j, k] = 2.0 * np.sum(diff[:, j, k] * U_c[:, j] * W_c[k, :])
                    V_grad[:, c] = V_grad_c.flatten()
                    
                    # W_grad[k,i,c] = 2 * sum_j diff[i,j,k] * U_c[i,j] * V_c[j,k]
                    W_grad_c = np.zeros((p, n))
                    for k in range(p):
                        for i in range(n):
                            W_grad_c[k, i] = 2.0 * np.sum(diff[i, :, k] * U_c[i, :] * V_c[:, k])
                    W_grad[:, c] = W_grad_c.flatten()
                
                # Update with learning rate
                U = U - lr * U_grad
                V = V - lr * V_grad
                W = W - lr * W_grad
                
                # Decay learning rate
                lr = max(0.01, lr * 0.99)
            
            # Check if we found a good solution
            Ua = U
            Va = V
            Wa = W
            recon = np.zeros((n, m, p))
            for c in range(target_rank):
                U_c = Ua[:, c].reshape(n, m)
                V_c = Va[:, c].reshape(m, p)
                W_c = Wa[:, c].reshape(p, n)
                recon += U_c[:, :, np.newaxis] * V_c[np.newaxis, :, :] * W_c.T[np.newaxis, :, :]
            residual = np.linalg.norm(recon - np.ones((n, m, p)))
            
            if residual < 1e-8 and target_rank < best_rank:
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = target_rank
                best_residual = residual
                break
            elif residual < 1e-4 and target_rank < best_rank:
                # Promising, do more iterations
                for extra_iter in range(500):
                    if time.time() > deadline - 30:
                        break
                    recon = np.zeros((n, m, p))
                    for c in range(target_rank):
                        U_c = Ua[:, c].reshape(n, m)
                        V_c = Va[:, c].reshape(m, p)
                        W_c = Wa[:, c].reshape(p, n)
                        recon += U_c[:, :, np.newaxis] * V_c[np.newaxis, :, :] * W_c.T[np.newaxis, :, :]
                    diff = recon - np.ones((n, m, p))
                    residual = np.linalg.norm(diff)
                    
                    if residual < 1e-10:
                        break
                    
                    U_grad = np.zeros_like(U)
                    V_grad = np.zeros_like(V)
                    W_grad = np.zeros_like(W)
                    
                    for c in range(target_rank):
                        U_c = Ua[:, c].reshape(n, m)
                        V_c = Va[:, c].reshape(m, p)
                        W_c = Wa[:, c].reshape(p, n)
                        
                        U_grad_c = np.zeros((n, m))
                        for i in range(n):
                            for j in range(m):
                                U_grad_c[i, j] = 2.0 * np.sum(diff[i, j, :] * V_c[j, :] * W_c[:, i])
                        U_grad[:, c] = U_grad_c.flatten()
                        
                        V_grad_c = np.zeros((m, p))
                        for j in range(m):
                            for k in range(p):
                                V_grad_c[j, k] = 2.0 * np.sum(diff[:, j, k] * U_c[:, j] * W_c[k, :])
                        V_grad[:, c] = V_grad_c.flatten()
                        
                        W_grad_c = np.zeros((p, n))
                        for k in range(p):
                            for i in range(n):
                                W_grad_c[k, i] = 2.0 * np.sum(diff[i, :, k] * U_c[i, :] * V_c[:, k])
                        W_grad[:, c] = W_grad_c.flatten()
                    
                    U = U - lr * U_grad
                    V = V - lr * V_grad
                    W = W - lr * W_grad
                    lr = max(0.001, lr * 0.995)
                
                recon = np.zeros((n, m, p))
                for c in range(target_rank):
                    U_c = Ua[:, c].reshape(n, m)
                    V_c = Va[:, c].reshape(m, p)
                    W_c = Wa[:, c].reshape(p, n)
                    recon += U_c[:, :, np.newaxis] * V_c[np.newaxis, :, :] * W_c.T[np.newaxis, :, :]
                residual = np.linalg.norm(recon - np.ones((n, m, p)))
                
                if residual < 1e-8 and target_rank < best_rank:
                    best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = target_rank
                    best_residual = residual
                    break
    
    return best
# EVOLVE-BLOCK-END
