"""Find a rank-minimal exact tensor decomposition for 3x5 * 5x6 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    """
    Attempt to find a lower-rank decomposition using gradient descent on the 
    tensor residual. If no better rank is found, fall back to schoolbook.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # The tensor T has shape (n, m, p) in the standard matrix multiplication sense
    # But the interface defines U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # The tensor equation is: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 if (i,j,k) corresponds to C[i,k] = sum_j A[i,j]B[j,k]
    # Wait, standard matrix mult C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor T_{i,j,k} = 1 if the term A_{i,j}B_{j,k} contributes to C_{i,k}.
    # So T_{i,j,k} = 1 for all i,j,k? No.
    # The tensor is T_{i,j,k} where i is row of A, j is col of A/row of B, k is col of B.
    # The product is C_{i,k}.
    # The decomposition is T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}? 
    # Let's check the indices in the prompt.
    # U shape (n*m, rank) -> index (i, j)
    # V shape (m*p, rank) -> index (j, k)
    # W shape (p*n, rank) -> index (k, i)  <-- Note: "W[k*n+i,r]"
    # The term is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This sums to 1 if the standard multiplication rule holds?
    # Actually, the "tensor" of matrix multiplication is defined such that 
    # T_{i,j,k} = 1. The rank is the minimum number of rank-1 tensors summing to T.
    # A rank-1 tensor is u_{i,j} v_{j,k} w_{k,i}.
    # So we want sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1 for all i,j,k.
    
    # Target tensor T is all ones of shape (n, m, p).
    # Let's verify: if we use schoolbook, U_{i,j,r} is 1 for r=(i,j,k), V_{j,k,r} is 1, W_{k,i,r} is 1.
    # Sum over r: for a fixed (i,j,k), only one r matches all three. So sum is 1.
    # So the target is indeed the all-ones tensor of shape (n, m, p).
    
    target = np.ones((n, m, p))
    
    def residual_norm(U_flat, V_flat, W_flat, rank):
        """Calculate Frobenius norm of (Target - sum_r U_r V_r W_r)."""
        # Reshape
        U = U_flat.reshape(n, m, rank)
        V = V_flat.reshape(m, p, rank)
        W = W_flat.reshape(p, n, rank)
        
        # Compute sum_r U[:,:,r] * V[:,:,r] * W[:,:,r]
        # Using einsum: 'ijr,jkr,kin->ijk'
        prod = np.einsum('ijr,jkr,kin->ijk', U, V, W)
        diff = target - prod
        return np.linalg.norm(diff)
    
    def gradient(U_flat, V_flat, W_flat, rank):
        """Calculate gradients of the squared residual norm."""
        U = U_flat.reshape(n, m, rank)
        V = V_flat.reshape(m, p, rank)
        W = W_flat.reshape(p, n, rank)
        
        prod = np.einsum('ijr,jkr,kin->ijk', U, V, W)
        diff = target - prod
        
        # Gradients:
        # d/dU_{i,j,r} sum_{i',j',k'} (T_{i'j'k'} - P_{i'j'k'})^2
        # = -2 * (T - P)_{i,j,k} * dP_{i,j,k}/dU_{i,j,r}
        # P_{i,j,k} = sum_r' U_{i,j,r'} V_{j,k,r'} W_{k,i,r'}
        # dP_{i,j,k}/dU_{i,j,r} = V_{j,k,r} W_{k,i,r}
        
        # grad_U_{i,j,r} = -2 * sum_k (T_{i,j,k} - P_{i,j,k}) * V_{j,k,r} * W_{k,i,r}
        # Using einsum: 'ijk,jkr,kin->ijr' for the product part
        # Let E = T - P (shape i,j,k)
        # term_VW = einsum('jkr,kin->jkn', V, W) -- wait, indices need to align for contraction with E.
        # We need sum_k E_{i,j,k} * V_{j,k,r} * W_{k,i,r}
        # This is einsum('ijk,jkr,kin->ijr', E, V, W)
        
        grad_U = -2.0 * np.einsum('ijk,jkr,kin->ijr', diff, V, W)
        
        # Similarly for V:
        # grad_V_{j,k,r} = -2 * sum_i E_{i,j,k} * U_{i,j,r} * W_{k,i,r}
        # einsum('ijk,ijr,kin->jkr', E, U, W)
        grad_V = -2.0 * np.einsum('ijk,ijr,kin->jkr', diff, U, W)
        
        # Similarly for W:
        # grad_W_{k,i,r} = -2 * sum_j E_{i,j,k} * U_{i,j,r} * V_{j,k,r}
        # einsum('ijk,ijr,jkr->kin', E, U, V)
        grad_W = -2.0 * np.einsum('ijk,ijr,jkr->kin', diff, U, V)
        
        return grad_U.flatten(), grad_V.flatten(), grad_W.flatten()
    
    best_rank = schoolbook_rank
    best_result = None
    
    # Try ranks from a small number up to schoolbook rank - 1
    # Start with low ranks to see if we can find a solution
    # Note: Random initialization might get stuck in local minima.
    # We will try a few random restarts for each rank.
    
    for rank in range(1, schoolbook_rank):
        if time.time() > deadline - 5.0:
            break
            
        # Try a few random initializations for this rank
        found_solution = False
        for attempt in range(5):
            if time.time() > deadline - 5.0:
                break
                
            U_init = np.random.randn(n * m * rank) * 0.1
            V_init = np.random.randn(m * p * rank) * 0.1
            W_init = np.random.randn(p * n * rank) * 0.1
            
            # Simple Gradient Descent
            lr = 0.01
            for step in range(100):
                if time.time() > deadline - 5.0:
                    break
                    
                norm = residual_norm(U_init, V_init, W_init, rank)
                if norm < 1e-5:
                    # Found a solution!
                    found_solution = True
                    break
                    
                gU, gV, gW = gradient(U_init, V_init, W_init, rank)
                
                U_init -= lr * gU
                V_init -= lr * gV
                W_init -= lr * gW
                
            if found_solution:
                best_rank = rank
                # Convert to list of lists as required
                U_list = U_init.reshape(n*m, rank).tolist()
                V_list = V_init.reshape(m*p, rank).tolist()
                W_list = W_init.reshape(p*n, rank).tolist()
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                break
        
        if found_solution:
            # If we found a solution for this rank, can we go lower?
            # Yes, continue the loop to try lower ranks? 
            # Wait, the loop is increasing rank. 
            # We should check if we can find a solution for *this* rank.
            # If we do, we should try to find an even *lower* rank?
            # The loop goes 1, 2, 3... 
            # If we find a solution at rank 10, we should continue to see if rank 9 works? 
            # No, the loop is `for rank in range(1, schoolbook_rank)`.
            # So it tries 1, then 2, etc.
            # If it finds a solution at rank 10, it has already tried 1-9 and failed.
            # So rank 10 is the minimum found so far.
            # We should break and return this, or continue to see if a *lower* rank was missed?
            # No, if it failed for 1-9, it's unlikely to succeed later unless we change strategy.
            # However, random restarts might have failed for 1-9.
            # Let's just return the first one found in ascending order.
            break
            
    if best_result is not None:
        return best_result
    else:
        # Fall back to schoolbook
        U = [[0] * schoolbook_rank for _ in range(n * m)]
        V = [[0] * schoolbook_rank for _ in range(m * p)]
        W = [[0] * schoolbook_rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

