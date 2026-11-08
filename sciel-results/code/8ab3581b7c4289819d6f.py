"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Start with the known rank-42 block-Strassen decomposition
    # as a baseline, then try to find lower rank via optimization.
    
    # Build the target tensor T of shape (n*m, m*p, p*n) = (8, 24, 12)
    # T[a, b, c] = 1 if a=i*m+j, b=j'*p+k, c=k'*n+i' with i=i', j=j', k=k'
    # i.e., T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Flatten T to 1D for objective computation
    T_flat = T.flatten()
    
    def tensor_product(U, V, W, R):
        """Compute sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] as flat vector."""
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Result: (n*m, m*p, p*n)
        result = np.zeros((n * m, m * p, p * n), dtype=np.float64)
        for r in range(R):
            result += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        return result
    
    def objective(U, V, W, R):
        """Compute ||sum_r U_r ⊗ V_r ⊗ W_r - T||^2."""
        prod = tensor_product(U, V, W, R)
        diff = prod.flatten() - T_flat
        return np.dot(diff, diff)
    
    def objective_and_grads(U, V, W, R):
        """Compute objective and gradients w.r.t. U, V, W."""
        prod = tensor_product(U, V, W, R)
        diff = prod - T  # (n*m, m*p, p*n)
        
        # Gradient w.r.t. U[:,r]: for each r, grad_U[:,r] = einsum('ijk,jk->i', diff, V[:,r], W[:,r])
        # More precisely: d/dr sum_s U_s ⊗ V_s ⊗ W_s - T, then project
        # grad_U[:,r] = einsum('ijk,jk->i', diff, V[:,r], W[:,r])
        # But that's not right. Let me think again.
        
        # F(U,V,W) = sum_r U_r ⊗ V_r ⊗ W_r
        # dF/du_i (for U[i,r]) = V_r[k] * W_r[l] summed over k,l... no.
        
        # F[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # dF[i,j,k]/dU[i',r'] = delta_{i,i'} * V[j,r'] * W[k,r']
        # grad_U[i',r'] = sum_{j,k} diff[i',j,k] * V[j,r'] * W[k,r']
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for r in range(R):
            # grad_U[:,r] = einsum('j,k->', ...) no
            # grad_U[i,r] = sum_{j,k} diff[i,j,k] * V[j,r] * W[k,r]
            # = (diff * (V[:,r] * W[:,r])) summed over j,k for each i
            # Using einsum: grad_U[:,r] = einsum('ijk,j,k->i', diff, V[:,r], W[:,r])
            grad_U[:, r] = np.einsum('ijk,j,k->i', diff, V[:, r], W[:, r])
            grad_V[:, r] = np.einsum('ijk,i,k->j', diff, U[:, r], W[:, r])
            grad_W[:, r] = np.einsum('ijk,i,j->k', diff, U[:, r], V[:, r])
        
        obj = np.dot(diff.flatten(), diff.flatten())
        return obj, grad_U, grad_V, grad_W
    
    def verify_decomp(U, V, W):
        """Verify that the decomposition exactly reconstructs T."""
        prod = tensor_product(U, V, W, U.shape[1])
        return np.max(np.abs(prod - T))
    
    # Build the rank-42 baseline
    rank_42 = 42
    U_base = np.zeros((n * m, rank_42), dtype=np.float64)
    V_base = np.zeros((m * p, rank_42), dtype=np.float64)
    W_base = np.zeros((p * n, rank_42), dtype=np.float64)
    
    U_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (0, 0): -1},
        {(0, 1): 1, (1, 1): -1}
    ]
    
    V_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (0, 0): -1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (1, 1): 1}
    ]
    
    W_contrib = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): -1},
        {(0, 1): 1, (1, 1): 1},
        {(0, 0): 1, (1, 0): 1},
        {(0, 0): -1, (0, 1): 1},
        {(1, 1): 1},
        {(0, 0): 1}
    ]
    
    col = 0
    for a in range(2):
        for b in range(3):
            for s in range(7):
                for (i_loc, j_loc), coeff in U_coeffs[s].items():
                    i = i_loc
                    j = a * 2 + j_loc
                    U_base[i * m + j, col] += coeff
                
                for (j_loc, k_loc), coeff in V_coeffs[s].items():
                    j = a * 2 + j_loc
                    k = b * 2 + k_loc
                    V_base[j * p + k, col] += coeff
                
                for (i_loc, k_loc), coeff in W_contrib[s].items():
                    i = i_loc
                    k = b * 2 + k_loc
                    W_base[k * n + i, col] += coeff
                
                col += 1
    
    best_rank = rank_42
    best_U, best_V, best_W = U_base.copy(), V_base.copy(), W_base.copy()
    
    # Try to find lower rank via gradient-based optimization
    # The tensor rank minimization problem is non-convex, but for small ranks
    # we can try random initializations with gradient descent.
    
    np.random.seed(42)
    
    for R in range(41, 30, -1):
        if time.time() > deadline - 30:
            break
        
        found = False
        max_attempts = 3 if R >= 38 else 2
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 25:
                break
            
            # Initialize with small random values
            scale = 0.3
            U_try = np.random.randn(n * m, R) * scale
            V_try = np.random.randn(m * p, R) * scale
            W_try = np.random.randn(p * n, R) * scale
            
            lr = 0.05
            obj_prev = float('inf')
            stall_count = 0
            best_obj = float('inf')
            
            max_steps = 300
            for step in range(max_steps):
                if time.time() > deadline - 20:
                    break
                
                obj, dU, dV, dW = objective_and_grads(U_try, V_try, W_try, R)
                
                if obj < 1e-10:
                    found = True
                    break
                
                if obj < best_obj:
                    best_obj = obj
                    stall_count = 0
                else:
                    stall_count += 1
                
                if stall_count > 50:
                    lr *= 0.5
                    if lr < 1e-6:
                        break
                
                # Adaptive learning rate
                U_try -= lr * dU
                V_try -= lr * dV
                W_try -= lr * dW
                
                # Periodic restart with smaller lr if stuck
                if step > 0 and step % 50 == 0 and obj > obj_prev * 0.9:
                    lr *= 0.7
                
                obj_prev = obj
            
            if found:
                break
        
        if found:
            max_err = verify_decomp(U_try, V_try, W_try)
            if max_err < 1e-8:
                best_rank = R
                best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
                break
    
    # Final verification
    if time.time() < deadline - 5:
        final_err = verify_decomp(best_U, best_V, best_W)
        if final_err > 1e-6:
            # Fall back to the known-good rank-42 solution
            best_rank = rank_42
            best_U, best_V, best_W = U_base.copy(), V_base.copy(), W_base.copy()
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
