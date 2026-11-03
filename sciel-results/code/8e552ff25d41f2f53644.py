import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    target_tensor = np.ones((n, m, p))
    
    # We want to find rank R such that T = sum_r U_r V_r W_r
    # T is the tensor of matrix multiplication, which is all 1s for the specific indices?
    # No, T_{ijk} = 1 if the term A_ij B_jk contributes to C_ik.
    # C_ik = sum_j A_ij B_jk.
    # So T_{ijk} = 1 for all i, j, k.
    
    # Try ranks from low to high? Or high to low?
    # We want the MINIMAL rank. So we should start from a lower bound and go up.
    # But finding a low rank is harder.
    # Let's try a range of ranks.
    
    # Known lower bound: max(n*m, m*p, p*n) = 25.
    # Trivial: 100.
    
    # Let's try to find a solution for ranks 80..100.
    # If we find one for 80, we return it.
    # If we fail, we return the trivial one.
    
    best_rank = n * m * p
    best_solution = None
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer
    
    # Ranks to try. We prioritize lower ranks.
    # It's expensive to try very low ranks if they don't exist.
    # Let's try a few specific ranks that are plausible.
    ranks_to_try = [85, 88, 90, 92, 94, 96, 98, 100]
    
    for R in ranks_to_try:
        if time.time() > deadline:
            break
            
        # Multiple restarts for this rank
        for restart in range(5):
            if time.time() > deadline:
                break
                
            np.random.seed(42 + restart * 1000 + R)
            
            # Initialize U, V, W
            # Scale initialization to help convergence
            U = np.random.randn(n * m, R) * 0.1
            V = np.random.randn(m * p, R) * 0.1
            W = np.random.randn(p * n, R) * 0.1
            
            # Reshape for easier access
            # U: (n, m, R)
            # V: (m, p, R)
            # W: (p, n, R)
            
            # Gradient Descent
            # Error E = 0.5 * sum (1 - sum_r U V W)^2
            # Grad U = - sum_k (1 - val) V W
            # Grad V = - sum_i (1 - val) U W
            # Grad W = - sum_j (1 - val) U V
            
            lr = 0.01
            for step in range(500):
                if time.time() > deadline:
                    break
                
                U_r = U.reshape(n, m, R)
                V_r = V.reshape(m, p, R)
                W_r = W.reshape(p, n, R)
                
                # Calculate current values val[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                val = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                
                # Calculate error
                err = np.sum((val - 1.0) ** 2)
                
                if err < 1e-8:
                    # Check if we can round
                    U_int = np.round(U * 2) / 2
                    V_int = np.round(V * 2) / 2
                    W_int = np.round(W * 2) / 2
                    
                    # Verify exactness
                    U_r_int = U_int.reshape(n, m, R)
                    V_r_int = V_int.reshape(m, p, R)
                    W_r_int = W_int.reshape(p, n, R)
                    val_int = np.einsum('ijr,jkr,kir->ijk', U_r_int, V_r_int, W_r_int)
                    
                    if np.max(np.abs(val_int - 1.0)) < 1e-9:
                        return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                    # If rounding failed, continue optimizing
                    # Maybe we need to be closer to integers.
                    # Add a penalty? Or just continue.
                
                # Calculate gradients
                # diff[i,j,k] = 1 - val[i,j,k]
                diff = 1.0 - val
                
                # Grad U: sum_k diff[i,j,k] V[j,k,r] W[k,i,r]
                # Indices: diff (i,j,k), V (j,k,r), W (k,i,r) -> Result (i,j,r)
                grad_U = np.einsum('ijk,jkr,kir->ijr', diff, V_r, W_r)
                
                # Grad V: sum_i diff[i,j,k] U[i,j,r] W[k,i,r]
                # Indices: diff (i,j,k), U (i,j,r), W (k,i,r) -> Result (j,k,r)
                grad_V = np.einsum('ijk,ijr,kir->jkr', diff, U_r, W_r)
                
                # Grad W: sum_j diff[i,j,k] U[i,j,r] V[j,k,r]
                # Indices: diff (i,j,k), U (i,j,r), V (j,k,r) -> Result (k,i,r)
                grad_W = np.einsum('ijk,ijr,jkr->kir', diff, U_r, V_r)
                
                # Update
                U -= lr * grad_U.reshape(n * m, R)
                V -= lr * grad_V.reshape(m * p, R)
                W -= lr * grad_W.reshape(p * n, R)
                
                # Decay learning rate
                lr *= 0.995
                
                if step % 10 == 0:
                    # Optional: check for rounding earlier if error is small
                    if err < 1e-6:
                         U_int = np.round(U * 2) / 2
                         V_int = np.round(V * 2) / 2
                         W_int = np.round(W * 2) / 2
                         U_r_int = U_int.reshape(n, m, R)
                         V_r_int = V_int.reshape(m, p, R)
                         W_r_int = W_int.reshape(p, n, R)
                         val_int = np.einsum('ijr,jkr,kir->ijk', U_r_int, V_r_int, W_r_int)
                         if np.max(np.abs(val_int - 1.0)) < 1e-9:
                             return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}

    # Fallback to schoolbook
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
# EVOLVE-BLOCK-END
