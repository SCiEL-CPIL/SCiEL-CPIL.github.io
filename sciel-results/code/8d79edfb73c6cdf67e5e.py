import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook initialization
    rank0 = n * m * p
    U0 = np.zeros((nm, rank0))
    V0 = np.zeros((mp, rank0))
    W0 = np.zeros((pn, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m + j, col] = 1
                V0[j*p + k, col] = 1
                W0[k*n + i, col] = 1
                col += 1
                
    best_rank = rank0
    best_U, best_V, best_W = U0, V0, W0

    # Try to find a lower rank decomposition using block structure
    # Split p=9 into 3 blocks of 3
    block_size = 3
    num_blocks = p // block_size
    
    sub_n, sub_m, sub_p = 2, 3, 3
    sub_nm, sub_mp, sub_pn = sub_n*sub_m, sub_m*sub_p, sub_p*sub_n
    
    def get_sub_tensor_target():
        target = np.zeros((sub_nm, sub_mp, sub_pn))
        for i in range(sub_n):
            for j in range(sub_m):
                for k in range(sub_p):
                    target[i*sub_m + j, j*sub_p + k, k*sub_n + i] = 1.0
        return target

    target_sub = get_sub_tensor_target()
    
    def find_sub_decomposition_als(rank_sub, max_iters=500):
        # Initialize with random values
        np.random.seed(42)
        U = np.random.randn(sub_nm, rank_sub)
        V = np.random.randn(sub_mp, rank_sub)
        W = np.random.randn(sub_pn, rank_sub)
        
        # Normalize to prevent divergence
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        for _ in range(max_iters):
            if time.time() > deadline - 10:
                break
                
            # Update U
            # T_{a,b,c} = sum_l U_{a,l} V_{b,l} W_{c,l}
            # For fixed b,c, T_{a,b,c} = sum_l U_{a,l} (V_{b,l} W_{c,l})
            # Let M_{a, l} = U_{a,l}
            # Let P_{b,c,l} = V_{b,l} W_{c,l}
            # T_{a,b,c} = sum_l M_{a,l} P_{b,c,l}
            # This is not a standard linear system for U directly because of the coupling.
            # However, we can view it as:
            # T (sub_nm x sub_mp x sub_pn)
            # We want to minimize || T - sum_l u_l v_l w_l ||^2
            # This is not convex. ALS works by fixing two and solving for the third.
            # If V and W are fixed, the term is sum_l u_l (v_l \otimes w_l)
            # Let Z_l = v_l \otimes w_l (shape sub_mp x sub_pn)
            # Then T = sum_l u_l \otimes Z_l
            # This is linear in u_l.
            # T_{a,b,c} = sum_l U_{a,l} Z_{l,b,c}
            # Reshape T to (sub_nm, sub_mp*sub_pn)
            # Reshape Z to (rank_sub, sub_mp*sub_pn)
            # T_flat = U @ Z_flat^T
            # U = T_flat @ (Z_flat^T @ Z_flat)^-1
            
            Z_flat = np.einsum('bl,cl->bcl', V, W).reshape(rank_sub, sub_mp * sub_pn)
            T_flat = target_sub.reshape(sub_nm, sub_mp * sub_pn)
            
            # Solve U @ Z_flat^T = T_flat
            # U = T_flat @ pinv(Z_flat^T)
            try:
                U_new = T_flat @ np.linalg.pinv(Z_flat.T)
            except:
                break
            U = U_new
            
            # Update V
            # T_{a,b,c} = sum_l V_{b,l} (U_{a,l} W_{c,l})
            # Let Y_l = u_l \otimes w_l (shape sub_nm x sub_pn)
            # T = sum_l v_l \otimes Y_l
            # Reshape T to (sub_mp, sub_nm*sub_pn)
            # Reshape Y to (rank_sub, sub_nm*sub_pn)
            # T_flat_v = V @ Y_flat^T
            Y_flat = np.einsum('al,cl->acl', U, W).reshape(rank_sub, sub_nm * sub_pn)
            T_flat_v = target_sub.transpose(1, 0, 2).reshape(sub_mp, sub_nm * sub_pn)
            
            try:
                V_new = T_flat_v @ np.linalg.pinv(Y_flat.T)
            except:
                break
            V = V_new
            
            # Update W
            # T_{a,b,c} = sum_l W_{c,l} (U_{a,l} V_{b,l})
            # Let X_l = u_l \otimes v_l (shape sub_nm x sub_mp)
            # T = sum_l w_l \otimes X_l
            # Reshape T to (sub_pn, sub_nm*sub_mp)
            # Reshape X to (rank_sub, sub_nm*sub_mp)
            X_flat = np.einsum('al,bl->abl', U, V).reshape(rank_sub, sub_nm * sub_mp)
            T_flat_w = target_sub.transpose(2, 0, 1).reshape(sub_pn, sub_nm * sub_mp)
            
            try:
                W_new = T_flat_w @ np.linalg.pinv(X_flat.T)
            except:
                break
            W = W_new
            
            # Check convergence
            T_approx = np.einsum('al,bl,cl->abc', U, V, W)
            err = np.linalg.norm(T_approx - target_sub)
            if err < 1e-8:
                return U, V, W, err
        
        T_approx = np.einsum('al,bl,cl->abc', U, V, W)
        err = np.linalg.norm(T_approx - target_sub)
        if err < 1e-8:
            return U, V, W, err
        return None

    # Try to find sub-decomposition with ALS
    sub_decomp = None
    # Try rank 6 first (best case), then 12, then 18
    for r in [6, 12, 18]:
        if time.time() > deadline - 20:
            break
        sub_decomp = find_sub_decomposition_als(r)
        if sub_decomp:
            print(f"Found sub-decomp with rank {r}, loss {sub_decomp[3]}")
            break

    if sub_decomp:
        U_sub, V_sub, W_sub, _ = sub_decomp
        rank_sub = U_sub.shape[1]
        total_rank = rank_sub * num_blocks
        
        # Assemble full decomposition
        U_full = np.zeros((nm, total_rank))
        V_full = np.zeros((mp, total_rank))
        W_full = np.zeros((pn, total_rank))
        
        for b in range(num_blocks):
            # Copy U
            U_full[:, b*rank_sub:(b+1)*rank_sub] = U_sub
            
            # Copy V, shifting k index
            for j in range(sub_m):
                for k_prime in range(sub_p):
                    V_full[j*p + b*sub_p + k_prime, b*rank_sub:(b+1)*rank_sub] = V_sub[j*sub_p + k_prime, :]
            
            # Copy W, shifting k index
            for k_prime in range(sub_p):
                for i in range(sub_n):
                    W_full[b*sub_p + k_prime, b*rank_sub:(b+1)*rank_sub] = W_sub[k_prime*sub_n + i, :]
        
        # Verify
        def verify_full(U, V, W):
            r = U.shape[1]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        if abs(s - 1.0) > 1e-6:
                            return False
            return True

        if verify_full(U_full, V_full, W_full):
            best_rank = total_rank
            best_U, best_V, best_W = U_full, V_full, W_full
            print(f"Constructed full decomposition with rank {total_rank}")

    # Return best
    def to_list(U, V, W):
        return {
            "U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]
        }
    
    return to_list(best_U, best_V, best_W)
