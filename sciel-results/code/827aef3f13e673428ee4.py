import numpy as np
import time

def construct():
    n, m, p = 2, 3, 9
    deadline = time.time() + 140  # safety margin
    
    # Sub-problem: 2x3x3
    n2, m2, p2 = 2, 3, 3
    # Tensor T of shape (n2*m2, m2*p2, p2*n2)
    # Indices: a = i*m2 + j, b = j*p2 + k, c = k*n2 + i
    T = np.zeros((n2*m2, m2*p2, p2*n2))
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                T[i*m2+j, j*p2+k, k*n2+i] = 1.0
                
    # Schoolbook rank for sub-problem is 18.
    # Total schoolbook rank is 54.
    
    best_sub_decomp = None
    best_sub_rank = n2*m2*p2
    
    # Try ranks from 12 up to 17
    # Rank 11 might be impossible or very hard.
    for target_rank in range(12, n2*m2*p2 + 1):
        if time.time() > deadline - 20:
            break
            
        found = False
        # Try a few seeds
        for seed in range(50):
            if time.time() > deadline - 20:
                break
                
            rng = np.random.RandomState(seed)
            U = rng.randn(n2*m2, target_rank)
            V = rng.randn(m2*p2, target_rank)
            W = rng.randn(p2*n2, target_rank)
            
            # ALS iterations
            for it in range(1000):
                if time.time() > deadline - 20:
                    break
                    
                # Update U
                # For each a, solve M * U_a = X_a
                # M is (m2*p2 * p2*n2) x target_rank ? No.
                # M_{(b,c), r} = V_{b,r} * W_{c,r}
                M_mat = np.einsum('br,cr->bcr', V, W).reshape(m2*p2 * p2*n2, target_rank)
                
                for a in range(n2*m2):
                    X_a = T[a, :, :].reshape(-1)
                    # Solve least squares
                    U[a, :] = np.linalg.lstsq(M_mat, X_a, rcond=None)[0]
                    
                # Update V
                # For each b, solve N * V_b = Y_b
                # N_{(a,c), r} = U_{a,r} * W_{c,r}
                N_mat = np.einsum('ar,cr->acr', U, W).reshape(n2*m2 * p2*n2, target_rank)
                for b in range(m2*p2):
                    Y_b = T[:, b, :].reshape(-1)
                    V[b, :] = np.linalg.lstsq(N_mat, Y_b, rcond=None)[0]
                    
                # Update W
                # For each c, solve P * W_c = Z_c
                # P_{(a,b), r} = U_{a,r} * V_{b,r}
                P_mat = np.einsum('ar,br->abr', U, V).reshape(n2*m2 * m2*p2, target_rank)
                for c in range(p2*n2):
                    Z_c = T[:, :, c].reshape(-1)
                    W[c, :] = np.linalg.lstsq(P_mat, Z_c, rcond=None)[0]
                    
            # Check convergence
            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            if np.allclose(T_approx, T, atol=1e-6):
                best_sub_decomp = (U, V, W)
                best_sub_rank = target_rank
                found = True
                break
                
        if found:
            break
            
    if best_sub_decomp is not None and best_sub_rank < n2*m2*p2:
        U2, V2, W2 = best_sub_decomp
        r = best_sub_rank
        
        # Construct full 2x3x9 decomposition
        # Total rank = 3 * r
        total_rank = 3 * r
        U_full = np.zeros((n*m, total_rank))
        V_full = np.zeros((m*p, total_rank))
        W_full = np.zeros((p*n, total_rank))
        
        for b_block in range(3):
            for r_local in range(r):
                col = b_block * r + r_local
                
                # U is the same for all blocks (indices i, j)
                for a in range(n*m):
                    U_full[a, col] = U2[a, r_local]
                    
                # V indices: j, k_full = b_block*3 + k_local
                # Sub V indices: j, k_local
                for j in range(m2):
                    for k_local in range(p2):
                        b_full = j * p + b_block * p2 + k_local
                        b_sub = j * p2 + k_local
                        V_full[b_full, col] = V2[b_sub, r_local]
                        
                # W indices: k_full, i
                # k_full = b_block*3 + k_local
                # Sub W indices: k_local, i
                for k_local in range(p2):
                    for i in range(n2):
                        c_full = (b_block * p2 + k_local) * n2 + i
                        c_sub = k_local * n2 + i
                        W_full[c_full, col] = W2[c_sub, r_local]
                        
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
