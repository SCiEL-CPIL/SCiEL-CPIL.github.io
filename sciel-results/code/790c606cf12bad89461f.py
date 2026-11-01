import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Target tensor T of shape (n*p, n*m, m*p)
    # T[i*k, i*j, j*k] = 1 if indices match?
    # No, T is defined by T_{ik, ij, jk} = 1 for the schoolbook terms.
    # Actually, T is the tensor such that <T, A, B> = vec(AB).
    # T_{ik, ij, jk} is 1 if the indices correspond to a valid term in the sum.
    # In the schoolbook decomposition, the term for (i,j,k) is 1 at (ik, ij, jk).
    # So T has 1s at positions (i*p+k, i*m+j, j*p+k) for all i,j,k.
    
    T = np.zeros((n*p, n*m, m*p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*p+k, i*m+j, j*p+k] = 1.0

    # Try to find lower rank decompositions
    # We try ranks from 22 down to 20 (arbitrary lower bound for search)
    # If we find one, we return it.
    
    best_result = None
    start_time = time.time()
    deadline = start_time + 100 # 20s safety margin
    
    for rank in range(22, 15, -1): # Try 22, 21, 20...
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        for seed in range(10):
            if time.time() > deadline:
                break
            
            np.random.seed(seed)
            # Initialize U, V, W
            U = np.random.randn(n*p, rank)
            V = np.random.randn(n*m, rank)
            W = np.random.randn(m*p, rank)
            
            # Simple gradient descent or ALS
            # Let's use a simple iterative refinement:
            # Fix V, W, update U. Fix U, W, update V. Fix U, V, update W.
            
            # Residual R = T - sum_r U[:, r] \otimes V[:, r] \otimes W[:, r]
            # Update U[:, r] = argmin ||T - ... - U[:, r] \otimes V[:, r] \otimes W[:, r]||
            
            # We can use least squares for each column.
            # Let's implement a simple ALS.
            
            max_iters = 50
            for _ in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Update U
                for r in range(rank):
                    # T - sum_{r' != r} U[:, r'] \otimes V[:, r'] \otimes W[:, r']
                    # This is expensive if done naively. 
                    # But dimensions are small.
                    # Residual = T
                    # for r2 in range(rank):
                    #   if r2 == r: continue
                    #   Residual -= np.einsum('ij,jk,kl->ilk', U[:, r2], V[:, r2], W[:, r2])
                    #   Wait, shapes: U is (np, 1), V is (nm, 1), W is (mp, 1).
                    #   Outer product is (np, nm, mp).
                    #   Residual is (np, nm, mp).
                    
                    # Let's precompute the contribution of other terms?
                    # Or just recompute. 22 * 576 ops is small.
                    
                    Residual = T.copy()
                    for r2 in range(rank):
                        if r2 == r:
                            continue
                        # Add back the term? No, Residual = T - Sum_{r != r2} Term_r2
                        # So Residual = T
                        # for r2 != r: Residual -= Term_r2
                        # Then U[:, r] = argmin ||Residual - U[:, r] \otimes V[:, r] \otimes W[:, r]||
                        
                        # This is a linear least squares problem for U[:, r].
                        # Let A = V[:, r] \otimes W[:, r] (vector of size nm*mp)
                        # Let b = Residual (vector of size np*nm*mp)
                        # We want U_vec such that U_vec \otimes A_approx b?
                        # No, the term is U[:, r] \otimes V[:, r] \otimes W[:, r].
                        # Flatten U[:, r] to u (size np).
                        # Flatten V[:, r] \otimes W[:, r] to vw (size nm*mp).
                        # The term is u \otimes vw.
                        # We want to minimize ||b - u \otimes vw||.
                        # This is not a standard linear LS for u because of the tensor product.
                        # However, if we fix V and W, the term is linear in U.
                        # Term = np.einsum('i,jk->ijk', U[:, r], V[:, r] * W[:, r]?) No.
                        # V[:, r] is vector of size nm. W[:, r] is vector of size mp.
                        # The term is U[:, r] (size np) \otimes V[:, r] (size nm) \otimes W[:, r] (size mp).
                        # Let's reshape Residual to (np, nm*mp).
                        # Res_2d = Residual.reshape(np, nm*mp)
                        # The term for r is U[:, r] (size np) \otimes (V[:, r] \otimes W[:, r]) (size nm*mp).
                        # Let vw = np.kron(V[:, r], W[:, r])
                        # Then Term_2d = np.outer(U[:, r], vw)
                        # We want to minimize ||Res_2d - np.outer(u, vw)||_F.
                        # This is a rank-1 approximation problem.
                        # The best u is given by:
                        # u = (Res_2d @ vw) / (vw @ vw)
                        
                        vw = np.kron(V[:, r], W[:, r])
                        Res_2d = Residual.reshape(n*p, n*m*m*p)
                        # Wait, Residual shape is (np, nm, mp). Reshape to (np, nm*mp).
                        Res_2d = Residual.reshape(n*p, n*m*m*p)
                        
                        u = (Res_2d @ vw) / (np.dot(vw, vw) + 1e-12)
                        U[:, r] = u
                        
                # Update V
                for r in range(rank):
                    Residual = T.copy()
                    for r2 in range(rank):
                        if r2 == r:
                            continue
                        Residual -= np.einsum('i,j,k->ijk', U[:, r2], V[:, r2], W[:, r2])
                    
                    # Term is U[:, r] \otimes V[:, r] \otimes W[:, r]
                    # Fix U, W. Update V.
                    # Reshape Residual to (nm, np*mp)
                    Res_2d = Residual.reshape(n*m, n*p*m*p)
                    uw = np.kron(U[:, r], W[:, r])
                    # Wait, order of kron.
                    # Term[i, j, k] = U[i, r] V[j, r] W[k, r]
                    # Reshape to (j, i*k)?
                    # Res[j, i*k] = Sum_{i,k} Res[i, j, k] delta_{i*k, i*k}
                    # Let's be careful with indices.
                    # Res is (i, j, k).
                    # We want to group (i, k) and keep j.
                    # Res_2d = Res.transpose(1, 0, 2).reshape(n*m, n*p*m*p)
                    Res_2d = Residual.transpose(1, 0, 2).reshape(n*m, n*p*m*p)
                    
                    # The term is V[:, r] (size nm) \otimes (U[:, r] \otimes W[:, r]) (size np*mp)
                    # Let uw = np.kron(U[:, r], W[:, r])
                    # But we need to match the reshaping.
                    # Res[i, j, k] -> Res_2d[j, i*p+k]
                    # Term[i, j, k] = U[i, r] V[j, r] W[k, r]
                    # Term_2d[j, i*p+k] = V[j, r] * (U[i, r] W[k, r])
                    # Let vec_uw[i*p+k] = U[i, r] W[k, r]
                    # This is not a simple kron if indices are interleaved?
                    # U is size np. W is size mp.
                    # vec_uw = np.kron(U[:, r], W[:, r])?
                    # np.kron(u, w) has size np*mp.
                    # Index x = i*mp + k.
                    # But we reshaped to i*p+k.
                    # So we need to reshape U and W appropriately or just compute the vector directly.
                    
                    vec_uw = np.zeros(n*p*m*p)
                    for i in range(n*p):
                        for k in range(m*p):
                            # Wait, U index is i (0..np-1). W index is k (0..mp-1).
                            # In Res[i, j, k], i is from 0 to np-1, k is from 0 to mp-1.
                            # So vec_uw[i*mp + k] = U[i, r] * W[k, r]
                            pass
                    # This is just np.kron(U[:, r], W[:, r]) if we define the index order correctly.
                    # np.kron(u, w) [i*mp + k] = u[i] * w[k].
                    # Yes.
                    uw = np.kron(U[:, r], W[:, r])
                    
                    v = (Res_2d @ uw) / (np.dot(uw, uw) + 1e-12)
                    V[:, r] = v

                # Update W
                for r in range(rank):
                    Residual = T.copy()
                    for r2 in range(rank):
                        if r2 == r:
                            continue
                        Residual -= np.einsum('i,j,k->ijk', U[:, r2], V[:, r2], W[:, r2])
                    
                    # Fix U, V. Update W.
                    # Reshape to (mp, np*nm)
                    Res_2d = Residual.transpose(2, 0, 1).reshape(m*p, n*p*n*m)
                    uv = np.kron(U[:, r], V[:, r])
                    
                    w = (Res_2d @ uv) / (np.dot(uv, uv) + 1e-12)
                    W[:, r] = w
            
            # Check error
            # Compute sum
            Sum = np.zeros((n*p, n*m, m*p))
            for r in range(rank):
                Sum += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            
            error = np.linalg.norm(Sum - T)
            
            if error < 1e-6:
                # Try to round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                Sum_r = np.zeros((n*p, n*m, m*p))
                for r in range(rank):
                    Sum_r += np.einsum('i,j,k->ijk', U_r[:, r], V_r[:, r], W_r[:, r])
                
                error_r = np.linalg.norm(Sum_r - T)
                if error_r < 1e-9:
                    # Found a valid solution
                    # Convert to lists
                    U_list = U_r.tolist()
                    V_list = V_r.tolist()
                    W_list = W_r.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
            
            # If we didn't return, continue searching

    # If no better solution found, return schoolbook
    rank = rank_schoolbook
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
