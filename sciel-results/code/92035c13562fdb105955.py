import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Target ranks to try, from lowest to highest
    # Known upper bound is 60. Schoolbook is 72.
    target_ranks = [54, 56, 58, 60, 62, 64, 66, 68, 70, 72]
    
    # Create the full tensor T of shape (n, m, p)
    # T[i, j, k] = 1 if this is a valid multiplication term, 0 otherwise
    # For matrix multiplication A (n x m) * B (m x p) = C (n x p)
    # The tensor is T[i, j, k] = 1 for all i, j, k? 
    # No, the tensor T for matrix multiplication is defined such that 
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # The tensor T has components T_{i, j, k} = 1.
    # Wait, the standard tensor for matrix multiplication M_{n,m,p} is:
    # T_{i, j, k} = 1 for all i in 1..n, j in 1..m, k in 1..p.
    # The contraction is sum_j A_{ij} B_{jk} = C_{ik}.
    # So T is a 3D array of ones.
    
    T = np.ones((n, m, p))
    
    start_time = time.time()
    deadline = start_time + 120  # 120s safety margin
    
    for rank in target_ranks:
        if time.time() > deadline - 10:
            break
            
        # Try to find a rank decomposition
        success = False
        for attempt in range(10): # 10 restarts per rank
            if time.time() > deadline - 10:
                break
                
            # Initialize U, V, W
            U = np.random.rand(n, rank) * 0.1
            V = np.random.rand(m, rank) * 0.1
            W = np.random.rand(p, rank) * 0.1
            
            # ALS Iterations
            for _ in range(100):
                if time.time() > deadline - 10:
                    break
                    
                # Solve for U: T_{i,j,k} = sum_r U_{i,r} V_{j,r} W_{k,r}
                # Reshape T to (n, m*p)
                # We want U (n x rank) such that U @ (V*W) approx T
                # Let M = V (m x rank) and N = W (p x rank)
                # T_{i,j,k} = sum_r U_{i,r} (V_{j,r} W_{k,r})
                # This is not a simple matrix multiplication.
                # We can flatten V and W.
                # Let's use the standard ALS update.
                
                # Update U:
                # For each i, U[i, :] minimizes || T[i, :, :] - U[i, :] contracted with V, W ||
                # This is equivalent to:
                # Let X = V (m x rank), Y = W (p x rank)
                # We want to solve for U[i, r] such that sum_r U[i, r] X[:, r] Y[:, r] = T[i, :, :]
                # This is a linear system.
                # Vectorize: T_i (m*p) = U_i (rank) @ (X * Y) (m*p x rank)
                # (X * Y)_{j,k,r} = X_{j,r} Y_{k,r}
                
                X = V
                Y = W
                # Construct the design matrix for U
                # D_U (m*p x rank) where D_U[(j,k), r] = X[j, r] * Y[k, r]
                D_U = (X[:, np.newaxis, :] * Y[np.newaxis, :, :]).reshape(m * p, rank)
                # T_flat (n x m*p)
                T_flat = T.reshape(n, m * p)
                # Solve U = T_flat @ D_U^+
                try:
                    U = T_flat @ np.linalg.pinv(D_U)
                except:
                    break
                
                # Update V:
                # D_V (n*p x rank) where D_V[(i,k), r] = U[i, r] * W[k, r]
                D_V = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(n * p, rank)
                T_flat_V = T.transpose(0, 2, 1).reshape(n * p, m) # T[i,k,j]
                try:
                    V = T_flat_V @ np.linalg.pinv(D_V)
                except:
                    break
                    
                # Update W:
                # D_W (n*m x rank) where D_W[(i,j), r] = U[i, r] * V[j, r]
                D_W = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(n * m, rank)
                T_flat_W = T.transpose(1, 0, 2).reshape(n * m, p) # T[j,i,k]
                try:
                    W = T_flat_W @ np.linalg.pinv(D_W)
                except:
                    break
            
            # Check convergence and snap to grid
            # Calculate error
            T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            err = np.linalg.norm(T_recon - T)
            
            if err < 1e-6:
                # Snap to nearest 0.5
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                # Verify exactness
                T_recon_snap = np.einsum('ir,jr,kr->ijk', U_snap, V_snap, W_snap)
                if np.allclose(T_recon_snap, T, atol=1e-9):
                    # Success! Convert to lists
                    U_list = U_snap.tolist()
                    V_list = V_snap.tolist()
                    W_list = W_snap.tolist()
                    
                    # The interface requires U (n*m, rank), V (m*p, rank), W (p*n, rank)
                    # But the tensor definition was T[i,j,k] = 1.
                    # The standard matrix multiplication tensor is T[i,j,k].
                    # The output format in the prompt:
                    # U (n*m, rank), V (m*p, rank), W (p*n, rank)
                    # This implies the indices are flattened differently.
                    # Let's check the prompt's schoolbook:
                    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
                    # This corresponds to T[i, j, k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
                    # Wait, the prompt says W[k*n+i, r].
                    # So the tensor is T[i, j, k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
                    # This is the standard matrix multiplication tensor.
                    
                    # My U, V, W are of shape (n, rank), (m, rank), (p, rank).
                    # I need to reshape them to (n*m, rank), (m*p, rank), (p*n, rank).
                    # U_flat[i*m+j, r] = U[i, r] ? No, that's not how it works.
                    # In the tensor decomposition T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r],
                    # The factors are vectors.
                    # The prompt's U is a matrix of size (n*m) x rank.
                    # The schoolbook U[i*m+j, c] = 1 if c corresponds to (i,j,k).
                    # This means the "index" of the row in U is the pair (i,j).
                    # This is NOT a standard tensor decomposition of T[i,j,k] into U[i,r]V[j,r]W[k,r].
                    # The prompt's U, V, W are not the factors of the tensor T in the standard sense.
                    # They are factors of the **bilinear map** (A, B) -> C.
                    # A is n x m. B is m x p.
                    # U is (n*m) x rank.
                    # V is (m*p) x rank.
                    # W is (p*n) x rank.
                    
                    # The decomposition is:
                    # C_{ik} = sum_r (U_{i*m+j, r} A_{ij}) (V_{j*p+k, r} B_{jk}) W_{k*n+i, r} ?
                    # No, the standard form is:
                    # C_{ik} = sum_r (sum_j U_{i*m+j, r} A_{ij}) (sum_j V_{j*p+k, r} B_{jk}) W_{k*n+i, r}
                    # This is equivalent to:
                    # C = (A U^T) * (V^T B) * W^T ?
                    
                    # Let's stick to the schoolbook structure.
                    # The tensor T[i,j,k] = 1.
                    # The decomposition is T[i,j,k] = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
                    # where U_{i,j,r} is the (i*m+j)-th row of U, r-th col.
                    
                    # My ALS found U (n x rank), V (m x rank), W (p x rank) such that
                    # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r].
                    # This is a decomposition of the tensor T.
                    # Can we convert this to the prompt's format?
                    # The prompt's format requires U to depend on (i,j).
                    # My U depends on i.
                    # This is a different decomposition.
                    # The prompt's format is:
                    # U is (n*m) x rank.
                    # V is (m*p) x rank.
                    # W is (p*n) x rank.
                    
                    # If I have U[i,r], V[j,r], W[k,r], I can construct:
                    # U_prompt[i*m+j, r] = U[i,r]
                    # V_prompt[j*p+k, r] = V[j,r]
                    # W_prompt[k*n+i, r] = W[k,r]
                    
                    # Let's check if this works.
                    # Sum_r U_prompt[i*m+j, r] V_prompt[j*p+k, r] W_prompt[k*n+i, r]
                    # = Sum_r U[i,r] V[j,r] W[k,r] = T[i,j,k] = 1.
                    # Yes, it works!
                    
                    # So I just need to expand my U, V, W to the required shapes.
                    U_prompt = np.zeros((n*m, rank))
                    for i in range(n):
                        for j in range(m):
                            U_prompt[i*m+j, :] = U_snap[i, :]
                            
                    V_prompt = np.zeros((m*p, rank))
                    for j in range(m):
                        for k in range(p):
                            V_prompt[j*p+k, :] = V_snap[j, :]
                            
                    W_prompt = np.zeros((p*n, rank))
                    for k in range(p):
                        for i in range(n):
                            W_prompt[k*n+i, :] = W_snap[k, :]
                            
                    return {
                        "U": U_prompt.tolist(),
                        "V": V_prompt.tolist(),
                        "W": W_prompt.tolist()
                    }
    
    # Fallback: Schoolbook Rank 72
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
