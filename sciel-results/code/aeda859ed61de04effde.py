import numpy as np
import time

DIMENSIONS = [2, 2, 15]

def _solve_m223_rank10():
    """
    Attempts to find a rank 10 decomposition for 2x2x3 matrix multiplication.
    Returns (U, V, W) if found, else None.
    """
    n, m, p = 2, 2, 3
    dim_u = n * m  # 4
    dim_v = m * p  # 6
    dim_w = p * n  # 6
    
    # Construct target tensor T of shape (dim_u, dim_v, dim_w)
    T = np.zeros((dim_u, dim_v, dim_w))
    for r in range(n):
        for mm in range(m):
            for c in range(p):
                idx_u = r * m + mm
                idx_v = mm * p + c
                idx_w = c * n + r  # Wait, W index convention?
                # Task says: W[k*n+i, r]. 
                # C is n x p. C[r, c].
                # W index in task description: W[k*n+i, r].
                # Usually W corresponds to C.
                # Let's check the seed program's W index.
                # Seed: W[14*n + 0][col] = 1 for C[0, 14].
                # Here n=2, p=15.
                # C[0, 14] -> row 0, col 14.
                # W index = 14*2 + 0 = 28.
                # So W index = col * n + row.
                # In our 2x2x3 case:
                # C[r, c] -> W index = c * n + r.
                idx_w = c * n + r
                T[idx_u, idx_v, idx_w] = 1.0

    rank = 10
    # Initialize U, V, W
    np.random.seed(42)
    U = np.random.randn(dim_u, rank)
    V = np.random.randn(dim_v, rank)
    W = np.random.randn(dim_w, rank)
    
    # Normalize to prevent scale issues
    for k in range(rank):
        norm = np.sqrt(np.sum(U[:, k]**2) + np.sum(V[:, k]**2) + np.sum(W[:, k]**2))
        if norm > 0:
            U[:, k] /= norm
            V[:, k] /= norm
            W[:, k] /= norm
    
    # Alternating Least Squares
    # We want to minimize || T - sum_k U_k V_k^T W_k^T ||_F^2 ?
    # Actually T is 3D. We flatten T to 1D of size dim_u*dim_v*dim_w.
    # And the model is sum_k outer(U_k, V_k, W_k).
    # This is equivalent to minimizing || T_flat - sum_k vec(outer(U_k, V_k)) * W_k^T || ? No.
    # Let's use the property: T = sum_k U_k \otimes V_k \otimes W_k.
    # We can fix W and solve for U, V? No, that's bilinear.
    # Better: Fix V, W. Solve for U?
    # T_{ijk} = sum_k U_{ik} V_{jk} W_{lk} (using l for w index).
    # Reshape T to (dim_u, dim_v * dim_w).
    # Then T_{i, (j,l)} = sum_k U_{ik} (V_{jk} W_{lk}).
    # Let M_{kl} = V_k W_l? No, V is (dim_v, rank), W is (dim_w, rank).
    # We need to form a matrix for each i?
    # Let's use the standard ALS for tensor decomposition.
    
    tol = 1e-8
    max_iter = 2000
    start_time = time.time()
    deadline = start_time + 40  # 40 seconds limit for solver
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
            
        # Update U
        # Fix V, W.
        # For each i, we want U_{i, :} such that U_{i, :} \cdot (V \otimes W)_{: , :} approximates T_{i, :, :}
        # Let A_{i, k} = sum_{j,l} T_{i,j,l} V_{j,k} W_{l,k}
        # Then U_{i, :} should be A_{i, :} / sum_k V_{j,k}^2 W_{l,k}^2 ?
        # Actually, we minimize sum_{j,l} (T_{i,j,l} - sum_k U_{i,k} V_{j,k} W_{l,k})^2
        # This is a linear system for U_{i, :}.
        # Let M_{k, k'} = sum_{j,l} V_{j,k} W_{l,k} V_{j,k'} W_{l,k'}
        # Let b_{i, k} = sum_{j,l} T_{i,j,l} V_{j,k} W_{l,k}
        # Then M U_{i, :}^T = b_{i, :}^T
        
        # Compute M (rank x rank)
        # M_{k, k'} = < V_k \otimes W_k, V_{k'} \otimes W_{k'} > = (V^T V)_{k,k'} (W^T W)_{k,k'}? No.
        # sum_{j,l} V_{j,k} W_{l,k} V_{j,k'} W_{l,k'} = (sum_j V_{j,k} V_{j,k'}) (sum_l W_{l,k} W_{l,k'})
        # So M = (V^T V) * (W^T W) (element-wise product? No, Hadamard product)
        VtV = V.T @ V
        WtW = W.T @ W
        M = VtV * WtW  # Hadamard product
        
        # Compute b (dim_u x rank)
        # b_{i, k} = sum_{j,l} T_{i,j,l} V_{j,k} W_{l,k}
        # T is (dim_u, dim_v, dim_w)
        # We can do this efficiently:
        # T_flat = T.reshape(dim_u, -1)
        # We need sum_{j,l} T_{i, j, l} (V[:, k] W[:, k])_{j, l}
        # Let X_k = V[:, k] W[:, k] (Hadamard product of column k of V and W)
        # X is (dim_v, dim_w) reshaped to (dim_v*dim_w, )
        # b_{i, k} = dot(T_{i, :, :}.flatten(), X_k)
        
        T_flat = T.reshape(dim_u, dim_v * dim_w)
        # Precompute X for all k
        # X is (rank, dim_v * dim_w)
        X = np.zeros((rank, dim_v * dim_w))
        for k in range(rank):
            X[k] = (V[:, k] * W[:, k]).flatten()
        
        b = T_flat @ X.T  # (dim_u, rank)
        
        # Solve M U^T = b^T
        # M is (rank, rank)
        try:
            U_new = np.linalg.solve(M, b.T).T
        except np.linalg.LinAlgError:
            U_new = np.linalg.lstsq(M, b.T, rcond=None)[0].T
        
        # Update V
        # Fix U, W.
        # M_{k, k'} = (U^T U)_{k,k'} (W^T W)_{k,k'}
        UtU = U_new.T @ U_new
        WtW = W.T @ W
        M = UtU * WtW
        
        # b_{j, k} = sum_{i,l} T_{i,j,l} U_{i,k} W_{l,k}
        # T is (dim_u, dim_v, dim_w)
        # Transpose to (dim_v, dim_u * dim_w)
        T_trans = np.transpose(T, (1, 0, 2)).reshape(dim_v, dim_u * dim_w)
        
        X = np.zeros((rank, dim_u * dim_w))
        for k in range(rank):
            X[k] = (U_new[:, k] * W[:, k]).flatten()
            
        b = T_trans @ X.T
        try:
            V_new = np.linalg.solve(M, b.T).T
        except np.linalg.LinAlgError:
            V_new = np.linalg.lstsq(M, b.T, rcond=None)[0].T
            
        # Update W
        # Fix U, V.
        # M_{k, k'} = (U^T U)_{k,k'} (V^T V)_{k,k'}
        UtU = U_new.T @ U_new
        VtV = V_new.T @ V_new
        M = UtU * VtV
        
        # b_{l, k} = sum_{i,j} T_{i,j,l} U_{i,k} V_{j,k}
        # T is (dim_u, dim_v, dim_w)
        # Transpose to (dim_w, dim_u * dim_v)
        T_trans2 = np.transpose(T, (2, 0, 1)).reshape(dim_w, dim_u * dim_v)
        
        X = np.zeros((rank, dim_u * dim_v))
        for k in range(rank):
            X[k] = (U_new[:, k] * V_new[:, k]).flatten()
            
        b = T_trans2 @ X.T
        try:
            W_new = np.linalg.solve(M, b.T).T
        except np.linalg.LinAlgError:
            W_new = np.linalg.lstsq(M, b.T, rcond=None)[0].T
            
        U, V, W = U_new, V_new, W_new
        
        # Check convergence
        # Compute residual
        # Residual = T - sum_k U_k V_k^T W_k^T
        # This is expensive to compute fully every iteration.
        # We can estimate it.
        if it % 100 == 0:
            # Quick check
            # Reconstruct a few entries
            pass

    # Final check
    # Reconstruct tensor
    T_rec = np.zeros_like(T)
    for k in range(rank):
        T_rec += np.einsum('i,j,l->ijl', U[:, k], V[:, k], W[:, k])
    
    error = np.linalg.norm(T - T_rec)
    if error < 1e-4:
        # Try to round to 0.5
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        T_rec_rounded = np.zeros_like(T)
        for k in range(rank):
            T_rec_rounded += np.einsum('i,j,l->ijl', U_rounded[:, k], V_rounded[:, k], W_rounded[:, k])
        
        error_rounded = np.linalg.norm(T - T_rec_rounded)
        if error_rounded < 1e-6:
            return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
            
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try to find rank 10 decomposition for 2x2x3
    # If found, use 5 blocks
    # If not, use 7 blocks of 2x2x2 (Strassen) + 1 block of 2x2x1 (Schoolbook)
    
    # Deadline guard
    start_time = time.time()
    total_time_limit = 145  # Leave 5s margin
    
    m223_decomp = None
    if time.time() - start_time < 60:
        m223_decomp = _solve_m223_rank10()
        
    if m223_decomp is not None:
        U_block, V_block, W_block = m223_decomp
        rank_block = len(U_block[0]) # 10
        total_rank = 5 * rank_block
        rank = total_rank
        
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        for b in range(5):
            col_offset = b * rank_block
            # Map indices
            # Block b uses columns 3b, 3b+1, 3b+2 of B
            # B indices in V: row * p + col
            # V_block has indices for 2x3 block.
            # V_block[i, k] corresponds to B_{i, 0}, B_{i, 1}, B_{i, 2} of the block.
            # We need to map V_block indices to global V indices.
            # V_block row 0 -> B[0, 3b], B[0, 3b+1], B[0, 3b+2]
            # V_block row 1 -> B[1, 3b], B[1, 3b+1], B[1, 3b+2]
            # V_block row 2 -> B[2, ...]? No, B is 2x3. Rows 0,1. Cols 0,1,2.
            # V_block shape (6, 10).
            # Rows 0-2 correspond to B[0, :], Rows 3-5 correspond to B[1, :].
            
            for k in range(rank_block):
                # U
                for i in range(n * m):
                    U[i][col_offset + k] = U_block[i][k]
                
                # V
                for i in range(6):
                    # i=0 -> B[0, 0] -> global 0*p + 3b
                    # i=1 -> B[0, 1] -> global 0*p + 3b+1
                    # i=2 -> B[0, 2] -> global 0*p + 3b+2
                    # i=3 -> B[1, 0] -> global 1*p + 3b
                    # i=4 -> B[1, 1] -> global 1*p + 3b+1
                    # i=5 -> B[1, 2] -> global 1*p + 3b+2
                    if i < 3:
                        global_idx = 0 * p + (3 * b + i)
                    else:
                        global_idx = 1 * p + (3 * b + (i - 3))
                    
                    V[global_idx][col_offset + k] = V_block[i][k]
                
                # W
                # W_block shape (6, 10).
                # W index convention: c * n + r.
                # Block W indices:
                # 0: C[0, 0] -> global (3b)*n + 0
                # 1: C[0, 1] -> global (3b+1)*n + 0
                # 2: C[0, 2] -> global (3b+2)*n + 0
                # 3: C[1, 0] -> global (3b)*n + 1
                # 4: C[1, 1] -> global (3b+1)*n + 1
                # 5: C[1, 2] -> global (3b+2)*n + 1
                
                for i in range(6):
                    if i < 3:
                        c = 3 * b + i
                        r = 0
                    else:
                        c = 3 * b + (i - 3)
                        r = 1
                    
                    global_idx = c * n + r
                    W[global_idx][col_offset + k] = W_block[i][k]
        
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to rank 53 solution
        rank = 53
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        col = 0
        U_coeffs = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1]
        ]
        
        def get_V_coeffs(block_idx):
            c0 = 2 * block_idx
            c1 = 2 * block_idx + 1
            sparse = []
            sparse.append([(0*p + c0, 1), (1*p + c1, 1)])
            sparse.append([(0*p + c0, 1)])
            sparse.append([(0*p + c1, 1), (1*p + c1, -1)])
            sparse.append([(1*p + c0, 1), (0*p + c0, -1)])
            sparse.append([(1*p + c1, 1)])
            sparse.append([(0*p + c0, 1), (0*p + c1, 1)])
            sparse.append([(1*p + c0, 1), (1*p + c1, 1)])
            return sparse

        def get_W_coeffs(block_idx):
            c0 = 2 * block_idx
            c1 = 2 * block_idx + 1
            sparse = []
            sparse.append([(c0*n + 0, 1), (c1*n + 1, 1)])
            sparse.append([(c0*n + 1, 1), (c1*n + 1, -1)])
            sparse.append([(c1*n + 0, 1), (c1*n + 1, 1)])
            sparse.append([(c0*n + 0, 1), (c0*n + 1, 1)])
            sparse.append([(c0*n + 0, -1), (c1*n + 0, 1)])
            sparse.append([(c1*n + 1, 1)])
            sparse.append([(c0*n + 0, 1)])
            return sparse

        for b in range(7):
            V_sparse = get_V_coeffs(b)
            W_sparse = get_W_coeffs(b)
            for k in range(7):
                for idx, val in enumerate(U_coeffs[k]):
                    if val != 0:
                        U[idx][col] = val
                for idx, val in V_sparse[k]:
                    V[idx][col] = val
                for idx, val in W_sparse[k]:
                    W[idx][col] = val
                col += 1

        # Last column
        U[0][col] = 1
        V[0*p + 14][col] = 1
        W[14*n + 0][col] = 1
        col += 1
        
        U[1][col] = 1
        V[1*p + 14][col] = 1
        W[14*n + 0][col] = 1
        col += 1
        
        U[2][col] = 1
        V[0*p + 14][col] = 1
        W[14*n + 1][col] = 1
        col += 1
        
        U[3][col] = 1
        V[1*p + 14][col] = 1
        W[14*n + 1][col] = 1
        col += 1

        return {"U": U, "V": V, "W": W}

