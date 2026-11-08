import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Target tensor T is defined by T[i, j, k] = 1 if indices align
    # We don't build the full dense tensor if possible, but for ALS we need to contract.
    # T_{a,b,c} where a=(i,j), b=(j,k), c=(k,i)
    # We can represent T as a list of non-zero entries.
    # Entry (a, b, c) is 1.
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    # i in 0..n-1, j in 0..m-1, k in 0..p-1
    
    non_zeros = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                non_zeros.append((a, b, c, 1.0))
                
    # Try ranks from lower bound upwards
    min_rank = max(dim_U, dim_V, dim_W)
    max_rank = 60 # Upper limit for search
    
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    best_result = None
    best_rank = n * m * p
    
    # Fallback schoolbook
    U_school = np.zeros((dim_U, best_rank))
    V_school = np.zeros((dim_V, best_rank))
    W_school = np.zeros((dim_W, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_school[a, col] = 1
                V_school[b, col] = 1
                W_school[c, col] = 1
                col += 1
                
    for R in range(min_rank, max_rank + 1):
        if time.time() > deadline:
            break
            
        # Try a few random restarts for this rank
        found = False
        for restart in range(5):
            if time.time() > deadline:
                break
                
            # Initialize U, V, W
            np.random.seed(42 + R + restart) # Reproducibility for debugging, but random is better for search
            # Actually, use random seed
            U = np.random.randn(dim_U, R)
            V = np.random.randn(dim_V, R)
            W = np.random.randn(dim_W, R)
            
            # ALS iterations
            for it in range(200):
                # Solve for U
                # Residual R_{a,b,c} = T_{a,b,c} - sum_r U_{a,r} V_{b,r} W_{c,r}
                # For fixed V, W, the term is U_{a,r} * (sum_r' V_{b,r'} W_{c,r'}?) No.
                # The model is sum_r U_{a,r} V_{b,r} W_{c,r}.
                # For a fixed a, we want to minimize sum_{b,c} (T_{a,b,c} - sum_r U_{a,r} V_{b,r} W_{c,r})^2.
                # This is linear in U_{a,r}.
                # Let M_{b,c} = sum_r V_{b,r} W_{c,r}.
                # We want U_{a} to approximate T_{a, :, :} / M?
                # Actually, we can solve for all U at once.
                # Construct matrix A_U of size (dim_U * dim_V * dim_W) x R? Too big.
                
                # Efficient ALS:
                # 1. Compute M = V^T @ W ? No.
                # The tensor product is sum_r U_r \otimes V_r \otimes W_r.
                # Flatten T to vector t.
                # Flatten U, V, W?
                
                # Let's use the property:
                # For U:
                # We need to solve min_U || T - sum_r U_r \otimes (V_r \otimes W_r) ||_F^2
                # Let Z_r = V_r \otimes W_r (flattened vector of size dim_V * dim_W)
                # Then T_flattened (size dim_U * dim_V * dim_W) = sum_r U_r (flattened? No, U_r is vector dim_U) \otimes Z_r
                # This is a linear system:
                # Let T be a 3D array.
                # T[a, b, c] = sum_r U[a, r] * (V[b, r] * W[c, r])
                # For each a, T[a, :, :] = sum_r U[a, r] * (V[:, r] \otimes W[:, r])
                # Let M be a matrix of size (dim_V * dim_W) x R where M[:, r] = V[:, r] \otimes W[:, r]
                # Then for each a, T[a, :, :].flatten() = M @ U[a, :]
                # So U[a, :] = (M^T M)^{-1} M^T T[a, :, :].flatten()
                # This can be done for all a at once.
                
                # Compute M_VW
                # V is dim_V x R, W is dim_W x R
                # We need V_r \otimes W_r for each r.
                # This is a matrix of size (dim_V * dim_W) x R.
                # M_VW = np.einsum('br,cr->bc r', V, W).reshape(dim_V * dim_W, R)
                
                M_VW = np.einsum('br,cr->bcr', V, W).reshape(dim_V * dim_W, R)
                
                # T_flat_a is dim_V * dim_W for each a.
                # We need T as a 3D array.
                # Build T once outside the loop?
                # T is sparse.
                # Let's build dense T for simplicity, size 12 x 28 x 21 = 7056. Small enough.
                
                # Pre-compute T dense
                if 'T_dense' not in locals():
                    T_dense = np.zeros((dim_U, dim_V, dim_W))
                    for a, b, c, val in non_zeros:
                        T_dense[a, b, c] = val
                    T_flat = T_dense.reshape(dim_U, dim_V * dim_W)
                
                # Solve for U
                # T_flat[a, :] = M_VW.T @ U[a, :] ?
                # T_flat[a, b*c] = sum_r U[a, r] * M_VW[b*c, r]
                # So T_flat.T = U.T @ M_VW
                # U.T = T_flat.T @ (M_VW.T M_VW)^{-1} M_VW.T
                # Or use least squares: U = np.linalg.lstsq(M_VW.T, T_flat.T, rcond=None)[0].T
                # Wait, dimensions:
                # M_VW is (dim_V*dim_W) x R.
                # T_flat is dim_U x (dim_V*dim_W).
                # We want U (dim_U x R) such that U @ M_VW.T = T_flat.
                # So U = T_flat @ (M_VW.T M_VW)^{-1} M_VW.T
                # Or U = np.linalg.lstsq(M_VW.T, T_flat.T, rcond=None)[0].T
                
                try:
                    U_new = np.linalg.lstsq(M_VW.T, T_flat.T, rcond=None)[0].T
                except:
                    break
                    
                # Solve for V
                # T[a, b, c] = sum_r V[b, r] * (U[a, r] * W[c, r])
                # For each b, T[:, b, :].flatten() = M_UW @ V[b, :]
                # M_UW is (dim_U * dim_W) x R, M_UW[:, r] = U[:, r] \otimes W[:, r]
                M_UW = np.einsum('ar,cr->acr', U_new, W).reshape(dim_U * dim_W, R)
                T_flat_V = T_dense.transpose(1, 0, 2).reshape(dim_V, dim_U * dim_W)
                try:
                    V_new = np.linalg.lstsq(M_UW.T, T_flat_V.T, rcond=None)[0].T
                except:
                    break
                    
                # Solve for W
                # T[a, b, c] = sum_r W[c, r] * (U[a, r] * V[b, r])
                # For each c, T[:, :, c].flatten() = M_UV @ W[c, :]
                M_UV = np.einsum('ar,br->abr', U_new, V_new).reshape(dim_U * dim_V, R)
                T_flat_W = T_dense.transpose(2, 0, 1).reshape(dim_W, dim_U * dim_V)
                try:
                    W_new = np.linalg.lstsq(M_UV.T, T_flat_W.T, rcond=None)[0].T
                except:
                    break
                    
                # Check convergence
                U = U_new
                V = V_new
                W = W_new
                
                # Compute error
                # Reconstruct and check
                # T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                # err = np.linalg.norm(T_recon - T_dense)
                # If err < 1e-6, check exactness
                
            # After ALS, check if we have a good solution
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T_recon - T_dense)
            
            if err < 1e-5:
                # Try to snap to half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                T_recon_snap = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
                err_snap = np.linalg.norm(T_recon_snap - T_dense)
                
                if err_snap < 1e-9:
                    # Found a valid decomposition
                    # Convert to lists for return
                    U_list = U_snap.tolist()
                    V_list = V_snap.tolist()
                    W_list = W_snap.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
                    
    # Fallback
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
