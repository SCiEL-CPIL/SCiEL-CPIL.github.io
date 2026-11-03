import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    # Target Tensor T
    # T[i, j, k] = 1 if indices match multiplication pattern
    # Actually, T is the tensor such that sum_c U[i,c] V[j,c] W[k,c] = T[i,j,k]
    # For matrix multiplication, T[i,j,k] is 1 if the term A[i,j] B[j,k] contributes to C[i,k].
    # In the standard decomposition, T[i,j,k] = 1 for all i,j,k.
    # Wait, the tensor T for matrix multiplication is:
    # T = sum_{i,j,k} e_{i,j} (in U) * e_{j,k} (in V) * e_{k,i} (in W)?
    # No, the indices in W are (k, i) in the prompt's convention W[k*n+i].
    # So W index is (k, i).
    # The term is A[i,j] B[j,k] -> C[i,k].
    # So the tensor entry corresponding to (i,j) in U, (j,k) in V, (i,k) in W is 1.
    # Note: W index is k*n + i. So W row is (k, i).
    # So T[i*m+j, j*p+k, k*n+i] = 1.
    
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    # Try to find lower rank decomposition
    # We try ranks from 47 down to 42
    # But we must be fast.
    
    def try_decompose(rank):
        # Initialize U, V, W with random values
        # Use small random values to start
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        for it in range(50):
            if time.time() > deadline:
                return None
            
            # Update U
            # B shape: (m*p * p*n) x rank
            # B[v_idx, w_idx, c] = V[v_idx, c] * W[w_idx, c]
            B = np.einsum('vc,wc->vwc', V, W).reshape(m * p * p * n, rank)
            # T_flat shape: (n*m, m*p*p*n)
            # For each i, solve B * U[i,:] = T[i, :, :] flattened
            T_flat = T.reshape(n * m, m * p * p * n)
            for i in range(n * m):
                # Solve least squares
                U[i, :] = np.linalg.lstsq(B, T_flat[i, :], rcond=None)[0]
                
            # Update V
            # C shape: (n*m * p*n) x rank
            # C[u_idx, w_idx, c] = U[u_idx, c] * W[w_idx, c]
            C = np.einsum('uc,wc->uwc', U, W).reshape(n * m * p * n, rank)
            T_flat_v = T.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
            for j in range(m * p):
                V[j, :] = np.linalg.lstsq(C, T_flat_v[j, :], rcond=None)[0]
                
            # Update W
            # D shape: (n*m * m*p) x rank
            # D[u_idx, v_idx, c] = U[u_idx, c] * V[v_idx, c]
            D = np.einsum('uc,vc->uvc', U, V).reshape(n * m * m * p, rank)
            T_flat_w = T.transpose(2, 0, 1).reshape(p * n, n * m * m * p)
            for k in range(p * n):
                W[k, :] = np.linalg.lstsq(D, T_flat_w[k, :], rcond=None)[0]
                
        # Check residual
        # Reconstruct T_recon
        T_recon = np.einsum('ic,jc,kc->ijk', U, V, W)
        residual = np.linalg.norm(T_recon - T)
        
        if residual < 1e-5:
            # Snap to half-integers
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            T_recon_snap = np.einsum('ic,jc,kc->ijk', U_snap, V_snap, W_snap)
            residual_snap = np.linalg.norm(T_recon_snap - T)
            
            if residual_snap < 1e-5:
                return {
                    "U": U_snap.tolist(),
                    "V": V_snap.tolist(),
                    "W": W_snap.tolist()
                }
        return None

    # Try ranks
    for rank in range(47, 41, -1):
        if time.time() > deadline:
            break
        res = try_decompose(rank)
        if res:
            return res
            
    return {"U": U_school, "V": V_school, "W": W_school}
