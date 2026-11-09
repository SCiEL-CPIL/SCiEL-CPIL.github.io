import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Precompute indices for the non-zero entries of the tensor
    # Tensor T has T[idx_u, idx_v, idx_w] = 1 if (idx_u, idx_v, idx_w) corresponds to (i,j), (j,k), (k,i)
    # idx_u = i*m + j
    # idx_v = j*p + k
    # idx_w = k*n + i
    
    support_u = []
    support_v = []
    support_w = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                support_u.append(i*m + j)
                support_v.append(j*p + k)
                support_w.append(k*n + i)
                
    support_u = np.array(support_u)
    support_v = np.array(support_v)
    support_w = np.array(support_w)
    
    # Total entries
    total_entries = nm * mp * pn
    # We don't need to store the full tensor, just the objective function
    
    def residual(params):
        r = params.shape[0] // (nm + mp + pn)
        # Reshape params into U, V, W
        # U is (nm, r), V is (mp, r), W is (pn, r)
        # params is flat array of size r*(nm+mp+pn)
        
        # This reshape is tricky if we don't know r inside residual easily without passing it.
        # Better to define residual inside a closure that captures r.
        
        pass

    # We will iterate over candidate ranks
    start_time = time.time()
    deadline = start_time + 110 # Leave 10s margin
    
    # Lower bound is max(nm, mp, pn) = 24
    min_rank = max(nm, mp, pn)
    
    best_result = None
    
    for r in range(min_rank, 50): # Try up to rank 50
        if time.time() > deadline:
            break
            
        # Try a few random restarts for this rank
        for attempt in range(10):
            if time.time() > deadline:
                break
                
            # Initialize U, V, W
            # Scale initialization can help
            U = np.random.randn(nm, r) * 0.5
            V = np.random.randn(mp, r) * 0.5
            W = np.random.randn(pn, r) * 0.5
            
            # Flatten
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def res_func(x):
                u = x[:nm*r].reshape(nm, r)
                v = x[nm*r:nm*r+mp*r].reshape(mp, r)
                w = x[nm*r+mp*r:].reshape(pn, r)
                
                # Calculate the reconstructed tensor values for the support
                # We need to check all 5184 entries? That's slow.
                # Let's check the 72 support entries and a random sample of zero entries.
                # Or better, calculate the full tensor contraction?
                # T_recon[i, j, k] = sum_c U[i,c] V[j,c] W[k,c]
                # We can compute this using broadcasting if we are careful, but memory might be an issue for 12x24x18.
                # 12*24*18 = 5184 floats. That's small.
                
                # Compute full tensor
                # u is (nm, r), v is (mp, r), w is (pn, r)
                # We want T[i, j, k] = sum_c u[i,c] * v[j,c] * w[k,c]
                # This is a 3-tensor.
                # We can compute it as:
                # tmp1[i, j, c] = u[i,c] * v[j,c] -> (nm, mp, r)
                # tmp2[i, j, k] = sum_c tmp1[i,j,c] * w[k,c]
                
                # Using einsum
                # 'ic,jc,kc->ijk'
                T_recon = np.einsum('ic,jc,kc->ijk', u, v, w, optimize=True)
                
                # Reshape to 1D to match the flattened index space?
                # The indices in U, V, W are flattened.
                # u[i] corresponds to i in 0..nm-1
                # v[j] corresponds to j in 0..mp-1
                # w[k] corresponds to k in 0..pn-1
                # So T_recon[i, j, k] is the value at index (i, j, k) in the tensor product space.
                
                # Flatten T_recon
                T_flat = T_recon.flatten()
                
                # Target tensor T_target
                # T_target is 0 everywhere, except at support indices where it is 1.
                # We need to map (i,j,k) to flat index.
                # The flattening order in numpy is C-order (row-major) by default.
                # Index = i * (mp * pn) + j * pn + k
                
                # Create target vector
                T_target = np.zeros(total_entries)
                # Map support indices to flat indices
                # support_u is i, support_v is j, support_w is k
                flat_support_idx = support_u * (mp * pn) + support_v * pn + support_w
                T_target[flat_support_idx] = 1.0
                
                return T_flat - T_target

            try:
                # Use least_squares
                # Bounds can help, but let's try unconstrained first
                result = least_squares(res_func, params, method='trf', max_nfev=1000)
                
                if result.success or result.cost < 1e-10:
                    # Extract solution
                    u_sol = result.x[:nm*r].reshape(nm, r)
                    v_sol = result.x[nm*r:nm*r+mp*r].reshape(mp, r)
                    w_sol = result.x[nm*r+mp*r:].reshape(pn, r)
                    
                    # Snap to nearest 0.5
                    u_snap = np.round(u_sol * 2) / 2
                    v_snap = np.round(v_sol * 2) / 2
                    w_snap = np.round(w_sol * 2) / 2
                    
                    # Verify snapped solution
                    T_snap = np.einsum('ic,jc,kc->ijk', u_snap, v_snap, w_snap, optimize=True).flatten()
                    err = np.linalg.norm(T_snap - T_target)
                    
                    if err < 1e-9:
                        # Found a valid decomposition
                        return {
                            "U": u_snap.tolist(),
                            "V": v_snap.tolist(),
                            "W": w_snap.tolist()
                        }
                        
            except Exception:
                pass

    # If no better solution found, return schoolbook
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
