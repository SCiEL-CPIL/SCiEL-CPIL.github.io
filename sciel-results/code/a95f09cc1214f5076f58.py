import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    # Populate schoolbook
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    # Verify schoolbook (should be perfect)
    # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    
    # Try to find rank 74
    target_rank = rank_school - 1
    if target_rank < 1:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
        
    start_time = time.time()
    time_limit = 100 # Leave margin for verification and fallback
    
    # We will try to optimize rank 74
    # Flatten U, V, W into a single vector
    size_u = n * m * target_rank
    size_v = m * p * target_rank
    size_w = p * n * target_rank
    total_size = size_u + size_v + size_w
    
    # Initial guess: Random small noise? Or based on schoolbook?
    # Random initialization is often better for non-convex problems to escape local minima,
    # but structured initialization might be faster.
    # Let's try a structured initialization: take the first 74 schoolbook terms and add noise?
    # No, that leaves one term missing.
    # Better: Random initialization with small variance.
    
    np.random.seed(42)
    x0 = np.random.randn(total_size) * 0.1
    
    def unpack(x):
        u = x[:size_u].reshape(n * m, target_rank)
        v = x[size_u:size_u + size_v].reshape(m * p, target_rank)
        w = x[size_u + size_v:].reshape(p * n, target_rank)
        return u, v, w

    def objective(x):
        u, v, w = unpack(x)
        # Compute tensor T_reconstructed
        # T[i, j, k] = sum_r u[i*m+j, r] * v[j*p+k, r] * w[k*n+i, r]
        # We can vectorize this using einsum
        # u: (n*m, r), v: (m*p, r), w: (p*n, r)
        # Reshape u to (n, m, r), v to (m, p, r), w to (p, n, r)
        u_3d = u.reshape(n, m, target_rank)
        v_3d = v.reshape(m, p, target_rank)
        w_3d = w.reshape(p, n, target_rank)
        
        # T[i, j, k] = sum_r u_3d[i, j, r] * v_3d[j, k, r] * w_3d[k, i, r]
        # Using einsum: 'ijr,jkr,kr i -> ijk' ? No, w is k*n+i, so index is (k, i).
        # w_3d is (p, n, r). So w_3d[k, i, r].
        # T = np.einsum('ijr,jkr,kir->ijk', u_3d, v_3d, w_3d)
        
        # Target tensor is 1 if (i,j,k) is a valid schoolbook term?
        # No, the target tensor T_target has T_target[i,j,k] = 1 if the term exists in the multiplication?
        # The multiplication tensor is defined such that the product C = AB has C[i,k] = sum_j A[i,j]B[j,k].
        # The tensor decomposition represents the bilinear (trilinear) map.
        # The "target" tensor is the identity map on the tensor product space?
        # Actually, the tensor T for matrix multiplication has entries T[i,j,k] such that
        # sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C^*[k,i] = 0?
        # No.
        # The condition is: For any A, B, C, sum_{r} U_{ij,r} V_{jk,r} W_{ki,r} A_{ij} B_{jk} = C_{ik} ... wait.
        # The standard definition:
        # We want to compute C_{ik} = sum_j A_{ij} B_{jk}.
        # The tensor decomposition provides a set of linear forms.
        # The condition for exact decomposition is that the tensor T (where T_{ijk} = 1 if we are computing the term A_{ij}B_{jk} for C_{ik}?)
        # Let's check the schoolbook.
        # U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1.
        # Sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 if there is a c such that all are 1.
        # In schoolbook, for each (i,j,k), there is exactly one c.
        # So the reconstructed tensor R[i,j,k] = 1 for all i,j,k.
        # The "target" tensor is the all-ones tensor of size n x m x p.
        
        target = np.ones((n, m, p))
        reconstructed = np.einsum('ijr,jkr,kir->ijk', u_3d, v_3d, w_3d)
        
        diff = reconstructed - target
        return np.sum(diff * diff)

    # Gradient
    def gradient(x):
        u, v, w = unpack(x)
        u_3d = u.reshape(n, m, target_rank)
        v_3d = v.reshape(m, p, target_rank)
        w_3d = w.reshape(p, n, target_rank)
        
        target = np.ones((n, m, p))
        reconstructed = np.einsum('ijr,jkr,kir->ijk', u_3d, v_3d, w_3d)
        diff = reconstructed - target
        
        # Gradient w.r.t u_3d[i,j,r]:
        # d/du diff^2 = 2 * diff[i,j,k] * v_3d[j,k,r] * w_3d[k,i,r]
        # grad_u_3d[i,j,r] = 2 * sum_k diff[i,j,k] v_3d[j,k,r] w_3d[k,i,r]
        # Using einsum: 'ijk,jkr,kir -> ijr'
        grad_u_3d = 2 * np.einsum('ijk,jkr,kir->ijr', diff, v_3d, w_3d)
        
        # Gradient w.r.t v_3d[j,k,r]:
        # grad_v_3d[j,k,r] = 2 * sum_i diff[i,j,k] u_3d[i,j,r] w_3d[k,i,r]
        # Using einsum: 'ijk,ijr,kir -> jkr'
        grad_v_3d = 2 * np.einsum('ijk,ijr,kir->jkr', diff, u_3d, w_3d)
        
        # Gradient w.r.t w_3d[k,i,r]:
        # grad_w_3d[k,i,r] = 2 * sum_j diff[i,j,k] u_3d[i,j,r] v_3d[j,k,r]
        # Using einsum: 'ijk,ijr,jkr -> kir'
        grad_w_3d = 2 * np.einsum('ijk,ijr,jkr->kir', diff, u_3d, v_3d)
        
        grad_u = grad_u_3d.reshape(size_u)
        grad_v = grad_v_3d.reshape(size_v)
        grad_w = grad_w_3d.reshape(size_w)
        
        return np.concatenate([grad_u, grad_v, grad_w])

    best_result = None
    best_score = float('inf')
    
    # Try a few random restarts if time permits
    max_restarts = 5
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        np.random.seed(42 + restart)
        x0 = np.random.randn(total_size) * (0.1 if restart == 0 else 0.5)
        
        # Use L-BFGS-B
        try:
            res = minimize(objective, x0, jac=gradient, method='L-BFGS-B', options={'maxiter': 500, 'gtol': 1e-8, 'disp': False})
            
            if time.time() - start_time > time_limit:
                break
                
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
                
            if best_score < 1e-8:
                break
        except Exception:
            pass
            
    if best_result is not None and best_score < 1e-6:
        u, v, w = unpack(best_result)
        # Snap to half-integers
        u_snapped = np.round(u * 2) / 2
        v_snapped = np.round(v * 2) / 2
        w_snapped = np.round(w * 2) / 2
        
        # Verify
        u_3d = u_snapped.reshape(n, m, target_rank)
        v_3d = v_snapped.reshape(m, p, target_rank)
        w_3d = w_snapped.reshape(p, n, target_rank)
        reconstructed = np.einsum('ijr,jkr,kir->ijk', u_3d, v_3d, w_3d)
        
        if np.allclose(reconstructed, np.ones((n, m, p)), atol=1e-9):
            return {
                "U": u_snapped.tolist(),
                "V": v_snapped.tolist(),
                "W": w_snapped.tolist()
            }
            
    # Fallback to schoolbook rank 75
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
