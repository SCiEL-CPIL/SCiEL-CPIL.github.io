"""Program: rank-28 exact tensor decomposition for 2x4 * 4x4 matrix multiplication.
Attempts to find a lower rank decomposition (e.g., 24) using optimization, falling back to Strassen rank 28."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 4]

def _verify(U, V, W):
    """Verify if the decomposition is exact."""
    n, m, p = DIMENSIONS
    R = len(U[0]) if U else 0
    # Convert to numpy for speed
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Check shapes
    if U_np.shape != (n*m, R) or V_np.shape != (m*p, R) or W_np.shape != (p*n, R):
        return False
        
    # Check if entries are close to half-integers
    # The problem says "integer/half-integer entries". 
    # We allow a small tolerance for the verification of the snapped values.
    # But for the "exact" check, we should use the snapped values.
    # Let's assume the input U, V, W are already snapped or we check them here.
    
    # Reconstruct the tensor
    # T[i, j, k, l] = sum_r W[r, k*n+i] * U[r, i*m+j] * V[r, j*p+k]
    # We want T[i, j, k, l] to be 1 if j==l, 0 otherwise.
    # Wait, the indices in the prompt:
    # U is (n*m, R). Row i*m+j.
    # V is (m*p, R). Row j*p+k.
    # W is (p*n, R). Row k*n+i.
    
    # Let's compute the product for all i, j, k, l
    # Actually, we can compute the full tensor contraction.
    # C[i, k] = sum_j A[i, j] B[j, k]
    # Decomposition: C[i, k] = sum_r W[r, k*n+i] * (sum_j U[r, i*m+j] A[i,j]) * (sum_l V[r, l*p+k] B[l,k])
    # Coeff of A[i, j] B[l, k] is sum_r W[r, k*n+i] U[r, i*m+j] V[r, l*p+k]
    # This must be delta_{j, l}.
    
    # We can check this by iterating over i, k, j, l
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for l in range(m):
                    target = 1 if j == l else 0
                    val = 0
                    for r in range(R):
                        val += W_np[r, k*n+i] * U_np[r, i*m+j] * V_np[r, l*p+k]
                    if abs(val - target) > 1e-6:
                        return False
    return True

def _snap_to_half_integers(U, V, W):
    """Snap values to nearest half-integer and verify."""
    def snap_matrix(M):
        M_snapped = np.round(np.array(M) * 2) / 2.0
        return M_snapped
        
    U_s = snap_matrix(U)
    V_s = snap_matrix(V)
    W_s = snap_matrix(W)
    
    # Convert to list of lists for return
    U_list = U_s.tolist()
    V_list = V_s.tolist()
    W_list = W_s.tolist()
    
    if _verify(U_list, V_list, W_list):
        return U_list, V_list, W_list
    else:
        return None

def construct():
    """
    Construct a rank-minimal exact tensor decomposition.
    Tries to find rank 24, 25, 26, 27. Falls back to 28.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Fallback: Strassen rank 28
    def get_strassen():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_strassen_block(u_idx, v_idx, w_idx, col_offset):
            # M1 = (A00 + A11)(B00 + B11)
            U[u_idx[0]][col_offset + 0] += 1
            U[u_idx[3]][col_offset + 0] += 1
            V[v_idx[0]][col_offset + 0] += 1
            V[v_idx[3]][col_offset + 0] += 1
            W[w_idx[0]][col_offset + 0] += 1
            W[w_idx[3]][col_offset + 0] += 1
            
            # M2 = (A10 + A11)B00
            U[u_idx[2]][col_offset + 1] += 1
            U[u_idx[3]][col_offset + 1] += 1
            V[v_idx[0]][col_offset + 1] += 1
            W[w_idx[2]][col_offset + 1] += 1
            W[w_idx[3]][col_offset + 1] -= 1
            
            # M3 = A00(B01 - B11)
            U[u_idx[0]][col_offset + 2] += 1
            V[v_idx[1]][col_offset + 2] += 1
            V[v_idx[3]][col_offset + 2] -= 1
            W[w_idx[1]][col_offset + 2] += 1
            W[w_idx[3]][col_offset + 2] += 1
            
            # M4 = A11(B10 - B00)
            U[u_idx[3]][col_offset + 3] += 1
            V[v_idx[2]][col_offset + 3] += 1
            V[v_idx[0]][col_offset + 3] -= 1
            W[w_idx[0]][col_offset + 3] += 1
            W[w_idx[2]][col_offset + 3] += 1
            
            # M5 = (A00 + A01)B11
            U[u_idx[0]][col_offset + 4] += 1
            U[u_idx[1]][col_offset + 4] += 1
            V[v_idx[3]][col_offset + 4] += 1
            W[w_idx[0]][col_offset + 4] -= 1
            W[w_idx[1]][col_offset + 4] += 1
            
            # M6 = (A10 - A00)(B00 + B01)
            U[u_idx[2]][col_offset + 5] += 1
            U[u_idx[0]][col_offset + 5] -= 1
            V[v_idx[0]][col_offset + 5] += 1
            V[v_idx[1]][col_offset + 5] += 1
            W[w_idx[3]][col_offset + 5] += 1
            
            # M7 = (A01 - A11)(B10 + B11)
            U[u_idx[1]][col_offset + 6] += 1
            U[u_idx[3]][col_offset + 6] -= 1
            V[v_idx[2]][col_offset + 6] += 1
            V[v_idx[3]][col_offset + 6] += 1
            W[w_idx[0]][col_offset + 6] += 1
        
        # Block 1
        u1 = [0, 1, 4, 5]
        v1 = [0, 1, 4, 5]
        w1 = [0, 2, 1, 3]
        add_strassen_block(u1, v1, w1, 0)
        
        # Block 2
        u2 = [2, 3, 6, 7]
        v2 = [8, 9, 12, 13]
        w2 = [0, 2, 1, 3]
        add_strassen_block(u2, v2, w2, 7)
        
        # Block 3
        u3 = [0, 1, 4, 5]
        v3 = [2, 3, 6, 7]
        w3 = [4, 6, 5, 7]
        add_strassen_block(u3, v3, w3, 14)
        
        # Block 4
        u4 = [2, 3, 6, 7]
        v4 = [10, 11, 14, 15]
        w4 = [4, 6, 5, 7]
        add_strassen_block(u4, v4, w4, 21)
        
        return {"U": U, "V": V, "W": W}

    def try_rank(R, max_restarts=5):
        """Try to find a decomposition of rank R."""
        if time.time() > deadline:
            return None
            
        # Variables: U (n*m*R), V (m*p*R), W (p*n*R)
        size_u = n * m * R
        size_v = m * p * R
        size_w = p * n * R
        total_size = size_u + size_v + size_w
        
        def objective(x):
            U_flat = x[:size_u]
            V_flat = x[size_u:size_u+size_v]
            W_flat = x[size_u+size_v:]
            
            U = U_flat.reshape(n*m, R)
            V = V_flat.reshape(m*p, R)
            W = W_flat.reshape(p*n, R)
            
            # Compute residuals
            # Residual for (i, k, j, l) is sum_r W[r, k*n+i] U[r, i*m+j] V[r, l*p+k] - delta_{j,l}
            # We can vectorize this.
            # T[i, k, j, l] = sum_r W[r, k*n+i] U[r, i*m+j] V[r, l*p+k]
            
            # Reshape for einsum?
            # U: (i, j, r)
            # V: (l, k, r)
            # W: (k, i, r)
            
            U_3d = U.reshape(n, m, R)
            V_3d = V.reshape(m, p, R)
            W_3d = W.reshape(p, n, R)
            
            # T[i, k, j, l] = sum_r W_3d[k, i, r] U_3d[i, j, r] V_3d[l, k, r]
            # Note: V index is l*p+k, so V[l, k, r].
            # U index is i*m+j, so U[i, j, r].
            # W index is k*n+i, so W[k, i, r].
            
            # np.einsum('kir,ijr,lkr->ikjl', W_3d, U_3d, V_3d)
            # Wait, V is (m, p, R) which is (l, k, r).
            # So 'kir,ijr,lkr->ikjl'
            T = np.einsum('kir,ijr,lkr->ikjl', W_3d, U_3d, V_3d)
            
            # Target: delta_{j,l} for all i,k
            # Target[i, k, j, l] = 1 if j==l else 0
            # We can create a target tensor of shape (n, p, m, m)
            # Target[i, k, j, l] = I[j, l]
            I = np.eye(m)
            Target = np.zeros((n, p, m, m))
            for i in range(n):
                for k in range(p):
                    Target[i, k, :, :] = I
                    
            diff = T - Target
            return np.sum(diff**2)
            
        best_result = None
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Random initialization
            np.random.seed(42 + restart)
            x0 = np.random.randn(total_size) * 0.5
            
            res = minimize(objective, x0, method='BFGS', options={'maxiter': 200, 'gtol': 1e-8})
            
            if res.success and res.fun < 1e-6:
                # Extract U, V, W
                U_flat = res.x[:size_u]
                V_flat = res.x[size_u:size_u+size_v]
                W_flat = res.x[size_u+size_v:]
                
                U = U_flat.reshape(n*m, R).tolist()
                V = V_flat.reshape(m*p, R).tolist()
                W = W_flat.reshape(p*n, R).tolist()
                
                # Try to snap
                snapped = _snap_to_half_integers(U, V, W)
                if snapped:
                    return snapped
                else:
                    # If snapping fails, maybe we can return the real values?
                    # The prompt says "integer/half-integer entries".
                    # So we MUST return integer/half-integer.
                    # If we can't snap, this solution is invalid for the interface.
                    pass
                    
        return None

    # Try ranks from 24 to 27
    for R in [24, 25, 26, 27]:
        result = try_rank(R)
        if result:
            return result
            
    # Fallback
    return get_strassen()
