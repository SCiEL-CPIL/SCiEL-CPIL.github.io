import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 5, 5]
DEADLINE = 120.0 # Safety margin before 150s

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # The known valid rank 71 decomposition
    # We will try to find rank 70, 69, 68
    # If found, return it. Otherwise return rank 71.
    
    # Pre-compute the target tensor T of shape (n, m, p)
    # T[i, j, k] = 1
    T = np.ones((n, m, p), dtype=float)
    
    # Flattened indices for U, V, W
    # U: (n*m, R) -> index i*m + j
    # V: (m*p, R) -> index j*p + k
    # W: (p*n, R) -> index k*n + i
    
    def error_func(params, R):
        """
        params: flattened array of size (n*m + m*p + p*n) * R
        Returns: squared Frobenius error
        """
        # Reshape
        U_size = n * m
        V_size = m * p
        W_size = p * n
        
        U = params[:U_size * R].reshape(U_size, R)
        V = params[U_size * R : (U_size + V_size) * R].reshape(V_size, R)
        W = params[(U_size + V_size) * R :].reshape(W_size, R)
        
        # Compute tensor sum: sum_r U[:, r] (outer) V[:, r] (outer) W[:, r]
        # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Efficient computation using einsum
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # We want T[i, j, k]
        
        # Let's compute the sum of rank-1 tensors
        # T = sum_r u_r (outer) v_r (outer) w_r
        # u_r is vector of length nm
        # v_r is vector of length mp
        # w_r is vector of length pn
        
        # We can compute this by iterating over r, but that's slow.
        # Better:
        # T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        
        # We can reshape U to (n, m, R)
        U_3d = U.reshape(n, m, R)
        V_3d = V.reshape(m, p, R)
        W_3d = W.reshape(p, n, R)
        
        # T[i, j, k] = sum_r U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
        
        # This is a contraction over r.
        # We can use einsum: 'ijr,jkr,kir->ijk'
        # But we need to be careful with indices.
        
        try:
            T_calc = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d, optimize=True)
        except Exception:
            # Fallback
            T_calc = np.zeros((n, m, p))
            for r in range(R):
                u = U_3d[:, :, r]
                v = V_3d[:, :, r]
                w = W_3d[:, :, r]
                # u[i, j] * v[j, k] * w[k, i]
                # Sum over i, j, k? No, we want the tensor.
                # The term is u[i,j] * v[j,k] * w[k,i]
                # This is not a standard outer product because of the cyclic index i in w.
                # Wait, W is indexed by k*n+i.
                # So W[k, i] is the value.
                # The term is U[i, j] * V[j, k] * W[k, i].
                # This is exactly the trace convention.
                
                # We can compute this as:
                # Temp[j, k, i] = sum_r ... no, r is fixed here.
                # For fixed r, the tensor is u (outer) v (outer) w?
                # u[i, j] * v[j, k] * w[k, i]
                # This is not separable in a simple outer product sense for the whole tensor at once,
                # but for a fixed r, it is a rank-1 tensor in the space of matrices?
                # No, it's a specific 3-way tensor.
                
                # Let's just loop over r for simplicity in the fallback, it's small.
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_calc[i, j, k] += u[i, j] * v[j, k] * w[k, i]
        
        return np.sum((T_calc - T) ** 2)
    
    # Try to find lower rank
    best_result = None
    best_rank = 71
    
    # We will try ranks 70, 69
    for R in [70, 69]:
        if time.time() - start_time > DEADLINE - 20:
            break
            
        # Multiple random restarts
        for restart in range(5):
            if time.time() - start_time > DEADLINE - 10:
                break
                
            # Initialize random parameters
            # Scale: entries are around 1, so init with small random values
            np.random.seed(42 + R + restart)
            U_size = n * m
            V_size = m * p
            W_size = p * n
            total_size = (U_size + V_size + W_size) * R
            
            # Start with random values in [-1, 1]
            x0 = np.random.uniform(-1, 1, total_size)
            
            # Optimize
            # Use L-BFGS-B
            # We need to be careful with the number of iterations
            res = minimize(error_func, x0, args=(R,), method='L-BFGS-B', 
                           options={'maxiter': 50, 'ftol': 1e-6})
            
            if res.fun < 1e-4:
                # Potential solution found. Try to snap to half-integers
                U_size = n * m
                V_size = m * p
                W_size = p * n
                
                U = res.x[:U_size * R].reshape(U_size, R)
                V = res.x[U_size * R : (U_size + V_size) * R].reshape(V_size, R)
                W = res.x[(U_size + V_size) * R :].reshape(W_size, R)
                
                # Snap to half-integers
                # Multiply by 2, round, divide by 2
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                # Verify exactness
                T_calc = np.einsum('ijr,jkr,kir->ijk', 
                                   U_snap.reshape(n, m, R), 
                                   V_snap.reshape(m, p, R), 
                                   W_snap.reshape(p, n, R), 
                                   optimize=True)
                
                if np.allclose(T_calc, T, atol=1e-9):
                    best_rank = R
                    best_result = {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
                    break # Found valid lower rank, stop searching for this R
        
        if best_rank < 71:
            break # Found a lower rank, stop
    
    if best_result:
        return best_result
    else:
        # Fallback to the known rank 71 decomposition
        # Re-implement the rank 71 logic from the prompt to ensure correctness
        total_rank = 71
        U = np.zeros((n * m, total_rank), dtype=float)
        V = np.zeros((m * p, total_rank), dtype=float)
        W = np.zeros((p * n, total_rank), dtype=float)
        
        blocks = [
            (0, 0, 0),
            (0, 2, 0),
            (0, 0, 2),
            (0, 2, 2)
        ]
        
        col = 0
        for (i0, j0, k0) in blocks:
            # Strassen M1
            U[i0*m + j0, col] += 1.0
            U[(i0+1)*m + (j0+1), col] += 1.0
            V[j0*p + k0, col] += 1.0
            V[(j0+1)*p + (k0+1), col] += 1.0
            W[k0*n + i0, col] += 1.0
            W[(k0+1)*n + (i0+1), col] += 1.0
            col += 1
            
            # Strassen M2
            U[(i0+1)*m + j0, col] += 1.0
            U[(i0+1)*m + (j0+1), col] += 1.0
            V[j0*p + k0, col] += 1.0
            W[k0*n + (i0+1), col] += 1.0
            W[(k0+1)*n + (i0+1), col] -= 1.0
            col += 1
            
            # Strassen M3
            U[i0*m + j0, col] += 1.0
            V[j0*p + (k0+1), col] += 1.0
            V[(j0+1)*p + (k0+1), col] -= 1.0
            W[(k0+1)*n + i0, col] += 1.0
            W[(k0+1)*n + (i0+1), col] += 1.0
            col += 1
            
            # Strassen M4
            U[(i0+1)*m + (j0+1), col] += 1.0
            V[(j0+1)*p + k0, col] += 1.0
            V[j0*p + k0, col] -= 1.0
            W[k0*n + i0, col] += 1.0
            W[k0*n + (i0+1), col] += 1.0
            col += 1
            
            # Strassen M5
            U[i0*m + j0, col] += 1.0
            U[i0*m + (j0+1), col] += 1.0
            V[(j0+1)*p + (k0+1), col] += 1.0
            W[k0*n + i0, col] -= 1.0
            W[(k0+1)*n + i0, col] += 1.0
            col += 1
            
            # Strassen M6
            U[(i0+1)*m + j0, col] += 1.0
            U[i0*m + j0, col] -= 1.0
            V[j0*p + k0, col] += 1.0
            V[j0*p + (k0+1), col] += 1.0
            W[(k0+1)*n + (i0+1), col] += 1.0
            col += 1
            
            # Strassen M7
            U[i0*m + (j0+1), col] += 1.0
            U[(i0+1)*m + (j0+1), col] -= 1.0
            V[(j0+1)*p + k0, col] += 1.0
            V[(j0+1)*p + (k0+1), col] += 1.0
            W[k0*n + i0, col] += 1.0
            col += 1
        
        covered = np.zeros((n, m, p), dtype=bool)
        for (i0, j0, k0) in blocks:
            covered[i0:i0+2, j0:j0+2, k0:k0+2] = True
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if not covered[i, j, k]:
                        U[i*m + j, col] = 1.0
                        V[j*p + k, col] = 1.0
                        W[k*n + i, col] = 1.0
                        col += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    
    errors = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(U.shape[1]):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-9:
                    errors += 1
    
    print(f"Verification Errors: {errors}")
    print(f"Rank: {U.shape[1]}")
