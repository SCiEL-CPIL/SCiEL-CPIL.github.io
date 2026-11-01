"""Matrix multiplication 2x3x9: rank-minimal tensor decomposition."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 9]

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3x9 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Try to find a lower rank decomposition using optimization
    # Start from R = 53 down to 50
    for R in range(53, 49, -1):
        if time.time() > deadline:
            break
        result = _try_rank_optimization(n, m, p, R, deadline)
        if result is not None:
            return result
            
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_rank_optimization(n, m, p, R, deadline):
    """Attempt to find a rank R decomposition using least squares."""
    # Initialize with random values
    np.random.seed(42)
    U = np.random.randn(n * m, R)
    V = np.random.randn(m * p, R)
    W = np.random.randn(p * n, R)
    
    # Reshape to 3D tensors for easier handling
    # T_ijk = sum_r U_ir V_jr W_kr
    # We need to match the multiplication tensor T_mult
    
    # Define the target tensor T_mult
    # T_mult[i, j, k] = 1 if j==k? No.
    # C_ik = sum_j A_ij B_jk
    # T[i, j, k] corresponds to A_ij, B_jk, C_ik
    # The tensor T has T[i, j, k] = 1?
    # Actually, the tensor is T[i, j, k] = delta_{i,i'} delta_{j,j'} delta_{k,k'}?
    # No, the tensor is defined by the bilinear map.
    # The entries of the tensor T are T_{i,j,k} where i in n, j in m, k in p.
    # T_{i,j,k} = 1 if the indices match the multiplication pattern.
    # For M_{n,m,p}, T_{i,j,k} = 1 if j is the shared index?
    # Wait, the indices of U, V, W in the prompt are:
    # U: (n*m, R), index a = i*m + j
    # V: (m*p, R), index b = j*p + k
    # W: (p*n, R), index c = k*n + i
    # The contraction is sum_r U[a, r] V[b, r] W[c, r]
    # This should equal 1 if (a,b,c) form a valid multiplication term, 0 otherwise.
    # Valid term: a=(i,j), b=(j,k), c=(k,i).
    # So T[a,b,c] = 1 if b's first index == a's second index, c's first index == b's second index, c's second index == a's first index.
    
    def compute_residual(params):
        U_flat, V_flat, W_flat = params[:n*m*R], params[n*m*R:n*m*R + m*p*R], params[n*m*R + m*p*R:]
        U = U_flat.reshape(n*m, R)
        V = V_flat.reshape(m*p, R)
        W = W_flat.reshape(p*n, R)
        
        # Compute the tensor T = sum_r U_r o V_r o W_r
        # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # We only need to check the non-zero entries of the target tensor.
        # The target tensor has 1s at (i*m+j, j*p+k, k*n+i) for all i,j,k.
        # And 0 elsewhere?
        # Actually, the tensor is sparse.
        # We can compute the residual only on the support.
        
        residual = 0.0
        # Iterate over all valid (i, j, k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    # T_target[a, b, c] = 1
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    residual += (val - 1.0) ** 2
                    
                    # Also need to ensure other entries are 0?
                    # The rank R decomposition defines a tensor T'.
                    # We need T' = T_target.
                    # T_target is zero elsewhere.
                    # Checking all (n*m * m*p * p*n) entries is too slow.
                    # But for small n,m,p, it might be okay?
                    # n*m=6, m*p=27, p*n=18. 6*27*18 = 2916 entries.
                    # R is ~50.
                    # 2916 * 50 = 145800 operations per residual evaluation.
                    # This is fast.
                    
        # Add penalty for non-zero entries elsewhere?
        # If we only minimize the error on the support, we might get a tensor that matches on support but has garbage elsewhere.
        # However, if the rank is low enough, the "garbage" might be zero?
        # No, we need to enforce T' = T_target exactly.
        # So we must minimize the error on ALL entries.
        
        # Let's compute the full tensor T' and compare to T_target.
        # T'[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # This is a 3D tensor of size 6x27x18.
        # We can compute it using einsum.
        T_prime = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Construct T_target
        T_target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T_target[a, b, c] = 1.0
                    
        # Residual is the flattened difference
        return (T_prime - T_target).flatten()

    initial_params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    try:
        res = least_squares(compute_residual, initial_params, method='lm', max_nfev=1000)
        if res.cost < 1e-8:
            # Extract U, V, W
            U_sol = res.x[:n*m*R].reshape(n*m, R)
            V_sol = res.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
            W_sol = res.x[n*m*R + m*p*R:].reshape(p*n, R)
            # Round to integers if close
            U_sol = np.round(U_sol * 2) / 2
            V_sol = np.round(V_sol * 2) / 2
            W_sol = np.round(W_sol * 2) / 2
            
            # Verify
            T_prime = np.einsum('ar,br,cr->abc', U_sol, V_sol, W_sol)
            T_target = np.zeros((n*m, m*p, p*n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        T_target[a, b, c] = 1.0
                        
            if np.allclose(T_prime, T_target, atol=1e-6):
                return {"U": U_sol, "V": V_sol, "W": W_sol}
    except Exception:
        pass
        
    return None

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0]) if result else 'None'}")
