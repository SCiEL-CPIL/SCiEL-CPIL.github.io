import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def _verify(U, V, W, n, m, p):
    """Return True iff the decomposition is exact."""
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    T = np.einsum('ar,br,cr->abc', Ua, Va, Wa)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i * m + j, j * p + k, k * n + i] - 1.0) > 1e-9:
                    return False
    T2 = T.copy()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T2[i * m + j, j * p + k, k * n + i] = 0.0
    if np.max(np.abs(T2)) > 1e-9:
        return False
    return True

def _build_schoolbook(n, m, p):
    """Guaranteed rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    rank = 135
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    tensor_dim = n * m * p

    # Precompute the mapping from tensor index (i,j,k) to linear indices in U, V, W
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    # We need to solve for coefficients in U, V, W such that
    # sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
    # This is a bilinear system, which is hard.
    # However, the directive says: "solving the resulting linear system for the coefficients using Gaussian elimination"
    # This implies fixing the support pattern (which entries are non-zero) and then the system becomes linear?
    # No, matrix multiplication tensor decomposition is inherently nonlinear (trilinear).
    # Unless... we fix two of the matrices and solve for the third?
    # If we fix U and V, then W is determined by:
    # W[c, r] = sum_{a,b} T[a,b,c] U[a,r] V[b,r] / (sum_{a,b} U[a,r]^2 V[b,r]^2) ? No, that's least squares.
    #
    # Let's re-read the directive carefully: "sampling sparse support patterns for the factor matrices U, V, and W with a fixed rank of 135, then solving the resulting linear system for the coefficients using Gaussian elimination to find an exact integer or half-integer solution"
    #
    # This strategy is theoretically flawed for general matrix multiplication because the tensor is not linear in the factors simultaneously.
    # However, if we assume a specific structure, e.g., we fix U and V to be sparse random matrices, we can solve for W in a least-squares sense, but exactness is not guaranteed.
    #
    # Wait, is there a way to make it linear?
    # If we fix the *values* of U and V, then for each rank term r, we have:
    # T[:, :, c] = sum_r W[c, r] * (U[:, r] outer V[:, r])
    # This is a linear system for W[c, r] if U and V are fixed!
    # Yes! If we fix U and V, the equation is linear in W.
    # So the strategy is:
    # 1. Randomly generate sparse U and V (e.g., entries in {-1, 0, 1}).
    # 2. Construct the linear system for W.
    # 3. Solve for W using Gaussian elimination (or least squares, but we want exact).
    # 4. Check if W is "nice" (integer/half-integer) and if the decomposition is exact.
    #
    # Let's implement this.
    # We need to solve: sum_r W[c, r] * (U[:, r] outer V[:, r]) = T[:, :, c] for each c.
    # Actually, it's a single big linear system for all W entries.
    # Let M be a matrix of size (n*m*p) x (rank * p).
    # The unknowns are W[c, r] for c in 0..p-1, r in 0..rank-1.
    # The equations are for each (i, j, k) in the tensor.
    # Equation (i, j, k): sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = T[i, j, k]
    #
    # This is still nonlinear because W is multiplied by U and V.
    # BUT, if we fix U and V, then for a fixed r, the term is U[i*m+j, r] V[j*p+k, r] * W[k*n+i, r].
    # This is linear in W[k*n+i, r].
    # So yes, if U and V are fixed, it is a linear system for W.
    #
    # Let's try this.
    
    best_result = _build_schoolbook(n, m, p)
    
    for _ in range(100): # Limit iterations to fit in time
        if time.time() > deadline - 10:
            break
            
        # Generate sparse random U and V
        # Use {-1, 0, 1} with some density
        U = np.zeros((U_dim, rank), dtype=float)
        V = np.zeros((V_dim, rank), dtype=float)
        
        # Randomly set entries
        for r in range(rank):
            # Pick a few non-zero entries
            num_nonzero_u = np.random.randint(1, 4)
            indices_u = np.random.choice(U_dim, num_nonzero_u, replace=False)
            U[indices_u, r] = np.random.choice([-1, 1], num_nonzero_u)
            
            num_nonzero_v = np.random.randint(1, 4)
            indices_v = np.random.choice(V_dim, num_nonzero_v, replace=False)
            V[indices_v, r] = np.random.choice([-1, 1], num_nonzero_v)
        
        # Construct the linear system for W
        # Unknowns: W[k*n+i, r] for all k, i, r. Total unknowns: p * n * rank
        # Equations: for all i, j, k. Total equations: n * m * p
        
        A = np.zeros((tensor_dim, W_dim * rank))
        b = np.zeros(tensor_dim)
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    b[idx] = 1.0 # Target tensor value is 1 for (i,j,k)
                    
                    for r in range(rank):
                        u_val = U[i*m + j, r]
                        v_val = V[j*p + k, r]
                        if u_val != 0 and v_val != 0:
                            # Coefficient for W[k*n+i, r]
                            col_idx = (k*n + i) * rank + r
                            A[idx, col_idx] = u_val * v_val
                    idx += 1
        
        # Solve A * w_flat = b
        # We want an exact solution. If the system is underdetermined (more unknowns than equations),
        # we might find a solution.
        # n*m*p = 144 equations.
        # W_dim * rank = 36 * 135 = 4860 unknowns.
        # Highly underdetermined. We can use least squares, but we need exactness.
        # If a solution exists, it might have rational entries.
        
        try:
            # Use pinv for a quick check, or solve least squares
            w_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            
            # Check if it's an exact solution
            if np.max(np.abs(A @ w_flat - b)) > 1e-9:
                continue
                
            # Reshape w_flat to W
            W = w_flat.reshape(W_dim, rank)
            
            # Check if W entries are "nice" (close to integer or half-integer)
            # The directive says "integer or half-integer".
            # Let's check if 2*W is close to integer.
            W_scaled = 2 * W
            W_rounded = np.round(W_scaled)
            if np.max(np.abs(W_scaled - W_rounded)) > 1e-6:
                continue
            
            W_final = W_rounded / 2.0
            
            # Verify the full decomposition with U, V, W_final
            # U and V are integers, W_final is half-integer.
            # We need to return U, V, W as lists of lists.
            
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W_final.tolist()
            
            if _verify(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
                
        except Exception:
            continue
            
    return best_result
