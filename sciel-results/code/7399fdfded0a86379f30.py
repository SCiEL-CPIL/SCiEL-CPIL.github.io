"""Rewritten program using alternating least squares for matrix multiplication tensor decomposition."""
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a low-rank tensor decomposition for multiplying a 3x4 matrix by a 4x8 matrix.
    Uses alternating least squares to solve for U, V, W given a target rank R.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Precompute the index mapping for the tensor entries
    # The tensor T has T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # We need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Store the triplets (u_idx, v_idx, w_idx) where the tensor value is 1
    triplets = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                triplets.append((u_idx, v_idx, w_idx))
    
    num_equations = len(triplets)
    
    def try_rank(R, max_iters=50, n_restarts=5):
        """Try to find a decomposition of rank R using ALS."""
        for restart in range(n_restarts):
            # Initialize with random values
            U = np.random.randn(nm, R)
            V = np.random.randn(mp, R)
            W = np.random.randn(pn, R)
            
            for iteration in range(max_iters):
                # Solve for U: fix V and W
                # For each (i,j), we have sum_k V[j*p+k, r] * W[k*n+i, r] * U[i*m+j, r] = 1
                # This is a linear system for each U row: U[u_idx, :] * M_u = 1
                # where M_u[r, s] = sum over k of V[j*p+k, r] * W[k*n+i, r]
                # Actually, for a fixed (i,j), the equation is:
                # sum_r U[i*m+j, r] * (sum_k V[j*p+k, r] * W[k*n+i, r]) = 1
                # So U[i*m+j, :] * A = 1 where A[r] = sum_k V[j*p+k, r] * W[k*n+i, r]
                
                # We can solve for all U rows simultaneously using least squares
                # Build the system: for each equation (i,j,k), we have
                # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r = 1
                
                # For U: fix V, W. For each (i,j), sum over k:
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for each k
                # This gives p equations for U[i*m+j, :]
                # Let's build a matrix M_u of shape (p, R) for each (i,j)
                # M_u[k, r] = V[j*p+k, r] * W[k*n+i, r]
                # Then M_u @ U[i*m+j, :].T = ones(p)
                
                U_new = np.zeros((nm, R))
                for i in range(n):
                    for j in range(m):
                        M_u = np.zeros((p, R))
                        for k in range(p):
                            v_idx = j * p + k
                            w_idx = k * n + i
                            M_u[k, :] = V[v_idx, :] * W[w_idx, :]
                        # Solve M_u @ x = ones(p)
                        x, residuals, rank_u, sv = np.linalg.lstsq(M_u, np.ones(p), rcond=None)
                        U_new[i * m + j, :] = x
                
                # Solve for V: fix U and W
                # For each (j,k), sum over i:
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for each i
                # M_v[i, r] = U[i*m+j, r] * W[k*n+i, r]
                # M_v @ V[j*p+k, :].T = ones(n)
                
                V_new = np.zeros((mp, R))
                for j in range(m):
                    for k in range(p):
                        M_v = np.zeros((n, R))
                        for i in range(n):
                            u_idx = i * m + j
                            w_idx = k * n + i
                            M_v[i, :] = U_new[u_idx, :] * W[w_idx, :]
                        x, residuals, rank_v, sv = np.linalg.lstsq(M_v, np.ones(n), rcond=None)
                        V_new[j * p + k, :] = x
                
                # Solve for W: fix U and V
                # For each (k,i), sum over j:
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for each j
                # M_w[j, r] = U[i*m+j, r] * V[j*p+k, r]
                # M_w @ W[k*n+i, :].T = ones(m)
                
                W_new = np.zeros((pn, R))
                for k in range(p):
                    for i in range(n):
                        M_w = np.zeros((m, R))
                        for j in range(m):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            M_w[j, :] = U_new[u_idx, :] * V_new[v_idx, :]
                        x, residuals, rank_w, sv = np.linalg.lstsq(M_w, np.ones(m), rcond=None)
                        W_new[k * n + i, :] = x
                
                U, V, W = U_new, V_new, W_new
            
            # Check if the decomposition is valid
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                        error += (val - 1.0) ** 2
            
            if error < 1e-8:
                # Try rounding to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                error_r = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            val = np.dot(U_r[u_idx, :], V_r[v_idx, :] * W_r[w_idx, :])
                            error_r += (val - 1.0) ** 2
                
                if error_r < 1e-8:
                    return {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }
        
        return None
    
    # Try ranks from low to high
    for R in range(41, 100):
        result = try_rank(R)
        if result is not None:
            return result
    
    # Fallback: schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
