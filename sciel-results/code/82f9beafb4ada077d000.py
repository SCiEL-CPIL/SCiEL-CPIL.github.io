import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build the target tensor matrix M_mat (mode-1 unfolding)
    # Size: (n*m) x (m*p * p*n)
    # M_mat[u, v*w_index]
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    # The target tensor has 1 at (u,v,w) if indices match.
    
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    M_mat = np.zeros((size_u, size_v * size_w))
    
    # Populate M_mat
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                # In the flattened mode-1 matrix, the column index is v * size_w + w?
                # Yes, if we flatten (v, w) into a single index.
                M_mat[u, v * size_w + w] = 1.0

    # Helper to verify solution
    def verify(U, V, W):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Reconstruct tensor and check against M_mat
        R = U.shape[1]
        # Compute M_recon_mat
        # M_recon[u, v*w] = sum_r U[u,r] * (V[r] otimes W[r])[v,w]
        # V[r] otimes W[r] is a vector of length size_v * size_w
        # We can compute this as:
        # Temp = np.einsum('ur, vr, wr -> uvw', U, V, W)
        # But it's faster to use matrix multiplication.
        # Let A_VW be a matrix of size (size_v * size_w, R) where column r is V[r] otimes W[r]
        # Actually, V[r] is length size_v, W[r] is length size_w.
        # Kronecker product.
        
        # Efficient check:
        # Reshape M_mat to (size_u, size_v, size_w)
        M_tensor = M_mat.reshape(size_u, size_v, size_w)
        
        # Compute sum of outer products
        # U, V, W are (size, R)
        # Term r: U[:, r] otimes V[:, r] otimes W[:, r]
        
        # We can do this iteratively or with einsum
        # einsum is likely fast enough for small sizes
        recon_tensor = np.einsum('ur,vr,wr->uvw', U, V, W)
        
        # Check exact match
        # Since we expect integer/half-integer, we can check diff
        diff = np.abs(recon_tensor - M_tensor)
        return np.all(diff < 1e-9)

    # ALS Solver
    def solve_als(rank, max_iter=100, tol=1e-12):
        # Initialize
        U = np.random.randn(size_u, rank)
        V = np.random.randn(size_v, rank)
        W = np.random.randn(size_w, rank)
        
        # Precompute M_mat
        # For U update: M_mat (size_u x size_v*size_w)
        # A_VW (size_v*size_w x rank)
        
        for it in range(max_iter):
            # Update U
            # M_mat @ A_VW.T (size_u x rank)
            # A_VW @ A_VW.T (rank x rank)
            # A_VW column r is V[:, r] otimes W[:, r]
            # We can form A_VW efficiently?
            # V is (size_v, rank), W is (size_w, rank)
            # A_VW[:, r] = np.kron(V[:, r], W[:, r])
            
            # Construct A_VW
            # This is a bit expensive to do every iteration if we build it from scratch.
            # But size_v * size_w * rank is small (300 * 60 = 18000).
            
            A_VW = np.zeros((size_v * size_w, rank))
            for r in range(rank):
                A_VW[:, r] = np.kron(V[:, r], W[:, r])
            
            # Solve U = M_mat @ A_VW.T @ inv(A_VW @ A_VW.T)
            # Use least squares
            try:
                U_new = np.linalg.lstsq(A_VW.T, M_mat.T, rcond=None)[0].T
            except:
                return None, None, None, 1.0
                
            # Update V
            # Need mode-2 unfolding of M_mat
            # M_mat is (size_u, size_v * size_w)
            # We need M_mat reshaped to (size_u, size_v, size_w)
            M_tensor = M_mat.reshape(size_u, size_v, size_w)
            # Mode-2 unfolding: (size_v, size_u * size_w)
            # M_mat2[v, u*w]
            M_mat2 = np.transpose(M_tensor, (1, 0, 2)).reshape(size_v, size_u * size_w)
            
            A_UW = np.zeros((size_u * size_w, rank))
            for r in range(rank):
                A_UW[:, r] = np.kron(U_new[:, r], W[:, r])
                
            try:
                V_new = np.linalg.lstsq(A_UW.T, M_mat2.T, rcond=None)[0].T
            except:
                return None, None, None, 1.0

            # Update W
            # Mode-3 unfolding: (size_w, size_u * size_v)
            M_mat3 = np.transpose(M_tensor, (2, 0, 1)).reshape(size_w, size_u * size_v)
            
            A_UV = np.zeros((size_u * size_v, rank))
            for r in range(rank):
                A_UV[:, r] = np.kron(U_new[:, r], V_new[:, r])
                
            try:
                W_new = np.linalg.lstsq(A_UV.T, M_mat3.T, rcond=None)[0].T
            except:
                return None, None, None, 1.0

            U, V, W = U_new, V_new, W_new
            
            # Check convergence
            # Compute residual
            recon_tensor = np.einsum('ur,vr,wr->uvw', U, V, W)
            residual = np.linalg.norm(recon_tensor - M_tensor)
            
            if residual < tol:
                return U, V, W, residual
                
            if time.time() > deadline:
                return U, V, W, residual
                
        return U, V, W, residual

    # Search for rank
    # Try ranks from 30 to 59
    for rank in range(30, rank_schoolbook):
        # Try a few restarts
        for restart in range(5):
            U, V, W, res = solve_als(rank, max_iter=50, tol=1e-10)
            if U is None:
                continue
            if res < 1e-8:
                # Round to nearest 0.5
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                if verify(U_r, V_r, W_r):
                    # Convert to lists
                    return {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }
                    
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    # Fallback to schoolbook
    rank = rank_schoolbook
    U = np.zeros((size_u, rank))
    V = np.zeros((size_v, rank))
    W = np.zeros((size_w, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U[u, col] = 1
                V[v, col] = 1
                W[w, col] = 1
                col += 1
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
