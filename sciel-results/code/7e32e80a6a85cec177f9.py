import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x5 * 5x5 matrix multiplication.
    
    Uses ALS (Alternating Least Squares) with random restarts to find a low-rank
    exact decomposition, then rounds to nearest half-integer and verifies.
    Falls back to schoolbook (rank 50) if no better exact decomposition is found.
    """
    n, m, p = 2, 5, 5
    dim_U = n * m  # 10
    dim_V = m * p  # 25
    dim_W = p * n  # 10
    
    # Build the target tensor T[i, j, k] = 1 if this corresponds to C[i,k] via A[i,j]*B[j,k]
    # T is indexed by (i*m+j, j*p+k, k*n+i) for the flat indices
    # Actually, the tensor M has entries M[u, v, w] where:
    #   u = i*m + j  (A index)
    #   v = j*p + k  (B index)  
    #   w = k*n + i  (C index, trace convention W[k*n+i, r])
    # M[u, v, w] = 1 if u, v, w are consistent (i.e., the j in u matches j in v, and k in v matches k in w)
    
    # For each (i, j, k) in range(n) x range(m) x range(p):
    #   u = i*m + j
    #   v = j*p + k
    #   w = k*n + i
    #   M[u, v, w] = 1
    
    # We'll work with the tensor as a 3D array of size (dim_U, dim_V, dim_W)
    T = np.zeros((dim_U, dim_V, dim_W), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    # Schoolbook solution as fallback
    def schoolbook():
        rank = n * m * p  # 50
        U = np.zeros((dim_U, rank), dtype=np.float64)
        V = np.zeros((dim_V, rank), dtype=np.float64)
        W = np.zeros((dim_W, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    U[u, col] = 1.0
                    V[v, col] = 1.0
                    W[w, col] = 1.0
                    col += 1
        return U, V, W
    
    # Verify a decomposition
    def verify(U, V, W):
        """Check if U, V, W exactly decompose T. Return True if exact."""
        # Reconstruct: for each (u,v,w), sum_r U[u,r]*V[v,r]*W[w,r] should equal T[u,v,w]
        # Efficient check: compute the reconstructed tensor and compare
        # R[u,v,w] = sum_r U[u,r] * V[v,r] * W[w,r]
        # This is a tensor contraction. We can do it as:
        # For each r, outer product of U[:,r], V[:,r], W[:,r]
        # But that's O(rank * dim_U * dim_V * dim_W) which is fine for small dims
        
        # Alternative: use einsum
        # R = np.einsum('ur,vr,wr->uvw', U, V, W)
        # But let's do it more carefully to avoid memory issues
        rank = U.shape[1]
        # Build R by summing rank-1 tensors
        # For efficiency, do it in chunks or use einsum
        R = np.einsum('ur,vr,wr->uvw', U, V, W)
        return np.allclose(R, T, atol=1e-10, rtol=1e-10)
    
    # Round to nearest half-integer
    def round_half(X):
        return np.round(X * 2) / 2.0
    
    # ALS step: fix two matrices, solve for the third
    def solve_U(V, W, rank, T):
        """Solve for U given V and W. U[u,r] = sum_{v,w} T[u,v,w] * V[v,r] * W[w,r] / (sum_{v,w} V[v,r]^2 * W[w,r]^2)"""
        U = np.zeros((dim_U, rank), dtype=np.float64)
        # For each u and r: U[u,r] = sum_{v,w} T[u,v,w] * V[v,r] * W[w,r] / norm
        # Precompute for each r: W_r = W[:,r], V_r = V[:,r]
        for r in range(rank):
            Vr = V[:, r]
            Wr = W[:, r]
            # For each u: numerator = sum_{v,w} T[u,v,w] * Vr[v] * Wr[w]
            # = sum_v Vr[v] * (sum_w T[u,v,w] * Wr[w])
            # = sum_v Vr[v] * (T[u,:,:] @ Wr)
            # T[u,:,:] is (dim_V, dim_W)
            # So for all u: numerator[u] = sum_v Vr[v] * (T[u] @ Wr)[v]
            # = (Vr @ (T @ Wr)) where T @ Wr gives (dim_U, dim_V)
            TW = T @ Wr  # (dim_U, dim_V)
            numerator = Vr @ TW  # (dim_U,)
            # Denominator: sum_{v,w} Vr[v]^2 * Wr[w]^2 = (sum_v Vr[v]^2) * (sum_w Wr[w]^2)
            denom = np.sum(Vr**2) * np.sum(Wr**2)
            if denom > 1e-15:
                U[:, r] = numerator / denom
            else:
                U[:, r] = 0.0
        return U
    
    def solve_V(U, W, rank, T):
        """Solve for V given U and W."""
        V = np.zeros((dim_V, rank), dtype=np.float64)
        for r in range(rank):
            Ur = U[:, r]
            Wr = W[:, r]
            # V[v,r] = sum_{u,w} T[u,v,w] * Ur[u] * Wr[w] / norm
            # = sum_w Wr[w] * (sum_u T[u,v,w] * Ur[u])
            # T is (dim_U, dim_V, dim_W). T[u,v,w] for fixed v,w: sum_u T[u,v,w]*Ur[u] = (Ur @ T[:,:,w])
            # Actually: for fixed v, V[v,r] = sum_{u,w} T[u,v,w] Ur[u] Wr[w]
            # Let's compute: for each w, TW_w[v] = sum_u T[u,v,w] * Ur[u]
            # Then V[v,r] = sum_w TW_w[v] * Wr[w]
            # T[:,:,w] is (dim_U, dim_V). Ur @ T[:,:,w] gives (dim_V,)
            TW = np.zeros((dim_V,), dtype=np.float64)
            for w in range(dim_W):
                TW += Ur @ T[:, :, w] * Wr[w]
            V[:, r] = TW
            # Denominator: sum_{u,w} Ur[u]^2 * Wr[w]^2 = (sum_u Ur[u]^2) * (sum_w Wr[w]^2)
            denom = np.sum(Ur**2) * np.sum(Wr**2)
            if denom > 1e-15:
                V[:, r] /= denom
            else:
                V[:, r] = 0.0
        return V
    
    def solve_W(U, V, rank, T):
        """Solve for W given U and V."""
        W = np.zeros((dim_W, rank), dtype=np.float64)
        for r in range(rank):
            Ur = U[:, r]
            Vr = V[:, r]
            # W[w,r] = sum_{u,v} T[u,v,w] * Ur[u] * Vr[v] / norm
            # = sum_u Ur[u] * (sum_v T[u,v,w] * Vr[v])
            # For fixed u: TV_u[w] = sum_v T[u,v,w] * Vr[v]
            # Then W[w,r] = sum_u Ur[u] * TV_u[w]
            TW = np.zeros((dim_W,), dtype=np.float64)
            for u in range(dim_U):
                TW += Ur[u] * (T[u, :, :] @ Vr)
            W[:, r] = TW
            # Denominator: sum_{u,v} Ur[u]^2 * Vr[v]^2 = (sum_u Ur[u]^2) * (sum_v Vr[v]^2)
            denom = np.sum(Ur**2) * np.sum(Vr**2)
            if denom > 1e-15:
                W[:, r] /= denom
            else:
                W[:, r] = 0.0
        return W
    
    # Compute reconstruction error
    def compute_error(U, V, W):
        R = np.einsum('ur,vr,wr->uvw', U, V, W)
        return np.max(np.abs(R - T))
    
    # ALS optimization
    def als_optimize(U, V, W, max_iters=500, tol=1e-12):
        """Run ALS to minimize reconstruction error."""
        rank = U.shape[1]
        prev_err = compute_error(U, V, W)
        for it in range(max_iters):
            U = solve_U(V, W, rank, T)
            V = solve_V(U, W, rank, T)
            W = solve_W(U, V, rank, T)
            err = compute_error(U, V, W)
            if abs(prev_err - err) < tol:
                break
            prev_err = err
        return U, V, W, prev_err
    
    # Time management
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin before 150s kill
    
    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p  # 50
    
    # Try various ranks from low to high
    # Theoretical lower bound: max(dim_U, dim_V, dim_W) = max(10, 25, 10) = 25
    # But for matrix multiplication, the rank is at least n*m*p / something...
    # Let's try ranks from 25 up to 49
    random.seed(42)
    
    ranks_to_try = list(range(25, min(50, best_rank)))
    # Also try some specific ranks that might be good
    ranks_to_try = sorted(set(ranks_to_try + [30, 35, 40, 45, 48]))
    ranks_to_try = [r for r in ranks_to_try if r < best_rank]
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Multiple restarts for this rank
        n_restarts = 5
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(dim_U, rank) * 0.1
            V = np.random.randn(dim_V, rank) * 0.1
            W = np.random.randn(dim_W, rank) * 0.1
            
            # Run ALS
            U, V, W, err = als_optimize(U, V, W, max_iters=300)
            
            # Check if exact
            if err < 1e-10:
                # Try rounding to half-integers
                U_r = round_half(U)
                V_r = round_half(V)
                W_r = round_half(W)
                if verify(U_r, V_r, W_r):
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = rank
                    break
                # Also try without rounding
                if verify(U, V, W):
                    best_U, best_V, best_W = U, V, W
                    best_rank = rank
                    break
            else:
                # Try rounding even if not perfectly exact from ALS
                U_r = round_half(U)
                V_r = round_half(V)
                W_r = round_half(W)
                if verify(U_r, V_r, W_r):
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = rank
                    break
    
    # If we found a better rank, use it; otherwise use schoolbook
    if best_rank < 50:
        U, V, W = best_U, best_V, best_W
    else:
        U, V, W = schoolbook()
    
    # Convert to lists of lists (as required by the interface)
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
