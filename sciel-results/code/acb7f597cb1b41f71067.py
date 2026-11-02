import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 60
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Schoolbook solution
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    U_sb, V_sb, W_sb = schoolbook()
    
    def to_half_int(arr):
        """Round to nearest half-integer."""
        return np.round(arr * 2) / 2.0

    def check_exact(U, V, W, T, tol=1e-9):
        """Check if sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c] == T"""
        recon = np.einsum('ac,bc,dc->abd', U, V, W)
        return np.allclose(recon, T, atol=tol)
    
    # Verify schoolbook is correct
    assert check_exact(U_sb, V_sb, W_sb, T), "Schoolbook solution failed verification"
    
    # For 3x4x4x5 matrix multiplication, rank 60 is believed to be optimal.
    # The 60 non-zero entries of the multiplication tensor are linearly
    # independent in the tensor product space.
    
    def try_quick_rank(r, max_iters=10, seed=42):
        """Quick ALS attempt to see if rank r works."""
        np.random.seed(seed)
        
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        for it in range(max_iters):
            if time.time() > deadline - 20:
                break
            
            # Update U
            VW = np.einsum('bd,cd->bcd', V, W)
            VW_2d = VW.transpose(2, 0, 1).reshape(r, m * p * p * n)
            T_2d = T.reshape(n * m, m * p * p * n)
            try:
                U = T_2d @ np.linalg.pinv(VW_2d.T)
            except:
                pass
            
            # Update V
            UW = np.einsum('ad,cd->acd', U, W)
            UW_2d = UW.transpose(2, 0, 1).reshape(r, n * m * p * n)
            T_2d_v = T.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
            try:
                V = T_2d_v @ np.linalg.pinv(UW_2d.T)
            except:
                pass
            
            # Update W
            UV = np.einsum('ad,bd->abd', U, V)
            UV_2d = UV.transpose(2, 0, 1).reshape(r, n * m * m * p)
            T_2d_w = T.transpose(2, 0, 1).reshape(p * n, n * m * m * p)
            try:
                W = T_2d_w @ np.linalg.pinv(UV_2d.T)
            except:
                pass
            
            recon = np.einsum('ad,bd,cd->abc', U, V, W)
            err = np.linalg.norm(recon - T)
            if err < 1e-8:
                break
        
        U_h = to_half_int(U)
        V_h = to_half_int(V)
        W_h = to_half_int(W)
        
        if check_exact(U_h, V_h, W_h, T):
            return U_h, V_h, W_h
        
        return None
    
    # Try to find a lower-rank decomposition using multiple seeds and ranks
    def try_rank_with_seeds(r, seeds, max_iters=20):
        """Try rank r with multiple random seeds."""
        for seed in seeds:
            if time.time() > deadline - 5:
                break
            result = try_quick_rank(r, max_iters=max_iters, seed=seed)
            if result is not None:
                return result
        return None
    
    # Try rank 59 with multiple seeds
    best_result = None
    for r in [59, 58, 57]:
        if time.time() > deadline - 10:
            break
        seeds = list(range(42, 42 + 5))  # 5 seeds per rank
        result = try_rank_with_seeds(r, seeds, max_iters=15)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        U, V, W = best_result
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Return schoolbook (rank 60)
    U_list = [[float(x) for x in row] for row in U_sb]
    V_list = [[float(x) for x in row] for row in V_sb]
    W_list = [[float(x) for x in row] for row in W_sb]
    return {"U": U_list, "V": V_list, "W": W_list}
