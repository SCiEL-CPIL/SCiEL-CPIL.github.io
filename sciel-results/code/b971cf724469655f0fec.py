"""Bootstrap direct approach for 2x3x7 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need T[i,j,k] = 1 if (i,j,k) corresponds to a multiplication term, else 0
    # Actually, the tensor for matrix multiplication has T[i,j,k] = 1 for all valid (i,j,k)
    # because C[i,k] = sum_j A[i,j] B[j,k]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_rank_decomposition(n, m, p, target_rank, max_iters=500, seed=42):
    """Try to find a decomposition of given rank using random initialization and Newton-like refinement."""
    np.random.seed(seed)
    
    # Initialize with small random values
    U = np.random.randn(n * m, target_rank) * 0.5
    V = np.random.randn(m * p, target_rank) * 0.5
    W = np.random.randn(p * n, target_rank) * 0.5
    
    # Reshape for vectorized operations
    U_3d = U.reshape(n, m, target_rank)
    V_3d = V.reshape(m, p, target_rank)
    W_3d = W.reshape(p, n, target_rank)
    
    ones_p = np.ones(p)
    ones_n = np.ones(n)
    ones_m = np.ones(m)
    
    for it in range(max_iters):
        # 1. Solve for U
        # For each i, j: M[i,j] @ U[i,j,:]^T = ones(p)
        # M[i,j][k,r] = V[j,k,r] * W[k,i,r]
        U_new_3d = np.zeros_like(U_3d)
        for i in range(n):
            for j in range(m):
                # V[j, :, :] is (p, R)
                # W[:, i, :] is (p, R)
                M = V_3d[j, :, :] * W_3d[:, i, :]
                try:
                    u, _, _, _ = np.linalg.lstsq(M, ones_p, rcond=None)
                    U_new_3d[i, j, :] = u
                except:
                    U_new_3d[i, j, :] = U_3d[i, j, :]
        
        # 2. Solve for V
        # For each j, k: M[j,k] @ V[j,k,:]^T = ones(n)
        # M[j,k][i,r] = U[i,j,r] * W[k,i,r]
        V_new_3d = np.zeros_like(V_3d)
        for j in range(m):
            for k in range(p):
                # U[:, j, :] is (n, R)
                # W[k, :, :] is (n, R)
                M = U_3d[:, j, :] * W_3d[k, :, :]
                try:
                    v, _, _, _ = np.linalg.lstsq(M, ones_n, rcond=None)
                    V_new_3d[j, k, :] = v
                except:
                    V_new_3d[j, k, :] = V_3d[j, k, :]
        
        # 3. Solve for W
        # For each k, i: M[k,i] @ W[k,i,:]^T = ones(m)
        # M[k,i][j,r] = U[i,j,r] * V[j,k,r]
        W_new_3d = np.zeros_like(W_3d)
        for k in range(p):
            for i in range(n):
                # U[i, :, :] is (m, R)
                # V[:, k, :] is (m, R)
                M = U_3d[i, :, :] * V_3d[:, k, :]
                try:
                    w, _, _, _ = np.linalg.lstsq(M, ones_m, rcond=None)
                    W_new_3d[k, i, :] = w
                except:
                    W_new_3d[k, i, :] = W_3d[k, i, :]
        
        U_3d, V_3d, W_3d = U_new_3d, V_new_3d, W_new_3d
        U, V, W = U_3d.reshape(n*m, target_rank), V_3d.reshape(m*p, target_rank), W_3d.reshape(p*n, target_rank)
        
        # Check convergence
        # Compute T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        max_err = np.max(np.abs(T - 1.0))
        
        if max_err < 1e-9:
            # Snap to rational values
            # Try scales
            for scale in [1, 2, 4, 8, 16, 32, 64]:
                U_snap = np.round(U * scale) / scale
                V_snap = np.round(V * scale) / scale
                W_snap = np.round(W * scale) / scale
                
                # Verify
                if _verify_decomposition(
                    U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), n, m, p
                ):
                    return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
        
        # Early termination if error is not decreasing or is too high
        if it > 10 and max_err > 10.0:
            break
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Schoolbook as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb

    # Try various target ranks from low to high
    # Lower bound is max(2*7, 3*7, 2*3) = 21
    # Known constructions might go lower.
    target_ranks = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 20:  # Keep 20s buffer
            break
        
        # Try multiple random seeds
        # Increase seeds for lower ranks as they are harder
        num_seeds = 10 if target_rank < 21 else 5
        
        for seed in range(num_seeds):
            if time.time() > deadline - 10:
                break
            
            result = _try_rank_decomposition(n, m, p, target_rank, max_iters=500, seed=seed)
            if result is not None:
                U_res, V_res, W_res = result
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_result = {"U": U_res, "V": V_res, "W": W_res}
                    # If we found a good one, we can stop early
                    if target_rank <= 20:
                        break
        if target_rank <= 20 and best_rank == target_rank:
            break

    return best_result
# EVOLVE-BLOCK-END
