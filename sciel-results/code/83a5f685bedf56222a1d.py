"""Matrix multiplication 3,4,7 - tensor decomposition with rank minimization."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]
DEADLINE = 145.0
START_TIME = time.time()

def time_remaining():
    return DEADLINE - (time.time() - START_TIME)

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 84
    
    def schoolbook():
        U = np.zeros((n * m, rank_schoolbook), dtype=np.int64)
        V = np.zeros((m * p, rank_schoolbook), dtype=np.int64)
        W = np.zeros((p * n, rank_schoolbook), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    def build_tensor():
        T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1
        return T
    
    T = build_tensor()
    
    def verify(U, V, W):
        """Verify the decomposition is exact."""
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    pred = float(np.dot(U[a], V[b] * W[c]))
                    if abs(T[a, b, c] - pred) > 1e-6:
                        return False
        return True
    
    def try_als(target_rank, max_iter=20, seed=42):
        """Try ALS and round to integers."""
        if time_remaining() < 20:
            return None
        
        np.random.seed(seed)
        U = np.random.randn(n * m, target_rank)
        V = np.random.randn(m * p, target_rank)
        W = np.random.randn(p * n, target_rank)
        
        for it in range(max_iter):
            if time_remaining() < 20:
                return None
            
            M_U = (V[:, :, None] * W[None, :, :]).transpose(2, 0, 1).reshape(target_rank, -1)
            for a in range(n * m):
                U[a], _, _, _ = np.linalg.lstsq(M_U, T[a].ravel(), rcond=None)
            
            M_V = (U[:, :, None] * W[None, :, :]).transpose(2, 0, 1).reshape(target_rank, -1)
            for b in range(m * p):
                V[b], _, _, _ = np.linalg.lstsq(M_V, T[:, b, :].ravel(), rcond=None)
            
            M_W = (U[:, :, None] * V[None, :, :]).transpose(2, 0, 1).reshape(target_rank, -1)
            for c in range(p * n):
                W[c], _, _, _ = np.linalg.lstsq(M_W, T[:, :, c].ravel(), rcond=None)
        
        # Try rounding to integers
        U_int = np.round(U).astype(np.int64)
        V_int = np.round(V).astype(np.int64)
        W_int = np.round(W).astype(np.int64)
        
        if verify(U_int, V_int, W_int):
            return U_int, V_int, W_int
        
        # Try rounding to half-integers
        U_half = np.round(U * 2) / 2
        V_half = np.round(V * 2) / 2
        W_half = np.round(W * 2) / 2
        
        if verify(U_half, V_half, W_half):
            return U_half, V_half, W_half
        
        return None
    
    # Try ranks from 83 down to 70
    for target_rank in range(rank_schoolbook - 1, 69, -1):
        if time_remaining() < 25:
            break
        
        result = try_als(target_rank, max_iter=15)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    U, V, W = schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
