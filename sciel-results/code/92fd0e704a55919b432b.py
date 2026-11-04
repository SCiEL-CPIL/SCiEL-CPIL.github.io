import time
import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    nmp = n * m * p  # 84
    
    # Schoolbook fallback (always valid, rank 84)
    def _schoolbook():
        rank = nmp
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
    
    # Verification: check that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    # for all i,j,k, and entries are integer or half-integer
    def _verify(U, V, W):
        rank = len(U[0])
        # Check entries are integer or half-integer
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if abs(val * 2 - round(val * 2)) > 1e-9:
                        return False
        # Check tensor decomposition
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    # Strategy: Try to find a decomposition with rank < 84
    # The lower bound is max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    # But realistic lower bounds for M_{3,4,7} are higher.
    # 
    # Approach: Use structured random search with numpy for speed.
    # Initialize with sparse random ±1/2 entries, then verify.
    # Also try a "Strassen-like" block decomposition approach.
    
    # First, try a structured approach: 
    # For M_{3,4,7}, we can try to exploit the structure.
    # One idea: decompose M_{3,4,7} = M_{3,4,4} ⊗ M_{1,1,7} or similar,
    # but that doesn't help directly.
    #
    # Better: Use random search with numpy vectorized verification.
    
    # Precompute indices for fast verification
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need T[i,j,k] = 1 for all i,j,k
    
    # Vectorized verification using numpy
    def _verify_np(U_np, V_np, W_np):
        """U_np: (n*m, rank), V_np: (m*p, rank), W_np: (p*n, rank)"""
        rank = U_np.shape[1]
        # Reshape for broadcasting
        # U[i*m+j, c] -> shape (n, m, rank)
        U_r = U_np.reshape(n, m, rank)
        # V[j*p+k, c] -> shape (m, p, rank)
        V_r = V_np.reshape(m, p, rank)
        # W[k*n+i, c] -> shape (p, n, rank)
        W_r = W_np.reshape(p, n, rank)
        
        # For each (i,j,k): sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # U_r[i,j,:] * V_r[j,k,:] * W_r[k,i,:]
        # Need to broadcast: 
        # U_r[i, j, c], V_r[j, k, c], W_r[k, i, c]
        # Result shape: (n, m, p)
        
        # U_r: (n, m, rank) -> (n, m, 1, rank)
        # V_r: (m, p, rank) -> (1, m, p, rank)
        # W_r: (p, n, rank) -> (p, n, 1, rank) -> transpose to (n, 1, p, rank)
        
        U_b = U_r[:, :, np.newaxis, :]  # (n, m, 1, rank)
        V_b = V_r[np.newaxis, :, :, :]  # (1, m, p, rank)
        W_b = W_r.T[:, np.newaxis, :, :]  # W_r is (p, n, rank), W_r.T is (rank, p, n)... 
        # Actually W_r[k*n+i, c] so W_r reshaped is (p, n, rank)
        # We need W[k, i, c] for fixed (i, j, k)
        # W_b should be (n, 1, p, rank) where W_b[i, 0, k, c] = W[k*n+i, c]
        W_b = W_r.T  # (rank, p, n) -> no
        # W_r is (p, n, rank). We want (n, p, rank) with W_r[k, i, c] -> (i, k, c)
        W_b = np.transpose(W_r, (1, 0, 2))  # (n, p, rank)
        W_b = W_b[:, np.newaxis, :, :]  # (n, 1, p, rank)
        
        # Product: (n, m, p, rank)
        prod = U_b * V_b * W_b
        total = np.sum(prod, axis=3)  # (n, m, p)
        
        # Check all entries are 1
        if not np.allclose(total, 1.0, atol=1e-9):
            return False
        
        # Check half-integer constraint
        for mat in [U_np, V_np, W_np]:
            if not np.allclose(mat * 2, np.round(mat * 2), atol=1e-9):
                return False
        return True
    
    # Try random search for rank < 84
    # Use sparse initialization for better chance
    np.random.seed(42)
    
    best_result = None
    best_rank = nmp
    
    # Try ranks from a reasonable lower bound up to 83
    # Lower bound: max(n*m, m*p, p*n) = 28, but let's start higher
    # For M_{3,4,7}, known results suggest rank is around 40-50
    
    for target_rank in range(30, nmp):
        if time.time() > deadline - 20:
            break
        
        # Number of attempts: more for smaller ranks
        max_attempts = 50 if target_rank < 50 else 20
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Initialize with sparse random entries
            U_np = np.zeros((n * m, target_rank))
            V_np = np.zeros((m * p, target_rank))
            W_np = np.zeros((p * n, target_rank))
            
            # Sparse initialization: each row has a few non-zero entries
            for c in range(target_rank):
                # Each column of U, V, W gets a few non-zero entries
                # Choose random positions
                u_idx = np.random.choice(n * m, size=min(3, n * m), replace=False)
                v_idx = np.random.choice(m * p, size=min(3, m * p), replace=False)
                w_idx = np.random.choice(p * n, size=min(3, p * n), replace=False)
                
                for idx in u_idx:
                    U_np[idx, c] = np.random.choice([-1, 1]) * np.random.choice([0.5, 1.0])
                for idx in v_idx:
                    V_np[idx, c] = np.random.choice([-1, 1]) * np.random.choice([0.5, 1.0])
                for idx in w_idx:
                    W_np[idx, c] = np.random.choice([-1, 1]) * np.random.choice([0.5, 1.0])
            
            if _verify_np(U_np, V_np, W_np):
                best_rank = target_rank
                best_result = {
                    "U": U_np.tolist(),
                    "V": V_np.tolist(),
                    "W": W_np.tolist()
                }
                break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    return _schoolbook()
