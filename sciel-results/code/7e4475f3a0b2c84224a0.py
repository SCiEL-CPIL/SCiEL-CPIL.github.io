import numpy as np
import time
import random

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.int64)
        V = np.zeros((m * p, rank), dtype=np.int64)
        W = np.zeros((p * n, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    def check_decomposition(U, V, W, rank):
        """Vectorized check: Sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k"""
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        
        # Use einsum to compute the sum for all i,j,k at once
        # result[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
        # This is a contraction over c.
        # We can do: np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        # But W3 is (p,n,rank), so index is k,i,c.
        # U3 is (n,m,rank) -> i,j,c
        # V3 is (m,p,rank) -> j,k,c
        # W3 is (p,n,rank) -> k,i,c
        
        result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3, optimize=True)
        return np.all(result == 1)
    
    def random_attempt(rank, seed_offset=0):
        """Try a random integer decomposition."""
        np.random.seed(42 + seed_offset + rank * 1000)
        # Sparse random: mostly 0, some -1, 1
        U = np.zeros((n * m, rank), dtype=np.int64)
        V = np.zeros((m * p, rank), dtype=np.int64)
        W = np.zeros((p * n, rank), dtype=np.int64)
        
        density = 0.2
        for c in range(rank):
            mask_u = np.random.random(n * m) < density
            mask_v = np.random.random(m * p) < density
            mask_w = np.random.random(p * n) < density
            U[mask_u, c] = np.random.choice([-1, 1], size=mask_u.sum())
            V[mask_v, c] = np.random.choice([-1, 1], size=mask_v.sum())
            W[mask_w, c] = np.random.choice([-1, 1], size=mask_w.sum())
        return U, V, W

    # Search from high rank down
    best_rank = n * m * p
    best_U, best_V, best_W = None, None, None
    
    # Try a range of ranks
    ranks_to_try = list(range(74, 50, -1))
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
        
        # More attempts for higher ranks (easier to find)
        max_attempts = 100 if rank >= 70 else 50
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            U, V, W = random_attempt(rank, attempt)
            
            if check_decomposition(U, V, W, rank):
                best_rank = rank
                best_U, best_V, best_W = U, V, W
                break
        
        if best_rank < n * m * p:
            # Found something, try to go lower
            if rank <= 60:
                break
    
    if best_rank < n * m * p and best_U is not None:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        U, V, W, _ = schoolbook()
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
