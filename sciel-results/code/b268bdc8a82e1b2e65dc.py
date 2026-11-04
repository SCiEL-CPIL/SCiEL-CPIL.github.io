import numpy as np
import time

n, m, p = 2, 5, 6
nm, mp, pn = n * m, m * p, p * n

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s limit

    def get_schoolbook():
        """Return the schoolbook decomposition with rank n*m*p."""
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    # Pre-compute indices for the 60 tensor entries
    idx_u = np.zeros(n * m * p, dtype=np.int32)
    idx_v = np.zeros(n * m * p, dtype=np.int32)
    idx_w = np.zeros(n * m * p, dtype=np.int32)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u[idx] = i * m + j
                idx_v[idx] = j * p + k
                idx_w[idx] = k * n + i
                idx += 1

    def check_decomposition_fast(U, V, W):
        """
        Vectorized check using pre-computed indices.
        U: (nm, R), V: (mp, R), W: (pn, R)
        """
        vals = U[idx_u] * V[idx_v] * W[idx_w]
        sums = vals.sum(axis=1)
        return np.all(sums == 1.0)

    # Strategy:
    # 1. Try to find a rank < 60 using fast random search.
    # 2. If found, return it.
    # 3. Otherwise, return schoolbook (rank 60).
    
    # We focus on ranks 59 and 58 with high intensity, then fallback.
    # Total search time budget: 10 seconds.
    
    def try_rank_random_batch(R, time_budget):
        """Try random search for a valid decomposition of rank R."""
        start_search = time.time()
        batch_size = 500
        # Estimate cost: batch_size * 60 * R operations
        # For R=59, 500*60*59 ~ 1.8M ops. ~1ms per batch?
        # Let's run batches until time_budget is used.
        
        while time.time() < start_search + time_budget:
            if time.time() > deadline:
                return None
            
            # Generate a batch of random matrices
            # Using small integers -1, 0, 1, 2
            U_batch = np.random.randint(-1, 3, size=(batch_size, nm, R))
            V_batch = np.random.randint(-1, 3, size=(batch_size, mp, R))
            W_batch = np.random.randint(-1, 3, size=(batch_size, pn, R))
            
            # Check each in the batch
            U_sel = U_batch[:, idx_u]
            V_sel = V_batch[:, idx_v]
            W_sel = W_batch[:, idx_w]
            
            vals = U_sel * V_sel * W_sel
            sums = vals.sum(axis=2) # (batch, 60)
            
            # Find indices where all 60 sums are 1
            valid_mask = np.all(sums == 1.0, axis=1)
            valid_indices = np.where(valid_mask)[0]
            
            if len(valid_indices) > 0:
                idx_valid = valid_indices[0]
                return U_batch[idx_valid], V_batch[idx_valid], W_batch[idx_valid]
                
        return None

    # Try rank 59 for 5s, rank 58 for 5s
    for R in [59, 58]:
        if time.time() > deadline - 2:
            break
        result = try_rank_random_batch(R, 5.0)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook (Rank 60)
    U, V, W = get_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
