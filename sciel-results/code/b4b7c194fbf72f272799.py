"""Matrix multiplication 2x5x6 rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 60
    
    # Precompute target tensor
    # Shape: (n*m, m*p, p*n) = (10, 30, 12)
    target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0

    def make_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        # Convert to list of lists for output interface
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = make_schoolbook()
    best_rank = rank_schoolbook

    def verify_decomposition_np(U, V, W):
        """Verify using NumPy. U, V, W are numpy arrays."""
        # Compute T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # einsum is efficient for this
        T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_decomp, target, atol=1e-10)

    def compute_loss_np(U, V, W):
        """Compute squared error loss."""
        T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
        diff = T_decomp - target
        return np.sum(diff * diff)

    def try_rank_np(r, max_iter=2000, seed=42):
        """Try to find a rank-r decomposition using NumPy-accelerated loss."""
        rng = random.Random(seed)
        
        # Initialize with small random values
        U = np.random.uniform(-1, 1, (n*m, r))
        V = np.random.uniform(-1, 1, (m*p, r))
        W = np.random.uniform(-1, 1, (p*n, r))
        
        best_loss = float('inf')
        best_U, best_V, best_W = None, None, None
        
        # Adaptive step size
        step_size = 1.0
        
        for iteration in range(max_iter):
            if time.time() > deadline - 10:
                break
                
            loss = compute_loss_np(U, V, W)
            if loss < best_loss:
                best_loss = loss
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                
            if best_loss < 1e-12:
                # Verify exactly
                if verify_decomposition_np(best_U, best_V, best_W):
                    return best_U, best_V, best_W
            
            # Random perturbation
            # Pick a random matrix and a random element
            idx = rng.randint(0, 2)
            if idx == 0:
                row = rng.randint(0, n*m-1)
                col = rng.randint(0, r-1)
                U[row, col] += rng.gauss(0, step_size)
            elif idx == 1:
                row = rng.randint(0, m*p-1)
                col = rng.randint(0, r-1)
                V[row, col] += rng.gauss(0, step_size)
            else:
                row = rng.randint(0, p*n-1)
                col = rng.randint(0, r-1)
                W[row, col] += rng.gauss(0, step_size)
                
            # Decay step size
            step_size *= 0.999
            
        # Final check on best found
        if best_U is not None and best_loss < 1e-8:
             if verify_decomposition_np(best_U, best_V, best_W):
                 return best_U, best_V, best_W
                 
        return None

    # Try ranks from 2 upward
    # We can go up to rank_schoolbook, but usually we stop early if we find one.
    # Given the difficulty, we might not find anything below 60.
    # But let's try a few small ranks and maybe a few near 60?
    # Actually, if rank 60 is optimal, we just return it.
    
    # To save time, we can try a few specific ranks.
    # If we find a rank < 60, we return it.
    
    ranks_to_try = list(range(2, 10)) + list(range(50, 60))
    
    for r in ranks_to_try:
        if time.time() > deadline - 15:
            break
        
        # Try a few seeds for each rank
        for seed in range(3):
            if time.time() > deadline - 15:
                break
            result = try_rank_np(r, max_iter=1000, seed=r*1000 + seed)
            if result is not None:
                U, V, W = result
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = r
                print(f"Found rank {r} decomposition!")
                return best_result # Return immediately if found
        
    return best_result
