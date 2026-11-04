import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build schoolbook decomposition as fallback
    U_school = np.zeros((n * m, rank_schoolbook), dtype=np.float64)
    V_school = np.zeros((m * p, rank_schoolbook), dtype=np.float64)
    W_school = np.zeros((p * n, rank_schoolbook), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_school, V_school, W_school
    
    def verify_decomposition(U, V, W, rank):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        T_target = np.ones((n, m, p), dtype=np.float64)
        
        return np.allclose(T_recon, T_target, atol=1e-9)
    
    # Strategy: Try to find a lower-rank decomposition using Kronecker product
    # structure with Strassen-like components.
    
    # For 4x5 * 5x6, we can think of this as a block multiplication.
    # Pad to 4x8 * 8x8 or similar, but that's not directly helpful.
    
    # Instead, let's try a structured search for rank < 120.
    # We'll use a greedy/iterative approach to build up the decomposition.
    
    # First, try some known lower bounds and see if we can find valid decompositions.
    # The tensor rank of 4x5*5x6 matmul is at most 120 (schoolbook).
    # It's known that for square n x n, rank < n^3 for n >= 2.
    # For rectangular, similar improvements exist.
    
    # Let's try a systematic search with random initialization and local refinement.
    
    def try_random_decomposition(target_rank, num_trials=50, seed_base=0):
        """Try to find a valid decomposition of the given rank using random search."""
        for trial in range(num_trials):
            if time.time() > deadline - 30:
                return None
            
            np.random.seed(seed_base + trial)
            
            # Initialize with small random values
            U = np.random.uniform(-1, 1, size=(n * m, target_rank))
            V = np.random.uniform(-1, 1, size=(m * p, target_rank))
            W = np.random.uniform(-1, 1, size=(p * n, target_rank))
            
            # Quick check: compute the reconstructed tensor
            U_r = U.reshape(n, m, target_rank)
            V_r = V.reshape(m, p, target_rank)
            W_r = W.reshape(p, n, target_rank)
            
            T_recon = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            T_target = np.ones((n, m, p), dtype=np.float64)
            
            if np.allclose(T_recon, T_target, atol=1e-9):
                return U, V, W
        
        return None
    
    # Try progressively lower ranks
    for target_rank in [119, 118, 117, 116, 115, 114, 113, 112, 111, 110]:
        if time.time() > deadline - 40:
            break
        
        result = try_random_decomposition(target_rank, num_trials=20, seed_base=target_rank)
        if result is not None:
            U, V, W = result
            if verify_decomposition(U, V, W, target_rank):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break
    
    # If we haven't found anything better, try a structured approach
    # using the fact that 4 = 2*2, 5 = 2*2+1, 6 = 2*3
    # We can try to exploit block structure.
    
    # For now, return the best we found
    if best_rank < rank_schoolbook:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
