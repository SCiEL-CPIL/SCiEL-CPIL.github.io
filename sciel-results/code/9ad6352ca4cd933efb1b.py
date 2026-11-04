import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    deadline = time.time() + 140.0  # Safety margin for 150s limit
    
    def verify_numpy(U, V, W):
        """
        Efficiently verify tensor decomposition using numpy.
        U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        Check: for all i,j,k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
        """
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        rank = U.shape[1]
        
        # Build index arrays
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        
        # Compute all 125 values
        ui = U[i_idx * m + j_idx]  # (125, rank)
        vk = V[j_idx * p + k_idx]  # (125, rank)
        wk = W[k_idx * n + i_idx]  # (125, rank)
        
        # Sum over rank dimension
        totals = np.sum(ui * vk * wk, axis=1)
        
        return np.allclose(totals, 1.0, atol=1e-9)
    
    def get_schoolbook():
        """Return the schoolbook decomposition (rank 125)."""
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U.tolist(), V.tolist(), W.tolist()
    
    # Start with schoolbook as baseline
    U_best, V_best, W_best = get_schoolbook()
    best_rank = n * m * p
    
    # Strategy: Try to find lower-rank decompositions using structured approaches
    # For 5x5x5, we can try:
    # 1. Circulant-based decompositions
    # 2. Random sparse searches at various ranks
    # 3. Greedy column elimination from schoolbook
    
    # Approach 1: Try to eliminate columns from schoolbook by checking linear dependencies
    # If a column in the tensor can be expressed as a linear combination of others, we can reduce rank
    
    # Approach 2: Random search for lower rank
    # For rank R, we have (n*m + m*p + p*n) * R = 25*R variables
    # and 125 constraints (one per (i,j,k) pair)
    # For R < 125/3 ≈ 41.67, the system is overdetermined, so random search is unlikely to work
    # But we can try a few structured approaches
    
    # Let's try a simple structured approach: 
    # Use the fact that matrix multiplication can be decomposed using Kronecker products
    # For 5 = 2+3, we can potentially combine 2x2 and 3x3 decompositions
    
    # However, the most practical approach given time constraints is to:
    # 1. Verify schoolbook works (it does)
    # 2. Try a few random attempts at lower ranks
    # 3. Return the best found
    
    # Try random searches at a few target ranks
    target_ranks = [64, 50, 40, 35, 30, 25]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 10.0:
            break
        
        # Try a few random initializations
        for trial in range(3):
            if time.time() > deadline - 5.0:
                break
            
            # Random initialization with some structure
            np.random.seed(42 + target_rank + trial)
            
            # Initialize with small random values
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Verify
            if verify_numpy(U, V, W):
                U_best = U.tolist()
                V_best = V.tolist()
                W_best = W.tolist()
                best_rank = target_rank
                break
        
        if best_rank < target_rank:
            break
    
    # If we found an improvement, use it; otherwise use schoolbook
    if best_rank < n * m * p:
        return {"U": U_best, "V": V_best, "W": W_best}
    else:
        return {"U": U_best, "V": V_best, "W": W_best}
# EVOLVE-BLOCK-END
