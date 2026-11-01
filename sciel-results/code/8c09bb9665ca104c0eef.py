import numpy as np
import time

DIMENSIONS = [2, 2, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    # n=2, m=2, p=10
    # A is 2x2, B is 2x10, C is 2x10
    # U shape (4, R), V shape (20, R), W shape (20, R)
    
    # Known rank 35 construction:
    # Split B into 5 pairs of columns.
    # For each pair, use Strassen (rank 7).
    # Total rank = 5 * 7 = 35.
    
    # Strategy:
    # 1. Build the rank 35 solution.
    # 2. Attempt to reduce rank by finding linear dependencies or merging terms.
    #    Specifically, check if the 35 terms span a subspace of dimension < 35
    #    in the tensor product space, or if we can solve for a lower rank decomposition
    #    using a structured random search or local optimization.
    #    Given the time limit, we will focus on a quick check for rank 34 or 33
    #    by perturbing the Strassen solution and projecting, or by a small random search.
    #    However, exact rank reduction is hard. We will stick to the robust rank 35
    #    unless a very fast heuristic finds a better one.
    
    # The most reliable path is to return the known rank 35 solution.
    # Let's ensure the implementation is correct and efficient.
    
    def build_strassen_rank_35():
        U = np.zeros((4, 35), dtype=float)
        V = np.zeros((20, 35), dtype=float)
        W = np.zeros((20, 35), dtype=float)
        
        # Strassen 7 terms for 2x2
        # A indices: 0:00, 1:01, 2:10, 3:11
        # B indices for one block: 0:00, 1:01, 2:10, 3:11 (local)
        # C indices for one block: 0:00, 1:01, 2:10, 3:11 (local)
        
        # Standard Strassen terms (verified):
        terms = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
        ]
        
        col = 0
        for q in range(5):
            for a_vec, b_vec, c_vec in terms:
                # U
                U[0, col] = a_vec[0]
                U[1, col] = a_vec[1]
                U[2, col] = a_vec[2]
                U[3, col] = a_vec[3]
                
                # V
                # B indices: j*10+k.
                # For block q, k is 2q and 2q+1.
                # b_vec is [B00, B01, B10, B11] local.
                # Local 0 -> global j=0, k=2q
                # Local 1 -> global j=0, k=2q+1
                # Local 2 -> global j=1, k=2q
                # Local 3 -> global j=1, k=2q+1
                
                V[0*10 + 2*q, col] = b_vec[0]
                V[0*10 + 2*q+1, col] = b_vec[1]
                V[1*10 + 2*q, col] = b_vec[2]
                V[1*10 + 2*q+1, col] = b_vec[3]
                
                # W
                # C indices: k*2+i.
                # c_vec is [C00, C10, C01, C11] local?
                # Based on previous verification:
                # W[4*q + 0, col] = c_vec[0] -> C00 (k=2q, i=0)
                # W[4*q + 1, col] = c_vec[1] -> C10 (k=2q, i=1)
                # W[4*q + 2, col] = c_vec[2] -> C01 (k=2q+1, i=0)
                # W[4*q + 3, col] = c_vec[3] -> C11 (k=2q+1, i=1)
                
                W[4*q + 0, col] = c_vec[0]
                W[4*q + 1, col] = c_vec[1]
                W[4*q + 2, col] = c_vec[2]
                W[4*q + 3, col] = c_vec[3]
                
                col += 1
        
        return U, V, W

    U, V, W = build_strassen_rank_35()
    
    # Attempt to improve rank using a quick heuristic search if time permits.
    # We will try to see if we can find a rank 34 solution by random perturbation
    # and checking for linear dependencies or using a small-scale optimization.
    # Given the complexity, we will skip this and return the robust rank 35 solution.
    # This ensures correctness and avoids timeout risks.
    
    elapsed = time.time() - start_time
    if elapsed < 100:
        # Placeholder for potential rank reduction search
        # For now, we return the known solution.
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
