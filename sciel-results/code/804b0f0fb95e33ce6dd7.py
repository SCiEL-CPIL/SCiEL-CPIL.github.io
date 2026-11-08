"""Matrix multiplication 2x5x6 tensor decomposition - structured approach."""
import time
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 2, 5, 6
    trivial_rank = n * m * p  # 60
    
    # The tensor T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]B[j,k]
    # Indices: A has (i,j) for i in [0,2), j in [0,5)
    #          B has (j,k) for j in [0,5), k in [0,6)
    #          C has (i,k) for i in [0,2), k in [0,6)
    
    # For each output (i,k), we need sum over j of A[i,j]*B[j,k]
    # The trivial decomposition uses one rank-1 tensor per (i,j,k) triple
    
    # Strategy: exploit the small n=2 dimension
    # The matrix multiplication tensor for 2x5x6 can potentially be decomposed
    # using structured approaches. However, finding the exact minimal rank is hard.
    
    # Let's try a few structured approaches:
    
    # Approach 1: Trivial decomposition (rank 60)
    rank = trivial_rank
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Check if we have time to attempt improvements
    elapsed = time.time() - start_time
    if elapsed > 100:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Approach 2: Try to use Strassen-like techniques
    # For 2x2x2, Strassen achieves rank 7 instead of 8
    # Can we apply similar ideas to 2x5x6?
    
    # Partition the middle dimension j=5 into blocks
    # 5 = 2 + 2 + 1
    # Partition the output dimension k=6 into blocks
    # 6 = 2 + 2 + 2
    
    # Let's try to construct a decomposition using block structure
    # A is 2x5, B is 5x6, C is 2x6
    
    # Split A = [A1, A2, A3] where A1, A2 are 2x2, A3 is 2x1
    # Split B = [B1; B2; B3] where B1, B2 are 2x6, B3 is 1x6
    
    # C = A1*B1 + A2*B2 + A3*B3
    
    # For each block product, we can use known decompositions
    # But this gets complex. Let's try a simpler heuristic.
    
    # Approach 3: Try rank 59 by removing one term and adjusting
    # This is unlikely to work easily, but let's try a verification approach
    
    # For a given rank r, we need to find U, V, W such that:
    # sum_{r=1}^{R} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is a system of 2*5*6 = 60 nonlinear equations in R*(10+30+12) = 52R unknowns
    
    # For R=59, we have 59*52 = 3068 unknowns and 60 equations
    # This is underdetermined, so solutions likely exist, but finding them is hard
    
    # Let's try a random search with verification for lower ranks
    # We'll try ranks from 59 down to 50
    
    best_rank = trivial_rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to find a lower rank decomposition
    for target_rank in range(59, 49, -1):
        if time.time() > deadline - 5:
            break
        
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U_r = np.random.randn(n * m, target_rank) * 0.1
            V_r = np.random.randn(m * p, target_rank) * 0.1
            W_r = np.random.randn(p * n, target_rank) * 0.1
            
            # Verify: check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # for all i, j, k
            
            # Build the tensor from the decomposition
            tensor = np.zeros((n, m, p))
            for r in range(target_rank):
                u_col = U_r[:, r]
                v_col = V_r[:, r]
                w_col = W_r[:, r]
                
                # u_col has indices i*m+j
                # v_col has indices j*p+k
                # w_col has indices k*n+i
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            tensor[i, j, k] += u_col[i * m + j] * v_col[j * p + k] * w_col[k * n + i]
            
            # Check if tensor is all ones
            if np.allclose(tensor, 1.0, atol=1e-10):
                best_rank = target_rank
                best_U, best_V, best_W = U_r, V_r, W_r
                break
        
        if best_rank < trivial_rank:
            break
    
    # If we found a better rank, use it
    if best_rank < trivial_rank:
        U, V, W = best_U, best_V, best_W
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-START
# Entry point is construct(), returns dict with 'U', 'V', 'W' keys
# EVOLVE-BLOCK-END
