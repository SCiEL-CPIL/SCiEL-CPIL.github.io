import time
import numpy as np
import random

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s limit to be safe within 150s
    
    # Schoolbook decomposition (Rank 105)
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
    
    best_solution = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    # Function to verify a decomposition
    def verify(U, V, W, rank):
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # Check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
        # Using numpy for speed
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Reshape to tensors
        # U: (n, m, rank)
        # V: (m, p, rank)
        # W: (p, n, rank)
        U_t = U_np.reshape(n, m, rank)
        V_t = V_np.reshape(m, p, rank)
        W_t = W_np.reshape(p, n, rank)
        
        # We want T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # This is a contraction over r and indices.
        # Let's iterate over k to be safe or use einsum if possible.
        # einsum: 'ijr,jkr,kin->ijk'
        try:
            T = np.einsum('ijr,jkr,kin->ijk', U_t, V_t, W_t)
            if not np.allclose(T, 1.0, atol=1e-9):
                return False
        except Exception:
            return False
        return True

    # Strategy:
    # 1. Try to find a solution with rank < 105.
    # 2. The "sliding window" or Toeplitz idea suggests exploiting structure.
    #    For matrix multiplication, the tensor is T[i,j,k] = 1.
    #    This is a "completely positive" tensor structure in some sense, but actually it's the multiplication tensor.
    #    
    #    One known technique for reducing rank is to use the "Strassen-like" recursion or specific algebraic identities.
    #    However, for arbitrary 3x5x7, the schoolbook is often optimal or near-optimal unless specific block structures are exploited.
    #    
    #    Let's try a local search starting from the schoolbook solution.
    #    We can try to merge two rank-1 terms into one? No, that increases rank usually or requires specific alignment.
    #    
    #    Actually, a better approach for small improvements is to look for linear dependencies.
    #    If we have a decomposition of rank R, we can try to reduce it to R-1 if the R-th term is a linear combination of the others in a specific tensor sense.
    #    
    #    Given the time limit, let's try a few random restarts with rank 104, 103, etc., using a smarter initialization.
    #    Instead of pure random, we can start with schoolbook and perturb it.
    #    
    #    But pure random search for rank 104 in 3x5x7 is unlikely to succeed quickly.
    #    
    #    Let's check if there's a known lower bound or better algorithm.
    #    For 3x3x3, rank is 23 (vs 27 schoolbook).
    #    For 3x5x7, we can decompose 5x5 into blocks?
    #    5 = 3 + 2.
    #    If we use rank 23 for 3x3 and rank 8 for 2x2 (schoolbook for 2x2 is 8, Strassen is 7), we might combine them.
    #    
    #    Let's try to construct a decomposition based on block decomposition.
    #    A is 3x5, B is 5x7.
    #    Split A into A1 (3x3) and A2 (3x2).
    #    Split B into B1 (3x7) and B2 (2x7).
    #    C = A1*B1 + A2*B2.
    #    
    #    We need a decomposition for 3x3x7 and 3x2x7.
    #    This doesn't immediately reduce the rank below 105 without sophisticated sharing.
    #    
    #    Let's stick to the local search / random search for small rank reductions, but make it efficient.
    #    
    #    We will try ranks from 104 down to, say, 90.
    #    For each rank, we try a few random initializations.
    #    To make it more likely to succeed, we can use a "greedy" approach:
    #    Start with schoolbook.
    #    Try to remove one term and see if the remaining terms can be adjusted to compensate.
    #    This is a non-linear optimization problem.
    #    
    #    Given the constraints, I will implement a limited random search for rank 104 and 103.
    #    If that fails, return schoolbook.
    
    random.seed(42)
    np.random.seed(42)
    
    # Try a few ranks below schoolbook
    for R in range(best_rank - 1, best_rank - 6, -1):
        if time.time() > deadline - 10:
            break
            
        trials = 10
        for _ in range(trials):
            # Random initialization with small integers
            # We bias towards 0 and 1 to keep numbers small
            U = np.zeros((n * m, R), dtype=float)
            V = np.zeros((m * p, R), dtype=float)
            W = np.zeros((p * n, R), dtype=float)
            
            # Fill with random -1, 0, 1
            # Use a sparse-like initialization to help convergence if we were doing gradient descent,
            # but here we are just checking.
            for r in range(R):
                # Ensure each column has some non-zero entries
                U[:, r] = np.random.choice([-1, 0, 1], size=n * m)
                V[:, r] = np.random.choice([-1, 0, 1], size=m * p)
                W[:, r] = np.random.choice([-1, 0, 1], size=p * n)
                
            if verify(U.tolist(), V.tolist(), W.tolist(), R):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
    # If no better solution found, return schoolbook
    return best_solution
# EVOLVE-BLOCK-END
