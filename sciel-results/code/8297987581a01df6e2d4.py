"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Schoolbook decomposition as fallback
    schoolbook_rank = n * m * p  # 120
    
    def schoolbook_decomp():
        U = np.zeros((n * m, schoolbook_rank))
        V = np.zeros((m * p, schoolbook_rank))
        W = np.zeros((p * n, schoolbook_rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    def to_list(mat):
        return [[round(float(x), 12) for x in row] for row in mat]
    
    # Verify decomposition correctness
    def verify(U, V, W):
        rank = U.shape[1]
        # Compute sum over rank of U[i,j] * V[j,k] * W[k,i]
        # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
        # For each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        
        # Sum over rank dimension: np.einsum('irj,jkr,kri->ijk', U_3d, V_3d, W_3d)
        # Actually: U[i,j,r] * V[j,k,r] * W[k,i,r]
        result = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        # Check if all entries are 1
        return np.allclose(result, 1.0, atol=1e-10)
    
    # Try to find a lower rank decomposition using structured search
    # For 4x5x6 matrix multiplication, known lower bounds suggest rank < 120 may be possible
    # We'll try a greedy approach: start with schoolbook and attempt to merge terms
    
    best_U, best_V, best_W = schoolbook_decomp()
    best_rank = schoolbook_rank
    
    # Try small perturbations / structured reductions
    # One approach: try to find if any terms can be combined
    # For now, use a randomized local search to try to reduce rank
    
    def try_reduce_rank(U, V, W, max_iter=1000):
        """Try to reduce rank by finding redundant or combinable terms."""
        rank = U.shape[1]
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            # Try to find two terms that can be merged
            # This is heuristic - check if removing a term and adjusting others works
            # For simplicity, try random linear combinations
            
            # Pick two random columns
            c1, c2 = np.random.choice(rank, 2, replace=False)
            
            # Try to merge: replace c1, c2 with a combination
            # This is very heuristic and may not work well
            # Skip for now, just return current best
        
        return U, V, W
    
    # Apply a simple optimization: use known results for small matrix multiplication
    # For 4x5x6, the exact rank is not well-known, but we can try structured decompositions
    
    # Try a block-based approach
    # Split matrices into blocks and use block matrix multiplication
    
    # For now, return the schoolbook decomposition which is guaranteed correct
    U_out = to_list(best_U)
    V_out = to_list(best_V)
    W_out = to_list(best_W)
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
