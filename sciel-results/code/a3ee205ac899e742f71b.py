"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Shapes: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # W[k*n + i, r] corresponds to output C[i, k]
    
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # The tensor T[i*m+j, j*p+k, k*n+i] = 1 for the matrix multiplication
    # Actually, the standard form: C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] in the "A,B,C" indices, but our format is:
    # U indexed by (i*m+j), V indexed by (j*p+k), W indexed by (k*n+i)
    # The tensor entry T[a, b, c] where a=i*m+j, b=j*p+k, c=k*n+i
    # T[a,b,c] = 1 if a,b,c correspond to valid (i,j,k) with i in [0,n), j in [0,m), k in [0,p)
    # and the decomposition requires: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    
    # Build the target tensor as a 3D array
    # T[a, b, c] = 1 if there exist i,j,k such that a=i*m+j, b=j*p+k, c=k*n+i
    # Note: for given a=i*m+j, b=j'*p+k, c=k'*n+i', we need j=j' and k=k' and i=i'
    # So T[a,b,c] = 1 iff j_a = j_b and k_b = k_c and i_a = i_c
    # where a=i_a*m+j_a, b=j_b*p+k_b, c=k_c*n+i_c
    
    target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
    
    # Try to find a rank-r decomposition
    # Start with a reasonable guess and search
    best_result = None
    
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
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
    
    def verify(U, V, W, rank):
        """Check if the decomposition is exact."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        # Compute sum_r U[a,r]*V[b,r]*W[c,r] for all a,b,c
        # This is a tensor product
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    val = np.dot(U_arr[a, :], V_arr[b, :] * W_arr[c, :])
                    if abs(val - target[a, b, c]) > 1e-9:
                        return False
        return True
    
    def try_rank(rank, max_iters=5000):
        """Try to find a rank-r decomposition via random search."""
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            # Random initialization with small integers/half-integers
            # Use values from {-2, -1, -0.5, 0, 0.5, 1, 2}
            vals = [-2, -1, -0.5, 0, 0.5, 1, 2]
            U = [[random.choice(vals) for _ in range(rank)] for _ in range(n * m)]
            V = [[random.choice(vals) for _ in range(rank)] for _ in range(m * p)]
            W = [[random.choice(vals) for _ in range(rank)] for _ in range(p * n)]
            
            if verify(U, V, W, rank):
                return {"U": U, "V": V, "W": W}
        
        return None
    
    # Try structured approach: exploit the block structure
    # For 2×5×5, think of it as two rows of 5×5 matrix multiplication
    # C[0,k] = sum_j A[0,j]*B[j,k]
    # C[1,k] = sum_j A[1,j]*B[j,k]
    
    # Try ranks from lower bound upward
    # Lower bound: max(n*m, m*p, n*p) = max(10, 25, 10) = 25
    # But we know schoolbook is 50, so try 25-50
    
    # First, try a few specific structured decompositions
    # Laderman-style for 2×2×2 blocks, then extend
    
    # Try random search for various ranks
    for rank in range(25, 51):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(rank, max_iters=2000)
        if result is not None:
            return result
    
    # If no improvement found, return schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
