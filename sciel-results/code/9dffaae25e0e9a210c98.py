"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    n, m, p = 2, 5, 5
    N = n * m  # 10
    M = m * p  # 25
    P = p * n  # 10

    def verify_fast(U, V, W):
        """
        Fast verification using vectorized operations.
        U: (N, rank), V: (M, rank), W: (P, rank)
        Returns True if the decomposition exactly represents matrix multiplication.
        """
        rank = U.shape[1]
        # We need to check that for all i, j, k:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        
        # Efficient check: 
        # For each pair of matrices U_c and V_c, their outer product gives a matrix over (i,j) x (j,k)
        # But the constraint couples i, j, k.
        
        # Let's reshape U, V, W to 3D:
        # U3d: (n, m, rank)
        # V3d: (m, p, rank)
        # W3d: (p, n, rank)
        
        U3d = U.reshape(n, m, rank)
        V3d = V.reshape(m, p, rank)
        W3d = W.reshape(p, n, rank)
        
        # The condition is: for all i, j, k:
        # sum_c U3d[i, j, c] * V3d[j, k, c] * W3d[k, i, c] == 1
        
        # We can compute the tensor T of shape (n, m, p) where:
        # T[i, j, k] = sum_c U3d[i, j, c] * V3d[j, k, c] * W3d[k, i, c]
        
        # This is a bit tricky to vectorize directly without huge memory usage if we do full outer products,
        # but n=2, m=5, p=5 is small.
        # We can iterate over c or use einsum.
        
        # Let's use einsum:
        # T[i, j, k] = U3d[i, j, c] * V3d[j, k, c] * W3d[k, i, c]
        # Note the index c is summed over.
        # Note the index i appears in U and W.
        # Note the index j appears in U and V.
        # Note the index k appears in V and W.
        
        # This is a cyclic dependency in indices i, j, k.
        # We can compute it by iterating over one index, say i.
        # For a fixed i:
        # T[i, j, k] = sum_c U3d[i, j, c] * V3d[j, k, c] * W3d[k, i, c]
        # Let A_i[j, c] = U3d[i, j, c]
        # Let B_jk[k, c] = V3d[j, k, c] -> depends on j
        # Let C_ki[c, k] = W3d[k, i, c] -> depends on i
        
        # Actually, for fixed i:
        # T[i, j, k] = sum_c U3d[i, j, c] * V3d[j, k, c] * W3d[k, i, c]
        
        # We can compute this for all j, k for a fixed i by:
        # Temp[j, k] = sum_c U3d[i, j, c] * V3d[j, k, c] * W3d[k, i, c]
        
        # Let's just use a loop over i, j, k for simplicity since 2*5*5 = 50 entries.
        # Or better, loop over c? No, rank can be large.
        # Loop over i, j, k is 50 iterations. Inside, sum over rank.
        # This is fast.
        
        T = np.zeros((n, m, p), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Vectorized sum over c
                    T[i, j, k] = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
        
        return np.allclose(T, np.ones((n, m, p)), atol=1e-9)

    def schoolbook():
        rank = n * m * p
        U = np.zeros((N, rank))
        V = np.zeros((M, rank))
        W = np.zeros((P, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_result = schoolbook()
    best_rank = n * m * p
    
    # Strategy: 
    # 1. Try to find a decomposition with rank < 50.
    # 2. Use a randomized search with verification.
    # 3. Use a structured search based on the small dimension n=2.
    
    # Theoretical lower bound is max(N, M, P) = 25.
    # Known rank for 2x5x5 is likely less than 50.
    # Let's try to find it.
    
    # Approach: Random initialization with sparse patterns.
    # Since n=2, the structure might allow rank ~ 2*m*p - something?
    # For 2x2x2, rank is 7 (Strassen).
    # For 2xNxN, rank is often around 2*N^2 - N or similar.
    # For N=5, 2*25 = 50. Strassen-like might be lower.
    
    # Let's try a few specific ranks.
    # We will try ranks from 49 down to 30.
    # For each rank, we try random initializations.
    
    np.random.seed(42)
    
    # To make search more effective, we can use the fact that the tensor is "all ones" in the (i,j,k) space
    # but with specific index mappings.
    
    # A better heuristic: 
    # Start with schoolbook and try to merge terms? 
    # Or use ALS? ALS is hard because of the integer/half-integer constraint? 
    # The task says "integer/half-integer entries". 
    # Wait, "exactly decompose". 
    # If we use random floats, we won't get exact integer/half-integer easily.
    # But the evaluator checks `np.allclose`. 
    # The prompt says "integer/half-integer entries that exactly decompose".
    # This implies the solution should have these properties.
    # However, the seed uses integers.
    # If I find a floating point solution that is very close to integers, I can round it.
    
    # Let's try to find a solution with rank < 50 using random search over integers {-1, 0, 1}.
    # This is a combinatorial search.
    
    # For rank R, we have 3 * R * (N+M+P) variables if sparse?
    # Let's try a simple random search for small ranks.
    
    for target_rank in range(49, 24, -1):
        if time.time() > deadline - 10:
            break
        
        # Try to find a decomposition of this rank
        # We will try random sparse matrices with entries in {-1, 1}
        
        found = False
        # Number of trials depends on time remaining
        trials = 100
        for t in range(trials):
            if time.time() > deadline - 5:
                break
            
            U = np.zeros((N, target_rank))
            V = np.zeros((M, target_rank))
            W = np.zeros((P, target_rank))
            
            # Initialize with random sparse entries
            # Each column c has a few non-zero entries
            for c in range(target_rank):
                # Choose random indices for U, V, W
                # To be efficient, we can choose a random pattern
                # For U (size 10), choose 2-3 non-zeros
                # For V (size 25), choose 2-3 non-zeros
                # For W (size 10), choose 2-3 non-zeros
                
                u_idx = np.random.choice(N, size=np.random.randint(2, 4), replace=False)
                v_idx = np.random.choice(M, size=np.random.randint(2, 4), replace=False)
                w_idx = np.random.choice(P, size=np.random.randint(2, 4), replace=False)
                
                for idx in u_idx:
                    U[idx, c] = np.random.choice([-1, 1])
                for idx in v_idx:
                    V[idx, c] = np.random.choice([-1, 1])
                for idx in w_idx:
                    W[idx, c] = np.random.choice([-1, 1])
            
            if verify_fast(U, V, W):
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    # If we didn't find anything better, return schoolbook
    return best_result
# EVOLVE-BLOCK-END
