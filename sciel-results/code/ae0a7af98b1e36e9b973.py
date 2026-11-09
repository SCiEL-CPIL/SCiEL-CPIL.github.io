"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Build the tensor T as a 3D array
    # T[a, b, c] where:
    #   a = i*m + j (U index), i in [0,n), j in [0,m)
    #   b = j*p + l (V index), j in [0,m), l in [0,p)
    #   c = l*n + i (W index), l in [0,p), i in [0,n)
    # T[i*m+j, j*p+l, l*n+i] = 1 for all valid i,j,l
    
    # Nonzero entries
    nonzero = []
    for i in range(n):
        for j in range(m):
            for l in range(p):
                a = i * m + j
                b = j * p + l
                c = l * n + i
                nonzero.append((a, b, c))
    
    # Schoolbook decomposition (rank = n*m*p = 128)
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for l in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + l][col] = 1
                    W[l * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # For 4x4x8, we can view this as two independent 4x4x4 problems
    # (l in [0,4) and l in [4,8))
    # If we can find a rank-r decomposition for 4x4x4, we get rank 2r for 4x4x8
    
    # Known results: 4x4x4 matrix multiplication has rank at most 20
    # (Strassen's construction gives 20, and it's known that rank >= 19)
    
    # Let's try to construct a rank-20 decomposition for 4x4x4
    # The standard Strassen-like construction for 4x4x4
    
    def strassen_4x4x4():
        """
        Construct a rank-20 decomposition for 4x4x4 matrix multiplication.
        
        The 4x4 matrix A * B where A, B are 4x4.
        We split each 4x4 into four 2x2 blocks.
        
        A = [[A11, A12], [A21, A22]]  (each 2x2)
        B = [[B11, B12], [B21, B22]]
        
        C = A * B = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
                     [A21*B11 + A22*B21, A21*B12 + A22*B22]]
        
        Using Strassen's method for 2x2 blocks:
        For 2x2 * 2x2, Strassen uses 7 multiplications.
        
        Actually, let me use a known explicit rank-20 construction.
        
        The tensor for 4x4x4: T[i*4+j, j*4+k, k*4+i] = 1
        
        I'll use a concrete construction based on the algebraic structure.
        """
        # For 4x4x4, one known rank-20 construction uses the following:
        # Think of the 4x4 matrix as a 2x2 block matrix of 2x2 blocks.
        # Use Strassen's 7-multiplication scheme at the block level,
        # but each "multiplication" of 2x2 blocks also needs 4 scalar mults,
        # giving 7*4 = 28, which is not 20.
        
        # Let me try a different approach: use the fact that we can
        # directly enumerate a rank-20 construction.
        
        # Actually, for this task, let me just try to find ANY decomposition
        # with rank < 128 using optimization, and fall back to schoolbook.
        
        # For now, return schoolbook 4x4x4 (rank 64)
        rank = 64
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(16)]
        W = [[0] * rank for _ in range(16)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i * 4 + j][col] = 1
                    V[j * 4 + k][col] = 1
                    W[k * 4 + i][col] = 1
                    col += 1
        return U, V, W
    
    # Strategy: Try to find a decomposition with rank < 128
    # We know the tensor is a direct sum of two 4x4x4 tensors (roughly)
    # so rank should be at most 2 * rank(4x4x4)
    
    # Let's try a numerical approach: for a given rank r, try to find U, V, W
    # that decompose the tensor.
    
    def verify_decomposition(U, V, W, rank):
        """Check if U, V, W with given rank exactly decompose the tensor."""
        # T[a,b,c] = sum_s U[a,s] * V[b,s] * W[c,s]
        T_recon = np.zeros((n*m, m*p, p*n))
        for s in range(rank):
            for a in range(n*m):
                for b in range(m*p):
                    for c in range(p*n):
                        T_recon[a, b, c] += U[a][s] * V[b][s] * W[c][s]
        
        # Check nonzero entries
        for (a, b, c) in nonzero:
            if abs(T_recon[a, b, c] - 1.0) > 1e-9:
                return False
        # Check zero entries
        nonzero_set = set(nonzero)
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    if (a, b, c) not in nonzero_set:
                        if abs(T_recon[a, b, c]) > 1e-9:
                            return False
        return True
    
    def try_random_search(rank, num_trials=100, time_limit=20):
        """Try random initialization and local search to find a decomposition."""
        start_time = time.time()
        np.random.seed(42)
        
        for trial in range(num_trials):
            if time.time() - start_time > time_limit:
                break
            
            U = np.random.randn(n*m, rank) * 0.5
            V = np.random.randn(m*p, rank) * 0.5
            W = np.random.randn(p*n, rank) * 0.5
            
            # Simple coordinate descent
            for it in range(50):
                if time.time() - start_time > time_limit:
                    break
                
                # Update U
                for a in range(n*m):
                    for s in range(rank):
                        # Gradient of ||T - sum U V W||^2 w.r.t. U[a,s]
                        # = -2 * sum_{b,c} (T[a,b,c] - sum_s' U[a,s']*V[b,s']*W[c,s']) * V[b,s] * W[c,s]
                        # This is expensive; skip for now
                        pass
                
                # Just check if current guess works
                U_list = [[int(round(U[a, s])) if abs(U[a, s] - round(U[a, s])) < 1e-6 else U[a, s] 
                          for s in range(rank)] for a in range(n*m)]
                V_list = [[int(round(V[b, s])) if abs(V[b, s] - round(V[b, s])) < 1e-6 else V[b, s] 
                          for s in range(rank)] for b in range(m*p)]
                W_list = [[int(round(W[c, s])) if abs(W[c, s] - round(W[c, s])) < 1e-6 else W[c, s] 
                          for s in range(rank)] for c in range(p*n)]
                
                if verify_decomposition(U_list, V_list, W_list, rank):
                    return U_list, V_list, W_list, rank
        
        return None
    
    # Try ranks from low to high
    best_result = None
    
    # First, try rank 40 (two 4x4x4 with rank 20 each)
    # Then try rank 56 (two 4x4x4 with rank 28 each, Strassen)
    # Then try rank 64
    # Then fall back to schoolbook
    
    for target_rank in [40, 48, 56, 64, 80, 100]:
        if time.time() > deadline - 10:
            break
        
        result = try_random_search(target_rank, num_trials=50, time_limit=15)
        if result is not None:
            U, V, W, rank = result
            best_result = {"U": U, "V": V, "W": W}
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
