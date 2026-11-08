"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Strategy: Fix V to have an identity-like structure to decouple the problem.
    # For a 2x3 * 3x7 multiplication, we can try to find a decomposition where
    # V has a specific structure that simplifies the tensor.
    
    # First, let's try to find a low-rank decomposition using the Kronecker product idea.
    # The tensor T_{i,j,k} = delta_{j,k} for matrix multiplication.
    # We want T_{i,j,k} = sum_r U_{i,r} * V_{j,r} * W_{k,r}
    
    # For 2x3x7, the schoolbook rank is 42.
    # Let's try to see if we can do better by fixing V.
    
    # Try rank 21 (half of schoolbook) by exploiting symmetry.
    # Actually, for matrix multiplication, the rank is at least max(n*m, m*p, p*n) = max(6, 21, 14) = 21.
    # But the actual rank for 2x3x7 might be higher.
    
    # Let's use a systematic approach: try to find a decomposition with rank r
    # by solving a linear system for each r.
    
    # Start with the guaranteed schoolbook decomposition as fallback
    rank_fallback = n * m * p
    U_fallback = [[0] * rank_fallback for _ in range(n * m)]
    V_fallback = [[0] * rank_fallback for _ in range(m * p)]
    W_fallback = [[0] * rank_fallback for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U_fallback[u_idx][col] = 1
                V_fallback[v_idx][col] = 1
                W_fallback[w_idx][col] = 1
                col += 1
    
    # Now try to find a lower rank decomposition
    # The tensor has size n*m*p = 42
    # We want to find rank r such that the tensor can be decomposed.
    
    # Use the fact that for matrix multiplication, the rank is related to the 
    # structure of the multiplication.
    
    # Try ranks from 21 to 42
    best_rank = rank_fallback
    best_U, best_V, best_W = U_fallback, V_fallback, W_fallback
    
    # For each candidate rank, try to solve for the decomposition
    # This is a rank-r decomposition problem.
    
    # Use a simple approach: try to find a decomposition by fixing one matrix
    # and solving for the others.
    
    # Fix W to be random (or structured) and solve for U and V.
    # Actually, let's use a more principled approach.
    
    # The matrix multiplication tensor for (n,m,p) has a known structure.
    # For 2x3x7, let's try to find if rank 21 is achievable.
    
    # Construct the tensor as a 3D array
    tensor = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if j == k:
                    tensor[i, j, k] = 1
    
    # Reshape to a matrix for SVD
    # T_{i,j,k} = sum_r U_{i,r} V_{j,r} W_{k,r}
    # This is a 3-way tensor decomposition.
    
    # For a quick check, let's see if the tensor has a low-rank structure.
    # Reshape to (n, m*p) and check rank
    mat_np = tensor.reshape(n, m * p)
    rank_np = np.linalg.matrix_rank(mat_np)
    
    # The rank of the 2D slice gives a lower bound
    # But we need the 3-way tensor rank.
    
    # Let's try a simple iterative approach: start with rank 21 and see if we can solve.
    
    # Actually, for this specific case, let's use a known result:
    # The rank of matrix multiplication for (n,m,p) is at least max(n*m, m*p, p*n).
    # For (2,3,7), that's max(6, 21, 14) = 21.
    
    # Let's try to construct a rank-21 decomposition if possible.
    
    # One approach: use the fact that we can embed the 2x3 * 3x7 multiplication
    # into a larger structure.
    
    # For now, return the schoolbook decomposition as it's guaranteed to work.
    # A more sophisticated search would be needed to find lower rank.
    
    return {"U": U_fallback, "V": V_fallback, "W": W_fallback}
# EVOLVE-BLOCK-END
