"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from collections import defaultdict

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x5 matrix.
    
    Returns {'U', 'V', 'W'}: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    with integer/half-integer entries that exactly decompose the matrix multiplication tensor.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # The matrix multiplication tensor T has T[i,j,k] = delta where:
    # U has shape (n*m, rank), indexed by (i,j)
    # V has shape (m*p, rank), indexed by (j,k)  
    # W has shape (p*n, rank), indexed by (k,i)
    # The decomposition satisfies: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # for the standard matrix multiplication tensor where T[i,j,k]=1 iff the (i,k)-entry of A@B
    # depends on A[i,j]*B[j,k].
    
    # For matrix multiplication, the tensor T is defined such that:
    # (A @ B)[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all valid (i,j,k) in the sense that each term A[i,j]*B[j,k] contributes.
    # Actually, the standard tensor for matmul has T[i,j,k] = 1 for all i,j,k, and the decomposition
    # U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should reconstruct the identity mapping.
    
    # Let me reconsider. The matrix multiplication tensor M of format (n,m,p) has:
    # M[i,j,k] = 1 for all (i,j,k) — no wait.
    
    # The correct formulation: We want to compute C = A * B where A is n×m, B is m×p, C is n×p.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # The tensor T is a 3-tensor with T[i,j,k] = 1 for all i in [n], j in [m], k in [p].
    # Wait, that's not right either. Let me think more carefully.
    
    # The matrix multiplication tensor has entries indexed by (i, j, k) where i indexes rows of A,
    # j indexes columns of A / rows of B, k indexes columns of B. The tensor entry T[i,j,k] = 1.
    # 
    # A rank-1 decomposition term is: u_{i,j} * v_{j,k} * w_{k,i}
    # The sum over all terms should give T[i,j,k] = 1 for all i,j,k.
    # 
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    
    # The schoolbook solution uses rank = n*m*p with U[i*m+j, i*m+j] = 1, etc.
    
    # Let's try to find a lower rank solution using optimization/search.
    
    # First, let's try known results. For 3x3 * 3x5, the minimum rank might be less than 45.
    # Let's use a systematic search approach.
    
    # Strategy: Use random initialization and gradient-free optimization (or structured search)
    # to find a low-rank decomposition.
    
    # Let's first try to verify if rank 45 (schoolbook) is optimal or if we can do better.
    
    # Approach: Try to find a decomposition with rank < 45 using:
    # 1. Random initialization with integer/half-integer entries
    # 2. Check if the tensor equation is satisfied
    
    # The constraint is: for all (i,j,k) in [n]x[m]x[p]:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a system of n*m*p = 45 equations with 3*rank unknowns (well, rank*9 + rank*15 + rank*15 = rank*39 variables).
    
    # Let's try a structured approach: search over small integer/half-integer values.
    
    best_rank = n * m * p  # schoolbook
    best_solution = None
    
    # Generate schoolbook solution as fallback
    U_school = [[0] * best_rank for _ in range(n * m)]
    V_school = [[0] * best_rank for _ in range(m * p)]
    W_school = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_solution = (U_school, V_school, W_school)
    
    # Try to find lower rank solutions
    # Use random search with integer/half-integer entries in a small range
    
    def check_decomposition(U, V, W, rank, n, m, p):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(total - 1) > 1e-9:
                        return False
        return True
    
    def check_decomposition_numpy(U, V, W, rank, n, m, p):
        """Faster check using numpy."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Reshape for efficient computation
        # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # We need: for each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        
        # Create index arrays
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(total - 1) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        return valid
    
    # Try ranks from lower to higher
    # Start with a reasonable lower bound. The rank must be at least max(n*m, m*p, p*n) = max(9, 15, 15) = 15
    # but likely much higher. Let's try from 20 upward.
    
    # Use a smarter approach: structured random search
    # Values from {-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2}
    values = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    
    # For efficiency, let's try a few random attempts at various ranks
    # Focus on ranks that are plausible: 20-40
    
    attempts_per_rank = 50
    max_rank_to_try = 40
    
    for rank in range(20, min(max_rank_to_try + 1, best_rank)):
        if time.time() > deadline:
            break
        
        for attempt in range(attempts_per_rank):
            if time.time() > deadline:
                break
            
            # Random initialization with small integer/half-integer values
            # Use sparsity to help
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            # Fill with random values, keeping some zeros for sparsity
            density = 0.3  # fraction of non-zero entries
            
            for c in range(rank):
                # Each column gets some non-zero entries
                for idx in range(n * m):
                    if random.random() < density:
                        U[idx][c] = random.choice(values)
                for idx in range(m * p):
                    if random.random() < density:
                        V[idx][c] = random.choice(values)
                for idx in range(p * n):
                    if random.random() < density:
                        W[idx][c] = random.choice(values)
            
            if check_decomposition(U, V, W, rank, n, m, p):
                best_rank = rank
                best_solution = (U, V, W)
                break
        
        if best_rank < n * m * p:
            # Found a better solution, continue searching for even lower rank
            continue
    
    # If we found a better solution, use it; otherwise use schoolbook
    if best_solution is not None:
        U, V, W = best_solution
    else:
        U, V, W = best_solution
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
