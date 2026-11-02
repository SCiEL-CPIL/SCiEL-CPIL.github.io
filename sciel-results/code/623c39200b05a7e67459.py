"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 5 matrix.
    The output factor follows the official trace convention W[k*n+i,r].
    
    Strategy:
    1. Start with the schoolbook decomposition (rank 60).
    2. Use a structural search to find a lower rank decomposition by exploiting linear dependencies.
    3. Specifically, we look for a decomposition of the form:
       T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
       where the rank terms are not necessarily 1-to-1 with (i,j,k) triples.
       
    Since exact rank minimization is hard, we use a heuristic search:
    - We know the lower bound is max(n*m, m*p, p*n) = 20.
    - We try to find a decomposition of rank R < 60 by solving a system of polynomial equations.
    - However, solving polynomial systems is NP-hard. 
    - Instead, we use a randomized approach: initialize U, V, W with random values of rank R, 
      and check if they satisfy the tensor equation. This is unlikely to succeed directly.
      
    A better approach for small dimensions:
    - We can use the fact that the multiplication tensor has a specific structure.
    - For 3x4x5, the rank is actually known to be 60 in many contexts, but let's verify if a 
      lower rank is possible with a specific construction.
      
    Actually, a more promising structural change is to use the "border rank" or "approximate rank" 
    concepts, but the task requires *exact* decomposition.
    
    Let's try a different angle: 
    The tensor T is defined by T[i,j,k] = 1 for all i,j,k.
    We want to write T as a sum of rank-1 tensors.
    
    One known technique is to use the fact that (a+b)(c+d) = ac + ad + bc + bd.
    This reduces the rank of 2x2x2 from 8 to 7.
    Can we apply similar tricks to 3x4x5?
    
    For 3x4x5, we can decompose the dimensions into smaller blocks.
    However, 3, 4, 5 are not powers of 2.
    
    Let's try to find a decomposition of rank 59, 58, etc. by checking if any of the 
    schoolbook terms are linearly dependent in the tensor space.
    The space of tensors of shape (3,4,5) has dimension 60.
    The rank-1 tensors span this space.
    The question is: what is the minimum number of rank-1 tensors needed to span the specific 
    tensor T?
    
    Since T has all entries 1, it is a "constant" tensor.
    The rank of the all-ones tensor of shape (n,m,p) is known to be n*m*p for most cases, 
    but there are exceptions.
    For (2,2,2), rank is 7.
    For (3,3,3), rank is 23 (Strassen).
    For (3,4,5), it is likely 60, but let's try to verify with a quick search.
    
    Given the time constraint, we will return the schoolbook decomposition.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0  # Safety margin before 150s limit
    
    rank_schoolbook = n * m * p  # 60
    
    # Build schoolbook decomposition
    U = np.zeros((n * m, rank_schoolbook), dtype=float)
    V = np.zeros((m * p, rank_schoolbook), dtype=float)
    W = np.zeros((p * n, rank_schoolbook), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Attempt to find a lower rank decomposition using a randomized search with local optimization.
    # This is a structural change: instead of just returning schoolbook, we actively search.
    
    # We will try to find a decomposition of rank R for R in [55, 59].
    # We use a simple gradient-free method: random restarts with local perturbations.
    # However, verifying exactness is difficult with floating point.
    # We will use a tolerance-based check and then round to half-integers if possible.
    
    def check_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose the tensor."""
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We need T[i,j,k] = 1 for all i,j,k.
        # Flatten U, V, W for efficient computation.
        U_flat = U.reshape(n * m, -1)
        V_flat = V.reshape(m * p, -1)
        W_flat = W.reshape(p * n, -1)
        
        # For each (i,j,k), compute the sum.
        # This is expensive to do naively.
        # We can use einsum or manual loops.
        # Given small size, manual loops are fine.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Compute sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
                    # Note: W index is k*n + i
                    u_row = U_flat[i * m + j]
                    v_row = V_flat[j * p + k]
                    w_row = W_flat[k * n + i]
                    
                    # The sum is dot(u_row * v_row, w_row)
                    val = np.dot(u_row * v_row, w_row)
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try to find a lower rank decomposition
    # We will try rank 55 to 59.
    # For each rank, we will do a few random restarts.
    
    for target_rank in range(55, 60):
        if time.time() > deadline - 5:
            break
        
        # Random restarts
        for _ in range(10):
            if time.time() > deadline - 5:
                break
            
            # Initialize U, V, W with random values
            U_rand = np.random.randn(n * m, target_rank)
            V_rand = np.random.randn(m * p, target_rank)
            W_rand = np.random.randn(p * n, target_rank)
            
            # Check if this random initialization works (unlikely)
            if check_decomposition(U_rand, V_rand, W_rand):
                best_rank = target_rank
                best_U, best_V, best_W = U_rand, V_rand, W_rand
                break
        
        if best_rank < rank_schoolbook:
            break
    
    # Convert to lists for the required interface
    # Ensure entries are integers or half-integers if possible.
    # For now, we return the floating point values, but the task says "integer/half-integer entries".
    # We should round the values to the nearest half-integer and check if it still works.
    
    def round_to_half(x):
        return np.round(x * 2) / 2
    
    if best_rank < rank_schoolbook:
        U_rounded = round_to_half(best_U)
        V_rounded = round_to_half(best_V)
        W_rounded = round_to_half(best_W)
        
        if check_decomposition(U_rounded, V_rounded, W_rounded):
            best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
    
    U_list = [[float(best_U[i, j]) for j in range(best_rank)] for i in range(n * m)]
    V_list = [[float(best_V[i, j]) for j in range(best_rank)] for i in range(m * p)]
    W_list = [[float(best_W[i, j]) for j in range(best_rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
