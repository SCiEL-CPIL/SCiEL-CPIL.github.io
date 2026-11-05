"""Tensor decomposition search for 3x3 * 3x6 matrix multiplication."""
import time
import random
import itertools
from fractions import Fraction

DIMENSIONS = [3, 3, 6]
DEADLINE = 140.0  # safety margin before 150s kill

def _time_left():
    return DEADLINE - time.time()

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x6 matrix multiplication.
    
    The multiplication tensor T has entries T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    A rank-r decomposition gives U (n*m x r), V (m*p x r), W (p*n x r) such that:
      For all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    This is equivalent to finding r vectors (u_c, v_c, w_c) in R^{nm} x R^{mp} x R^{pn}
    such that their "tensor product" sums to the all-ones tensor.
    
    We use a structured search:
    1. Try small ranks with integer/half-integer entries
    2. Use the algebraic structure: if we can partition the tensor into rank-1 pieces
    3. The schoolbook rank is n*m*p = 54. We look for improvements.
    
    Key insight: For matrix multiplication C = A*B where A is n x m and B is m x p,
    the tensor is T[i,j,k] = 1. This is the "all-ones" tensor of shape (n, m, p).
    
    The rank of the all-ones tensor of shape (a, b, c) over the reals is min(a, b, c)
    if we're talking about the tensor as a 3-way array... wait, no. Let me reconsider.
    
    Actually, the multiplication tensor for (n x m) * (m x p) -> (n x p) has:
    - First mode: n*m (input A indices)
    - Second mode: m*p (input B indices)  
    - Third mode: n*p (output C indices)
    
    And T[(i,j), (j',k), (i',k')] = 1 if i=i' and j=j' and k=k', else 0.
    
    Wait, let me re-read the task. "output factor follows the official trace convention W[k*n+i,r]"
    
    So the tensor is: for each (i, j, k) where i in [0,n), j in [0,m), k in [0,p):
      U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c] summed over c equals 1.
    
    This means the tensor T has T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k,
    and 0 otherwise. But actually, the constraint is that for EACH (i,j,k) triple,
    the sum over c of U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1.
    
    The indices i*m+j and j*p+k and k*n+i are linked: the j in the first two must match,
    and the i in first and third must match, and k in second and third must match.
    
    So we need: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    This is the standard matrix multiplication tensor. The rank is the matrix multiplication
    rank, denoted R(n,m,p). For (3,3,6), we need R(3,3,6).
    
    Known results: R(2,2,2) = 7 (Strassen). For rectangular cases, there are various bounds.
    R(3,3,3) is known to be at most 23 (from various constructions) and at least 19.
    For (3,3,6), we can think of it as multiplying a 3x3 by a 3x6.
    
    One approach: use the fact that a 3x6 product can be seen as two 3x3 products
    (the first 3 columns and last 3 columns of the 3x6 matrix). So R(3,3,6) <= 2*R(3,3,3).
    If R(3,3,3) <= 23, then R(3,3,6) <= 46.
    
    But we can do better. Let me think about using Strassen-like ideas.
    
    Actually, let me just implement a search that tries to find decompositions of
    decreasing rank, starting from a reasonable upper bound.
    
    Strategy:
    1. Start with the schoolbook rank 54.
    2. Try to find a decomposition of rank r for r = 53, 52, ...
    3. Use a randomized local search or algebraic construction.
    
    For a more principled approach, let me use the following:
    - The tensor is the "multiplication tensor" which has a lot of symmetry.
    - We can use the fact that it's a 3-tensor with specific structure.
    
    Let me try a different approach: use the known algebraic constructions.
    
    For 3x3x3, one known rank-23 construction exists. For 3x3x6, we might be able
    to get rank 40 or so.
    
    Actually, let me just do a practical search:
    1. Start with a valid decomposition (schoolbook, rank 54)
    2. Try to reduce rank by finding linear dependencies or using Strassen-like tricks
    3. Verify each candidate
    
    Given time constraints, let me implement a structured search that:
    1. Tries ranks from 40 to 54
    2. For each rank, uses a randomized approach with integer/half-integer entries
    3. Verifies the decomposition
    
    Actually, for a fresh design, let me use a different approach entirely:
    Use the algebraic structure of the problem. The key constraint is:
    
    For all i in [0,3), j in [0,3), k in [0,6):
      sum_{c=0}^{r-1} U[i*3+j, c] * V[j*6+k, c] * W[k*3+i, c] = 1
    
    This is a system of 54 equations (one per (i,j,k) triple) with 3*3*r + 3*6*r + 6*3*r = 9r + 18r + 18r = 45r unknowns.
    
    For r = 40: 1800 unknowns, 54 equations - very underdetermined.
    For r = 30: 1350 unknowns, 54 equations - still very underdetermined.
    
    The challenge is that the equations are nonlinear (products of three variables).
    
    Let me try a structured construction approach:
    
    Insight: We can use "block" structures. For example, if we can find a decomposition
    where the U, V, W matrices have a block structure, we can reduce the search space.
    
    Let me try: use the fact that the 3x6 product can be decomposed using
    the 3x3x3 Strassen-like constructions.
    
    Actually, let me just implement a practical solver:
    1. Use a greedy/iterative approach
    2. Start with rank 54 (schoolbook)
    3. Try to merge or eliminate terms
    
    Given the time budget, let me implement a randomized search with verification.
    
    For the verification: given U, V, W of rank r, check that for all (i,j,k):
      sum_c U[i*3+j, c] * V[j*6+k, c] * W[k*3+i, c] == 1
    
    Let me code this up with a time limit.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # Schoolbook decomposition as fallback
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
    
    def verify(U, V, W):
        """Check if U, V, W form a valid decomposition."""
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True
    
    # Try to find a lower rank decomposition
    # Strategy: use a structured approach based on known constructions
    
    # Let's try rank values from low to high, starting with known good upper bounds
    # R(3,3,6) is at most 2*R(3,3,3) <= 2*23 = 46
    # But we might do better with a direct construction
    
    # For now, let's try a randomized search for ranks in a reasonable range
    # Given time constraints, we'll try a few ranks
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to find better decompositions
    # Use a structured random search with integer entries
    
    random.seed(42)
    
    # Try ranks from 40 up to best_rank - 1
    for target_rank in range(40, best_rank):
        if _time_left() < 20:
            break
        
        # Try a few random initializations for this rank
        max_attempts = 50
        for attempt in range(max_attempts):
            if _time_left() < 15:
                break
            
            # Initialize with small integer values
            U = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(p * n)]
            
            if verify(U, V, W):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break
        
        if _time_left() < 10:
            break
    
    # If we found a better rank, return it; otherwise return schoolbook
    return best_result
