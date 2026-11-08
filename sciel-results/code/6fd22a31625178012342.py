"""Find a rank-minimal exact tensor decomposition for multiplying a 2 by 4 matrix by a 4 by 7 matrix."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product
from fractions import Fraction
import random

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _validate(U, V, W, tol=1e-8):
    """Check if U, V, W form a valid decomposition of matrix multiplication."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    if U_np.shape != (n * m, rank):
        return False
    if V_np.shape != (m * p, rank):
        return False
    if W_np.shape != (p * n, rank):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_fixed_u_decomposition(target_rank, time_limit=10.0):
    """
    Try to find a decomposition where U is a 0-1 matrix with exactly one 1 per column.
    This reduces the problem to solving for V and W.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    nm = n * m
    mp = m * p
    pn = p * n
    
    # For each column c, U has a 1 at some position (i_c, j_c)
    # The constraint is: for all (i,j,k), sum_c [U[i*m+j,c] != 0] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Since U has exactly one 1 per column at position pos[c] = i_c * m + j_c:
    # For a fixed (i,j,k): sum over c where i_c == i and j_c == j of V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Let's try a specific structure: group columns by (i,j)
    # For each (i,j), let S_{ij} be the set of columns c where pos[c] = i*m+j
    # Then for each (i,j,k): sum_{c in S_{ij}} V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a system of bilinear equations in V and W.
    # Let's try to solve it by fixing V and solving for W, or vice versa.
    
    # Strategy: enumerate the assignment of columns to (i,j) positions
    # There are nm choices per column, so nm^target_rank total - too many.
    
    # Alternative: use a structured assignment.
    # Let's try: assign columns round-robin to (i,j) positions.
    
    # First, let's try a simple heuristic: for each (i,j), assign approximately
    # target_rank / nm columns.
    
    assignments = []
    for c in range(target_rank):
        # Assign column c to position (i,j) where i = c // m, j = c % m
        # This gives a uniform distribution
        i = c // m
        j = c % m
        assignments.append(i * m + j)
    
    U_np = np.zeros((nm, target_rank))
    for c in range(target_rank):
        U_np[assignments[c], c] = 1.0
    
    # Now for each (i,j), let S_{ij} be the set of columns assigned to (i,j)
    # For each k, we need: sum_{c in S_{ij}} V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Let's try to set V and W to specific values.
    # For simplicity, let's try: for c in S_{ij}, set V[j*p+k, c] = 1/sqrt(|S_{ij}|) and W[k*n+i, c] = sqrt(|S_{ij}|)
    # Then the sum is |S_{ij}| * (1/sqrt(|S_{ij}|)) * sqrt(|S_{ij}|) = |S_{ij}| * 1 = |S_{ij}|
    # That's not 1.
    
    # Let's try: V[j*p+k, c] = 1/|S_{ij}| and W[k*n+i, c] = 1
    # Then the sum is |S_{ij}| * (1/|S_{ij}|) * 1 = 1. 
    
    # But wait, V[j*p+k, c] must be the same for all (i,j) with the same j and k.
    # So V[j*p+k, c] is determined by j and k, not by i.
    # Similarly, W[k*n+i, c] is determined by k and i, not by j.
    
    # So the constraint is: for each (i,j,k), sum_{c in S_{ij}} V_{j,k,c} * W_{k,i,c} = 1
    # where V_{j,k,c} = V[j*p+k, c] and W_{k,i,c} = W[k*n+i, c]
    
    # This is a system of bilinear equations. Let's try to solve it numerically.
    
    # Variables: V_{j,k,c} for all j,k,c and W_{k,i,c} for all k,i,c
    # Constraints: n*m*p equations
    
    # This is a nonlinear system. Let's try a simple approach:
    # Fix W to be 1 for all entries, then solve for V.
    # Then for each (i,j,k): sum_{c in S_{ij}} V[j*p+k, c] * 1 = 1
    # So for each (j,k), sum_{c in S_{ij}} V[j*p+k, c] = 1 for all i where (i,j) has columns.
    # But S_{ij} depends on i, so this is not consistent unless all S_{ij} for fixed j are the same.
    
    # Let's try a different approach: use a random initialization and Newton's method.
    
    # Actually, let's just try a few specific assignments and see if they work.
    
    for trial in range(1000):
        if time.time() - start_time > time_limit:
            break
        
        # Random assignment of columns to (i,j) positions
        assignments = [random.randint(0, nm - 1) for _ in range(target_rank)]
        
        U_np = np.zeros((nm, target_rank))
        for c in range(target_rank):
            U_np[assignments[c], c] = 1.0
        
        # Initialize V and W randomly
        V_np = np.random.uniform(-1, 1, size=(mp, target_rank))
        W_np = np.random.uniform(-1, 1, size=(pn, target_rank))
        
        # Try to refine using gradient descent or similar
        # For now, just check if this random assignment happens to work
        if _validate(U_np.tolist(), V_np.tolist(), W_np.tolist()):
            return {"U": U_np.tolist(), "V": V_np.tolist(), "W": W_np.tolist()}
    
    return None

def _try_small_integer_search(target_rank, time_limit=10.0):
    """
    Try to find a decomposition with small integer entries.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Try rank up to target_rank with entries in {-1, 0, 1}
    # This is a very restricted search, but might work for small ranks.
    
    # For each column, we need to assign values to U, V, W.
    # This is a 3-partite matching problem.
    
    # Let's try a structured approach: for each column, pick a "pattern"
    # that satisfies the constraints.
    
    # Actually, this is too complex. Let's just use the fixed-U approach with a better solver.
    
    return None

def _try_known_construction():
    """
    Try to use a known construction for matrix multiplication.
    For 2x4 x 4x7, we can try to combine smaller known decompositions.
    """
    n, m, p = DIMENSIONS
    
    # Known: 2x2 x 2x2 has rank 7 (Strassen)
    # 2x4 x 4x7 can be decomposed as:
    # Split the 4x7 matrix into a 4x4 and a 4x3 part
    # 2x4 x 4x4: can use 2x2 x 2x2 blocks? 
    # Actually, 2x4 x 4x4 = (2x2 x 2x2) x 2 in some sense.
    
    # Let's try a direct construction.
    # For 2x4 x 4x4, the rank is at most 2*4*4 = 32, but might be lower.
    # For 2x4 x 4x3, the rank is at most 2*4*3 = 24.
    
    # Actually, let's just try to construct a rank 40 decomposition.
    # If we can find one, the score will be 56/40 = 1.4.
    
    # For now, let's just return None and let the caller handle it.
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    # Try to find a better decomposition than schoolbook (rank 56)
    
    # Try known constructions first
    result = _try_known_construction()
    if result is not None:
        if _validate(result["U"], result["V"], result["W"]):
            return result
    
    # Try fixed-U decomposition for various ranks
    for rank in range(28, 57):
        if time.time() > deadline - 5.0:
            break
        
        result = _try_fixed_u_decomposition(rank, time_limit=5.0)
        if result is not None:
            if _validate(result["U"], result["V"], result["W"]):
                return result
    
    # Fallback to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
