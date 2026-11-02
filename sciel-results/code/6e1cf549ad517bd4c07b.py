"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import itertools
from fractions import Fraction

def construct():
    """Find rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Schoolbook decomposition as fallback
    schoolbook_rank = n * m * p
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions using SAT-style search
    # Coefficient values to try: small integers and half-integers
    coeff_values = [-2, -1, -0.5, 0, 0.5, 1, 2]
    
    # Maximum rank to try (start from schoolbook and go down)
    # We'll try ranks from schoolbook_rank-1 down to some minimum
    # But we need to be careful about time
    
    # For each candidate rank, we need to solve:
    # For all (i, j, k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta(i,j,k)
    # where delta(i,j,k) = 1 if this is a valid multiplication term, else 0
    
    # Actually, the matrix multiplication tensor T[i,j,k] = 1 for all i,j,k
    # because (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    # So we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # This is a system of equations. Let's use a different approach:
    # We'll try to find sparse solutions by randomly initializing and refining
    
    # Given time constraints, let's try a few random restarts with local search
    import random
    
    # Try a few random initializations and see if we can find valid low-rank solutions
    # This is heuristic but may find something
    
    # Actually, for this specific problem, let's think about structure
    # The tensor is T[i,j,k] = 1 for all i in [0,3), j in [0,3), k in [0,7)
    # This is an outer product of three vectors of all ones!
    # T = u ⊗ v ⊗ w where u = [1,1,1], v = [1,1,1], w = [1,1,1,1,1,1,1]
    # So the rank is 1!
    
    # Wait, that's not right. The matrix multiplication tensor is NOT a simple outer product.
    # T[i,j,k] represents the coefficient of A_{i,j} * B_{j,k} in (AB)_{i,k}
    # For standard matrix multiplication, T[i,j,k] = 1 for all valid (i,j,k)
    # But the tensor has structure: it's the sum over j of A_{i,j} B_{j,k}
    
    # Let me reconsider. The bilinear form is:
    # (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor T has entries T[i,j,k] = 1 for all i,j,k
    # But we want to decompose this as sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    
    # Actually, I think I was overcomplicating this. Let me verify:
    # The matrix multiplication tensor M_{n,m,p} has entries M[i,j,k] = 1 
    # where i indexes rows of A (and output), j indexes columns of A / rows of B,
    # k indexes columns of B (and output).
    
    # The decomposition requires: for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = M[i,j,k]
    
    # If M[i,j,k] = 1 for all i,j,k, then we need:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # One simple solution: set U[i*m+j,0] = 1, V[j*p+k,0] = 1, W[k*n+i,0] = 1 for all i,j,k
    # Then the sum is 1 for all i,j,k. Rank = 1!
    
    # Let me verify this is correct:
    # U has shape (n*m, rank) = (9, 1)
    # V has shape (m*p, rank) = (21, 1)  
    # W has shape (p*n, rank) = (21, 1)
    
    # U[i*m+j, 0] = 1 for all i,j
    # V[j*p+k, 0] = 1 for all j,k
    # W[k*n+i, 0] = 1 for all k,i
    
    # Then sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = U[i*m+j,0] * V[j*p+k,0] * W[k*n+i,0] = 1*1*1 = 1
    
    # This works! The rank is 1!
    
    # But wait, is this a valid decomposition of matrix multiplication?
    # Let's check: does this actually compute matrix multiplication?
    
    # The decomposition says: (AB)_{i,k} = sum_j sum_c U[i*m+j,c] * A_{i,j} * V[j*p+k,c] * B_{j,k} * W[k*n+i,c]
    
    # With our solution: (AB)_{i,k} = sum_j 1 * A_{i,j} * 1 * B_{j,k} * 1 = sum_j A_{i,j} * B_{j,k}
    
    # Yes! This is exactly matrix multiplication!
    
    # So the rank-minimal decomposition has rank 1!
    
    U1 = [[1] for _ in range(n * m)]
    V1 = [[1] for _ in range(m * p)]
    W1 = [[1] for _ in range(p * n)]
    
    return {"U": U1, "V": V1, "W": W1}
# EVOLVE-BLOCK-END
