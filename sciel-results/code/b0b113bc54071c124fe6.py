"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 4x4 matrix by a 4x9 matrix.
    
    Starts with schoolbook (rank 144) and attempts to find lower-rank
    exact decompositions via randomized search with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    nm = n * m  # 16
    mp = m * p  # 36
    pn = p * n  # 36
    
    deadline = time.time() + 130  # 130s to be safe before 150s kill
    
    # First, build the schoolbook decomposition as fallback
    rank_school = n * m * p  # 144
    U_school = np.zeros((nm, rank_school), dtype=object)
    V_school = np.zeros((mp, rank_school), dtype=object)
    W_school = np.zeros((pn, rank_school), dtype=object)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # The tensor T[i, j, k] = 1 for all i,j,k
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Try to find lower rank decompositions
    # Strategy: try ranks from 143 downward, use randomized search
    
    # For efficiency, work with the constraint that entries are integers or half-integers
    # Scale by 2 to work with integers: U', V', W' where U = U'/2, etc.
    # Then sum_r (U'/2) * (V'/2) * (W'/2) = 1 => sum_r U'*V'*W' = 8
    
    # Try a few lower ranks with randomized search
    # Focus on ranks that might be achievable
    
    # Known bounds: for 4x4*4x9, the minimum rank is at least max(nm, mp, pn) = 36
    # and at most 144
    
    # Try ranks: 100, 90, 80, 70, 60, 50, 45, 40, 38, 36
    candidate_ranks = [120, 100, 90, 80, 70, 60, 50, 45, 42, 40, 38, 36]
    
    for target_rank in candidate_ranks:
        if time.time() > deadline - 5:
            break
        
        # Try to find a decomposition of this rank
        # Use randomized approach: initialize with random sparse matrices
        # and try to satisfy the constraints
        
        # For this small problem, we can try a more systematic approach
        # The constraint is: for each (i,j,k), sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] = 1
        
        # Let's try a structured approach: 
        # Use the fact that we can combine some schoolbook terms
        
        # Actually, let's try a simple heuristic: 
        # Start with schoolbook and try to merge terms
        
        # Alternative: use a known result - for matrix multiplication,
        # the rank of C_{n,m,p} is at most n*m*p, and there are known
        # improvements for specific cases.
        
        # For 4x4*4x9, let's try to find if rank < 144 is possible
        # by checking if we can express the tensor with fewer terms.
        
        # Simple test: can we do rank 143? 
        # This would require one of the tensor entries to be expressible
        # as a sum of products from the other terms.
        
        # Given the complexity, let's try a randomized search for lower ranks
        # with a time limit per rank attempt
        
        rank_attempt_deadline = min(time.time() + 20, deadline - 5)
        
        # Try a few random initializations
        for attempt in range(50):
            if time.time() > rank_attempt_deadline:
                break
            
            # Initialize with small random integer values
            # Scale: we want U*V*W sum to equal 1, so use small values
            U_r = np.zeros((nm, target_rank), dtype=object)
            V_r = np.zeros((mp, target_rank), dtype=object)
            W_r = np.zeros((pn, target_rank), dtype=object)
            
            # Initialize sparsely - each row gets a few non-zero entries
            for r in range(target_rank):
                # Assign each term to cover some (i,j,k) combinations
                # Randomly choose which (i,j,k) this term helps with
                i = random.randint(0, n-1)
                j = random.randint(0, m-1)
                k = random.randint(0, p-1)
                
                # Set the corresponding entries
                U_r[i*m+j, r] = random.choice([1, 1, 1, -1])
                V_r[j*p+k, r] = random.choice([1, 1, 1, -1])
                W_r[k*n+i, r] = random.choice([1, 1, 1, -1])
            
            # Check if this satisfies the constraints
            # For each (i,j,k), compute sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r]
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for r in range(target_rank):
                            s += U_r[i*m+j, r] * V_r[j*p+k, r] * W_r[k*n+i, r]
                        if s != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_rank = target_rank
                best_U = U_r
                best_V = V_r
                best_W = W_r
                break
        
        if best_rank == target_rank:
            break
    
    # Convert to lists for return
    U_list = [[int(U_list_val) for U_list_val in row] for row in best_U.tolist()]
    V_list = [[int(V_list_val) for V_list_val in row] for row in best_V.tolist()]
    W_list = [[int(W_list_val) for W_list_val in row] for row in best_W.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
