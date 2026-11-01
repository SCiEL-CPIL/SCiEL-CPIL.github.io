"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x4 matrix.
    Uses a structured search over factor matrices with small integer/half-integer entries.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=4
    
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    def validate(U, V, W, rank):
        """Check if U, V, W with given rank exactly decompose the multiplication tensor."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check valid entries: T[i*m+j, j*p+k, k*n+i] should be 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U_arr[a] * V_arr[b] * W_arr[c])
                    if abs(val - 1) > 1e-9:
                        return False
        
        # Check that T[a,b,c] = 0 when (a,b,c) doesn't correspond to valid multiplication
        for a in range(n * m):
            i, j = divmod(a, m)
            for b in range(m * p):
                j2, k = divmod(b, p)
                if j2 != j:
                    for c in range(p * n):
                        val = np.sum(U_arr[a] * V_arr[b] * W_arr[c])
                        if abs(val) > 1e-9:
                            return False
                else:
                    valid_c = k * n + i
                    for c in range(p * n):
                        val = np.sum(U_arr[a] * V_arr[b] * W_arr[c])
                        if c == valid_c:
                            if abs(val - 1) > 1e-9:
                                return False
                        else:
                            if abs(val) > 1e-9:
                                return False
        return True

    # Create schoolbook solution of rank 24
    rank_schoolbook = n * m * p  # 24
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_solution = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook
    
    random.seed(42)
    
    # Candidate values for entries
    candidates = [0, 0.5, -0.5, 1, -1]
    
    # Try to find lower rank decompositions using a more systematic approach
    # Based on the structure of matrix multiplication tensors, we can exploit
    # the fact that each output element (i,k) depends on a sum over j
    
    # For 2x3 * 3x4, the tensor has 24 nonzero entries
    # A rank-r decomposition means we can write the tensor as sum of r rank-1 tensors
    
    # Let's try a structured approach: for each rank component r, we choose
    # a subset of (i,j) pairs for U, (j,k) pairs for V, and (i,k) pairs for W
    # such that the contributions add up correctly
    
    # The key insight: for each (i,k) pair, we need sum_j U[i,j,r] * V[j,k,r] * W[i,k,r] = 1
    # This is a system of bilinear equations in the variables U, V, W
    
    # Let's try to solve this systematically for small ranks
    # We'll use a randomized local search approach with better initialization
    
    # First, let's try to find a rank-18 or lower solution using a more informed search
    # The structure suggests that we can group terms by the middle index j
    
    # Alternative approach: use the fact that for fixed (i,k), the equation is
    # sum_j (U[i,j,r] * V[j,k,r]) * W[i,k,r] = 1
    # If we fix W[i,k,r], then we need sum_j (U[i,j,r] * V[j,k,r]) = 1/W[i,k,r]
    
    # Let's try a different strategy: start from a known decomposition and try to merge terms
    # or use a greedy approach to find sparse solutions
    
    # For now, let's improve the random search with better heuristics
    for target_rank in range(12, 24):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random sparse initializations with better structure
        num_attempts = 30 if target_rank <= 15 else 15
        for attempt in range(num_attempts):
            if time.time() > deadline - 3:
                break
            
            U = [[0.0] * target_rank for _ in range(n * m)]
            V = [[0.0] * target_rank for _ in range(m * p)]
            W = [[0.0] * target_rank for _ in range(p * n)]
            
            # Initialize with a more structured approach
            # Each rank component should contribute to a consistent set of (i,j,k) triples
            for r in range(target_rank):
                # Choose a "pattern" for this rank component
                # Pattern: which (i,j,k) triples this component affects
                # For efficiency, each component should affect a small number of triples
                
                # Choose 2-4 (i,j,k) triples for this component
                num_triples = random.choice([2, 3, 4])
                triples = []
                for _ in range(num_triples):
                    i = random.randint(0, n-1)
                    j = random.randint(0, m-1)
                    k = random.randint(0, p-1)
                    triples.append((i, j, k))
                
                # For each triple, set the corresponding U, V, W entries
                for (i, j, k) in triples:
                    # Choose values such that the product contributes meaningfully
                    # We want U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] to be nonzero
                    val_u = random.choice([0.5, -0.5, 1, -1])
                    val_v = random.choice([0.5, -0.5, 1, -1])
                    val_w = random.choice([0.5, -0.5, 1, -1])
                    
                    U[i * m + j][r] += val_u
                    V[j * p + k][r] += val_v
                    W[k * n + i][r] += val_w
            
            if validate(U, V, W, target_rank):
                best_solution = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break
        
        if best_rank == target_rank:
            break
    
    if best_rank < rank_schoolbook:
        return best_solution
    else:
        return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
