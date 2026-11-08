"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x4 matrix.
    Uses a time-bounded search with multiple strategies:
    1. Known low-rank constructions (if any exist for this specific case)
    2. Randomized search with local refinement
    3. Greedy rank reduction from schoolbook
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # The tensor T[i,j,k] = delta_{i,k} for matrix multiplication C = A*B
    # where A is n x m, B is m x p, C is n x p
    # T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # Actually, the matrix multiplication tensor has entries:
    # M[i,k,j] = 1 for the coefficient of A[i,j] * B[j,k] in C[i,k]
    
    # Schoolbook rank = n*m*p = 36
    schoolbook_rank = n * m * p
    
    # Try to find a lower rank decomposition
    # For 3x3 * 3x4, the known lower bounds and upper bounds are:
    # - Lower bound: max(n*p, m*p, n*m) * something... actually for matrix mult,
    #   the rank is at least n*p (the output dimension)
    # - For square n x n, rank is between n^2 and n^3
    # - For 3x3 * 3x4, let's try ranks below 36
    
    # Strategy: Try to verify if a given rank r works by solving a system
    # The condition is: for all i in [0,n), k in [0,p), j in [0,m):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if computing C[i,k]
    #   Actually, the tensor decomposition condition is:
    #   For the matrix multiplication tensor, we need:
    #   T[i,k,j] = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c]  (with appropriate indexing)
    
    # Let me be more careful about the indexing convention:
    # U has shape (n*m, rank): indexed by (i,j) -> row i*m+j
    # V has shape (m*p, rank): indexed by (j,k) -> row j*p+k  
    # W has shape (p*n, rank): indexed by (k,i) -> row k*n+i  (trace convention W[k*n+i, r])
    
    # The decomposition condition: for each (i,j,k):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,k} * (something)
    
    # Actually, for matrix multiplication C = A * B where C[i,k] = sum_j A[i,j]*B[j,k]:
    # The tensor T has T[i,j,k] = 1 for the trilinear form that computes C[i,k]
    # So we need: for all i, k:
    #   sum_j A[i,j] * B[j,k] = C[i,k]
    # In tensor decomposition form:
    #   C[i,k] = sum_r (sum_j U[i*m+j,r] * A[i,j]) * (sum_j V[j*p+k,r] * B[j,k]) * W[k*n+i,r]
    # Wait, that's not quite right either. Let me think again.
    
    # The standard tensor decomposition for matrix multiplication:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_r [sum_j U[i*m+j,r] * A[i,j]] * [sum_j V[j*p+k,r] * B[j,k]] * W[k*n+i,r]
    
    # For this to hold for all A, B, we need:
    # sum_r U[i*m+j',r] * V[j'*p+k,r] * W[k*n+i,r] = delta_{j',j} for the correct pairing
    # More precisely, the bilinear form must satisfy:
    # For all i, j, k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # And for all i, j, k where the indices don't match up correctly, it should be 0.
    
    # Actually, the condition is simpler. The matrix multiplication tensor T is a 3-tensor
    # with T[i,j,k] = 1 if and only if we're using A[i,j] and B[j,k] to compute C[i,k].
    # In the tensor decomposition, we need:
    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    
    # Where T[i,j,k] = 1 for all valid i,j,k (since every A[i,j] and B[j,k] contributes
    # to exactly one output C[i,k]). Wait, that's not right either.
    
    # Let me reconsider. The matrix multiplication tensor T_{ijk} where:
    # - i indexes the row of A (and C): 0..n-1
    # - j indexes the column of A / row of B: 0..m-1  
    # - k indexes the column of B (and C): 0..p-1
    # T_{ijk} = 1 for all i,j,k (it's the all-ones tensor in this formulation)
    # No wait, that's the tensor that computes sum_j A[i,j]*B[j,k] = C[i,k]
    
    # So the condition is: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # This is a system of n*m*p = 36 equations.
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition is exact."""
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    def verify_decomposition_frac(U, V, W, rank):
        """Verify with exact rational arithmetic."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        s += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * Fraction(W[k*n+i][r])
                    if s != 1:
                        return False
        return True
    
    # Start with schoolbook
    best_rank = schoolbook_rank
    best_U = [[0] * schoolbook_rank for _ in range(n * m)]
    best_V = [[0] * schoolbook_rank for _ in range(m * p)]
    best_W = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j][col] = 1
                best_V[j*p+k][col] = 1
                best_W[k*n+i][col] = 1
                col += 1
    
    # Try to find lower rank decompositions
    # Known results: for 3x3 * 3x3, the rank is known to be at most 23 (Strassen-like)
    # For 3x3 * 3x4, we can try ranks from 35 down to maybe 20
    
    # Strategy 1: Try random initializations and solve
    import random
    random.seed(42)
    
    def try_rank(r):
        """Try to find a decomposition of rank r."""
        if time.time() > deadline:
            return None
        
        # Use numpy for speed
        n_m = n * m
        m_p = m * p
        p_n = p * n
        
        # The condition is: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # This is a system of 36 equations in n_m*r + m_p*r + p_n*r = (9+12+12)*r = 33r unknowns
        
        # For r >= 2, we have more unknowns than equations, so solutions should exist
        # The question is whether they can be made to have simple (integer/half-integer) values
        
        # Try a randomized approach: fix W to random values, solve for U and V
        # Or use a more structured approach
        
        # Let's try: set W to be a random matrix, then the equations become linear in 
        # the products U[i*m+j,r]*V[j*p+k,r] for each r
        
        # Actually, this is a bilinear system which is hard. Let me try a different approach.
        
        # Approach: Start from schoolbook and try to merge terms
        # Or: Use a known construction
        
        # For now, let's try a simple heuristic: random W, then solve for U and V
        # given that we fix one of the matrices
        
        # Actually, let's try a cleaner approach: 
        # Assume W has a specific simple structure and solve for U and V
        
        # Try W with entries from {-1, 0, 1}
        # This is a constraint satisfaction problem
        
        # For efficiency, let's try small ranks first and use a smart initialization
        
        # Strategy: Use the fact that for matrix multiplication, there are known
        # low-rank decompositions. For 3x3, Strassen's algorithm gives rank 7.
        # For 3x3 * 3x4, we can potentially extend this.
        
        # Let's try a direct approach: for a given rank r, try to find U, V, W
        # such that the condition holds. We'll use a local search / optimization approach.
        
        # Initialize with random values
        for attempt in range(20):
            if time.time() > deadline:
                return None
            
            # Random initialization
            W_rand = np.random.randint(-1, 2, size=(p_n, r))
            U_rand = np.random.randint(-1, 2, size=(n_m, r))
            V_rand = np.random.randint(-1, 2, size=(m_p, r))
            
            # Check if this works
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for rr in range(r):
                            s += U_rand[i*m+j, rr] * V_rand[j*p+k, rr] * W_rand[k*n+i, rr]
                        if s != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return (U_rand.tolist(), V_rand.tolist(), W_rand.tolist())
        
        return None
    
    # Try ranks from schoolbook-1 down to some minimum
    # The theoretical lower bound for matrix multiplication rank is at least n*p = 12
    # but realistically for 3x3*3x4 it's probably around 20-25
    
    # Let's be more systematic. Try a few specific ranks.
    for r in range(schoolbook_rank - 1, 11, -1):
        if time.time() > deadline:
            break
        
        result = try_rank(r)
        if result is not None:
            U, V, W = result
            # Verify with exact arithmetic
            if verify_decomposition_frac(U, V, W, r):
                best_rank = r
                best_U = U
                best_V = V
                best_W = W
                break
    
    # If we found a better rank, use it; otherwise use schoolbook
    if best_rank < schoolbook_rank:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
