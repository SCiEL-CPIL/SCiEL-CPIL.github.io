"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    # Need: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Try to find minimal rank decomposition
    # Lower bound: max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    
    # Strategy: Try constructive decompositions for various ranks
    # Start with schoolbook and try to improve
    
    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]
        # Check all (i,j,k) triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr] for rr in range(r))
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def to_half_int_list(mat):
        """Convert numpy array to list of lists, ensuring half-integer values."""
        return [[float(mat[i][j]) for j in range(mat.shape[1])] for i in range(mat.shape[0])]
    
    # Schoolbook as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    best = (rank_sb, U_sb, V_sb, W_sb)
    
    # Try to find better decompositions using algebraic structure
    # Approach: Use the fact that we can combine terms
    
    # Try rank = m*p = 28
    # For this, we need U (12x28), V (28x28), W (21x28)
    # Idea: For each (j,k) pair, have a term, but share across i
    
    # Let's try a structured approach:
    # Group by k: for each k, we need to handle all (i,j) for that k
    # W[k*n+i, r] depends on k and i
    
    # Try a decomposition based on separating the j dimension
    # For each j, we have a 3x7 sub-problem (n x p)
    
    # Alternative: Use bilinear structure
    # C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_j A[i,j] * (row j of B, column k)
    
    # Let me try a direct algebraic search for small improvements over schoolbook
    # by combining terms that have the same (i,k) but different j
    
    # Key insight: For fixed (i,k), the equation is:
    # sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # If W[k*n+i,r] is the same for all j (which it is, since W doesn't depend on j),
    # then: W[k*n+i,r] * sum_j U[i*m+j,r] * V[j*p+k,r] contributes to the (i,k) entry
    
    # So for each (i,k) and r: W[k*n+i,r] * (sum over j of U[i*m+j,r]*V[j*p+k,r]) 
    # and the sum over r of these must equal 1
    
    # This suggests: for each r, define for each (i,k): 
    #   S_r[i,k] = sum_j U[i*m+j,r] * V[j*p+k,r]
    # Then we need: sum_r W[k*n+i,r] * S_r[i,k] = 1 for all (i,k)
    
    # This is a system where S_r are 3x7 matrices and W columns are 21-vectors
    # that must satisfy a bilinear constraint
    
    # Let me try a constructive approach with specific structure
    
    # Try: rank = 42 (2 * n * p)
    # For each (i,k), use 2 terms
    
    # Actually, let me try a more systematic search
    # For rank r, try to find solutions with small integer entries
    
    def try_rank(r, max_entries=3):
        """Try to find a rank-r decomposition with entries in {-max_entries,...,max_entries} * 0.5"""
        if time.time() > DEADLINE - 10:
            return None
        
        # This is a hard nonlinear system. Let's use a heuristic approach.
        # Initialize with random values and try to refine.
        
        # Actually, for exact solutions with small integers, let's try a structured approach.
        
        # Try: U has entries in {0, 1}, V has entries in {0, 1}, W has entries in {0, 1}
        # This gives: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # Each term is 0 or 1, so we need exactly one r where all three are 1.
        # This is like a set cover problem.
        
        # For binary entries, the equation becomes:
        # For each (i,j,k): exactly one r has U[i*m+j,r]=V[j*p+k,r]=W[k*n+i,r]=1
        
        # This is equivalent to: the sets {r : U[i*m+j,r]=1} for different (i,j) 
        # intersect with {r : V[j*p+k,r]=1} and {r : W[k*n+i,r]=1} in exactly one element
        
        # Let's try a different structure: allow entries in {0, 1, -1}
        
        # For now, return None if we can't find a quick solution
        return None
    
    # Let's try a specific constructive decomposition
    # Based on the structure of the problem
    
    # Approach: Use the fact that n=3, m=4, p=7
    # Try to find a decomposition of rank 42 or lower
    
    # Method: For each (i,k) pair (21 pairs), we need the sum over j and r to give 1.
    # If we can make W[k*n+i,r] = 1 for r in some set and 0 otherwise,
    # and arrange U and V appropriately...
    
    # Let me try: rank = 42, with the following structure:
    # For r = 0..20: handle one (i,k) pair, using all j
    # For r = 21..41: handle corrections
    
    # Actually, let me try a simpler approach: just use schoolbook but with
    # some terms combined if possible.
    
    # For now, return schoolbook if we can't find better quickly.
    if time.time() > DEADLINE - 5:
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try a few specific rank values with a simple search
    for r in [28, 29, 30, 35, 40, 42, 49, 56, 63, 70, 77]:
        if time.time() > DEADLINE - 5:
            break
        result = try_rank(r)
        if result is not None:
            U, V, W = result
            if verify(U, V, W):
                best = (r, U, V, W)
                break
    
    return {"U": best[1], "V": best[2], "W": best[3]}
# EVOLVE-BLOCK-END
