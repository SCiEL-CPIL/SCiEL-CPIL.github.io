"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Attempt to find a rank-minimal exact tensor decomposition for 2x4 * 4x6 matrix multiplication.
    
    The multiplication tensor T has indices (i, j, k) with i in [0,n), j in [0,m), k in [0,p).
    A decomposition of rank r consists of matrices U (n*m x r), V (m*p x r), W (p*n x r) such that
    for all i,j,k: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    Since T[i,j,k] = 1 for all valid i,j,k, we need:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    We use a randomized search with exact rational arithmetic to find a low-rank decomposition.
    
    Strategy:
    1. Start with the schoolbook rank (n*m*p = 48) as a baseline.
    2. Use a structured ansatz: try rank r from a lower bound up to 48.
    3. For each rank r, use a parameterized family of decompositions and solve for parameters.
    4. Use random initialization + Newton-style refinement in exact rationals.
    
    For small dimensions like 2x4x6, the known minimum rank might be achievable through
    structured decompositions. We try:
    - Rank 12: if possible, this would be a 4x improvement over schoolbook.
    - More realistically, we search for ranks between known lower bounds and 48.
    
    A key insight: for matrix multiplication of (n x m) by (m x p), the tensor has a
    "triangular" structure. We can use block-circulant or other structured forms.
    
    Let's use a direct approach: parameterize U, V, W with free variables and solve the
    system of n*m*p equations.
    """
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 48
    
    # Deadline guard: 150s limit, use 140s as safety margin
    start_time = time.time()
    deadline = start_time + 140
    
    def elapsed():
        return time.time() - start_time
    
    # Build the multiplication tensor T[i,j,k] = 1 for all i,j,k
    # We need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = max(8, 24, 12) = 24
    # But we can do better with structured decompositions
    
    best_rank = schoolbook_rank
    best_result = None
    
    # First, create the schoolbook decomposition as a fallback
    def schoolbook():
        r = schoolbook_rank
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try a structured decomposition based on Kronecker products
    # For 2x4 * 4x6, we can think of this as combining smaller multiplications
    
    # Ansatz: use a sum of Kronecker products
    # U = sum_a U_a ⊗ something, etc.
    
    # Let's try rank r = 12, 16, 20, 24, 28, 32, 36, 40, 44, 48
    # and for each, try random rational initializations and solve
    
    def try_rank(r, max_attempts=50):
        """Try to find a rank-r decomposition using randomized search."""
        if r >= best_rank:
            return None
        
        n_rows_U = n * m  # 8
        n_rows_V = m * p  # 24
        n_rows_W = p * n  # 12
        
        # We have r columns in each of U, V, W
        # Total unknowns: r * (n_rows_U + n_rows_V + n_rows_W) = r * 44
        # Constraints: n*m*p = 48 equations
        # So for r >= 2, we have more unknowns than constraints (for r=2: 88 unknowns, 48 constraints)
        # But the equations are nonlinear (trilinear)
        
        # Use a different approach: fix W to have simple values and solve for U, V
        # Or use a tensor network contraction approach
        
        # Let's try: assume W[k*n+i, c] takes values in {-1, 0, 1}
        # Then for each c, we have a rank-1 contribution
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        
        # This is still hard. Let's use a simpler structured ansatz.
        
        # Structured ansatz: block decomposition
        # Think of the 4x6 product as two 4x3 blocks, or the 2x4 as two 2x2 blocks
        
        # Ansatz 1: U[i*m+j, c] = a[i,j,c], V[j*p+k,c] = b[j,k,c], W[k*n+i,c] = d[k,i,c]
        # with the constraint that they factorize in a structured way
        
        # For 2x4 * 4x6, one natural structure:
        # Split the inner dimension 4 into 2+2
        # Then we can use a "block" decomposition
        
        # Let's try rank = 12 with a specific structure:
        # c indexes (i_block, j_block, k_block) where i_block in {0,1}, j_block in {0,1}, k_block in {0,1,2}
        # No, that gives 2*2*3 = 12
        
        # Actually, let's just do a direct linear algebra approach for small ranks.
        
        # For rank r, we can set up the equations as a system and use Groebner bases
        # or iterative refinement. For now, let's try a greedy approach.
        
        # Greedy: build the decomposition one column at a time
        # At each step, choose the next column to minimize the residual
        
        # Initialize residual R[i,j,k] = 1 for all i,j,k
        # For c in range(r):
        #   Find u, v, w that best match the residual
        #   R -= outer(u, v, w)
        
        # But "best match" for a rank-1 tensor is non-trivial.
        
        # Alternative: use the fact that for matrix multiplication, there are known
        # decompositions. For 2x2 * 2x2, Strassen's algorithm gives rank 7.
        # For 2x4 * 4x6, we can try to compose smaller decompositions.
        
        # Let's try to use a Kronecker product structure:
        # (2x4) * (4x6) can be seen as (2x2) * (2x6) with some duplication,
        # or (2x4) * (4x2) * (2x3), etc.
        
        # Simplest: just return schoolbook if we can't find better
        return None
    
    # Try a few specific structured decompositions
    
    # Structure 1: Use the fact that 4 = 2*2 and 6 = 2*3
    # Represent the 4x6 matrix as a 2x2 block matrix of 2x3 blocks
    # This might allow a rank reduction
    
    # For now, let's try a randomized search with exact rationals for small ranks
    # This is computationally expensive, so we limit the search
    
    # Try rank 12, 16, 20 with a simple randomized approach
    for r in [12, 14, 16, 18, 20, 22, 24]:
        if elapsed() > deadline - 20:
            break
        
        # Random initialization with small rationals
        found = False
        for attempt in range(10):
            if elapsed() > deadline - 10:
                break
            
            # Randomly initialize U, V, W with entries in {-1, 0, 1}
            U = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(n*m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(m*p)]
            W = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(p*n)]
            
            # Check if this satisfies the equations
            valid = True
            for i in range(n):
                if not valid:
                    break
                for j in range(m):
                    if not valid:
                        break
                    for k in range(p):
                        s = 0
                        for c in range(r):
                            s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                        if s != 1:
                            valid = False
                            break
            
            if valid:
                return {"U": U, "V": V, "W": W}
        
        # Try a more structured approach: use the multiplication table structure
        # For c in range(r), let U[i*m+j,c] = f_c(i,j), V[j*p+k,c] = g_c(j,k), W[k*n+i,c] = h_c(k,i)
        # We need sum_c f_c(i,j) * g_c(j,k) * h_c(k,i) = 1
    
    # If we didn't find a better decomposition, return schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
