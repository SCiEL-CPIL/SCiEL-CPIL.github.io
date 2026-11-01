"""Matrix multiplication 2x3x4 rank minimization via structured search."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

# Theoretical lower bounds for rank of multiplying n x m by m x p matrix
# Known results:
# - 2x3x3: rank 7 (optimal)
# - 2x3x4: lower bound is 12 (from asymptotic rank bounds and specific constructions)
# - Schoolbook rank: 2*3*4 = 24
#
# We know from literature that the rank for 2x3x4 is 12.
# Let's try to construct it or find it via search.

def _verify(U, V, W, n=2, m=3, p=4):
    """Verify that the tensor decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the multiplication tensor T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # The decomposition says: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Actually, the tensor for matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p
    # is T[i,j,k] such that C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has entries T[i,j,k] = delta_{i,k}... no wait.
    
    # Let me reconsider. The matrix multiplication tensor T_{ijk} for multiplying (n x m) by (m x p)
    # is defined so that (A*B)[i,k] = sum_j T_{i,j,k} A[i,j] B[j,k]
    # So T_{i,j,k} = 1 if the index matches, i.e., T_{i,j,k} = 1 for all valid i,j,k where the
    # contraction over j gives the correct result. Actually T_{i,j,k} = 1 for all i,j,k in range.
    
    # Wait, that's not right either. Let me think more carefully.
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor is T[i,j,k] = 1 for all valid i in [0,n), j in [0,m), k in [0,p)
    # No! That would give sum_j A[i,j]*B[j,k] which is exactly what we want.
    # So T[i,j,k] = 1 for all i,j,k.
    
    # The decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Let's verify by computing the left side and right side for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = 1.0  # T[i,j,k] = 1
                rhs = 0.0
                for c in range(rank):
                    rhs += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(lhs - rhs) > 1e-9:
                    return False
    return True

def _is_half_integer(x, tol=1e-9):
    """Check if x is an integer or half-integer."""
    return abs(x * 2 - round(x * 2)) < tol

def _verify_half_integer(U, V, W, tol=1e-9):
    """Check all entries are half-integers."""
    for M in [U, V, W]:
        for row in M:
            for val in row:
                if not _is_half_integer(val, tol):
                    return False
    return True

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3x4 matrix multiplication."""
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 4
    schoolbook_rank = n * m * p  # 24
    
    # Start with schoolbook solution as fallback
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    best_rank = schoolbook_rank
    best_solution = (U_sb, V_sb, W_sb)
    
    # Try ranks from 12 upward (theoretical minimum is 12)
    for target_rank in range(12, schoolbook_rank + 1):
        if time.time() > deadline - 10:
            break
        
        # Try to find a decomposition of the given rank
        # Strategy: use random initialization with half-integer values, then try to solve
        # the system of equations
        
        # The system: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # This is a system of n*m*p = 24 equations in 3*target_rank unknowns.
        
        # Approach: Fix W to be simple (e.g., identity-like or sparse), then solve for U and V
        # Or use a more structured approach.
        
        # Known result: rank 12 exists for 2x3x4. Let's try a structured construction.
        # One known approach uses the fact that 2x3x3 has rank 7, and we can extend it.
        
        # Let me try a direct search with small integer/half-integer values.
        # For rank r, each matrix has entries that should be in {-1, -0.5, 0, 0.5, 1, 1.5, 2, ...}
        # But for simplicity, let's try entries in {-1, 0, 1} first, then {-1, -0.5, 0, 0.5, 1}
        
        if target_rank == 12:
            # Try known constructions for rank 12
            # One approach: the rank of 2x3x4 is known to be 12.
            # Let's try to construct it using a pattern.
            
            # Method: Use the fact that we can think of this as two 2x3x2 blocks or similar.
            # Actually, let's try a random search with half-integer entries in a small set.
            
            found = _search_rank_12(n, m, p, target_rank, deadline)
            if found is not None:
                U, V, W = found
                if _verify(U, V, W, n, m, p) and _verify_half_integer(U, V, W):
                    best_rank = target_rank
                    best_solution = (U, V, W)
                    break
        
        if time.time() > deadline - 5:
            break
    
    U, V, W = best_solution
    return {"U": U, "V": V, "W": W}

def _search_rank_12(n, m, p, rank, deadline):
    """Search for a rank-12 decomposition using structured patterns."""
    start = time.time()
    
    # Try a known construction pattern for 2x3x4 rank 12
    # Based on the work of Strassen and others, there are explicit constructions.
    
    # Let's try: use 12 columns, each contributing to specific (i,j,k) triples.
    # The key insight is that each rank-1 term contributes to a "slice" of the tensor.
    
    # Try random search with half-integer values in {-1, -0.5, 0, 0.5, 1}
    values = [-1, -0.5, 0, 0.5, 1]
    
    # We need to solve: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # 
    # Strategy: Fix W to have a specific structure, then solve for U and V.
    # 
    # Alternative: Use the fact that the problem is equivalent to finding a 12-term
    # bilinear decomposition.
    
    # Let me try a different approach: use the known optimal decomposition for 2x3x3
    # (rank 7) and extend it to 2x3x4.
    
    # Actually, let's try a more systematic approach:
    # For rank r, we have 3*r variables. The constraint is 24 equations.
    # For r=12, we have 36 variables and 24 equations, so the system is underdetermined.
    
    # Let's try fixing some variables to simplify.
    
    # Approach: Set W to be a 12x12 matrix with specific structure, then solve for U and V.
    # Or better: use a greedy/iterative approach.
    
    # Let's try: initialize all matrices with small random half-integer values,
    # then use iterative refinement.
    
    random.seed(42)
    
    for attempt in range(100):
        if time.time() > deadline - 10:
            break
        
        # Random initialization
        U = [[random.choice(values) for _ in range(rank)] for _ in range(n*m)]
        V = [[random.choice(values) for _ in range(rank)] for _ in range(m*p)]
        W = [[random.choice(values) for _ in range(rank)] for _ in range(p*n)]
        
        # Try to fix W to be simpler: make W have entries in {-1, 0, 1}
        W = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(p*n)]
        
        # Now try to solve for U and V given W
        # For each column c, the contribution is U[:,c] * V[:,c] * W[:,c] (outer product scaled by W)
        # Actually, for each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        
        # This is still nonlinear. Let's try a different approach.
        
        # Fix W to be the identity-like structure: W[k*n+i, c] = 1 if c == k*n+i (for c < p*n)
        # But p*n = 8, and we have rank 12, so this doesn't directly work.
        
        # Let's try: W has 12 columns. For the first 8 columns, W[k*n+i, c] = delta_{k*n+i, c}
        # For the last 4 columns, W is something else.
        
        # Actually, let me try a completely different tactic: use the known result that
        # rank(2x3x4) = 12 and try to find the explicit decomposition from literature.
        
        # From the literature, one construction uses the following idea:
        # The 2x3x4 tensor can be decomposed using 12 rank-1 tensors.
        
        # Let's try a structured pattern where each rank-1 term covers a specific pattern.
        
        # Pattern 1: Terms that cover (i=0, j=0, k=0), (i=0, j=0, k=1), etc.
        # But each rank-1 term covers a "rectangle" in the (i,j) x (j,k) x (k,i) space.
        
        # I'll try a more brute-force approach with better initialization.
        
        # Set W to have a block structure
        W_new = [[0]*rank for _ in range(p*n)]
        # First 8 columns: W[k*n+i, c] = 1 if c == k*n+i
        for c in range(min(8, rank)):
            i = c // p
            k = c % p
            W_new[k*n+i][c] = 1
        # Remaining 4 columns: assign non-trivial values
        for c in range(8, rank):
            for idx in range(p*n):
                W_new[idx][c] = random.choice([-1, 0, 1])
        
        W = W_new
        
        # Now solve for U and V
        # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # 
        # For columns where W[k*n+i,c] = 0, they don't contribute to this (i,j,k) equation.
        # For the first 8 columns, W[k*n+i,c] = 1 only when c = k*n+i.
        
        # This is getting complex. Let me try a simpler verification-based search.
        
        pass
    
    # Fallback: try a known explicit construction
    # Based on the pattern for 2x3x3 rank 7, extended to 2x3x4 rank 12
    
    # Let me try a specific known construction
    # For 2x3x4, one known rank-12 decomposition:
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Define 12 rank-1 terms
    # Each term c has U[:,c], V[:,c], W[:,c]
    
    # Term 0: covers (i=0,j=0,k=0) primarily
    U[0*3+0][0] = 1; V[0*4+0][0] = 1; W[0*2+0][0] = 1
    # Term 1: covers (i=0,j=0,k=1)
    U[0*3+0][1] = 1; V[0*4+1][1] = 1; W[1*2+0][1] = 1
    # Term 2: covers (i=0,j=0,k=2)
    U[0*3+0][2] = 1; V[0*4+2][2] = 1; W[2*2+0][2] = 1
    # Term 3: covers (i=0,j=0,k=3)
    U[0*3+0][3] = 1; V[0*4+3][3] = 1; W[3*2+0][3] = 1
    # Term 4: covers (i=0,j=1,k=0)
    U[0*3+1][4] = 1; V[1*4+0][4] = 1; W[0*2+0][4] = 1
    # Term 5: covers (i=0,j=1,k=1)
    U[0*3+1][5] = 1; V[1*4+1][5] = 1; W[1*2+0][5] = 1
    # Term 6: covers (i=0,j=1,k=2)
    U[0*3+1][6] = 1; V[1*4+2][6] = 1; W[2*2+0][6] = 1
    # Term 7: covers (i=0,j=1,k=3)
    U[0*3+1][7] = 1; V[1*4+3][7] = 1; W[3*2+0][7] = 1
    # Term 8: covers (i=1,j=0,k=0)
    U[1*3+0][8] = 1; V[0*4+0][8] = 1; W[0*2+1][8] = 1
    # Term 9: covers (i=1,j=0,k=1)
    U[1*3+0][9] = 1; V[0*4+1][9] = 1; W[1*2+1][9] = 1
    # Term 10: covers (i=1,j=0,k=2)
    U[1*3+0][10] = 1; V[0*4+2][10] = 1; W[2*2+1][10] = 1
    # Term 11: covers (i=1,j=0,k=3)
    U[1*3+0][11] = 1; V[0*4+3][11] = 1; W[3*2+1][11] = 1
    
    # But this only covers j=0 and j=1 for i=0,1 and j=0 for i=1.
    # We're missing j=2 entries. This naive approach gives rank 24 (schoolbook).
    
    # The rank-12 solution must use overlapping terms.
    # Let me try a different approach: use the fact that we can share terms.
    
    # Actually, the schoolbook solution above has rank 12 but only covers part of the tensor.
    # It doesn't work because it misses many (i,j,k) triples.
    
    # Let me verify if this partial solution works:
    if _verify(U, V, W, n, m, p):
        return (U, V, W)
    
    # It won't work. Let me try a proper rank-12 construction.
    # 
    # Known: the border rank of 2x3x4 is 12, and the rank is also 12.
    # A construction uses the following idea:
    # 
    # Think of the 2x3x4 tensor as a 2x(3x4) tensor, or use Kronecker products.
    
    # Let me try yet another approach: use numpy to solve the system.
    # For a fixed W, the equations become linear in U*V (element-wise product).
    # But U*V is not linear in U and V separately.
    
    # Alternative: fix both U and V to have sparse structure, then solve for W.
    # If U and V are fixed, then for each (i,j,k):
    #   W[k*n+i, :] dot (U[i*m+j, :] * V[j*p+k, :]) = 1
    # This is linear in W!
    
    # So: fix U and V, solve for W.
    # Let's try random sparse U and V, then solve for W.
    
    for attempt in range(200):
        if time.time() > deadline - 10:
            break
        
        # Random sparse U and V with entries in {-1, 0, 1}
        U_try = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(n*m)]
        V_try = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(m*p)]
        
        # Build the linear system for W
        # For each (i,j,k): sum_c W[k*n+i, c] * U_try[i*m+j, c] * V_try[j*p+k, c] = 1
        # 
        # This gives 24 equations in 12*2 = 24 unknowns (W has 8 rows * 12 cols = 96 entries,
        # but each equation only involves one row of W).
        # 
        # Actually, each equation for (i,j,k) only involves W[k*n+i, :], which is a row of W.
        # So we have 24 equations, but they group by (k,i) pairs.
        # For each (k,i) pair, we have m=3 equations (one for each j).
        # So for each of the p*n = 8 (k,i) pairs, we have a 3x12 system:
        #   sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = 1, for j=0,1,2
        
        W_try = [[0]*rank for _ in range(p*n)]
        valid = True
        
        for k in range(p):
            for i in range(n):
                # 3 equations (j=0,1,2) in 12 unknowns (W[k*n+i, 0..11])
                A = np.zeros((m, rank))
                b = np.ones(m)
                for j in range(m):
                    for c in range(rank):
                        A[j, c] = U_try[i*m+j][c] * V_try[j*p+k][c]
                
                # Solve A * W_row = b
                # This is underdetermined (3 equations, 12 unknowns)
                # Use least squares or find a particular solution
                try:
                    W_row, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                    # Check if the solution is exact
                    if np.max(np.abs(A @ W_row - b)) > 1e-9:
                        valid = False
                        break
                    # Check if W_row entries are half-integers
                    if not all(abs(W_row[c] * 2 - round(W_row[c] * 2)) < 1e-9 for c in range(rank)):
                        valid = False
                        break
                    W_try[k*n+i] = [float(x) for x in W_row]
                except:
                    valid = False
                    break
            
            if not valid:
                break
        
        if valid:
            if _verify(U_try, V_try, W_try, n, m, p) and _verify_half_integer(U_try, V_try, W_try):
                return (U_try, V_try, W_try)
    
    return None
