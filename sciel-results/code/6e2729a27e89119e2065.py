"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction
from itertools import product, combinations
import sys

DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x2 * 2x5 matrix multiplication.
    
    The tensor T has indices (i,j,k) where:
    - i in {0,1} (row of first matrix, size n=2)
    - j in {0,1} (col of first / row of second, size m=2)  
    - k in {0,1,2,3,4} (col of second matrix, size p=5)
    
    Output factor W follows trace convention W[k*n+i, r] = W[k*2+i, r]
    
    We use randomized algebraic elimination:
    - Start with the schoolbook rank-20 decomposition
    - Try to find lower-rank decompositions by solving sparse linear systems
    - Use Gaussian elimination over rationals with sparsity constraints
    """
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=5
    
    # The multiplication tensor: T[i,j,k] = 1 for all i,j,k
    # We want rank-r decomposition: T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # where U has shape (n*m, r) = (4, r)
    #       V has shape (m*p, r) = (10, r)
    #       W has shape (p*n, r) = (10, r)
    # and T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Known: schoolbook gives rank 20
    # Strassen-like methods might give lower rank
    # For 2x2 * 2x5, the minimum rank is at most 20, possibly lower
    
    # Let's try to find a rank-18 or rank-17 decomposition
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if total != 1:
                        return False
        return True
    
    def try_rank(rank, max_iterations=5000):
        """Try to find a rank-r decomposition using randomized elimination."""
        for iter_num in range(max_iterations):
            if time.time() > deadline:
                return None
            
            # Randomize the structure: choose which (i,j,k) triples get non-zero contributions
            # from which rank components, with some sparsity pattern
            
            # Strategy: start with a random assignment of which rank component handles
            # which (i,j,k) triples, then solve for the values
            
            # For rank r, we have r terms. Each term (i,j,k) can be covered by multiple terms.
            # Let's use a structured approach:
            
            # Random sparsity pattern for U, V, W
            U = [[Fraction(0) for _ in range(rank)] for _ in range(n*m)]
            V = [[Fraction(0) for _ in range(rank)] for _ in range(m*p)]
            W = [[Fraction(0) for _ in range(rank)] for _ in range(p*n)]
            
            # Assign each (i,j,k) to be primarily handled by a random subset of rank components
            # This breaks the block-diagonal symmetry
            
            # Random permutation of rank indices
            perm = list(range(rank))
            random.shuffle(perm)
            
            # For each (i,j,k), choose which rank components contribute
            # Use a structured pattern to make the system solvable
            assignments = {}
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Each (i,j,k) needs sum over r of U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
                        # Randomly assign to 1-3 rank components
                        num_contributors = random.randint(1, min(3, rank))
                        contributors = random.sample(perm, num_contributors)
                        assignments[(i,j,k)] = contributors
            
            # Now set up the linear system
            # Unknowns: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] for all i,j,k,r
            # But we have bilinear/quadratic constraints, not linear!
            
            # Alternative approach: fix some values and solve linearly
            # Fix U to have a specific structure, then solve for V and W
            
            # Let's try: fix U to be a random sparse matrix, then the constraints become
            # bilinear in V and W. Still nonlinear.
            
            # Better: use the fact that if we fix two of the three factors, the third is linear
            # Fix U and V to random values, then solve for W linearly
            # But W[k*n+i, r] appears in multiple equations (for different j)
            
            # Actually, let's use a different strategy:
            # Fix the structure so that the problem becomes a system of polynomial equations
            # and use elimination
            
            # Simpler approach: try known small-rank constructions
            
            # For rank 18: we can try to merge some of the 20 schoolbook terms
            # The schoolbook has terms for each (i,j,k). If we can find linear dependencies
            # in the tensor, we can reduce the rank.
            
            # Let's try a more direct approach: assume a specific sparse structure
            # and solve the resulting system
            
            # Structure: U has non-zeros only in certain positions
            # Let's try: U[i*m+j, r] is non-zero only if r is in a certain set depending on i,j
            
            # Actually, let's just try to find ANY rank-r solution by solving the
            # polynomial system using a random linearization
            
            # Fix W to random values, then the system is bilinear in U and V
            # Fix V to random values, then it's linear in U (and W)
            
            # Let's fix V to a random sparse matrix and solve for U and W
            # The constraint is: for all i,j,k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            
            # If we fix V, this is still bilinear in U and W.
            # If we also fix U, it's linear in W.
            
            # So: fix U and V to random sparse matrices, solve linear system for W
            # Then check if the solution is valid (all constraints satisfied)
            
            # But we need U and V such that the system for W is consistent.
            
            # Let's try: choose U and V randomly, solve for W, verify
            for attempt in range(20):
                if time.time() > deadline:
                    return None
                
                # Random sparse U
                U_trial = [[Fraction(0) for _ in range(rank)] for _ in range(n*m)]
                for idx in range(n*m):
                    # Each row of U has 1-3 non-zeros
                    num_nz = random.randint(1, min(3, rank))
                    positions = random.sample(range(rank), num_nz)
                    for pos in positions:
                        U_trial[idx][pos] = Fraction(random.randint(1, 3), 1)
                
                # Random sparse V
                V_trial = [[Fraction(0) for _ in range(rank)] for _ in range(m*p)]
                for idx in range(m*p):
                    num_nz = random.randint(1, min(3, rank))
                    positions = random.sample(range(rank), num_nz)
                    for pos in positions:
                        V_trial[idx][pos] = Fraction(random.randint(1, 3), 1)
                
                # Now solve for W: for each (i,k), we need W[k*n+i, r] such that
                # for all j: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # This is a system of m equations in rank unknowns for each (i,k)
                # But W[k*n+i, r] is shared across j for fixed (i,k)
                
                # For fixed (i,k), the unknowns are W[k*n+i, 0], ..., W[k*n+i, rank-1]
                # The equations are: for j=0,...,m-1:
                #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # This is m linear equations in rank unknowns.
                # If m < rank, we have more unknowns than equations, so many solutions.
                # If m = rank, we can solve uniquely (if consistent).
                # If m > rank, we may not have a solution.
                
                # Here m=2, so for each (i,k), we have 2 equations in rank unknowns.
                # We can pick rank-2 of the unknowns as free and solve for the rest,
                # or just set most to zero and solve for 2.
                
                W_trial = [[Fraction(0) for _ in range(rank)] for _ in range(p*n)]
                
                feasible = True
                for i in range(n):
                    for k in range(p):
                        # Solve for W[k*n+i, :]
                        # Equations: for j in range(m):
                        #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                        
                        # Build the 2 x rank matrix A where A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                        A = [[Fraction(0) for _ in range(rank)] for _ in range(m)]
                        for j in range(m):
                            for r in range(rank):
                                A[j][r] = U_trial[i*m+j][r] * V_trial[j*p+k][r]
                        
                        # Solve A * w = [1, 1, ..., 1] (m ones)
                        # Use Gaussian elimination
                        # We have m=2 equations, rank unknowns
                        # Find a particular solution
                        
                        # Pick 2 columns to be the "pivot" columns, set others to 0
                        # Find 2 linearly independent columns
                        pivot_cols = []
                        for r in range(rank):
                            if len(pivot_cols) == m:
                                break
                            col = [A[j][r] for j in range(m)]
                            # Check if this column is linearly independent of pivot_cols
                            independent = True
                            if pivot_cols:
                                # Check if col is a multiple of any existing pivot
                                for pc in pivot_cols:
                                    pc_col = [A[j][pc] for j in range(m)]
                                    # Check if col = c * pc_col for some c
                                    if all(pc_col[j] == 0 for j in range(m)):
                                        continue
                                    # Find c such that col[0] = c * pc_col[0]
                                    if pc_col[0] != 0:
                                        c = col[0] / pc_col[0]
                                        if all(col[j] == c * pc_col[j] for j in range(m)):
                                            independent = False
                                            break
                                    else:
                                        # pc_col[0] = 0, so we need col[0] = 0
                                        if col[0] != 0:
                                            independent = False
                                            break
                                        # Check remaining
                                        for j in range(1, m):
                                            if pc_col[j] != 0:
                                                c = col[j] / pc_col[j]
                                                if any(col[jj] != c * pc_col[jj] for jj in range(m)):
                                                    independent = False
                                                    break
                                        if not independent:
                                            break
                            if independent:
                                pivot_cols.append(r)
                        
                        if len(pivot_cols) < m:
                            feasible = False
                            break
                        
                        # Solve the m x m system for the pivot columns
                        # A_pivot * w_pivot = [1, 1, ..., 1]
                        A_pivot = [[A[j][r] for r in pivot_cols] for j in range(m)]
                        b = [Fraction(1) for _ in range(m)]
                        
                        # Gaussian elimination
                        # Make a copy
                        Aug = [A_pivot[j][:] + [b[j]] for j in range(m)]
                        
                        for col in range(m):
                            # Find pivot
                            pivot_row = None
                            for row in range(col, m):
                                if Aug[row][col] != 0:
                                    pivot_row = row
                                    break
                            if pivot_row is None:
                                feasible = False
                                break
                            # Swap
                            Aug[col], Aug[pivot_row] = Aug[pivot_row], Aug[col]
                            # Normalize
                            pivot_val = Aug[col][col]
                            for jj in range(col, m+1):
                                Aug[col][jj] /= pivot_val
                            # Eliminate
                            for row in range(m):
                                if row != col:
                                    factor = Aug[row][col]
                                    for jj in range(col, m+1):
                                        Aug[row][jj] -= factor * Aug[col][jj]
                        
                        if not feasible:
                            break
                        
                        # Solution
                        w_pivot = [Aug[j][m] for j in range(m)]
                        
                        # Set W[k*n+i, r] for pivot columns
                        for idx, r in enumerate(pivot_cols):
                            W_trial[k*n+i][r] = w_pivot[idx]
                        # Other entries stay 0
                
                if not feasible:
                    continue
                
                # Verify the full decomposition
                if verify_decomposition(U_trial, V_trial, W_trial, rank):
                    # Convert to integers if possible, otherwise keep fractions
                    # The problem allows integer/half-integer entries
                    # Check if all entries are integers or half-integers
                    valid_entries = True
                    for matrix in [U_trial, V_trial, W_trial]:
                        for row in matrix:
                            for val in row:
                                # Check if val is integer or half-integer
                                num = val.numerator
                                den = val.denominator
                                if den > 2:
                                    valid_entries = False
                                    break
                            if not valid_entries:
                                break
                        if not valid_entries:
                            break
                    
                    if valid_entries:
                        # Convert to float or keep as fractions? The problem says integer/half-integer
                        # Let's convert to a list of lists of numbers
                        def convert_matrix(mat):
                            result = []
                            for row in mat:
                                new_row = []
                                for val in row:
                                    if val.denominator == 1:
                                        new_row.append(int(val))
                                    else:
                                        # Half-integer
                                        new_row.append(float(val))
                                result.append(new_row)
                            return result
                        
                        return {
                            "U": convert_matrix(U_trial),
                            "V": convert_matrix(V_trial),
                            "W": convert_matrix(W_trial)
                        }
        
        return None
    
    # Try ranks from low to high
    # Known lower bounds: for 2x2*2x5, the rank is at least max(n*m, m*p, p*n) = max(4,10,10) = 10
    # But that's a weak bound. Better bounds exist.
    # Let's try ranks 10 through 19
    
    for rank in range(10, 20):
        if time.time() > deadline:
            break
        
        result = try_rank(rank)
        if result is not None:
            # Verify one more time
            U = result["U"]
            V = result["V"]
            W = result["W"]
            
            # Convert to fractions for verification
            def to_fraction(x):
                if isinstance(x, float):
                    # Check if it's a half-integer
                    if abs(x * 2 - round(x * 2)) < 1e-9:
                        return Fraction(int(x * 2), 2)
                    else:
                        return Fraction(x).limit_denominator(1000)
                elif isinstance(x, int):
                    return Fraction(x)
                else:
                    return Fraction(x)
            
            U_frac = [[to_fraction(x) for x in row] for row in U]
            V_frac = [[to_fraction(x) for x in row] for row in V]
            W_frac = [[to_fraction(x) for x in row] for row in W]
            
            if verify_decomposition(U_frac, V_frac, W_frac, rank):
                return result
    
    # Fallback: return the schoolbook decomposition
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

# EVOLVE-BLOCK-END
