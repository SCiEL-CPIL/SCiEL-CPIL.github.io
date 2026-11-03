"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s limit to be safe (150s wall-clock)
    
    n, m, p = DIMENSIONS
    
    # Helper to add a Strassen 2x2x2 block
    def add_strassen(U, V, W, i0, j0, k0, col):
        # M1
        U[i0 * m + j0][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[j0 * p + k0][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[k0 * n + i0][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M2
        U[(i0 + 1) * m + j0][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[j0 * p + k0][col] += 1
        W[k0 * n + i0 + 1][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] -= 1
        col += 1
        
        # M3
        U[i0 * m + j0][col] += 1
        V[j0 * p + k0 + 1][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] -= 1
        W[(k0 + 1) * n + i0][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M4
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[(j0 + 1) * p + k0][col] += 1
        V[j0 * p + k0][col] -= 1
        W[k0 * n + i0][col] += 1
        W[k0 * n + i0 + 1][col] += 1
        col += 1
        
        # M5
        U[i0 * m + j0][col] += 1
        U[i0 * m + j0 + 1][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[(k0 + 1) * n + i0][col] += 1
        W[k0 * n + i0][col] -= 1
        col += 1
        
        # M6
        U[(i0 + 1) * m + j0][col] += 1
        U[i0 * m + j0][col] -= 1
        V[j0 * p + k0][col] += 1
        V[j0 * p + k0 + 1][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M7
        U[i0 * m + j0 + 1][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] -= 1
        V[(j0 + 1) * p + k0][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[k0 * n + i0][col] += 1
        col += 1
        
        return col

    # Helper to add schoolbook terms for a sub-block
    def add_schoolbook(U, V, W, i_range, j_range, k_range, col):
        for i in i_range:
            for j in j_range:
                for k in k_range:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return col

    # First, build the baseline Strassen+Schoolbook decomposition (rank 99)
    rank_baseline = 99
    U_base = [[0] * rank_baseline for _ in range(n * m)]
    V_base = [[0] * rank_baseline for _ in range(m * p)]
    W_base = [[0] * rank_baseline for _ in range(p * n)]
    
    col = 0
    # Part 1: j in {0, 1}
    for k0 in [0, 2, 4]:
        col = add_strassen(U_base, V_base, W_base, 0, 0, k0, col)
    col = add_schoolbook(U_base, V_base, W_base, [0, 1], [0, 1], [6], col)
    col = add_schoolbook(U_base, V_base, W_base, [2], [0, 1], range(7), col)
    
    # Part 2: j in {2, 3, 4}
    for k0 in [0, 2, 4]:
        col = add_strassen(U_base, V_base, W_base, 0, 2, k0, col)
    col = add_schoolbook(U_base, V_base, W_base, [0, 1], [2, 3], [6], col)
    col = add_schoolbook(U_base, V_base, W_base, [0, 1], [4], range(7), col)
    col = add_schoolbook(U_base, V_base, W_base, [2], [2, 3, 4], range(7), col)
    
    # Now attempt to reduce rank via exact rational linear algebra
    # The tensor T_{i,j,k} = sum_r U_{i,j}^r * V_{j,k}^r * W_{k,i}^r
    # We need to find a smaller set of rank-1 tensors that span the same tensor space.
    
    # Strategy: Try to find linear dependencies among the 99 rank-1 components.
    # Each rank-1 component r defines a tensor T^r where T^r[i,j,k] = U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
    # If there exist coefficients c_r such that sum_r c_r * T^r = 0, we can eliminate one component.
    
    # Build the constraint matrix: for each (i,j,k) triple, we have:
    # sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = delta_{i,j,k} (the target tensor)
    # But actually, we want to check if the 99 tensors are linearly independent.
    
    # A simpler approach: try to find a rank-r decomposition by solving a system.
    # For rank r, we need U (15 x r), V (35 x r), W (21 x r) such that:
    # For all (i,j,k): sum_{r=1}^r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = 1 if this is a valid MM term
    
    # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p]
    # Wait no - the tensor for matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the bilinear form is: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T_{i,j,k} where the output index is (i,k) and input indices are (i,j) and (j,k)
    # T[i,j,k] = 1 if the output (i,k) depends on A[i,j] and B[j,k]
    # Actually, T[i,j,k] = 1 for all valid (i,j,k) since every (i,j) of A multiplies with every (j,k) of B
    
    # Let me reconsider. The tensor for multiplying n x m by m x p matrix:
    # Output C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has indices (i,j,k) where:
    # - i indexes rows of A (and C): 0..n-1
    # - j indexes cols of A / rows of B: 0..m-1  
    # - k indexes cols of B (and C): 0..p-1
    # T[i,j,k] = 1 for all (i,j,k) since C[i,k] = sum_j A[i,j]*B[j,k]
    
    # Wait, that's not right either. The tensor is defined such that:
    # C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all valid (i,j,k).
    
    # The rank-1 decomposition: T[i,j,k] = sum_r U[i,j]^r * V[j,k]^r * W[i,k]^r
    # where U has shape (n*m, r), V has shape (m*p, r), W has shape (n*p, r)
    # Note: W is indexed by (i,k) not (k,i) - let me check the convention.
    
    # From the task: "output factor follows the official trace convention W[k*n+i,r]"
    # So W[k*n+i, r] means W is indexed by (k, i) where k is col of B and i is row of A.
    # This matches: W[i,k] in my notation = W[k*n+i] in the flattened form.
    
    # So the constraint is:
    # For all (i,j,k) in [0,n)x[0,m)x[0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of n*m*p = 3*5*7 = 105 equations.
    # Unknowns: r * (n*m + m*p + n*p) = r * (15 + 35 + 21) = r * 71
    # For r = 99: 99 * 71 = 7029 unknowns, 105 equations - very underdetermined.
    
    # The question is: what is the minimum r such that a solution exists?
    # This is a hard problem in general, but we can try to find linear dependencies.
    
    # Alternative approach: Use the existing rank-99 solution and try to find
    # a linear combination of the rank-1 tensors that gives zero, then eliminate.
    
    # Let's try a different strategy: greedy elimination.
    # Start with the 99 terms. For each term r, check if it can be expressed
    # as a linear combination of the others. If so, remove it.
    
    # To check if term r is in the span of the others, we solve:
    # T^r = sum_{s != r} c_s * T^s
    # where T^r[i,j,k] = U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
    
    # This is a linear system in the c_s variables.
    
    # But this is expensive. Let's try a more practical approach:
    # Try specific structured reductions.
    
    # Approach: Try to merge schoolbook terms with Strassen terms.
    # For example, in the 2x2x7 block, the 7th k-index uses 4 schoolbook terms.
    # Can we use a different decomposition for 2x2x7 that uses fewer than 4+3*7 = 25 terms?
    
    # Known results:
    # 2x2x2: rank 7 (Strassen)
    # 2x2x3: rank ? (at most 7+6=13, possibly less)
    # 2x2x7: rank ? 
    
    # Let's try to find a better decomposition for 2x2x7.
    # The 2x2x7 tensor has 28 entries, all 1.
    # We need U (4 x r), V (14 x r), W (14 x r) such that
    # sum_r U[i,j]^r * V[j,k]^r * W[i,k]^r = 1 for all (i,j,k)
    
    # Try r = 24: Can we do 2x2x7 in 24 terms?
    # That would be 3 Strassen blocks (21 terms) + 3 extra terms for k=6.
    # But k=6 requires 4 terms in schoolbook (i=0,1; j=0,1).
    # Unless we can share structure...
    
    # Actually, let's try a completely different approach:
    # Use the fact that we have 105 constraints and try to find the minimal rank
    # by solving a sequence of linear systems.
    
    # For a given rank r, the problem is:
    # Find U (15 x r), V (35 x r), W (21 x r) such that
    # for all (i,j,k): sum_{r'=0}^{r-1} U[i*m+j, r'] * V[j*p+k, r'] * W[k*n+i, r'] = 1
    
    # This is a system of 105 polynomial equations (degree 3) in 71r variables.
    # Very hard to solve directly.
    
    # Practical approach: Start from the rank-99 solution and try to eliminate terms.
    
    # Let's try to find linear dependencies among the 99 rank-1 tensors.
    # Build a 105 x 99 matrix M where M[(i,j,k), r] = U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
    # Then check if the columns are linearly dependent.
    
    # If the columns are linearly dependent, there exist coefficients c_r (not all zero)
    # such that sum_r c_r * M[:,r] = 0, meaning sum_r c_r * T^r = 0.
    # We can then eliminate one term.
    
    # Let's build this matrix and compute its rank.
    
    def build_constraint_matrix(U, V, W, n, m, p):
        """Build the 105 x rank matrix where entry (i,j,k),r = U[i,j]^r * V[j,k]^r * W[k,i]^r"""
        r = len(U[0])
        M = [[Fraction(0)] * r for _ in range(n * m * p)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r_idx in range(r):
                        M[idx][r_idx] = Fraction(U[i * m + j][r_idx]) * Fraction(V[j * p + k][r_idx]) * Fraction(W[k * n + i][r_idx])
                    idx += 1
        return M
    
    def compute_rank_exact(M):
        """Compute the rank of a matrix of Fractions using Gaussian elimination."""
        if not M or not M[0]:
            return 0
        rows = len(M)
        cols = len(M[0])
        # Work on a copy
        A = [row[:] for row in M]
        rank = 0
        for col_idx in range(cols):
            if time.time() > deadline:
                break
            # Find pivot
            pivot_row = -1
            for row_idx in range(rank, rows):
                if A[row_idx][col_idx] != 0:
                    pivot_row = row_idx
                    break
            if pivot_row == -1:
                continue
            # Swap rows
            A[rank], A[pivot_row] = A[pivot_row], A[rank]
            # Eliminate
            pivot_val = A[rank][col_idx]
            for row_idx in range(rank + 1, rows):
                if A[row_idx][col_idx] != 0:
                    factor = A[row_idx][col_idx] / pivot_val
                    for c in range(col_idx, cols):
                        A[row_idx][c] -= factor * A[rank][c]
            rank += 1
        return rank
    
    # Build the constraint matrix for the baseline solution
    M_base = build_constraint_matrix(U_base, V_base, W_base, n, m, p)
    
    # Compute the rank
    rank_of_M = compute_rank_exact(M_base)
    
    # The rank of M_base tells us the dimension of the span of the 99 rank-1 tensors.
    # If rank_of_M < 99, then the tensors are linearly dependent and we can eliminate (99 - rank_of_M) terms.
    
    # But we need to be careful: even if the tensors are linearly dependent,
    # we need to find a subset that still spans the full tensor (which should be rank 105
    # since all entries are 1).
    
    # Actually, the target tensor T[i,j,k] = 1 for all (i,j,k) is a single vector in R^105.
    # We need sum_r T^r = T_target where T_target[i,j,k] = 1.
    # So we need the sum of the 99 tensors to equal the all-ones vector.
    # If there's a linear dependency sum_r c_r T^r = 0, we can't just remove a term
    # because we need the sum to be exactly the all-ones vector.
    
    # However, if T^r = sum_{s != r} c_s T^s, then we can replace T^r with the combination
    # and eliminate it, but this doesn't reduce the number of terms unless one of the c_s is 0
    # or we can combine terms.
    
    # Actually, if T^r is in the span of the others, we can express the all-ones vector
    # using only the other 98 tensors (by substituting). So we can eliminate term r.
    
    # Let's find which terms are redundant.
    
    # For each column r, check if it's in the span of the others.
    # This is expensive (99 * O(105 * 99^2) operations).
    
    # Instead, let's use the row-reduced form to identify pivot columns.
    # Non-pivot columns are linear combinations of pivot columns.
    
    # Let's redo Gaussian elimination tracking pivot columns.
    
    def find_redundant_columns(M):
        """Return the set of column indices that are linear combinations of earlier columns."""
        if not M or not M[0]:
            return set()
        rows = len(M)
        cols = len(M[0])
        A = [row[:] for row in M]
        pivot_cols = []
        redundant = set()
        
        current_row = 0
        for col_idx in range(cols):
            if time.time() > deadline:
                break
            if current_row >= rows:
                redundant.add(col_idx)
                continue
            # Find pivot in column col_idx, row >= current_row
            pivot_row = -1
            for row_idx in range(current_row, rows):
                if A[row_idx][col_idx] != 0:
                    pivot_row = row_idx
                    break
            if pivot_row == -1:
                redundant.add(col_idx)
                continue
            # Swap
            A[current_row], A[pivot_row] = A[pivot_row], A[current_row]
            # Eliminate below
            pivot_val = A[current_row][col_idx]
            for row_idx in range(current_row + 1, rows):
                if A[row_idx][col_idx] != 0:
                    factor = A[row_idx][col_idx] / pivot_val
                    for c in range(col_idx, cols):
                        A[row_idx][c] -= factor * A[current_row][c]
            pivot_cols.append(col_idx)
            current_row += 1
        
        # Columns not in pivot_cols are redundant
        pivot_set = set(pivot_cols)
        for c in range(cols):
            if c not in pivot_set:
                redundant.add(c)
        
        return redundant
    
    redundant_cols = find_redundant_columns(M_base)
    
    # If there are redundant columns, we can potentially reduce the rank.
    # But we need to verify that the remaining columns still sum to the all-ones vector.
    
    # Actually, the constraint is not just that the span contains the all-ones vector,
    # but that the specific sum equals the all-ones vector.
    # The 99 tensors sum to the all-ones vector by construction.
    # If column r is redundant (in the span of others), then T^r = sum_{s in pivot, s != r} c_s T^s
    # for some coefficients c_s.
    # Then sum_{all r} T^r = sum_{pivot} T^r + sum_{redundant} T^r
    # = sum_{pivot} T^r + sum_{redundant r} sum_{s in pivot, s != r} c_{r,s} T^s
    # = sum_{s in pivot} (1 + sum_{redundant r, r != s} c_{r,s}) T^s
    # This is not necessarily equal to the all-ones vector unless the coefficients work out.
    
    # So simply removing redundant columns doesn't work directly.
    
    # Alternative: If T^r is exactly equal to a linear combination of others with specific
    # coefficients, we can't just remove it. But if we can find a subset of the 99 terms
    # that still sums to the all-ones vector, we can use that subset.
    
    # This is getting complex. Let's try a different practical approach:
    # Try specific known better decompositions.
    
    # Known: 2x2x3 matrix multiplication can be done in 11 terms (not 13).
    # Reference: This is a known result.
    
    # If we can do 2x2x3 in 11 terms, then:
    # 2x2x7 = 2x2x3 + 2x2x2 + 2x2x2 = 11 + 7 + 7 = 25 terms (same as before)
    # Or 2x2x7 = 2x2x3 + 2x2x3 + 2x2x1 = 11 + 11 + 2 = 24 terms!
    
    # Let's try to construct a 2x2x3 decomposition in 11 terms.
    # And a 2x2x1 in 2 terms.
    
    # 2x2x1: C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0]
    # This is just 2 terms: (i=0,j=0,k=0) and (i=1,j=1,k=0)
    # Wait, no. C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # So we need 4 terms: (0,0,0), (0,1,0), (1,0,0), (1,1,0)
    # That's rank 4 for 2x2x1, not 2.
    
    # Hmm, so 2x2x1 has rank 4 (schoolbook).
    
    # Let me reconsider. The known result for 2x2x3:
    # Actually, I'm not sure about the exact rank. Let me just try to find
    # a better decomposition empirically.
    
    # Given the complexity, let's fall back to the baseline if we can't find improvements.
    # But first, let's try one more thing: check if we can use a 3x3x3 Strassen-like
    # decomposition that's better than 3x3x2 Strassen + schoolbook.
    
    # For now, return the baseline solution.
    return {"U": U_base, "V": V_base, "W": W_base}
# EVOLVE-BLOCK-END
