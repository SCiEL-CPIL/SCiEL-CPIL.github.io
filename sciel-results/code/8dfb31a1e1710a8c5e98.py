import numpy as np
import time
import random
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 130s limit, 20s safety margin

    # Schoolbook decomposition (Rank 42)
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    def is_halfint(x):
        """Check if x is an integer or half-integer (multiple of 0.5)"""
        return x * 2 == int(x * 2)

    def to_halfint_list(vec):
        """Convert a list of Fractions to a list of ints or half-ints (as Fraction)"""
        result = []
        for v in vec:
            if is_halfint(v):
                result.append(v)
            else:
                return None
        return result

    def solve_for_W(U, V, R):
        """
        Given U (nm x R) and V (mp x R) as lists of lists of Fractions,
        find W (pn x R) such that the tensor equation holds.
        Returns W as list of lists of Fractions, or None if no half-integer solution found.
        """
        W = [[0] * R for _ in range(p * n)]
        
        for k in range(p):
            for i in range(n):
                # We need to find w = W[k*n+i, :] (length R) such that:
                # For all j in [0, m): sum_r U[i*m+j, r] * V[j*p+k, r] * w[r] = 1
                # This is a linear system: A * w = 1 (vector of ones, length m)
                # where A[j][r] = U[i*m+j, r] * V[j*p+k, r]
                
                # Build the system using Fractions for exact arithmetic
                # A is m x R, b is m
                A = []
                b = []
                for j in range(m):
                    row = []
                    for r in range(R):
                        row.append(U[i*m+j][r] * V[j*p+k][r])
                    A.append(row)
                    b.append(Fraction(1))
                
                # Solve A * w = b exactly using fraction-based Gaussian elimination
                # Since R > m typically, the system is underdetermined.
                # We want to find a solution w with half-integer entries.
                
                # First, check if the system is consistent.
                # Use fraction-based row reduction.
                # We'll find the general solution: w = w_particular + t1*v1 + t2*v2 + ...
                # where v1, v2, ... span the nullspace of A.
                
                # Build augmented matrix [A | b]
                aug = []
                for j in range(m):
                    row = A[j] + [b[j]]
                    aug.append(row)
                
                # Gaussian elimination with partial pivoting (using Fractions)
                num_rows = len(aug)
                num_cols = R + 1
                rank_A = 0
                pivot_cols = []
                
                for col in range(R):
                    # Find pivot
                    pivot_row = None
                    for row in range(rank_A, num_rows):
                        if aug[row][col] != 0:
                            pivot_row = row
                            break
                    
                    if pivot_row is None:
                        continue
                    
                    # Swap rows
                    if pivot_row != rank_A:
                        aug[rank_A], aug[pivot_row] = aug[pivot_row], aug[rank_A]
                    
                    # Eliminate
                    pivot_val = aug[rank_A][col]
                    for row in range(num_rows):
                        if row != rank_A and aug[row][col] != 0:
                            factor = aug[row][col] / pivot_val
                            for c in range(col, num_cols):
                                aug[row][c] -= factor * aug[rank_A][c]
                    
                    pivot_cols.append(col)
                    rank_A += 1
                
                # Check consistency: any row with all zeros in A part but nonzero in b part?
                for row in range(num_rows):
                    all_zero = all(aug[row][c] == 0 for c in range(R))
                    if all_zero and aug[row][R] != 0:
                        return None  # Inconsistent
                
                # Now find a particular solution and nullspace.
                # Free variables: columns not in pivot_cols
                pivot_set = set(pivot_cols)
                free_cols = [c for c in range(R) if c not in pivot_set]
                
                # Particular solution: set free variables to 0
                w_particular = [Fraction(0)] * R
                for row_idx in range(rank_A):
                    col = pivot_cols[row_idx]
                    w_particular[col] = aug[row_idx][R]
                    # Subtract contributions from other pivot columns (should be 0 after elimination)
                    # Actually after full elimination, each pivot row has 1 in its pivot col and 0 in other pivot cols
                    # But it may have nonzero entries in free cols.
                    # Since we set free vars to 0, w_particular[col] = aug[row_idx][R]
                
                # Verify particular solution
                # (Skip for speed, trust the elimination)
                
                # Nullspace: for each free column f, set free var f = 1, others 0
                # Solve for pivot variables
                nullspace_basis = []
                for f in free_cols:
                    v = [Fraction(0)] * R
                    v[f] = Fraction(1)
                    for row_idx in range(rank_A):
                        col = pivot_cols[row_idx]
                        # v[col] = -aug[row_idx][f] (since aug[row_idx][col]*v[col] + ... + aug[row_idx][f]*1 = 0)
                        v[col] = -aug[row_idx][f]
                    nullspace_basis.append(v)
                
                # Now we need w = w_particular + sum(t_i * v_i) to have half-integer entries.
                # This is a linear constraint on the t_i's.
                # Each entry w[r] = w_particular[r] + sum_i t_i * v_i[r] must be in (1/2)Z.
                
                # If there are no free variables (full rank), just check if w_particular is half-integer.
                if not free_cols:
                    if all(is_halfint(x) for x in w_particular):
                        W[k*n+i] = w_particular
                    else:
                        return None
                    continue
                
                # We have len(free_cols) parameters. Try small rational values for t_i.
                # Since we want half-integer entries, and the coefficients are rational,
                # we can try t_i in {0, 1/2, 1, -1/2, -1} or similar.
                
                # Try all combinations of small values
                candidate_vals = [Fraction(0), Fraction(1,2), Fraction(1), Fraction(-1,2), Fraction(-1)]
                
                found = False
                num_free = len(free_cols)
                
                # Limit the search: if num_free is large, this explodes.
                # For our problem, R is at most ~42, m=3, so rank_A <= 3, free_cols >= R-3.
                # If R=30, free_cols >= 27. That's too many to enumerate.
                
                # Alternative: solve the linear constraints exactly.
                # For each entry r: w_particular[r] + sum_i t_i * v_i[r] = (2*z_r + 1)/2 or z_r/2
                # This is: sum_i t_i * v_i[r] = target_r - w_particular[r]
                # where target_r is some half-integer.
                
                # This is a system of R equations in num_free unknowns, with the right-hand side
                # being constrained to specific values.
                
                # Simpler approach: if num_free is small (say <= 4), enumerate.
                # If num_free is large, we have more freedom. Let's try a greedy approach:
                # Set t_i = 0 for all i, check if w_particular works. If not, try adjusting.
                
                # Actually, let's think differently. We have many degrees of freedom.
                # Let's try to find ANY half-integer solution by solving the linear system
                # with the constraint that entries are half-integers.
                
                # For each entry r, we need:
                # w_particular[r] + sum_i t_i * v_i[r] in (1/2)Z
                # => sum_i t_i * v_i[r] in (1/2)Z - w_particular[r]
                
                # Let's try t_i = 0 first.
                if all(is_halfint(x) for x in w_particular):
                    W[k*n+i] = w_particular
                    found = True
                    continue
                
                # Try setting some t_i to simple values.
                # If num_free <= 5, enumerate.
                if num_free <= 5:
                    for combo in product(candidate_vals, repeat=num_free):
                        w_test = [Fraction(0)] * R
                        for r in range(R):
                            w_test[r] = w_particular[r]
                            for idx in range(num_free):
                                w_test[r] += combo[idx] * nullspace_basis[idx][r]
                        if all(is_halfint(x) for x in w_test):
                            W[k*n+i] = w_test
                            found = True
                            break
                    if found:
                        continue
                
                # For larger num_free, try a heuristic:
                # Set t_i = 0 for all, then for each entry that's not half-integer,
                # try to fix it by adjusting one t_i.
                # This is complex. Let's just return None for now and try different U, V.
                return None
        
        return W

    # Try to find a lower rank decomposition
    # Lower bound: max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    # So rank >= 21.
    
    # Try ranks from 21 upward, but prioritize "nice" values.
    # Actually, we want the MINIMUM rank, so try from low to high.
    # But finding a valid decomposition is hard, so let's try a few ranks.
    
    # For each rank R, try random U, V with small integer/half-integer entries,
    # then solve for W and check if W has half-integer entries.
    
    # Entry choices for U and V: {-1, 0, 1} or {-0.5, 0, 0.5, 1}
    # Using Fractions for exact arithmetic.
    
    entries_frac = [Fraction(0), Fraction(1), Fraction(-1), Fraction(1,2), Fraction(-1,2)]
    
    # Try ranks from low to high
    ranks_to_try = list(range(21, 43))  # 21 to 42
    # Actually, let's be smarter. Try a subset first.
    # Known results: For 2x3x7, the rank is believed to be around 24-28.
    # Let's try: 21, 22, 23, 24, 25, 26, 27, 28, 30, 35
    
    ranks_to_try = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 35, 40]
    
    for R in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Try several random initializations
        num_attempts = 200 if R <= 25 else 100
        
        for attempt in range(num_attempts):
            if time.time() > deadline:
                break
            
            # Random U and V with small entries
            U = [[random.choice(entries_frac) for _ in range(R)] for _ in range(n * m)]
            V = [[random.choice(entries_frac) for _ in range(R)] for _ in range(m * p)]
            
            # Solve for W
            W = solve_for_W(U, V, R)
            
            if W is not None:
                # Verify the decomposition
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = Fraction(0)
                            for r in range(R):
                                s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                            if s != 1:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    # Convert to the required format (int or half-integer)
                    U_out = [[float(x) for x in row] for row in U]
                    V_out = [[float(x) for x in row] for row in V]
                    W_out = [[float(x) for x in row] for row in W]
                    
                    best_result = {"U": U_out, "V": V_out, "W": W_out}
                    best_rank = R
                    break  # Found a valid decomposition for this rank, move to next rank
        
        if best_rank == R:
            # Found the minimum rank (since we're going from low to high)
            break
    
    return best_result
# EVOLVE-BLOCK-END
