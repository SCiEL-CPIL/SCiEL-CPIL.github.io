import numpy as np
import time
import random
from fractions import Fraction

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

    def solve_for_W_int(U_int, V_int, R):
        """
        Given U_int (nm x R) and V_int (mp x R) as lists of lists of ints,
        find W_int (pn x R) as lists of lists of ints such that:
        sum_r U_int[i*m+j, r] * V_int[j*p+k, r] * W_int[k*n+i, r] = 8
        Returns W_int or None.
        """
        W_int = [[0] * R for _ in range(p * n)]
        
        for k in range(p):
            for i in range(n):
                # System: A * w = b
                # A[j, r] = U_int[i*m+j, r] * V_int[j*p+k, r]
                # b[j] = 8
                A = [[0] * R for _ in range(m)]
                for j in range(m):
                    for r in range(R):
                        A[j][r] = U_int[i*m+j][r] * V_int[j*p+k][r]
                
                b = [8] * m
                
                # Convert to Fractions for exact solving
                A_frac = [[Fraction(A[j][r]) for r in range(R)] for j in range(m)]
                b_frac = [Fraction(b[j]) for j in range(m)]
                
                # Augmented matrix
                aug = []
                for j in range(m):
                    aug.append(A_frac[j] + [b_frac[j]])
                
                # Gaussian Elimination
                num_rows = m
                num_cols = R + 1
                rank_A = 0
                pivot_cols = []
                
                for col in range(R):
                    pivot_row = None
                    for row in range(rank_A, num_rows):
                        if aug[row][col] != 0:
                            pivot_row = row
                            break
                    if pivot_row is None:
                        continue
                    if pivot_row != rank_A:
                        aug[rank_A], aug[pivot_row] = aug[pivot_row], aug[rank_A]
                    
                    pivot_val = aug[rank_A][col]
                    for row in range(num_rows):
                        if row != rank_A and aug[row][col] != 0:
                            factor = aug[row][col] / pivot_val
                            for c in range(col, num_cols):
                                aug[row][c] -= factor * aug[rank_A][c]
                    
                    pivot_cols.append(col)
                    rank_A += 1
                
                # Check consistency
                for row in range(num_rows):
                    if all(aug[row][c] == 0 for c in range(R)) and aug[row][R] != 0:
                        return None
                
                # Particular solution (free vars = 0)
                w_part = [Fraction(0)] * R
                for row_idx in range(rank_A):
                    col = pivot_cols[row_idx]
                    w_part[col] = aug[row_idx][R]
                
                # Check if w_part is integer
                if all(w_part[r].denominator == 1 for r in range(R)):
                    W_int[k*n+i] = [int(w_part[r]) for r in range(R)]
                    continue
                
                # If not integer, try to find an integer solution in the affine space
                # by searching over small integer shifts in the null space
                # Get null space basis (free variables)
                free_cols = [c for c in range(R) if c not in pivot_cols]
                
                if not free_cols:
                    return None
                
                # Try small integer values for free variables
                # Limit search to keep it fast
                max_search = 3
                found = False
                for free_vals in _generate_combinations(free_cols, max_search):
                    w_candidate = [Fraction(0)] * R
                    # Set free variables
                    for fc, fv in zip(free_cols, free_vals):
                        w_candidate[fc] = Fraction(fv)
                    
                    # Solve for pivot variables
                    for row_idx in range(rank_A):
                        col = pivot_cols[row_idx]
                        val = aug[row_idx][R]
                        for r in range(R):
                            if r != col:
                                val -= aug[row_idx][r] * w_candidate[r]
                        w_candidate[col] = val / aug[row_idx][col]
                    
                    # Check if all are integers
                    if all(w_candidate[r].denominator == 1 for r in range(R)):
                        W_int[k*n+i] = [int(w_candidate[r]) for r in range(R)]
                        found = True
                        break
                
                if not found:
                    return None
        
        return W_int

    def _generate_combinations(free_cols, max_val):
        """Generate all combinations of small integers for free columns."""
        if not free_cols:
            yield []
            return
        first = free_cols[0]
        rest = free_cols[1:]
        for val in range(-max_val, max_val + 1):
            for rest_combo in _generate_combinations(rest, max_val):
                yield [val] + rest_combo

    # Try to find a lower rank decomposition
    # Lower bound is max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    ranks_to_try = list(range(22, 43))
    
    # Integer entries for U, V (scaled by 2, so original entries are multiples of 0.5)
    # If we use integers for U', V', then U = U'/2.
    # We want U, V, W to be half-integers.
    # So U', V', W' should be integers.
    # Equation: sum U' V' W' = 8.
    
    entries_int = [-1, 0, 1]
    
    for R in ranks_to_try:
        if time.time() > deadline:
            break
        
        num_attempts = 800 if R <= 28 else 400
        
        for attempt in range(num_attempts):
            if time.time() > deadline:
                break
            
            # Use structured random initialization with bias toward sparse matrices
            U_int = [[random.choice(entries_int) for _ in range(R)] for _ in range(n * m)]
            V_int = [[random.choice(entries_int) for _ in range(R)] for _ in range(m * p)]
            
            # Additional heuristic: ensure some rows have non-zero entries
            # to avoid trivially zero systems
            for row_idx in range(n * m):
                if all(U_int[row_idx][r] == 0 for r in range(R)):
                    U_int[row_idx][random.randint(0, R-1)] = 1
            
            for row_idx in range(m * p):
                if all(V_int[row_idx][r] == 0 for r in range(R)):
                    V_int[row_idx][random.randint(0, R-1)] = 1
            
            W_int = solve_for_W_int(U_int, V_int, R)
            
            if W_int is not None:
                # Verify
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for r in range(R):
                                s += U_int[i*m+j][r] * V_int[j*p+k][r] * W_int[k*n+i][r]
                            if s != 8:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                
                if valid:
                    # Scale back to 0.5
                    U_out = [[x / 2.0 for x in row] for row in U_int]
                    V_out = [[x / 2.0 for x in row] for row in V_int]
                    W_out = [[x / 2.0 for x in row] for row in W_int]
                    
                    best_result = {"U": U_out, "V": V_out, "W": W_out}
                    best_rank = R
                    break
        
        if best_rank == R:
            break
    
    return best_result
# EVOLVE-BLOCK-END
