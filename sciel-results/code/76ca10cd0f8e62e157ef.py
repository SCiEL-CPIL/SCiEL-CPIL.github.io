import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Build the rank-28 construction as a fallback
    rank_28_result = _build_rank_28()
    
    # Try rank-27 feasibility test
    if time.time() < deadline - 5:
        rank_27_result = _try_rank_27()
        if rank_27_result is not None:
            return rank_27_result
    
    return rank_28_result

def _build_rank_28():
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: A_L (2x2) * B_T1 (2x2) -> C_1 (cols 0,1)
    # Col 0: M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[6][0] = 1
    W[0][0] = 1; W[3][0] = 1

    # Col 1: M2 = (A10+A11)B00
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1

    # Col 2: M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[6][2] = -1
    W[2][2] = 1; W[3][2] = 1

    # Col 3: M4 = A11(B10-B00)
    U[4][3] = 1
    V[5][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1

    # Col 4: M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[6][4] = 1
    W[0][4] = -1; W[2][4] = 1

    # Col 5: M6 = (A10-A00)(B00+B01)
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1

    # Col 6: M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[4][6] = -1
    V[5][6] = 1; V[6][6] = 1
    W[0][6] = 1

    # Part 2: A_L (2x2) * B_T2 (2x2) -> C_2 (cols 2,3)
    # Col 7: M1
    U[0][7] = 1; U[4][7] = 1
    V[2][7] = 1; V[8][7] = 1
    W[4][7] = 1; W[7][7] = 1

    # Col 8: M2
    U[3][8] = 1; U[4][8] = 1
    V[2][8] = 1
    W[5][8] = 1; W[7][8] = -1

    # Col 9: M3
    U[0][9] = 1
    V[3][9] = 1; V[8][9] = -1
    W[6][9] = 1; W[7][9] = 1

    # Col 10: M4
    U[4][10] = 1
    V[7][10] = 1; V[2][10] = -1
    W[4][10] = 1; W[5][10] = 1

    # Col 11: M5
    U[0][11] = 1; U[1][11] = 1
    V[8][11] = 1
    W[4][11] = -1; W[6][11] = 1

    # Col 12: M6
    U[3][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[7][12] = 1

    # Col 13: M7
    U[1][13] = 1; U[4][13] = -1
    V[7][13] = 1; V[8][13] = 1
    W[4][13] = 1

    # Part 3: A_L (2x2) * B_T3 (2x1) -> C_3 (col 4)
    # Col 14: A00*B04
    U[0][14] = 1
    V[4][14] = 1
    W[8][14] = 1

    # Col 15: A01*B14
    U[1][15] = 1
    V[9][15] = 1
    W[8][15] = 1

    # Col 16: A10*B04
    U[3][16] = 1
    V[4][16] = 1
    W[9][16] = 1

    # Col 17: A11*B14
    U[4][17] = 1
    V[9][17] = 1
    W[9][17] = 1

    # Part 4: A_R (2x1) * B_B (1x5) -> C_4 (all cols)
    # Col 18: A02*B20
    U[2][18] = 1
    V[10][18] = 1
    W[0][18] = 1

    # Col 19: A02*B21
    U[2][19] = 1
    V[11][19] = 1
    W[2][19] = 1

    # Col 20: A02*B22
    U[2][20] = 1
    V[12][20] = 1
    W[4][20] = 1

    # Col 21: A02*B23
    U[2][21] = 1
    V[13][21] = 1
    W[6][21] = 1

    # Col 22: A02*B24
    U[2][22] = 1
    V[14][22] = 1
    W[8][22] = 1

    # Col 23: A12*B20
    U[5][23] = 1
    V[10][23] = 1
    W[1][23] = 1

    # Col 24: A12*B21
    U[5][24] = 1
    V[11][24] = 1
    W[3][24] = 1

    # Col 25: A12*B22
    U[5][25] = 1
    V[12][25] = 1
    W[5][25] = 1

    # Col 26: A12*B23
    U[5][26] = 1
    V[13][26] = 1
    W[7][26] = 1

    # Col 27: A12*B24
    U[5][27] = 1
    V[14][27] = 1
    W[9][27] = 1

    return {"U": U, "V": V, "W": W}

def _try_rank_27():
    """
    Try to find a rank-27 decomposition by exploiting tensor symmetries
    and linear dependencies in the dual space.
    """
    n, m, p = DIMENSIONS
    target_rank = 27
    
    # Build the tensor T[i,j,k] = 1 if C_k depends on A_i * B_j
    tensor = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                tensor[(u_idx, v_idx, w_idx)] = 1
    
    # Start from rank-28 and try to find a linear dependency
    base = _build_rank_28()
    U_base = [row[:] for row in base["U"]]
    V_base = [row[:] for row in base["V"]]
    W_base = [row[:] for row in base["W"]]
    
    # Try to find a column that can be expressed as a linear combination
    # of the others by solving a linear system
    
    # For each pair of columns (a, b), try to express column a as
    # alpha * column b + (linear combo of others)
    # This is equivalent to finding a vector c in the null space of
    # the matrix formed by U and V columns
    
    # Build the "coefficient matrix" M where M[w, col] = W[w, col]
    # and check if there's a linear dependency
    
    # Alternative approach: try to merge two columns by adjusting
    # U and V entries to be in {-1, 0, 1} and solving for W
    
    # Try all pairs of columns to merge
    for col_a in range(target_rank):
        for col_b in range(col_a + 1, target_rank):
            # Try to merge col_b into col_a by setting
            # U_new[:, col_b] = U[:, col_a] + delta_U
            # V_new[:, col_b] = V[:, col_a] + delta_V
            # where delta_U and delta_V are small adjustments
            
            # For simplicity, try setting col_b = col_a (identical)
            # and see if we can solve for W
            U_new = [row[:] for row in U_base]
            V_new = [row[:] for row in V_base]
            
            # Set col_b to be identical to col_a
            for i in range(n * m):
                U_new[i][col_b] = U_new[i][col_a]
            for j in range(m * p):
                V_new[j][col_b] = V_new[j][col_a]
            
            # Now try to solve for W
            W_solution = [[Fraction(0)] * target_rank for _ in range(p * n)]
            
            valid = True
            for w in range(p * n):
                # Build system: for each (u,v) pair
                equations = []
                for u in range(n * m):
                    for v in range(m * p):
                        # Only consider (u,v) pairs where U[u][col] and V[v][col]
                        # are both non-zero for some col
                        has_support = False
                        for col in range(target_rank):
                            if U_new[u][col] != 0 and V_new[v][col] != 0:
                                has_support = True
                                break
                        if not has_support:
                            continue
                        
                        lhs = [Fraction(U_new[u][col]) * Fraction(V_new[v][col]) for col in range(target_rank)]
                        rhs = Fraction(tensor.get((u, v, w), 0))
                        equations.append((lhs, rhs))
                
                if not _solve_linear_system(equations, W_solution[w]):
                    valid = False
                    break
            
            if valid:
                # Verify the solution
                all_valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            computed = Fraction(0)
                            for col in range(target_rank):
                                computed += Fraction(U_new[u_idx][col]) * Fraction(V_new[v_idx][col]) * W_solution[w_idx][col]
                            expected = Fraction(tensor.get((u_idx, v_idx, w_idx), 0))
                            if computed != expected:
                                all_valid = False
                                break
                        if not all_valid:
                            break
                    if not all_valid:
                        break
                
                if all_valid:
                    U_final = [[float(U_new[i][c]) for c in range(target_rank)] for i in range(n * m)]
                    V_final = [[float(V_new[j][c]) for c in range(target_rank)] for j in range(m * p)]
                    W_final = [[float(W_solution[w][c]) for c in range(target_rank)] for w in range(p * n)]
                    
                    return {"U": U_final, "V": V_final, "W": W_final}
    
    return None

def _solve_linear_system(equations, solution):
    """
    Solve a linear system using Gaussian elimination with rationals.
    equations: list of (lhs, rhs) where lhs is list of coefficients
    solution: list to fill with the solution
    Returns True if a unique solution exists and was found.
    """
    if not equations:
        return True
    
    num_vars = len(equations[0][0])
    num_eqs = len(equations)
    
    # Create augmented matrix
    aug = []
    for lhs, rhs in equations:
        row = [Fraction(x) for x in lhs] + [Fraction(rhs)]
        aug.append(row)
    
    # Gaussian elimination
    n_rows = len(aug)
    n_cols = num_vars
    
    for col in range(n_cols):
        # Find pivot
        pivot_row = None
        for row in range(col, n_rows):
            if aug[row][col] != 0:
                pivot_row = row
                break
        
        if pivot_row is None:
            # No pivot - check if consistent
            for row in range(col, n_rows):
                if aug[row][n_cols] != 0:
                    return False
            continue
        
        # Swap
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        
        # Eliminate
        for row in range(n_rows):
            if row != col and aug[row][col] != 0:
                factor = aug[row][col] / aug[col][col]
                for j in range(col, n_cols + 1):
                    aug[row][j] -= factor * aug[col][j]
    
    # Back-substitution
    for i in range(n_cols):
        if i < n_rows and aug[i][i] != 0:
            solution[i] = aug[i][n_cols] / aug[i][i]
        elif i < n_rows:
            # Check consistency
            if aug[i][n_cols] != 0:
                return False
            solution[i] = Fraction(0)
        else:
            solution[i] = Fraction(0)
    
    return True
