"""Tensor decomposition for 3x4 x 4x6 matrix multiplication."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Schoolbook solution
    U_school = [[0] * schoolbook_rank for _ in range(n*m)]
    V_school = [[0] * schoolbook_rank for _ in range(m*p)]
    W_school = [[0] * schoolbook_rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = schoolbook_rank

    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if total != 1:
                        return False
        return True

    def try_rank_solving(rank, solve_for='W'):
        """Try to find a decomposition of given rank by fixing two matrices and solving for the third."""
        nonlocal best_result, best_rank, deadline
        
        num_vars = {'U': n*m*rank, 'V': m*p*rank, 'W': p*n*rank}
        num_eqs = n * m * p
        max_attempts = 150
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return False
            
            # Initialize all three with small random integers
            U = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(n*m)]
            V = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(m*p)]
            W = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(p*n)]
            
            # Build linear system for the third matrix
            A_mat = np.zeros((num_eqs, num_vars[solve_for]), dtype=float)
            b_vec = np.ones(num_eqs, dtype=float)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(rank):
                            if solve_for == 'W':
                                var_idx = (k * n + i) * rank + r
                                A_mat[eq_idx, var_idx] = U[i*m+j][r] * V[j*p+k][r]
                            elif solve_for == 'U':
                                var_idx = (i * m + j) * rank + r
                                A_mat[eq_idx, var_idx] = V[j*p+k][r] * W[k*n+i][r]
                            elif solve_for == 'V':
                                var_idx = (j * p + k) * rank + r
                                A_mat[eq_idx, var_idx] = U[i*m+j][r] * W[k*n+i][r]
                        eq_idx += 1
            
            # Solve least squares
            try:
                sol, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            except:
                continue
                
            # Round to half-integers
            sol_rounded = np.round(sol * 2) / 2.0
            
            # Convert to list of lists and verify
            if solve_for == 'W':
                sol_list = [[0] * rank for _ in range(p*n)]
                for idx in range(p*n):
                    for r in range(rank):
                        sol_list[idx][r] = Fraction(sol_rounded[idx * rank + r])
                if verify(U, V, sol_list, rank):
                    best_result = {"U": U, "V": V, "W": sol_list}
                    best_rank = rank
                    return True
            elif solve_for == 'U':
                sol_list = [[0] * rank for _ in range(n*m)]
                for idx in range(n*m):
                    for r in range(rank):
                        sol_list[idx][r] = Fraction(sol_rounded[idx * rank + r])
                if verify(sol_list, V, W, rank):
                    best_result = {"U": sol_list, "V": V, "W": W}
                    best_rank = rank
                    return True
            elif solve_for == 'V':
                sol_list = [[0] * rank for _ in range(m*p)]
                for idx in range(m*p):
                    for r in range(rank):
                        sol_list[idx][r] = Fraction(sol_rounded[idx * rank + r])
                if verify(U, sol_list, W, rank):
                    best_result = {"U": U, "V": sol_list, "W": W}
                    best_rank = rank
                    return True
                
        return False

    # Try ranks from 71 down to 60
    for rank in range(schoolbook_rank - 1, 59, -1):
        if time.time() > deadline - 10:
            break
        found = False
        for solve_for in ['W', 'U', 'V']:
            if time.time() > deadline - 10:
                break
            if try_rank_solving(rank, solve_for):
                found = True
                break
        if found:
            break

    return best_result
