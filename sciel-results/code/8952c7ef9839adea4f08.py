"""Tensor decomposition for 3x4 x 4x6 matrix multiplication."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin: 120s limit, use 115s
    
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

    def try_rank_solving(rank):
        """Try to find a decomposition of given rank by fixing U, V and solving for W."""
        nonlocal best_result, best_rank, deadline
        
        # Use more attempts for lower ranks, fewer for higher ranks
        if rank <= 64:
            max_attempts = 200
        elif rank <= 68:
            max_attempts = 100
        else:
            max_attempts = 50
            
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return False
            
            # Initialize U and V with small random integers
            # Use a wider range for lower ranks to increase chance of finding a solution
            if rank <= 64:
                val_range = 3
            else:
                val_range = 2
                
            U = [[random.randint(-val_range, val_range) for _ in range(rank)] for _ in range(n*m)]
            V = [[random.randint(-val_range, val_range) for _ in range(rank)] for _ in range(m*p)]
            
            # Build linear system for W
            # Variables: W[k*n + i, r] for k in p, i in n, r in rank
            # Eq for (i, j, k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            
            num_vars = p * n * rank
            num_eqs = n * m * p
            
            A_mat = np.zeros((num_eqs, num_vars), dtype=float)
            b_vec = np.ones(num_eqs, dtype=float)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(rank):
                            var_idx = (k * n + i) * rank + r
                            A_mat[eq_idx, var_idx] = U[i*m+j][r] * V[j*p+k][r]
                        eq_idx += 1
            
            # Solve least squares
            try:
                W_sol, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            except:
                continue
                
            # Check if the solution is close to exact (residuals should be near zero)
            # Only proceed if the solution is reasonably good
            if len(residuals) > 0 and residuals[0] > 1e-10:
                continue
                
            # Round to half-integers (multiply by 2, round, divide by 2)
            W_rounded = np.round(W_sol * 2) / 2.0
            
            # Convert to list of lists for verification
            W_list = [[0] * rank for _ in range(p*n)]
            for idx in range(p*n):
                for r in range(rank):
                    W_list[idx][r] = Fraction(W_rounded[idx * rank + r])
            
            # Verify
            if verify(U, V, W_list, rank):
                best_result = {"U": U, "V": V, "W": W_list}
                best_rank = rank
                return True
                
        return False

    # Try ranks from 71 down to 55
    for rank in range(schoolbook_rank - 1, 54, -1):
        if time.time() > deadline - 10:
            break
        if try_rank_solving(rank):
            break

    return best_result
