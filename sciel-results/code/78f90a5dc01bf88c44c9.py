import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    best_result = None
    U_school = [[0]*(n*m*p) for _ in range(n*m)]
    V_school = [[0]*(n*m*p) for _ in range(m*p)]
    W_school = [[0]*(n*m*p) for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    start_time = time.time()
    deadline = start_time + 110
    
    # Search for lower rank
    # The minimum rank is likely significantly lower than 63.
    # For 3x3x3 it is 23.
    # For 3x3x7, let's try from 62 down to 20.
    
    for r in range(62, 19, -1):
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        for trial in range(20):
            if time.time() > deadline:
                break
                
            # Random U, V with small integers
            # Using {-1, 0, 1}
            U_rand = np.random.randint(-1, 2, size=(n*m, r))
            V_rand = np.random.randint(-1, 2, size=(m*p, r))
            
            valid = True
            W_sol = []
            
            for i in range(n):
                for k in range(p):
                    # Build matrix A of size m x r
                    # A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    A = np.zeros((m, r))
                    for j in range(m):
                        for c in range(r):
                            A[j, c] = U_rand[i*m+j, c] * V_rand[j*p+k, c]
                    
                    # Solve A x = 1
                    # x is W[k*n+i, :]
                    # We need exact rational solution.
                    # Convert to Fractions for exactness?
                    # Or use numpy and check residual?
                    # Since entries are small integers, we can use Fraction.
                    
                    # Use Fraction for A and b
                    A_frac = [[Fraction(int(A[j, c])) for c in range(r)] for j in range(m)]
                    b_frac = [Fraction(1) for _ in range(m)]
                    
                    # Gaussian elimination to check consistency and find solution
                    # We just need to know if a solution exists.
                    # If it exists, we can find one.
                    
                    # Augmented matrix
                    aug = [row + [b_frac[j]] for j, row in enumerate(A_frac)]
                    
                    # Row reduce
                    rows = m
                    cols = r
                    rank = 0
                    for col in range(cols):
                        # Find pivot
                        pivot_row = -1
                        for row in range(rank, rows):
                            if aug[row][col] != 0:
                                pivot_row = row
                                break
                        if pivot_row == -1:
                            continue
                        # Swap
                        aug[rank], aug[pivot_row] = aug[pivot_row], aug[rank]
                        # Normalize
                        pivot_val = aug[rank][col]
                        for c in range(col, cols + 1):
                            aug[rank][c] /= pivot_val
                        # Eliminate
                        for row in range(rows):
                            if row != rank and aug[row][col] != 0:
                                factor = aug[row][col]
                                for c in range(col, cols + 1):
                                    aug[row][c] -= factor * aug[rank][c]
                        rank += 1
                        if rank == cols:
                            break
                    
                    # Check consistency
                    # If any row is [0, ..., 0, b] with b != 0, no solution
                    consistent = True
                    for row in range(rank, rows):
                        if aug[row][cols] != 0:
                            consistent = False
                            break
                    
                    if not consistent:
                        valid = False
                        break
                    
                    # Find a solution (set free variables to 0)
                    # This is a bit slow, but r is smallish.
                    # Actually, we just need to verify existence for now to filter.
                    # If we want to return the solution, we need to compute it.
                    # Let's assume if consistent, we can find it.
                    # For now, mark as valid.
                    
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Re-solve to get W
                W_list = [[0]*r for _ in range(p*n)]
                for i in range(n):
                    for k in range(p):
                        A = np.zeros((m, r))
                        for j in range(m):
                            for c in range(r):
                                A[j, c] = U_rand[i*m+j, c] * V_rand[j*p+k, c]
                        A_frac = [[Fraction(int(A[j, c])) for c in range(r)] for j in range(m)]
                        b_frac = [Fraction(1) for _ in range(m)]
                        aug = [row + [b_frac[j]] for j, row in enumerate(A_frac)]
                        
                        # Row reduce again to get solution
                        rows = m
                        cols = r
                        rank = 0
                        pivot_cols = []
                        for col in range(cols):
                            pivot_row = -1
                            for row in range(rank, rows):
                                if aug[row][col] != 0:
                                    pivot_row = row
                                    break
                            if pivot_row == -1:
                                continue
                            aug[rank], aug[pivot_row] = aug[pivot_row], aug[rank]
                            pivot_val = aug[rank][col]
                            for c in range(col, cols + 1):
                                aug[rank][c] /= pivot_val
                            for row in range(rows):
                                if row != rank and aug[row][col] != 0:
                                    factor = aug[row][col]
                                    for c in range(col, cols + 1):
                                        aug[row][c] -= factor * aug[rank][c]
                            rank += 1
                            pivot_cols.append(col)
                            if rank == cols:
                                break
                        
                        # Back substitution
                        x = [Fraction(0) for _ in range(cols)]
                        for i_pivot in range(rank-1, -1, -1):
                            col = pivot_cols[i_pivot]
                            val = aug[i_pivot][cols]
                            for c in range(col+1, cols):
                                val -= aug[i_pivot][c] * x[c]
                            x[col] = val
                            
                        W_list[k*n+i] = [float(val) for val in x]
                        
                return {"U": U_rand.tolist(), "V": V_rand.tolist(), "W": W_list}

    return best_result
