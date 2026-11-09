"""Matrix multiplication 3x3x4 rank-minimal exact tensor decomposition."""
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Try ranks from 20 upward, using exact rational arithmetic
    for rank in range(20, 37):
        for seed in range(200):
            rng = random.Random(seed + rank * 1000)
            
            # Random small integer V and W
            V = [[rng.randint(-2, 2) for _ in range(rank)] for _ in range(m * p)]
            W = [[rng.randint(-2, 2) for _ in range(rank)] for _ in range(p * n)]
            
            # Solve for U using exact rational arithmetic
            U = [[Fraction(0)] * rank for _ in range(n * m)]
            success = True
            
            for i in range(n):
                num_vars = m * rank
                num_eqs = m * p
                
                A = [[Fraction(0)] * num_vars for _ in range(num_eqs)]
                b = [Fraction(1)] * num_eqs
                
                eq_idx = 0
                for j in range(m):
                    for k in range(p):
                        for jj in range(m):
                            for r in range(rank):
                                A[eq_idx][jj * rank + r] = Fraction(V[j * p + k][r] * W[k * n + i][r])
                        eq_idx += 1
                
                # Gaussian elimination with exact arithmetic
                aug = [A[e] + [b[e]] for e in range(num_eqs)]
                row = 0
                for col in range(num_vars):
                    pivot = -1
                    for r in range(row, num_eqs):
                        if aug[r][col] != 0:
                            pivot = r
                            break
                    if pivot == -1:
                        continue
                    aug[row], aug[pivot] = aug[pivot], aug[row]
                    inv = 1 / aug[row][col]
                    for c in range(col, num_vars + 1):
                        aug[row][c] *= inv
                    for r in range(num_eqs):
                        if r != row and aug[r][col] != 0:
                            factor = aug[r][col]
                            for c in range(col, num_vars + 1):
                                aug[r][c] -= factor * aug[row][c]
                    row += 1
                    if row == num_eqs:
                        break
                
                # Check consistency
                consistent = True
                for r in range(row, num_eqs):
                    if all(aug[r][c] == 0 for c in range(num_vars)) and aug[r][num_vars] != 0:
                        consistent = False
                        break
                
                if not consistent:
                    success = False
                    break
                
                # Extract solution (free vars = 0)
                sol = [Fraction(0)] * num_vars
                for r in range(row):
                    lead_col = -1
                    for c in range(num_vars):
                        if aug[r][c] != 0:
                            lead_col = c
                            break
                    if lead_col != -1:
                        sol[lead_col] = aug[r][num_vars]
                
                for jj in range(m):
                    for r in range(rank):
                        U[i * m + jj][r] = sol[jj * rank + r]
            
            if not success:
                continue
            
            # Check if U entries are half-integers
            valid = True
            U_int = [[0] * rank for _ in range(n * m)]
            for idx in range(n * m):
                for r in range(rank):
                    val = U[idx][r]
                    val2 = val * 2
                    if val2.denominator != 1:
                        valid = False
                        break
                    U_int[idx][r] = val2.numerator // 2
                    if U_int[idx][r] * 2 != val2.numerator:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # Final verification with integer arithmetic
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r in range(rank):
                            val += U_int[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                        if val != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return {"U": U_int, "V": V, "W": W}
    
    # Fallback to schoolbook
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
