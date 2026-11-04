"""Matrix multiplication 4x5x6 rank minimization."""
import time
import sys
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Strategy: Exploit rectangular structure by fixing sparse patterns for U and V,
    # then solve for W using linear equations with exact arithmetic.
    
    # The matrix multiplication tensor M_{n,m,p} satisfies:
    # T[i,j,k] = delta_{i,k} for C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # A rank-r decomposition: T[i,j,k] = sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t]
    
    # Approach: Try different rank values and sparse patterns for U and V,
    # then check if a consistent W exists using linear algebra.
    
    def check_decomposition(U, V, W, n, m, p):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for t in range(len(U[0])):
                        total += Fraction(U[i*m+j][t]) * Fraction(V[j*p+k][t]) * Fraction(W[k*n+i][t])
                    expected = 1 if i == k else 0
                    if total != expected:
                        return False
        return True
    
    # Try to find a solution with rank < 120
    # Use a systematic approach: try rank values from 119 down to some minimum
    
    # For now, return the schoolbook solution which is guaranteed correct.
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
