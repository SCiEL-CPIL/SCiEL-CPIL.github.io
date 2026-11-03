import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # safety margin before 150s limit
    
    # Start with schoolbook (rank 50)
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    # Convert to Fraction for exact arithmetic
    U_sb_frac = [[Fraction(int(x)) for x in row] for row in U_sb.tolist()]
    V_sb_frac = [[Fraction(int(x)) for x in row] for row in V_sb.tolist()]
    W_sb_frac = [[Fraction(int(x)) for x in row] for row in W_sb.tolist()]
    
    # Try to reduce rank by removing linearly dependent columns
    # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We want to find a subset of columns that still spans the tensor
    
    # Strategy: use Strassen-like block decomposition
    # Partition A (2x5) into blocks, B (5x5) into blocks
    # A is 2x5: rows 0-1, cols 0-4
    # B is 5x5: rows 0-4, cols 0-4
    # C is 2x5: rows 0-1, cols 0-4
    
    # Let's try a direct approach: use the fact that for small matrices,
    # we can find rank via exhaustive search or known bounds.
    
    # Known: rank(2x5x5) <= 2*5*5 = 50 (schoolbook)
    # Can we do better? Let's try to use block structure.
    
    # Alternative: use the identity that if we can decompose into blocks
    # where each block uses Strassen, we might save rank.
    
    # Let's try: A = [A1 A2] where A1 is 2x2, A2 is 2x3
    # B = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x3, B21 is 3x2, B22 is 3x3
    # C = [A1 B11 + A2 B21, A1 B12 + A2 B22]
    
    # For each term:
    # A1*B11: 2x2 * 2x2, rank 4 or 7 (Strassen)
    # A2*B21: 2x3 * 3x2, rank 6 or less
    # A1*B12: 2x2 * 2x3, rank 4*3=12 or less
    # A2*B22: 2x3 * 3x3, rank 2*3*3=18 or less
    
    # Total with Strassen on 2x2: 7 + 6 + 12 + 18 = 43
    # But we need to be careful about the exact decomposition.
    
    # Let me try a different approach: use random initialization and try to find
    # a lower rank decomposition by solving the tensor equation.
    
    # Actually, let's just try rank 49 and see if we can solve for it.
    # This is a hard problem, so let's use a simpler heuristic.
    
    # Try to merge columns that are linearly dependent.
    # Build the tensor and check for dependencies.
    
    # For each column c, define the vector v_c in R^(n*m) x R^(m*p) x R^(p*n)
    # Actually, the tensor is T = sum_c u_c ⊗ v_c ⊗ w_c
    # where u_c is in R^(n*m), v_c in R^(m*p), w_c in R^(p*n)
    
    # We want to find a subset of columns that still gives the correct tensor.
    
    # Let's compute the full tensor and then try to remove columns one by one.
    
    # Compute tensor T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for c in range(rank_sb):
                    T[i, j, k] += U_sb_frac[i*m+j][c] * V_sb_frac[j*p+k][c] * W_sb_frac[k*n+i][c]
    
    # Now try to remove columns. Start with all columns included.
    included = list(range(rank_sb))
    
    # Try removing each column and checking if the tensor is still correct.
    # This is expensive, so let's do it greedily.
    
    def check_tensor(cols):
        T_test = np.zeros((n, m, p))
        for c in cols:
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_test[i, j, k] += U_sb_frac[i*m+j][c] * V_sb_frac[j*p+k][c] * W_sb_frac[k*n+i][c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if T_test[i, j, k] != T[i, j, k]:
                        return False
        return True
    
    # Greedy removal
    for c in range(rank_sb - 1, -1, -1):
        if time.time() > deadline - 5:
            break
        test_cols = [x for x in included if x != c]
        if check_tensor(test_cols):
            included = test_cols
    
    final_rank = len(included)
    
    # Build the result
    U = [[0] * final_rank for _ in range(n * m)]
    V = [[0] * final_rank for _ in range(m * p)]
    W = [[0] * final_rank for _ in range(p * n)]
    for new_c, old_c in enumerate(included):
        for i in range(n * m):
            U[i][new_c] = int(U_sb_frac[i][old_c]) if U_sb_frac[i][old_c].denominator == 1 else float(U_sb_frac[i][old_c])
        for j in range(m * p):
            V[j][new_c] = int(V_sb_frac[j][old_c]) if V_sb_frac[j][old_c].denominator == 1 else float(V_sb_frac[j][old_c])
        for k in range(p * n):
            W[k][new_c] = int(W_sb_frac[k][old_c]) if W_sb_frac[k][old_c].denominator == 1 else float(W_sb_frac[k][old_c])
    
    return {"U": U, "V": V, "W": W}
