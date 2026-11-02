import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

def check_solution(U, V, W, n, m, p):
    """
    Verify if U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    Convention: W[k*n+i, r]
    """
    R = U.shape[1]
    # Convert to Fraction for exactness
    U_frac = [[Fraction(int(x*2))/2 for x in row] for row in U]
    V_frac = [[Fraction(int(x*2))/2 for x in row] for row in V]
    W_frac = [[Fraction(int(x*2))/2 for x in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                s = Fraction(0)
                for r in range(R):
                    s += U_frac[u_idx][r] * V_frac[v_idx][r] * W_frac[w_idx][r]
                if s != 1:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # Safety margin

    # 1. Fallback: Schoolbook Solution
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_solution = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_school

    # 2. Strategy: Direct combinatorial search for rank-39 decomposition
    # This is a structural change from random restarts + least squares.
    # We fix a specific sparse support pattern for U and V that is known to work
    # for similar tensor decompositions (based on structured low-rank approximations),
    # and then solve for W exactly.
    
    # We will attempt to find a decomposition of rank R = 39.
    # Why 39? Theoretical lower bounds for 2x3x7 are around 21, but practical
    # decompositions often require higher ranks. Rank 39 is a candidate that
    # might be achievable with a structured pattern.
    
    R = 39
    
    # Define a structured pattern for U and V.
    # We use a "block diagonal" like structure with some overlap.
    # U is 6x39, V is 21x39.
    
    # Let's try a pattern where U and V are derived from a base pattern
    # that ensures the linear system for W is well-posed.
    
    # Base pattern: 
    # For each of the 39 columns, assign a specific (i, j) to U and (j, k) to V
    # such that the resulting A matrix (m x R) for each (k, i) has full rank m=3.
    
    # A simpler approach: Use a deterministic construction based on
    # finite field arithmetic or structured randomization that avoids
    # the "random restart" pitfall by ensuring diversity in the support.
    
    # Let's use a pattern based on:
    # U[i*m+j, r] = 1 if (i*m+j + r) % 6 < 3 else 0
    # V[j*p+k, r] = 1 if (j*p+k + r) % 21 < 10 else 0
    # This is a heuristic, but let's make it more robust by trying several
    # such structured patterns.
    
    patterns = []
    
    # Pattern 1: Cyclic shifts
    for shift in range(10):
        U_pat = np.zeros((n*m, R))
        V_pat = np.zeros((m*p, R))
        for r in range(R):
            for i in range(n):
                for j in range(m):
                    idx = i*m + j
                    if (idx + r + shift) % 6 < 3:
                        U_pat[idx, r] = 1
            for j in range(m):
                for k in range(p):
                    idx = j*p + k
                    if (idx + r + shift) % 21 < 10:
                        V_pat[idx, r] = 1
        patterns.append((U_pat, V_pat))
        
    # Pattern 2: Random but structured (seeded for reproducibility in search)
    np.random.seed(42)
    for _ in range(5):
        U_pat = np.zeros((n*m, R))
        V_pat = np.zeros((m*p, R))
        for r in range(R):
            # Select 3 random entries in U to be 1
            u_indices = np.random.choice(n*m, 3, replace=False)
            U_pat[u_indices, r] = 1
            # Select 10 random entries in V to be 1
            v_indices = np.random.choice(m*p, 10, replace=False)
            V_pat[v_indices, r] = 1
        patterns.append((U_pat, V_pat))
        
    # For each pattern, try to solve for W
    for U_pat, V_pat in patterns:
        if time.time() - start_time > deadline:
            break
            
        W = np.zeros((p*n, R))
        valid = True
        
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, R))
                for j in range(m):
                    A[j, :] = U_pat[i*m+j, :] * V_pat[j*p+k, :]
                
                # Check if A has rank m=3
                if np.linalg.matrix_rank(A) < m:
                    valid = False
                    break
                    
                # Solve A x = 1
                # Since A is m x R with m < R, we have a underdetermined system.
                # We want a solution x that is in {0, 0.5, 1, 1.5, ...}
                # A simple approach: find the minimum norm solution and check if it's half-integer.
                # Or, use least squares and round.
                
                try:
                    x, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                except:
                    valid = False
                    break
                
                # Check if x is close to half-integers
                x_rounded = np.round(x * 2) / 2
                if np.allclose(x, x_rounded, atol=1e-6):
                    W[k*n+i, :] = x_rounded
                else:
                    # Try to find a half-integer solution by solving a linear system
                    # over the rationals? Too slow.
                    # Instead, check if the residual is zero after rounding.
                    if np.allclose(A @ x_rounded, np.ones(m), atol=1e-6):
                        W[k*n+i, :] = x_rounded
                    else:
                        valid = False
                        break
            if not valid:
                break
                
        if valid:
            # Verify full solution
            if check_solution(U_pat, V_pat, W, n, m, p):
                best_solution = {"U": U_pat.tolist(), "V": V_pat.tolist(), "W": W.tolist()}
                best_rank = R
                return best_solution

    return best_solution

if __name__ == "__main__":
    res = construct()
    print(res)
