import numpy as np
import time
from fractions import Fraction
from itertools import product

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
    U_frac = [[Fraction(int(round(x*2)))/2 for x in row] for row in U]
    V_frac = [[Fraction(int(round(x*2)))/2 for x in row] for row in V]
    W_frac = [[Fraction(int(round(x*2)))/2 for x in row] for row in W]
    
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

    # Schoolbook solution as fallback
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

    # Lower bound: max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    # But tensor rank can be higher. Let's try from 21 upward.
    # However, for 2x3x7, known results suggest rank might be around 30-40.
    # Let's try a range of ranks, starting from a reasonable lower bound.
    
    # Theoretical lower bound for matrix multiplication tensor M_{n,m,p} is at least max(n*m, m*p, p*n)
    # but also at least n+m+p-2 (Strassen's bound for square, generalized).
    # For 2x3x7: n+m+p-2 = 2+3+7-2 = 10. That's very low.
    # A better lower bound is the slice rank or something. But let's just search.
    
    # We'll try ranks from 25 to 45, with multiple random restarts per rank.
    # For each rank R, we'll try several random initializations of U and V,
    # then solve for W exactly.
    
    np.random.seed(12345)
    
    for R in range(25, 46):
        if time.time() - start_time > deadline:
            break
            
        # Number of restarts per rank: more for lower R (harder to find)
        n_restarts = 5 if R < 30 else 3
        
        for restart in range(n_restarts):
            if time.time() - start_time > deadline:
                break
                
            # Generate random U and V with half-integer entries
            # Use small integers for better chance of exact solutions
            # U: (n*m, R) = (6, R)
            # V: (m*p, R) = (21, R)
            
            # Strategy: Generate U and V with entries from {-1, 0, 1} or {-0.5, 0, 0.5, 1}
            # Then solve for W.
            
            U = np.zeros((n * m, R))
            V = np.zeros((m * p, R))
            
            # Fill U and V with random values from a small set
            # To ensure the system is likely solvable, use structured randomness
            for r in range(R):
                # For U: each column should have some non-zero entries
                # Randomly choose which entries are non-zero
                u_mask = np.random.rand(n * m) < 0.5
                if not np.any(u_mask):
                    u_mask[np.random.randint(n * m)] = True
                U[u_mask, r] = np.random.choice([-1, 1], size=np.sum(u_mask))
                
                v_mask = np.random.rand(m * p) < 0.5
                if not np.any(v_mask):
                    v_mask[np.random.randint(m * p)] = True
                V[v_mask, r] = np.random.choice([-1, 1], size=np.sum(v_mask))
            
            # Now solve for W
            W = np.zeros((p * n, R))
            valid = True
            
            for k in range(p):
                for i in range(n):
                    # Build the linear system for W[k*n+i, :]
                    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
                    # This is: A @ w = ones(m), where A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    
                    A = np.zeros((m, R))
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    
                    # Solve A @ w = ones(m)
                    # This is an underdetermined system (m < R typically)
                    # We want a solution w that is half-integer.
                    
                    # First, check if a solution exists (rank condition)
                    b = np.ones(m)
                    
                    # Use least squares to get a candidate
                    try:
                        w_candidate, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                    except:
                        valid = False
                        break
                    
                    # Check if the residual is small (solution exists)
                    if residuals is not None and residuals[0] > 1e-10:
                        valid = False
                        break
                    
                    # Now we need to find a half-integer solution.
                    # The general solution is w = w_particular + null_space @ z
                    # For simplicity, let's first check if w_candidate is close to half-integers.
                    w_rounded = np.round(w_candidate * 2) / 2
                    if np.allclose(A @ w_rounded, b, atol=1e-8):
                        W[k * n + i, :] = w_rounded
                    else:
                        # Try to find a half-integer solution in the null space
                        # This is hard. Let's try a different approach:
                        # Solve the system over rationals by using the null space.
                        
                        # Get the null space of A
                        # A is m x R. Null space is (R-m) dimensional.
                        # We can use SVD to get the null space.
                        
                        # Actually, let's just try a few random combinations
                        # of the null space to see if we can get a half-integer solution.
                        
                        # For now, mark as invalid and try next restart.
                        valid = False
                        break
                
                if not valid:
                    break
            
            if valid:
                # Verify the full solution
                if check_solution(U, V, W, n, m, p):
                    best_solution = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = R
                    return best_solution
    
    return best_solution

if __name__ == "__main__":
    res = construct()
    print(res)
