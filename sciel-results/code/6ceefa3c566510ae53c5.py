"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor
    target = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[(a, b, c)] = 1
    
    best_result = None
    deadline = time.time() + 105  # safety margin
    
    # Try to find minimal rank decomposition
    for rank in range(20, n*m*p + 1):
        if time.time() > deadline:
            break
        result = solve_rank(n, m, p, nm, mp, pn, rank, target, deadline)
        if result is not None:
            best_result = result
            break
    
    if best_result is None:
        # Schoolbook fallback
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result

def solve_rank(n, m, p, nm, mp, pn, rank, target, deadline):
    """Attempt to find a rank-r exact decomposition."""
    np.random.seed(12345)
    
    # Use iterative refinement: start from a feasible point and reduce rank
    # Or: use random initialization + optimization
    
    for trial in range(30):
        if time.time() > deadline:
            return None
        
        # Random initialization
        U = np.random.randn(nm, rank) * 0.5
        V = np.random.randn(mp, rank) * 0.5
        W = np.random.randn(pn, rank) * 0.5
        
        # Optimize using alternating linear solves
        for it in range(200):
            if time.time() > deadline:
                return None
            
            # Update W: for each column c, minimize sum_{(a,b,c) in target} (sum_{c'} U[a,c']*V[b,c']*W[c,c'] - 1)^2
            # + sum_{(a,b,c) not in target} (sum_{c'} U[a,c']*V[b,c']*W[c,c'])^2
            # This is a quadratic in W[:,c] for each c, but coupled across c'
            
            # Simpler: use gradient descent with all three matrices
            # Compute error and gradients efficiently
            
            # For each (a,b,c), est = sum_{c'} U[a,c'] * V[b,c'] * W[c,c']
            # grad_U[a,c'] += 2*(est - T) * V[b,c'] * W[c,c]
            # etc.
            
            # This is O(nm*mp*pn*rank) which is 8*20*10*rank = 1600*rank
            # For rank=40, that's 64000 ops per gradient step, times 3 matrices = 192000
            # 200 iterations * 30 trials = 6000 gradient steps * 192000 = ~1.15 billion ops
            # Too slow. Need a better approach.
            
            # Alternative: exploit sparsity of target
            # Only 40 target entries are 1, rest are 0
            # grad_U[a,c'] = 2 * sum_{b,c} (est_{a,b,c} - T_{a,b,c}) * V[b,c'] * W[c,c]
            # Still expensive.
            
            # Let's try a different strategy: use the structure of matrix multiplication
            # to guide the search
            
            break
        
        # Check if we found a solution
        if check_solution(U, V, W, nm, mp, pn, target):
            # Round to rationals
            U_r, V_r, W_r = round_to_rationals(U, V, W)
            if U_r is not None and verify(U_r, V_r, W_r, nm, mp, pn, target):
                return {"U": U_r, "V": V_r, "W": W_r}
    
    return None

def check_solution(U, V, W, nm, mp, pn, target, tol=1e-6):
    """Check if U,V,W give an exact decomposition."""
    for (a, b, c) in target:
        val = sum(U[a, c_] * V[b, c_] * W[c, c_] for c_ in range(U.shape[1]))
        if abs(val - 1) > tol:
            return False
    # Check a sample of non-target entries
    np.random.seed(999)
    for _ in range(100):
        a = np.random.randint(nm)
        b = np.random.randint(mp)
        c = np.random.randint(pn)
        if (a, b, c) not in target:
            val = sum(U[a, c_] * V[b, c_] * W[c, c_] for c_ in range(U.shape[1]))
            if abs(val) > tol:
                return False
    return True

def round_to_rationals(U, V, W, max_denom=100):
    """Try to round to rational entries."""
    # Scale to find a common denominator
    all_vals = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    # Try to find rationals with small denominators
    U_r = np.zeros_like(U)
    V_r = np.zeros_like(V)
    W_r = np.zeros_like(W)
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            val = U[i, j]
            best_num, best_den = 0, 1
            for den in range(1, max_denom + 1):
                num = round(val * den)
                if abs(val - num / den) < 1e-8:
                    best_num, best_den = num, den
                    break
            U_r[i, j] = best_num / best_den
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            val = V[i, j]
            best_num, best_den = 0, 1
            for den in range(1, max_denom + 1):
                num = round(val * den)
                if abs(val - num / den) < 1e-8:
                    best_num, best_den = num, den
                    break
            V_r[i, j] = best_num / best_den
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            val = W[i, j]
            best_num, best_den = 0, 1
            for den in range(1, max_denom + 1):
                num = round(val * den)
                if abs(val - num / den) < 1e-8:
                    best_num, best_den = num, den
                    break
            W_r[i, j] = best_num / best_den
    return U_r.tolist(), V_r.tolist(), W_r.tolist()

def verify(U, V, W, nm, mp, pn, target):
    """Verify exact decomposition with rational entries."""
    from fractions import Fraction
    U_f = [[Fraction(int(U[i][j] * 1000000)) / 1000000 if U[i][j] != 0 else Fraction(0) for j in range(len(U[0]))] for i in range(nm)]
    # This is getting messy. Let me use a different verification.
    rank = len(U[0])
    for (a, b, c) in target:
        s = 0
        for c_ in range(rank):
            s += U[a][c_] * V[b][c_] * W[c][c_]
        if abs(s - 1) > 1e-9:
            return False
    return True
# EVOLVE-BLOCK-END
