import time
import numpy as np
from fractions import Fraction
import itertools
import random

DIMENSIONS = [3, 3, 7]

def _tensor_check(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n*m)]
    V = [[0] * rank for _ in range(m*p)]
    W = [[0] * rank for _ in range(p*n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return (U, V, W)

def _strassen_2x2_2x2_2x2():
    """Return the 7-term Strassen decomposition for 2x2 * 2x2 -> 2x2."""
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[3][3] = 1
    
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[2][5] = 1; W[3][5] = 1
    
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[1][6] = 1
    
    return U, V, W

def _embed_strassen_in_3x3x7(U_str, V_str, W_str, n, m, p):
    """
    Embed the 2x2x2x2 Strassen decomposition into the 3x3x3x7 problem.
    The Strassen part handles A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2].
    We need to add schoolbook terms for all other (i,j,k) triples.
    """
    rank = 7 + 63 - 8  # 7 Strassen terms + (63-8) schoolbook for remaining entries
    # Actually, let's be more careful. The Strassen handles 4*4=16 (i,j,k) combinations
    # where i,j in {0,1} and k in {0,1}. That's 2*2*2=8 entries.
    # Wait, no. For 2x2*2x2->2x2, the output is 2x2=4 entries, each computed from 2x2=4 inputs.
    # The tensor has 2*2*2=8 (i,j,k) indices. Strassen uses rank 7 for these 8.
    # So we save 1 term.
    
    # Total schoolbook: 3*3*7 = 63
    # Strassen replaces 8 schoolbook terms (for i,j in 0..1, k in 0..1) with 7 terms
    # So total rank = 63 - 8 + 7 = 62
    
    rank = 62
    U = [[0]*rank for _ in range(n*m)]  # 9 rows
    V = [[0]*rank for _ in range(m*p)]  # 21 rows
    W = [[0]*rank for _ in range(p*n)]  # 21 rows
    
    # Place Strassen terms in columns 0..6
    # Strassen U: 4x7, maps (i,j) for i,j in 0..1 to U[i*2+j]
    # In our 3x3, (i,j) with i,j in 0..1 maps to U[i*3+j]
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[i*m + j][c] = U_str[i*2 + j][c]
        for j in range(2):
            for k in range(2):
                V[j*p + k][c] = V_str[j*2 + k][c]
        for k in range(2):
            for i in range(2):
                W[k*n + i][c] = W_str[k*2 + i][c]
    
    # Place schoolbook terms for remaining (i,j,k) in columns 7..61
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if this is handled by Strassen
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    
    return U, V, W

def _try_random_search(n, m, p, target_rank, num_trials, deadline):
    """
    Try to find a rank-target_rank decomposition using random support patterns.
    For each trial, randomly choose which (row,col) positions are nonzero in U, V, W,
    then solve the linear system.
    """
    start = time.time()
    
    # The tensor has n*m*p = 63 entries
    # Each rank-1 term contributes to all 63 entries
    # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all (i,j,k)
    
    # This is a system of 63 equations in 9*r + 21*r + 21*r = 51*r unknowns
    # For r=56: 51*56 = 2856 unknowns, 63 equations - very underdetermined
    
    # Strategy: fix a sparse support pattern, then solve
    for trial in range(num_trials):
        if time.time() > deadline - 5:
            break
        
        # Random support: for each column c, randomly pick which entries are nonzero
        # Use a heuristic: each column should have ~3-5 nonzeros in each of U, V, W
        
        U = [[0]*target_rank for _ in range(n*m)]
        V = [[0]*target_rank for _ in range(m*p)]
        W = [[0]*target_rank for _ in range(p*n)]
        
        # Assign random supports
        for c in range(target_rank):
            # Pick 2-3 random positions in U
            u_positions = random.sample(range(n*m), random.randint(2,3))
            for pos in u_positions:
                U[pos][c] = random.choice([-1, 1])
            
            v_positions = random.sample(range(m*p), random.randint(2,3))
            for pos in v_positions:
                V[pos][c] = random.choice([-1, 1])
            
            w_positions = random.sample(range(p*n), random.randint(2,3))
            for pos in w_positions:
                W[pos][c] = random.choice([-1, 1])
        
        # Check if this support pattern can satisfy the tensor equations
        # Build the linear system: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # This is nonlinear in the variables, so we can't directly solve it.
        # Instead, we fix U and V, then solve for W linearly.
        # For fixed U and V, the equation becomes:
        # sum_c W[k*n+i,c] * (U[i*m+j,c] * V[j*p+k,c]) = 1
        # This is linear in W!
        
        # Build system: 63 equations, 21*target_rank unknowns for W
        # A_w[eq, unknown] * W_flat[unknown] = b[eq]
        num_eqs = n * m * p  # 63
        num_w_unknowns = p * n * target_rank  # 21 * target_rank
        
        A_w = [[0]*num_w_unknowns for _ in range(num_eqs)]
        b = [0]*num_eqs
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(target_rank):
                        coeff = U[i*m + j][c] * V[j*p + k][c]
                        if coeff != 0:
                            # W unknown index: (k*n+i) * target_rank + c
                            w_idx = (k*n + i) * target_rank + c
                            A_w[eq_idx][w_idx] = coeff
                    b[eq_idx] = 1
                    eq_idx += 1
        
        # Solve the linear system using numpy
        try:
            A_np = np.array(A_w, dtype=float)
            b_np = np.array(b, dtype=float)
            
            # Check if system is consistent
            # Use least squares as a first pass
            W_flat, residuals, rank_w, sv = np.linalg.lstsq(A_np, b_np, rcond=None)
            
            # Check residual
            if len(residuals) > 0 and residuals[0] > 1e-6:
                continue
            
            # Verify with exact arithmetic
            W_sol = [[0]*target_rank for _ in range(p*n)]
            for row in range(p*n):
                for c in range(target_rank):
                    val = W_flat[row * target_rank + c]
                    # Round to nearest half-integer or integer
                    val_rounded = round(val * 2) / 2
                    W_sol[row][c] = Fraction(int(val_rounded * 2), 2)
            
            # Verify the full decomposition
            if _tensor_check(U, V, W_sol, n, m, p):
                return U, V, W_sol
        except:
            continue
    
    return None

def _try_structured_search(n, m, p, target_rank, deadline):
    """
    Try structured support patterns based on block decomposition.
    """
    start = time.time()
    
    # Try embedding Strassen for the 2x2x2x2 core
    U_str, V_str, W_str = _strassen_2x2_2x2_2x2()
    U, V, W = _embed_strassen_in_3x3x7(U_str, V_str, W_str, n, m, p)
    rank = len(U[0])
    
    if rank < target_rank:
        if _tensor_check(U, V, W, n, m, p):
            return U, V, W
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 120

    n, m, p = DIMENSIONS  # 3, 3, 7

    # Fallback: Schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_result = (U_sb, V_sb, W_sb)
    best_rank = n * m * p

    # Try structured search first (fast)
    if time.time() < deadline - 10:
        result = _try_structured_search(n, m, p, best_rank, deadline)
        if result is not None:
            U_r, V_r, W_r = result
            rank_r = len(U_r[0])
            if rank_r < best_rank:
                best_result = (U_r, V_r, W_r)
                best_rank = rank_r

    # Try random search with decreasing target rank
    if time.time() < deadline - 10:
        for target_rank in range(best_rank - 1, 40, -1):
            if time.time() > deadline - 5:
                break
            
            # Limit trials based on remaining time
            remaining = deadline - 5 - time.time()
            if remaining < 2:
                break
            
            # Estimate trials we can afford (each trial takes ~0.1-0.5s)
            num_trials = max(1, int(remaining / 0.3))
            
            result = _try_random_search(n, m, p, target_rank, num_trials, deadline)
            if result is not None:
                U_r, V_r, W_r = result
                rank_r = len(U_r[0])
                if rank_r < best_rank:
                    best_result = (U_r, V_r, W_r)
                    best_rank = rank_r
                    break  # Found a valid decomposition, stop
    
    U, V, W = best_result
    return {"U": U, "V": V, "W": W}
