import time
import numpy as np
from itertools import product

def _verify_decomposition(n, m, p, U, V, W):
    """Verify that the tensor decomposition is exact."""
    rank = len(U[0])
    # Build the full multiplication tensor T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # T should equal the matrix multiplication tensor
    for i in range(n * m):
        for j in range(m * p):
            for k in range(p * n):
                val = sum(U[i][r] * V[j][r] * W[k][r] for r in range(rank))
                # The multiplication tensor: T[i,j,k] = 1 if i,j,k correspond to a valid product term
                # i = a*n + c (row a of A, col c of A)
                # j = b*m + d (row b of B, col d of B)  -- wait, B is m x p
                # Actually: A is n x m, B is m x p, C is n x p
                # U has n*m rows: index = a*m + c where a in [0,n), c in [0,m)
                # V has m*p rows: index = c*p + k where c in [0,m), k in [0,p)
                # W has p*n rows: index = k*n + a where k in [0,p), a in [0,n)
                # T[i,j,k] = 1 iff i = a*m+c, j = c'*p+k, k_idx = k'*n+a' and c=c' and a=a'
                # i = a*m + c
                # j = c'*p + k
                # k_idx = k'*n + a'
                # Match if c = c' and a = a'
                a, c = divmod(i, m)
                c2, k_col = divmod(j, p)
                k_row, a2 = divmod(k, n)
                expected = 1 if (c == c2 and a == a2 and k_col == k_row) else 0
                if val != expected:
                    return False
    return True

def _find_2x2x3_rank10():
    """Search for a rank-10 decomposition of M_{2,2,3}.
    
    A is 2x2 (4 entries), B is 2x3 (6 entries), C is 2x3 (6 entries).
    U: 4 x 10, V: 6 x 10, W: 6 x 10
    """
    # The multiplication tensor T[a,c,b,d,e] where:
    # a in {0,1} (row of A), c in {0,1} (col of A = row of B)
    # b in {0,1} (row of B), d in {0,2} (col of B)
    # e in {0,2} (col of C), f in {0,1} (row of C)
    # T = 1 iff c=b and a=f and d=e
    
    # U index: a*2 + c (4 entries)
    # V index: b*3 + d (6 entries)  
    # W index: e*2 + f (6 entries)
    
    # We need: sum_r U[a*2+c, r] * V[b*3+d, r] * W[e*2+f, r] = 1 iff c=b, a=f, d=e
    
    # Try to find integer/half-integer solutions
    # This is a hard combinatorial problem. Let's try a structured approach.
    
    # One idea: use Strassen-like combinations
    # For each pair (d, e) where d=e, we have a 2x2x1 problem in (a,c) and (b,f)
    # But we need to share across d values.
    
    # Let's try: for each of the 3 columns d=0,1,2, we have a 2x2x1 multiplication
    # C[:,d] = A * B[:,d]
    # Each 2x2x1 has rank 4. Total naive: 12.
    # Can we do better by sharing?
    
    # Actually, let me think about this differently.
    # The tensor is: T[a,c,b,d,e,f] = delta(c,b) * delta(a,f) * delta(d,e)
    # This is a "slice" structure.
    
    # Let me try a direct search with small integer values.
    # Given the complexity, let's try known constructions.
    
    # Known result: rank(M_{2,2,3}) = 10 is conjectured/proven.
    # Let me try to construct it.
    
    # Approach: Use the fact that we can write:
    # For each column pair, use Strassen-like tricks.
    
    # Actually, let me just do a brute-force search with limited values.
    # This may be too slow, so let's try a smarter approach.
    
    # Let's try to verify if rank 10 is achievable by attempting to solve
    # the system of equations.
    
    # The tensor has 4*6*6 = 144 entries, but only 2*3*2 = 12 are non-zero (value 1).
    # We need 4*10 + 6*10 + 6*10 = 160 variables.
    # 132 equations (the zero entries) + 12 equations (the one entries) = 144 equations.
    # 160 variables, 144 equations -> should be solvable.
    
    # Let's set up the linear system for the zero entries and check consistency.
    
    start_time = time.time()
    deadline = start_time + 30  # 30 second budget for this search
    
    # Try to find a solution with entries in {-1, 0, 1, 0.5, -0.5}
    # This is still a huge search space, so let's use a structured approach.
    
    # Let me try a specific construction:
    # Split the 3 columns into one pair and one single.
    # Pair (cols 0,1): use Strassen (rank 7)
    # Single (col 2): use schoolbook (rank 4)
    # Total: 11. Not 10.
    
    # Alternative: Can we share some terms between the pair and the single?
    
    # Let me try a different approach: assume the solution has a specific structure.
    # Let U, V, W have entries from {-1, 0, 1}.
    
    # Actually, let me just try to verify that rank 10 works by constructing
    # a specific example from literature.
    
    # From literature, one rank-10 decomposition of M_{2,2,3}:
    # This is non-trivial. Let me try a computational approach.
    
    # Set up as a system: for each (i,j,k) with T[i,j,k]=0, sum_r U[i,r]V[j,r]W[k,r] = 0
    # For each (i,j,k) with T[i,j,k]=1, sum_r U[i,r]V[j,r]W[k,r] = 1
    
    # This is a system of polynomial equations (cubic in the variables).
    # Let's try to find a solution with small integers.
    
    # Given time constraints, let me try a heuristic: fix U and V, solve for W.
    # Or use a genetic algorithm / random restart approach.
    
    # For now, let's fall back to rank 52 (6 Strassen + 1 rank-10 2x2x3) 
    # if we can't find rank-10 quickly, or rank 53.
    
    # Let me try a simpler approach: verify that a known rank-10 construction works.
    
    # Actually, I'll implement a randomized search for the 2x2x3 rank-10 case.
    
    best_solution = None
    
    # Random restart search
    np.random.seed(42)
    max_iter = 5000
    val_options = [-1, 0, 1]
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Random U, V with entries from val_options
        U_try = np.random.choice(val_options, size=(4, 10))
        V_try = np.random.choice(val_options, size=(6, 10))
        
        # Check if we can solve for W
        # For each (i,j,k) with T[i,j,k] = 1: sum_r U[i,r]*V[j,r]*W[k,r] = 1
        # For each (i,j,k) with T[i,j,k] = 0: sum_r U[i,r]*V[j,r]*W[k,r] = 0
        
        # This is still nonlinear in W. Let's instead check all entries.
        
        # Actually, let's just check if random U,V,W work
        W_try = np.random.choice(val_options, size=(6, 10))
        
        # Verify
        ok = True
        for a in range(2):
            for c in range(2):
                for b in range(2):
                    for d in range(3):
                        for e in range(3):
                            for f in range(2):
                                i = a * 2 + c
                                j = b * 3 + d
                                k = e * 2 + f
                                val = sum(U_try[i, r] * V_try[j, r] * W_try[k, r] for r in range(10))
                                expected = 1 if (c == b and a == f and d == e) else 0
                                if val != expected:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            best_solution = (U_try.tolist(), V_try.tolist(), W_try.tolist())
            break
    
    if best_solution is not None:
        return best_solution
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = 2, 2, 15
    
    # First, try to find a rank-10 decomposition for 2x2x3
    # If found, use 5 blocks of 3 columns -> rank 50
    # If not, fall back to 6 Strassen + 1 2x2x3 (if rank-11 available) or rank 53
    
    # Try to find 2x2x3 rank-10
    sol_223 = _find_2x2x3_rank10()
    
    if sol_223 is not None and time.time() < deadline - 10:
        U_223, V_223, W_223 = sol_223
        rank = 50  # 5 blocks x 10
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col_offset = 0
        for block in range(5):
            k0 = block * 3
            k1 = k0 + 1
            k2 = k0 + 2
            for r in range(10):
                c = col_offset + r
                # U: 4 rows (a*2+c)
                for i in range(4):
                    U[i][c] = U_223[i][r]
                # V: 6 rows (b*3+d), d in {k0,k1,k2}
                for b in range(2):
                    V[b * p + k0][c] = V_223[b * 3 + 0][r]
                    V[b * p + k1][c] = V_223[b * 3 + 1][r]
                    V[b * p + k2][c] = V_223[b * 3 + 2][r]
                # W: 6 rows (e*2+f), e in {k0,k1,k2}
                for f in range(2):
                    W[k0 * n + f][c] = W_223[0 * 2 + f][r]
                    W[k1 * n + f][c] = W_223[1 * 2 + f][r]
                    W[k2 * n + f][c] = W_223[2 * 2 + f][r]
            col_offset += 10
        
        return {"U": U, "V": V, "W": W}
    
    # Fallback: original rank-53 construction
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for M_{2,2,2}
    A_s = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    B_s = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    C_s = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    A_b = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    B_b = [
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 1]
    ]
    C_b = [
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1]
    ]
    
    col_offset = 0
    
    for block in range(7):
        k0 = block * 2
        k1 = k0 + 1
        for r in range(7):
            c = col_offset + r
            for i in range(4):
                U[i][c] = A_s[r][i]
            for j in range(2):
                V[j*p + k0][c] = B_s[r][j*2 + 0]
                V[j*p + k1][c] = B_s[r][j*2 + 1]
            W[k0*n + 0][c] = C_s[r][0]
            W[k1*n + 0][c] = C_s[r][1]
            W[k0*n + 1][c] = C_s[r][2]
            W[k1*n + 1][c] = C_s[r][3]
        col_offset += 7
        
    k_last = 14
    for r in range(4):
        c = col_offset + r
        for i in range(4):
            U[i][c] = A_b[r][i]
        V[0*p + k_last][c] = B_b[r][0]
        V[1*p + k_last][c] = B_b[r][1]
        W[k_last*n + 0][c] = C_b[r][0]
        W[k_last*n + 1][c] = C_b[r][1]
        
    return {"U": U, "V": V, "W": W}
