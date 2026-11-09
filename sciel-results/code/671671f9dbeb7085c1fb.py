import time
import numpy as np
from fractions import Fraction

DEADLINE = 135.0  # safety margin

def _find_2x2x2_rank7():
    """Find a rank-7 decomposition of the 2x2x2 matmul tensor."""
    # We need 7 terms where for all (i,j,k) in {0,1}^3:
    # sum_c U_c[i,j] * V_c[j,k] * W_c[k,i] = 1
    # 
    # Strategy: choose U_c and V_c sparsely, then solve for W_c.
    # 
    # Let's use the following structure (inspired by Strassen):
    # Each term c has U_c with 2 nonzeros, V_c with 1-2 nonzeros.
    
    # I'll use a specific construction. Let me define the terms as:
    # (U_nonzeros, V_nonzeros) where nonzeros are (index, value)
    # U indices: (i,j) -> 2i+j
    # V indices: (j,k) -> 2j+k
    # W indices: (k,i) -> 2k+i
    
    # Term structure (U pattern, V pattern):
    patterns = [
        # (U: {(i,j):val}, V: {(j,k):val})
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),  # M1-like
        ({(1,0):1, (1,1):1}, {(0,0):1}),            # M2-like
        ({(0,0):1}, {(1,0):1, (1,1):-1}),           # M3-like
        ({(1,1):1}, {(0,1):1, (0,0):-1}),           # M4-like
        ({(0,0):1, (0,1):1}, {(1,1):1}),            # M5-like
        ({(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),  # M6-like
        ({(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),  # M7-like
    ]
    
    # For each term, compute which (i,k) pairs get contributions:
    # contrib_c[i,k] = sum_j U_c[i,j] * V_c[j,k]
    # Then we need: sum_c contrib_c[i,k] * W_c[k,i] = 1 for all (i,k)
    # Wait no, the constraint is for all (i,j,k), not (i,k).
    # 
    # The constraint is: for all (i,j,k): sum_c U_c[i,j]*V_c[j,k]*W_c[k,i] = 1
    # 
    # For fixed U_c and V_c, this is linear in W_c.
    # Let's set up the system.
    
    # Unknowns: W_c[k,i] for c=0..6, k=0,1, i=0,1 -> 7*4 = 28 unknowns
    # Equations: 8 (one for each (i,j,k))
    
    # Let me use numpy to solve.
    A = np.zeros((8, 28))
    b = np.ones(8)
    
    for eq, (i, j, k) in enumerate([(i,j,k) for i in range(2) for j in range(2) for k in range(2)]):
        for c in range(7):
            Uval = patterns[c][0].get((i,j), 0)
            Vval = patterns[c][1].get((j,k), 0)
            if Uval != 0 and Vval != 0:
                # W_c[k,i] is at index c*4 + k*2 + i
                A[eq, c*4 + k*2 + i] = Uval * Vval
    
    # Solve the underdetermined system (8 equations, 28 unknowns)
    # Use least squares (will find a solution if one exists)
    x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    
    # Check if the solution is exact
    if np.allclose(A @ x, b, atol=1e-10):
        # Extract W values
        W = []
        for c in range(7):
            Wc = {}
            for k in range(2):
                for i in range(2):
                    val = x[c*4 + k*2 + i]
                    if abs(val) > 1e-10:
                        Wc[(k,i)] = round(val, 10)
            W.append(Wc)
        
        # Verify
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    total = 0
                    for c in range(7):
                        Uval = patterns[c][0].get((i,j), 0)
                        Vval = patterns[c][1].get((j,k), 0)
                        Wval = W[c].get((k,i), 0)
                        total += Uval * Vval * Wval
                    if abs(total - 1) > 1e-8:
                        return None
        
        return patterns, W
    else:
        return None

def _build_2x5x5():
    """Build a rank-46 decomposition for 2x5x5 using 2x2x2 Strassen blocks."""
    result = _find_2x2x2_rank7()
    if result is None:
        return None
    
    patterns, W_2x2 = result
    
    n, m, p = 2, 5, 5
    # Partition: j = {0,1},{2,3},{4}; k = {0,1},{2,3},{4}
    # Blocks: (2,2,2) for j in {0,1}x{k:0,1}, {0,1}x{2,3}, {2,3}x{0,1}, {2,3}x{2,3}
    #          (2,1,2) for j in {4}x{k:0,1}, {4}x{2,3}
    #          (2,2,1) for j in {0,1}x{k:4}, {2,3}x{k:4}
    #          (2,1,1) for j in {4}x{k:4}
    
    # Schoolbook for non-2x2x2 blocks:
    # (2,1,2): 2*1*2 = 4 terms each, 2 blocks = 8
    # (2,2,1): 2*2*1 = 4 terms each, 2 blocks = 8
    # (2,1,1): 2*1*1 = 2 terms
    # Total non-Strassen: 8 + 8 + 2 = 18
    # Strassen blocks: 4 * 7 = 28
    # Total: 46
    
    U = {}  # (i,j) -> list of (col, val)
    V = {}  # (j,k) -> list of (col, val)
    W = {}  # (k,i) -> list of (col, val)
    
    col = 0
    
    # Strassen blocks
    for j0 in [0, 2]:
        for k0 in [0, 2]:
            for c in range(7):
                Upat, Vpat = patterns[c]
                Wpat = W_2x2[c]
                for (di, dj), uval in Upat.items():
                    i, j = di, j0 + dj
                    U.setdefault((i,j), []).append((col, uval))
                for (dj, dk), vval in Vpat.items():
                    j, k = j0 + dj, k0 + dk
                    V.setdefault((j,k), []).append((col, vval))
                for (dk, di), wval in Wpat.items():
                    k, i = k0 + dk, di
                    W.setdefault((k,i), []).append((col, wval))
                col += 1
    
    # Schoolbook for remaining blocks
    # j=4, k in {0,1,2,3}
    for i in range(2):
        for k in range(4):
            U.setdefault((i,4), []).append((col, 1))
            V.setdefault((4,k), []).append((col, 1))
            W.setdefault((k,i), []).append((col, 1))
            col += 1
    
    # j in {0,1,2,3}, k=4
    for i in range(2):
        for j in range(4):
            U.setdefault((i,j), []).append((col, 1))
            V.setdefault((j,4), []).append((col, 1))
            W.setdefault((4,i), []).append((col, 1))
            col += 1
    
    # j=4, k=4
    for i in range(2):
        U.setdefault((i,4), []).append((col, 1))
        V.setdefault((4,4), []).append((col, 1))
        W.setdefault((4,i), []).append((col, 1))
        col += 1
    
    rank = col
    # Convert to matrix form
    U_mat = [[0]*rank for _ in range(n*m)]
    V_mat = [[0]*rank for _ in range(m*p)]
    W_mat = [[0]*rank for _ in range(p*n)]
    
    for (i,j), entries in U.items():
        for c, val in entries:
            U_mat[i*m+j][c] = val
    for (j,k), entries in V.items():
        for c, val in entries:
            V_mat[j*p+k][c] = val
    for (k,i), entries in W.items():
        for c, val in entries:
            W_mat[k*n+i][c] = val
    
    return U_mat, V_mat, W_mat

def _verify_numpy(U, V, W, n, m, p, tol=1e-8):
    """Fast verification using numpy."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(total - 1) > tol:
                    return False
    return True

def _verify_fraction(U, V, W, n, m, p):
    """Exact verification using Fraction."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if total != 1:
                    return False
    return True

def _try_merge(U, V, W, n, m, p, c1, c2, factor):
    """Try to merge columns c1 and c2."""
    rank = len(U[0])
    new_rank = rank - 1
    new_U = [[0]*new_rank for _ in range(n*m)]
    new_V = [[0]*new_rank for _ in range(m*p)]
    new_W = [[0]*new_rank for _ in range(p*n)]
    
    new_col = 0
    for c in range(rank):
        if c == c2:
            continue
        if c == c1:
            for i in range(n*m):
                new_U[i][new_col] = U[i][c1] + factor * U[i][c2]
            for j in range(m*p):
                new_V[j][new_col] = V[j][c1] + factor * V[j][c2]
            for k in range(p*n):
                new_W[k][new_col] = W[k][c1] + factor * W[k][c2]
        else:
            for i in range(n*m):
                new_U[i][new_col] = U[i][c]
            for j in range(m*p):
                new_V[j][new_col] = V[j][c]
            for k in range(p*n):
                new_W[k][new_col] = W[k][c]
        new_col += 1
    
    if _verify_numpy(new_U, new_V, new_W, n, m, p, tol=1e-6):
        return new_U, new_V, new_W
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try to build rank-46 decomposition
    result = _build_2x5x5()
    if result is not None:
        U, V, W = result
        if _verify_numpy(U, V, W, n, m, p, tol=1e-6):
            best_U, best_V, best_W = U, V, W
            best_rank = len(U[0])
        else:
            U, V, W = _schoolbook(n, m, p)
            best_U, best_V, best_W = U, V, W
            best_rank = len(U[0])
    else:
        U, V, W = _schoolbook(n, m, p)
        best_U, best_V, best_W = U, V, W
        best_rank = len(U[0])
    
    # Try merge search
    factors = [1, -1, 0.5, -0.5, 2, -2, 3, -3]
    current_U, current_V, current_W = best_U, best_V, best_W
    current_rank = best_rank
    
    while current_rank > 20 and time.time() < DEADLINE:
        merged = False
        for c1 in range(current_rank):
            if time.time() > DEADLINE:
                break
            for c2 in range(c1+1, current_rank):
                if time.time() > DEADLINE:
                    break
                for factor in factors:
                    result = _try_merge(current_U, current_V, current_W, n, m, p, c1, c2, factor)
                    if result is not None:
                        current_U, current_V, current_W = result
                        current_rank -= 1
                        best_U, best_V, best_W = current_U, current_V, current_W
                        best_rank = current_rank
                        merged = True
                        break
                if merged:
                    break
            if merged:
                break
        if not merged:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W
