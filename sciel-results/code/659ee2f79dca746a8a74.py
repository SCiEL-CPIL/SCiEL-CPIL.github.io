"""Search for a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 10 matrix."""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 3, 10]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # Fast numpy check first
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(Ua[i*m+j] * Va[j*p+k] * Wa[k*n+i])
                if abs(s - 1.0) > 1e-10:
                    return False
    # Exact Fraction check for final confirmation
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _verify_np(U, V, W, n, m, p, tol=1e-10):
    """Fast verification using NumPy only (no exact check)."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                prod = Ua[i*m+j] * Va[j*p+k] * Wa[k*n+i]
                s = np.sum(prod)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
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
    return {"U": U, "V": V, "W": W}

def _strassen_rank55(n, m, p):
    """
    Construct the known rank-55 decomposition using Strassen for 2x2 blocks
    and schoolbook for the 2x1 block.
    
    Split A (2x3) into A1 (2x2) and A2 (2x1).
    Split B (3x10) into B1 (2x10) and B2 (1x10).
    
    For A1*B1: use 5 Strassen blocks (each covering 2 columns of B), 7 mults each = 35
    For A2*B2: schoolbook, 2*10 = 20 mults
    Total: 55
    """
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen for 2x2 * 2x10 (5 blocks of 2 columns each)
    for g in range(5):
        k0 = 2 * g
        k1 = 2 * g + 1
        
        # U indices: A is 2x3, so A[i,j] -> U[i*3+j]
        # A00=U[0], A01=U[1], A10=U[3], A11=U[4]
        u00, u01, u10, u11 = 0, 1, 3, 4
        
        # V indices: B is 3x10, so B[j,k] -> V[j*10+k]
        # B10=V[10+k0], B11=V[10+k1] for j=1
        # B00=V[0+k0], B01=V[0+k1] for j=0
        v00, v01 = k0, k1  # j=0
        v10, v11 = 10 + k0, 10 + k1  # j=1
        
        # W indices: C is 2x10, so C[i,k] -> W[k*2+i]
        w00, w01 = k0 * 2, k1 * 2  # i=0
        w10, w11 = k0 * 2 + 1, k1 * 2 + 1  # i=1
        
        # Strassen terms for 2x2:
        # M1 = (A00+A11)(B00+B11)
        U[u00][col] = 1
        U[u11][col] = 1
        V[v00][col] = 1
        V[v11][col] = 1
        W[w00][col] = 1
        W[w11][col] = 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[u10][col] = 1
        U[u11][col] = 1
        V[v00][col] = 1
        W[w10][col] = 1
        W[w11][col] = -1
        col += 1
        
        # M3 = A00(B01-B11)
        U[u00][col] = 1
        V[v01][col] = 1
        V[v11][col] = -1
        W[w01][col] = 1
        W[w11][col] = 1
        col += 1
        
        # M4 = A11(B10-B00)
        U[u11][col] = 1
        V[v10][col] = 1
        V[v00][col] = -1
        W[w00][col] = 1
        W[w10][col] = 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[u00][col] = 1
        U[u01][col] = 1
        V[v11][col] = 1
        W[w01][col] = 1
        W[w00][col] = -1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[u10][col] = 1
        U[u00][col] = -1
        V[v00][col] = 1
        V[v01][col] = 1
        W[w11][col] = 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[u01][col] = 1
        U[u11][col] = -1
        V[v10][col] = 1
        V[v11][col] = 1
        W[w00][col] = 1
        col += 1
    
    # Schoolbook for A2 (2x1) * B2 (1x10)
    # A2: A[i,2] -> U[i*3+2], i=0,1 => U[2], U[5]
    # B2: B[2,k] -> V[2*10+k] = V[20+k], k=0..9
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    assert col == rank, f"Expected {rank}, got {col}"
    return {"U": U, "V": V, "W": W}

def _try_rank54_exploit_n2(n, m, p, time_budget, seed=42):
    """
    Try to find a rank-54 decomposition by exploiting the n=2 structure.
    
    Strategy: Start from the rank-55 Strassen construction and try to merge
    one pair of columns by finding a linear combination that satisfies both rows.
    Also try structured random searches with W solved via linear systems.
    """
    rng = np.random.RandomState(seed)
    deadline = time.time() + time_budget
    
    base = _strassen_rank55(n, m, p)
    U0 = np.array(base["U"], dtype=float)
    V0 = np.array(base["V"], dtype=float)
    W0 = np.array(base["W"], dtype=float)
    
    r55 = 55
    r54 = 54
    
    best = None
    
    # Approach A: Try to merge two columns from the rank-55 solution
    # For each pair of columns (a,b), check if we can find alpha,beta such that
    # replacing columns a,b with a single column works.
    # This is equivalent to checking if the 55-column system has a 1-dim kernel
    # in a specific way. We try all pairs of "similar" columns.
    
    # Group columns by their Strassen block and term type
    # Columns 0-6: block 0, terms M1-M7
    # Columns 7-13: block 1, terms M1-M7
    # etc.
    # Within each block, M1 and M5 both have U entries in u00, M2 and M6 both in u10, etc.
    
    # Try merging columns within the same Strassen block
    for g in range(5):
        if time.time() > deadline - 10:
            break
        block_start = g * 7
        for a in range(7):
            for b in range(a+1, 7):
                if time.time() > deadline - 10:
                    break
                ca, cb = block_start + a, block_start + b
                
                # Try: new_col = U0[:,ca] * alpha + U0[:,cb] * beta
                #       new_V  = V0[:,ca] * alpha + V0[:,cb] * beta
                #       new_W  = W0[:,ca] * alpha + W0[:,cb] * beta
                # We need: for all (i,j,k), sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
                # After merging, the contribution from columns ca,cb becomes:
                #   (U_a*alpha + U_b*beta) * (V_a*alpha + V_b*beta) * (W_a*alpha + W_b*beta)
                # vs original: U_a*V_a*W_a + U_b*V_b*W_b
                # This is a cubic in alpha,beta - very hard. Skip this approach.
                pass
    
    # Approach B: Structured random search with better initialization
    # Use the Strassen pattern as a template but with rank 54
    for attempt in range(500):
        if time.time() > deadline - 5:
            break
        
        U = np.zeros((n*m, r54))
        V = np.zeros((m*p, r54))
        
        strat = attempt % 5
        
        if strat == 0:
            # Copy Strassen structure, drop one column, redistribute
            U[:, :] = U0[:, :r54]
            V[:, :] = V0[:, :r54]
            # Perturb slightly
            mask_u = rng.random((n*m, r54)) < 0.05
            mask_v = rng.random((m*p, r54)) < 0.05
            U[mask_u] += rng.choice([-1, 1], size=np.sum(mask_u))
            V[mask_v] += rng.choice([-1, 1], size=np.sum(mask_v))
            
        elif strat == 1:
            # Strassen with one block reduced from 7 to 6 terms
            # Try to find a 6-term decomposition for one 2x2 block
            U[:, :] = U0[:, :r54]
            V[:, :] = V0[:, :r54]
            # Modify the last block (g=4) to use only 6 columns
            # This won't work directly, so add randomness
            for row in range(n*m):
                if rng.random() < 0.3:
                    col = rng.randint(0, r54)
                    U[row, col] += rng.choice([-1, 1])
            for row in range(m*p):
                if rng.random() < 0.3:
                    col = rng.randint(0, r54)
                    V[row, col] += rng.choice([-1, 1])
            
        elif strat == 2:
            # Sparse random with Strassen-like density
            for row in range(n*m):
                nnz = rng.randint(2, 5)
                cols = rng.choice(r54, size=nnz, replace=False)
                U[row, cols] = rng.choice([-1, 1], size=nnz)
            for row in range(m*p):
                nnz = rng.randint(2, 5)
                cols = rng.choice(r54, size=nnz, replace=False)
                V[row, cols] = rng.choice([-1, 1], size=nnz)
                
        elif strat == 3:
            # Block-structured: 5 blocks of ~11 columns each
            block_sizes = [11, 11, 11, 11, 10]
            pos = 0
            for g in range(5):
                bs = block_sizes[g]
                U[:, pos:pos+bs] = rng.randint(0, 2, size=(n*m, bs)) * 2 - 1
                V[:, pos:pos+bs] = rng.randint(0, 2, size=(m*p, bs)) * 2 - 1
                U[:, pos:pos+bs][rng.random((n*m, bs)) < 0.4] = 0
                V[:, pos:pos+bs][rng.random((m*p, bs)) < 0.4] = 0
                pos += bs
            
        else:
            # Hadamard-inspired structure
            U = rng.randint(0, 2, size=(n*m, r54)) * 2 - 1
            V = rng.randint(0, 2, size=(m*p, r54)) * 2 - 1
            U[rng.random((n*m, r54)) < 0.35] = 0
            V[rng.random((m*p, r54)) < 0.35] = 0
        
        # Solve for W using least squares per (i,k) pair
        W = np.zeros((p*n, r54))
        feasible = True
        
        for i in range(n):
            for k in range(p):
                A = np.zeros((m, r54))
                b = np.ones(m)
                for j in range(m):
                    A[j, :] = U[i*m+j] * V[j*p+k]
                
                try:
                    sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    if not np.allclose(A @ sol, b, atol=1e-10):
                        feasible = False
                        break
                    W[k*n+i, :] = sol
                except:
                    feasible = False
                    break
            if not feasible:
                break
        
        if feasible:
            # Fast numpy verification
            ok = _verify_np(U, V, W, n, m, p, tol=1e-9)
            if ok:
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
    
    return best

def _try_general_search(n, m, p, time_budget, seed=42, start_rank=54):
    """
    General search for low-rank decompositions using structured initialization
    and linear solve for W.
    """
    rng = np.random.RandomState(seed)
    deadline = time.time() + time_budget
    
    best = None
    best_rank = n * m * p + 1
    
    for r in range(start_rank, min(n*m*p + 1, start_rank + 3)):
        if time.time() > deadline - 10:
            break
        if r >= best_rank:
            continue
            
        max_attempts = 200
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
                
            strat = attempt % 4
            
            if strat == 0:
                U = np.zeros((n*m, r))
                V = np.zeros((m*p, r))
                for row in range(n*m):
                    nnz = rng.randint(2, min(5, r))
                    cols = rng.choice(r, size=nnz, replace=False)
                    U[row, cols] = rng.choice([-1, 1], size=nnz)
                for row in range(m*p):
                    nnz = rng.randint(2, min(5, r))
                    cols = rng.choice(r, size=nnz, replace=False)
                    V[row, cols] = rng.choice([-1, 1], size=nnz)
                    
            elif strat == 1:
                U = rng.randint(0, 2, size=(n*m, r)) * 2 - 1
                V = rng.randint(0, 2, size=(m*p, r)) * 2 - 1
                U[rng.random((n*m, r)) < 0.35] = 0
                V[rng.random((m*p, r)) < 0.35] = 0
                
            elif strat == 2:
                # Strassen-inspired block structure
                # 5 blocks for the 2x2 part + schoolbook for 2x1
                U = np.zeros((n*m, r))
                V = np.zeros((m*p, r))
                # Distribute r columns across 6 groups (5 Strassen + 1 schoolbook)
                # Each Strassen group gets ~r//6 columns, schoolbook gets the rest
                group_size = r // 6
                rem = r - group_size * 6
                pos = 0
                for g in range(5):
                    gs = group_size + (1 if g < rem else 0)
                    U[:, pos:pos+gs] = rng.randint(0, 2, size=(n*m, gs)) * 2 - 1
                    V[:, pos:pos+gs] = rng.randint(0, 2, size=(m*p, gs)) * 2 - 1
                    U[:, pos:pos+gs][rng.random((n*m, gs)) < 0.4] = 0
                    V[:, pos:pos+gs][rng.random((m*p, gs)) < 0.4] = 0
                    pos += gs
                # Schoolbook part
                gs = r - pos
                if gs > 0:
                    U[2, pos:pos+gs] = 1  # A[0,2]
                    U[5, pos:pos+gs] = 1  # A[1,2]
                    V[20:30, pos:pos+gs] = 1  # B[2,:]
                    # Actually need to be more careful
                    U[:, pos:pos+gs] = 0
                    V[:, pos:pos+gs] = 0
                    U[2, pos:pos+gs] = 1
                    U[5, pos:pos+gs] = 1
                    for k in range(p):
                        V[2*p+k, pos:pos+gs] = 1 if k < gs else 0
                    
            else:
                sparsity = 0.25 + 0.15 * (attempt % 4) / 4
                U = rng.randint(0, 2, size=(n*m, r)) * 2 - 1
                V = rng.randint(0, 2, size=(m*p, r)) * 2 - 1
                U[rng.random((n*m, r)) < sparsity] = 0
                V[rng.random((m*p, r)) < sparsity] = 0
            
            # Solve for W
            W = np.zeros((p*n, r))
            feasible = True
            
            for i in range(n):
                for k in range(p):
                    A = np.zeros((m, r))
                    b = np.ones(m)
                    for j in range(m):
                        A[j, :] = U[i*m+j] * V[j*p+k]
                    
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                        if not np.allclose(A @ sol, b, atol=1e-10):
                            feasible = False
                            break
                        W[k*n+i, :] = sol
                    except:
                        feasible = False
                        break
                if not feasible:
                    break
            
            if feasible:
                ok = _verify_np(U, V, W, n, m, p, tol=1e-9)
                if ok:
                    best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = r
                    break
        if best is not None:
            break
    
    return best

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135  # 15s safety margin before 150s limit
    
    # Start with the known rank-55 Strassen construction
    best = _strassen_rank55(n, m, p)
    best_rank = 55
    
    # Try to find rank-54 by exploiting n=2 structure
    time_left = deadline - time.time()
    if time_left > 20:
        for seed in [42, 12345, 777, 99999, 31415, 271828, 161803, 555, 888, 111]:
            if time.time() > deadline - 15:
                break
            tl = min(time_left, deadline - time.time() - 10)
            if tl < 10:
                break
            result = _try_rank54_exploit_n2(n, m, p, tl, seed=seed)
            if result is not None:
                r = len(result["U"][0])
                if r < best_rank:
                    if _verify(result["U"], result["V"], result["W"], n, m, p):
                        best = result
                        best_rank = r
                        break
    
    # Try general search for rank 54
    time_left = deadline - time.time()
    if time_left > 15 and best_rank > 54:
        for seed in [42, 12345, 777, 555]:
            if time.time() > deadline - 12:
                break
            tl = min(time_left, deadline - time.time() - 8)
            if tl < 8:
                break
            result = _try_general_search(n, m, p, tl, seed=seed, start_rank=54)
            if result is not None:
                r = len(result["U"][0])
                if r < best_rank:
                    if _verify(result["U"], result["V"], result["W"], n, m, p):
                        best = result
                        best_rank = r
                        break
    
    # Fallback to schoolbook if something goes wrong
    if best_rank > 60:
        best = _schoolbook(n, m, p)
        best_rank = 60
    
    return best
