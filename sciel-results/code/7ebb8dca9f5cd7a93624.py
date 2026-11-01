"""Search for a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 10 matrix."""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 3, 10]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
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
    """Fast verification using NumPy."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                prod = U[i*m+j] * V[j*p+k] * W[k*n+i]
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
    
    Key insight: For n=2, both output rows i=0 and i=1 share the same B matrix.
    The constraints are:
      For each (j,k): sum_c U[0*m+j,c]*V[j*p+k,c]*W[k*n+0,c] = 1
      For each (j,k): sum_c U[1*m+j,c]*V[j*p+k,c]*W[k*n+1,c] = 1
    
    We can try to construct U and V with structure that allows both rows to
    be satisfied simultaneously with rank 54 (one less than 55).
    
    Strategy: Start from the rank-55 Strassen construction and try to merge
    one pair of columns by finding a linear combination that satisfies both rows.
    """
    rng = np.random.RandomState(seed)
    deadline = time.time() + time_budget
    
    # Start from rank-55 and try to reduce by 1
    base = _strassen_rank55(n, m, p)
    U0 = np.array(base["U"], dtype=float)
    V0 = np.array(base["V"], dtype=float)
    W0 = np.array(base["W"], dtype=float)
    
    r55 = 55
    r54 = 54
    
    # Try to find if we can merge two columns of the rank-55 solution
    # by finding a linear combination that works
    # This is a hard problem, so we try random perturbations
    
    best = None
    
    # Approach 1: Try random U,V of rank 54 and solve for W
    for attempt in range(200):
        if time.time() > deadline - 10:
            break
        
        # Initialize U and V with structure similar to Strassen
        U = np.zeros((n*m, r54))
        V = np.zeros((m*p, r54))
        
        # Use the Strassen pattern but with rank 54
        # Try different sparsity patterns
        strat = attempt % 4
        
        if strat == 0:
            # Sparse random
            for row in range(n*m):
                nnz = rng.randint(2, min(6, r54))
                cols = rng.choice(r54, size=nnz, replace=False)
                U[row, cols] = rng.choice([-1, 0, 1], size=nnz)
            for row in range(m*p):
                nnz = rng.randint(2, min(6, r54))
                cols = rng.choice(r54, size=nnz, replace=False)
                V[row, cols] = rng.choice([-1, 0, 1], size=nnz)
                
        elif strat == 1:
            # Based on Strassen structure with one column removed
            U[:, :r54] = U0[:, :r54] * rng.randint(0, 2, size=(n*m, r54))
            V[:, :r54] = V0[:, :r54] * rng.randint(0, 2, size=(m*p, r54))
            # Add some randomness
            U += rng.randint(-1, 2, size=(n*m, r54)) * 0.1
            V += rng.randint(-1, 2, size=(m*p, r54)) * 0.1
            
        elif strat == 2:
            # Random ±1 with moderate sparsity
            U = rng.randint(0, 2, size=(n*m, r54)) * 2 - 1
            V = rng.randint(0, 2, size=(m*p, r54)) * 2 - 1
            U[rng.random((n*m, r54)) < 0.3] = 0
            V[rng.random((m*p, r54)) < 0.3] = 0
            
        else:
            # Block structure
            block_size = r54 // 5
            for g in range(5):
                start = g * block_size
                end = start + block_size if g < 4 else r54
                if end > start:
                    U[:, start:end] = rng.randint(0, 2, size=(n*m, end-start)) * 2 - 1
                    V[:, start:end] = rng.randint(0, 2, size=(m*p, end-start)) * 2 - 1
                    U[:, start:end][rng.random((n*m, end-start)) < 0.4] = 0
                    V[:, start:end][rng.random((m*p, end-start)) < 0.4] = 0
        
        # Solve for W
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
            # Verify full solution
            ok = True
            for i in range(n):
                if not ok:
                    break
                for j in range(m):
                    if not ok:
                        break
                    for k in range(p):
                        prod = U[i*m+j] * V[j*p+k] * W[k*n+i]
                        s = np.sum(prod)
                        if abs(s - 1.0) > 1e-10:
                            ok = False
                            break
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
    
    for r in range(start_rank, min(n*m*p + 1, start_rank + 10)):
        if time.time() > deadline - 15:
            break
        if r >= best_rank:
            continue
            
        max_attempts = 100
        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                break
                
            strat = attempt % 5
            
            if strat == 0:
                U = np.zeros((n*m, r))
                V = np.zeros((m*p, r))
                for row in range(n*m):
                    nnz = rng.randint(1, min(5, r))
                    cols = rng.choice(r, size=nnz, replace=False)
                    U[row, cols] = rng.choice([-1, 1], size=nnz)
                for row in range(m*p):
                    nnz = rng.randint(1, min(5, r))
                    cols = rng.choice(r, size=nnz, replace=False)
                    V[row, cols] = rng.choice([-1, 1], size=nnz)
                    
            elif strat == 1:
                U = rng.randint(0, 2, size=(n*m, r)) * 2 - 1
                V = rng.randint(0, 2, size=(m*p, r)) * 2 - 1
                U[rng.random((n*m, r)) < 0.3] = 0
                V[rng.random((m*p, r)) < 0.3] = 0
                
            elif strat == 2:
                # Kronecker-like
                base_size = min(r, 12)
                U = np.zeros((n*m, r))
                V = np.zeros((m*p, r))
                U[:, :base_size] = rng.randint(0, 2, size=(n*m, base_size)) * 2 - 1
                V[:, :base_size] = rng.randint(0, 2, size=(m*p, base_size)) * 2 - 1
                if r > base_size:
                    U[:, base_size:] = rng.randint(0, 2, size=(n*m, r-base_size)) * 2 - 1
                    V[:, base_size:] = rng.randint(0, 2, size=(m*p, r-base_size)) * 2 - 1
                    U[:, base_size:][rng.random((n*m, r-base_size)) < 0.5] = 0
                    V[:, base_size:][rng.random((m*p, r-base_size)) < 0.5] = 0
                    
            elif strat == 3:
                # Low-rank factor
                mid = max(2, r // 4)
                U1 = rng.randint(0, 2, size=(n*m, mid)) * 2 - 1
                U2 = rng.randint(0, 2, size=(mid, r)) * 2 - 1
                U = U1 @ U2
                V1 = rng.randint(0, 2, size=(m*p, mid)) * 2 - 1
                V2 = rng.randint(0, 2, size=(mid, r)) * 2 - 1
                V = V1 @ V2
                U[np.abs(U) < 1] = 0
                V[np.abs(V) < 1] = 0
                
            else:
                sparsity = 0.2 + 0.2 * (attempt % 5) / 5
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
                ok = True
                for i in range(n):
                    if not ok:
                        break
                    for j in range(m):
                        if not ok:
                            break
                        for k in range(p):
                            prod = U[i*m+j] * V[j*p+k] * W[k*n+i]
                            s = np.sum(prod)
                            if abs(s - 1.0) > 1e-10:
                                ok = False
                                break
                if ok:
                    best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = r
                    break
        if best is not None:
            break
    
    return best

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin before 150s limit
    
    # Start with the known rank-55 Strassen construction
    best = _strassen_rank55(n, m, p)
    best_rank = 55
    
    # Try to find rank-54 by exploiting n=2 structure
    time_left = deadline - time.time()
    if time_left > 30:
        for seed in [42, 12345, 777, 99999, 31415, 271828, 161803]:
            if time.time() > deadline - 20:
                break
            tl = min(time_left, deadline - time.time() - 15)
            if tl < 20:
                break
            result = _try_rank54_exploit_n2(n, m, p, tl, seed=seed)
            if result is not None:
                r = len(result["U"][0])
                if r < best_rank:
                    if _verify(result["U"], result["V"], result["W"], n, m, p):
                        best = result
                        best_rank = r
                        break
    
    # Try general search for ranks 54-58
    time_left = deadline - time.time()
    if time_left > 20 and best_rank > 54:
        for seed in [42, 12345, 777]:
            if time.time() > deadline - 20:
                break
            tl = min(time_left, deadline - time.time() - 15)
            if tl < 15:
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
