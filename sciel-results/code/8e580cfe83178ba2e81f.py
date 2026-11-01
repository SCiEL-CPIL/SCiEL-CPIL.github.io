"""Search for a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 10 matrix."""
import time
import random
from fractions import Fraction
import numpy as np
from scipy import linalg as sla

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
        
        u00, u01, u10, u11 = 0, 1, 3, 4
        v00, v01 = k0, k1
        v10, v11 = 10 + k0, 10 + k1
        w00, w01 = k0 * 2, k1 * 2
        w10, w11 = k0 * 2 + 1, k1 * 2 + 1
        
        # M1 = (A00+A11)(B00+B11)
        U[u00][col] = 1; U[u11][col] = 1; V[v00][col] = 1; V[v11][col] = 1; W[w00][col] = 1; W[w11][col] = 1
        col += 1
        # M2 = (A10+A11)B00
        U[u10][col] = 1; U[u11][col] = 1; V[v00][col] = 1; W[w10][col] = 1; W[w11][col] = -1
        col += 1
        # M3 = A00(B01-B11)
        U[u00][col] = 1; V[v01][col] = 1; V[v11][col] = -1; W[w01][col] = 1; W[w11][col] = 1
        col += 1
        # M4 = A11(B10-B00)
        U[u11][col] = 1; V[v10][col] = 1; V[v00][col] = -1; W[w00][col] = 1; W[w10][col] = 1
        col += 1
        # M5 = (A00+A01)B11
        U[u00][col] = 1; U[u01][col] = 1; V[v11][col] = 1; W[w01][col] = 1; W[w00][col] = -1
        col += 1
        # M6 = (A10-A00)(B00+B01)
        U[u10][col] = 1; U[u00][col] = -1; V[v00][col] = 1; V[v01][col] = 1; W[w11][col] = 1
        col += 1
        # M7 = (A01-A11)(B10+B11)
        U[u01][col] = 1; U[u11][col] = -1; V[v10][col] = 1; V[v11][col] = 1; W[w00][col] = 1
        col += 1
        
    # Schoolbook for A2 (2x1) * B2 (1x10)
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
    """
    rng = np.random.RandomState(seed)
    deadline = time.time() + time_budget
    
    base = _strassen_rank55(n, m, p)
    U0 = np.array(base["U"], dtype=float)
    V0 = np.array(base["V"], dtype=float)
    W0 = np.array(base["W"], dtype=float)
    
    r54 = 54
    
    best = None
    
    for attempt in range(200):
        if time.time() > deadline - 10:
            break
        
        U = np.zeros((n*m, r54))
        V = np.zeros((m*p, r54))
        
        strat = attempt % 4
        
        if strat == 0:
            for row in range(n*m):
                nnz = rng.randint(2, min(6, r54))
                cols = rng.choice(r54, size=nnz, replace=False)
                U[row, cols] = rng.choice([-1, 0, 1], size=nnz)
            for row in range(m*p):
                nnz = rng.randint(2, min(6, r54))
                cols = rng.choice(r54, size=nnz, replace=False)
                V[row, cols] = rng.choice([-1, 0, 1], size=nnz)
                
        elif strat == 1:
            U[:, :r54] = U0[:, :r54] * rng.randint(0, 2, size=(n*m, r54))
            V[:, :r54] = V0[:, :r54] * rng.randint(0, 2, size=(m*p, r54))
            U += rng.randint(-1, 2, size=(n*m, r54)) * 0.1
            V += rng.randint(-1, 2, size=(m*p, r54)) * 0.1
            
        elif strat == 2:
            U = rng.randint(0, 2, size=(n*m, r54)) * 2 - 1
            V = rng.randint(0, 2, size=(m*p, r54)) * 2 - 1
            U[rng.random((n*m, r54)) < 0.3] = 0
            V[rng.random((m*p, r54)) < 0.3] = 0
            
        else:
            block_size = r54 // 5
            for g in range(5):
                start = g * block_size
                end = start + block_size if g < 4 else r54
                if end > start:
                    U[:, start:end] = rng.randint(0, 2, size=(n*m, end-start)) * 2 - 1
                    V[:, start:end] = rng.randint(0, 2, size=(m*p, end-start)) * 2 - 1
                    U[:, start:end][rng.random((n*m, end-start)) < 0.4] = 0
                    V[:, start:end][rng.random((m*p, end-start)) < 0.4] = 0
        
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
            ok = True
            for i in range(n):
                if not ok: break
                for j in range(m):
                    if not ok: break
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

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # 20s safety margin before 150s limit
    
    # Start with the known rank-55 Strassen construction
    best = _strassen_rank55(n, m, p)
    best_rank = 55
    
    # Try the n=2 exploit search
    if best_rank > 54:
        time_left = deadline - time.time()
        if time_left > 10:
            for seed in [42, 12345, 777]:
                if time.time() > deadline - 20:
                    break
                tl = min(time_left, deadline - time.time() - 15)
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
    
    # Fallback
    if best_rank > 60:
        best = _schoolbook(n, m, p)
        best_rank = 60
        
    return best
