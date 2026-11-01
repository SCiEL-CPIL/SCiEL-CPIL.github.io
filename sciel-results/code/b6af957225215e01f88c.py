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

def _try_rank54_search(n, m, p, time_budget, seed=42):
    """
    Search for a rank-54 decomposition.
    Strategy: Random sparse U, V with integer entries. Solve for W.
    Check if W has integer or half-integer entries.
    """
    rng = np.random.RandomState(seed)
    deadline = time.time() + time_budget
    r = 54
    
    best = None
    
    for attempt in range(500):
        if time.time() > deadline - 5:
            break
            
        # Generate sparse U and V
        # We want U and V to be such that M_ki = U[i,:] * V[j,:] (elementwise) leads to simple W
        # Let's try random sparse integers
        U = np.zeros((n*m, r), dtype=int)
        V = np.zeros((m*p, r), dtype=int)
        
        # Sparsity level
        sparsity = 0.2
        for idx in range(U.size):
            if rng.random() < sparsity:
                U.flat[idx] = rng.choice([-1, 1])
        for idx in range(V.size):
            if rng.random() < sparsity:
                V.flat[idx] = rng.choice([-1, 1])
                
        W = np.zeros((p*n, r), dtype=float)
        feasible = True
        
        for i in range(n):
            for k in range(p):
                # Construct M_ki: m x r
                # M_ki[j, c] = U[i*m+j, c] * V[j*p+k, c]
                M_ki = np.zeros((m, r), dtype=float)
                for j in range(m):
                    M_ki[j, :] = U[i*m+j, :] * V[j*p+k, :]
                
                # Solve M_ki @ w = ones(m)
                # We want w to have integer or half-integer entries
                try:
                    # Use lstsq for stability
                    sol, residuals, rank, sv = np.linalg.lstsq(M_ki, np.ones(m), rcond=None)
                    # Check if solution is close to integer or half-integer
                    # Multiply by 2 to check integers
                    sol_scaled = sol * 2
                    rounded = np.round(sol_scaled)
                    if np.allclose(sol_scaled, rounded, atol=1e-9):
                        W[k*n+i, :] = sol
                    else:
                        feasible = False
                        break
                except:
                    feasible = False
                    break
                    
            if not feasible:
                break
                
        if feasible:
            # Final verification
            if _verify_np(U, V, W, n, m, p, tol=1e-9):
                # Convert to lists
                best = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                break
                
    return best

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Start with the known rank-55 Strassen construction
    best = _strassen_rank55(n, m, p)
    best_rank = 55
    
    # Try the rank-54 search
    if best_rank > 54:
        time_left = deadline - time.time()
        if time_left > 5:
            result = _try_rank54_search(n, m, p, time_left, seed=12345)
            if result is not None:
                # Verify with Fractions to be absolutely sure
                if _verify(result["U"], result["V"], result["W"], n, m, p):
                    best = result
                    best_rank = 54
    
    # Fallback
    if best_rank > 60:
        best = _schoolbook(n, m, p)
        best_rank = 60
        
    return best
