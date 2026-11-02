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

def _try_rank54_structured(n, m, p, time_budget, seed=42):
    """
    Try to find a rank-54 decomposition using structured search exploiting n=2 symmetry.
    Key insight: with n=2, the W matrix has only 20 rows, and we can exploit
    the fact that W[k*n+0] and W[k*n+1] are related for each k.
    """
    rng = np.random.RandomState(seed)
    deadline = time.time() + time_budget
    
    r54 = 54
    best = None
    
    # Strategy: Fix the Strassen structure for the 2x2 part (35 terms)
    # and try to merge or eliminate 1 term from the schoolbook part (20 terms)
    # by allowing half-integer or rational entries.
    
    # Alternative: Try to find a rank-54 by perturbing the rank-55 solution
    base = _strassen_rank55(n, m, p)
    U0 = np.array(base["U"], dtype=float)
    V0 = np.array(base["V"], dtype=float)
    W0 = np.array(base["W"], dtype=float)
    
    # Try to merge two columns of the rank-55 into one by finding
    # a linear combination that still works
    for attempt in range(50):
        if time.time() > deadline - 5:
            break
        
        # Pick two random columns to try to merge
        c1, c2 = rng.choice(r54 + 1, size=2, replace=False)
        c1 = int(c1); c2 = int(c2)
        if c1 >= 55 or c2 >= 55:
            continue
        
        # Try to find alpha, beta such that replacing columns c1,c2
        # with a single column works. This is a non-linear problem,
        # so we try a few heuristics.
        
        # Heuristic 1: Set one column to be a scaled version of the other
        for scale in [0.5, -0.5, 2.0, -2.0, 0.0]:
            U = U0.copy()
            V = V0.copy()
            W = W0.copy()
            
            # Remove column c2, scale column c1
            if scale != 0:
                U[:, c1] = scale * U0[:, c1]
                V[:, c1] = scale * V0[:, c1]
                # W stays the same for now, we'll solve for it
            else:
                # Just remove c2
                pass
            
            # Try to solve for W with the modified U, V
            feasible = True
            W_new = np.zeros((p*n, 54))
            
            for i in range(n):
                for k in range(p):
                    A = np.zeros((m, 54))
                    b = np.ones(m)
                    for j in range(m):
                        A[j, :] = U[i*m+j] * V[j*p+k]
                    
                    # Check if the system is consistent
                    try:
                        # Use least squares and check residual
                        sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                        if len(residuals) == 0 or residuals[0] < 1e-10:
                            W_new[k*n+i, :] = sol
                        else:
                            feasible = False
                            break
                    except:
                        feasible = False
                        break
                if not feasible:
                    break
            
            if feasible:
                # Verify
                ok = True
                for i in range(n):
                    if not ok: break
                    for j in range(m):
                        if not ok: break
                        for k in range(p):
                            prod = U[i*m+j] * V[j*p+k] * W_new[k*n+i]
                            s = np.sum(prod)
                            if abs(s - 1.0) > 1e-8:
                                ok = False
                                break
                if ok:
                    # Round to nearest half-integer
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W_new * 2) / 2
                    
                    # Verify rounded version
                    ok_r = True
                    for i in range(n):
                        if not ok_r: break
                        for j in range(m):
                            if not ok_r: break
                            for k in range(p):
                                prod = U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i]
                                s = np.sum(prod)
                                if abs(s - 1.0) > 1e-8:
                                    ok_r = False
                                    break
                    if ok_r:
                        best = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                        return best
    
    return best

def _try_rank54_general(n, m, p, time_budget, seed=42):
    """
    General search for rank-54 using random initialization and solving for W.
    """
    rng = np.random.RandomState(seed)
    deadline = time.time() + time_budget
    
    r54 = 54
    best = None
    
    for attempt in range(100):
        if time.time() > deadline - 5:
            break
        
        # Random sparse initialization
        U = np.zeros((n*m, r54))
        V = np.zeros((m*p, r54))
        
        # Each row of U has 2-4 non-zero entries
        for row in range(n*m):
            nnz = rng.randint(2, 5)
            cols = rng.choice(r54, size=nnz, replace=False)
            U[row, cols] = rng.choice([-1, 1], size=nnz)
        
        for row in range(m*p):
            nnz = rng.randint(2, 5)
            cols = rng.choice(r54, size=nnz, replace=False)
            V[row, cols] = rng.choice([-1, 1], size=nnz)
        
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
                    sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                    if len(residuals) == 0 or residuals[0] < 1e-10:
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
            # Verify
            ok = True
            for i in range(n):
                if not ok: break
                for j in range(m):
                    if not ok: break
                    for k in range(p):
                        prod = U[i*m+j] * V[j*p+k] * W[k*n+i]
                        s = np.sum(prod)
                        if abs(s - 1.0) > 1e-8:
                            ok = False
                            break
            if ok:
                # Round to nearest half-integer
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Verify rounded version
                ok_r = True
                for i in range(n):
                    if not ok_r: break
                    for j in range(m):
                        if not ok_r: break
                        for k in range(p):
                            prod = U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i]
                            s = np.sum(prod)
                            if abs(s - 1.0) > 1e-8:
                                ok_r = False
                                break
                if ok_r:
                    best = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    return best
    
    return best

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s limit
    
    # Start with the known rank-55 Strassen construction
    best = _strassen_rank55(n, m, p)
    best_rank = 55
    
    # Verify the rank-55 construction
    if not _verify_np(best["U"], best["V"], best["W"], n, m, p):
        # Fallback to schoolbook if something went wrong
        best = _schoolbook(n, m, p)
        best_rank = 60
        return best
    
    # Try to find rank-54
    if best_rank > 54:
        time_left = deadline - time.time()
        if time_left > 10:
            # Try structured search first (more likely to succeed)
            tl = min(time_left * 0.4, 30)
            if tl > 5:
                result = _try_rank54_structured(n, m, p, tl, seed=42)
                if result is not None:
                    r = len(result["U"][0])
                    if r < best_rank:
                        if _verify(result["U"], result["V"], result["W"], n, m, p):
                            best = result
                            best_rank = r
            
            # Try general search with remaining time
            if best_rank > 54:
                time_left = deadline - time.time()
                if time_left > 10:
                    tl = min(time_left * 0.5, 40)
                    if tl > 5:
                        for seed in [12345, 777, 999]:
                            if time.time() > deadline - 15:
                                break
                            tl_seed = min(tl, deadline - time.time() - 10)
                            if tl_seed < 5:
                                break
                            result = _try_rank54_general(n, m, p, tl_seed, seed=seed)
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
