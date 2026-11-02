"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _verify_exact_fraction(U, V, W, n, m, p):
    """Verify exact tensor decomposition using Fraction for exact arithmetic."""
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    rank = len(U_frac[0]) if U_frac else 0
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                s = Fraction(0)
                for c in range(rank):
                    s += U_frac[ui][c] * V_frac[vj][c] * W_frac[wk][c]
                if s != 1:
                    return False
    return True


def _fast_als(n, m, p, target_rank, max_iters=20, time_limit=10.0):
    """
    Fast ALS using vectorized operations.
    """
    start_time = time.time()
    
    # Initialize
    np.random.seed(42)
    U = np.random.randn(n * m, target_rank) * 0.1
    V = np.random.randn(m * p, target_rank) * 0.1
    W = np.random.randn(p * n, target_rank) * 0.1
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            return None
        
        # Update U using vectorized operations
        for i in range(n * m):
            ii = i // m
            jj = i % m
            # A[k, c] = V[jj*p+k, c] * W[k*n+ii, c]
            v_rows = V[jj*p:(jj+1)*p, :]  # shape (p, target_rank)
            w_rows = W[ii::n, :]  # shape (p, target_rank)
            A = v_rows * w_rows  # shape (p, target_rank)
            b = np.ones(p)
            try:
                U[i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
            except:
                pass
        
        # Update V
        for j in range(m * p):
            jj = j // p
            kk = j % p
            u_rows = U[jj::m, :]  # shape (n, target_rank)
            w_rows = W[kk::n, :]  # shape (n, target_rank)
            A = u_rows * w_rows  # shape (n, target_rank)
            b = np.ones(n)
            try:
                V[j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
            except:
                pass
        
        # Update W
        for k in range(p * n):
            kk = k // n
            ii = k % n
            u_rows = U[ii::m, :]  # shape (m, target_rank)
            v_rows = V[ii*p:(ii+1)*p, :]  # shape (m, target_rank)
            A = u_rows * v_rows  # shape (m, target_rank)
            b = np.ones(m)
            try:
                W[k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
            except:
                pass
        
        # Check residual (vectorized)
        residual = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = i * m + j
                    vj = j * p + k
                    wk = k * n + i
                    s = np.dot(U[ui, :], V[vj, :] * W[wk, :])
                    residual += (s - 1) ** 2
        
        if residual < 1e-10:
            return U, V, W
    
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110

    # Start with schoolbook
    result = _schoolbook(n, m, p)
    best_rank = n * m * p

    # Try to find lower rank
    for target_rank in range(83, 75, -1):
        if time.time() > deadline - 10:
            break
        
        remaining = deadline - time.time() - 5
        if remaining < 2:
            break
        
        sol = _fast_als(n, m, p, target_rank, max_iters=30, time_limit=remaining / 2)
        if sol is not None:
            U, V, W = sol
            # Snap to half-integers
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            U_list = U_snap.tolist()
            V_list = V_snap.tolist()
            W_list = W_snap.tolist()
            
            if _verify_exact_fraction(U_list, V_list, W_list, n, m, p):
                result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = target_rank
                # Continue to try lower ranks
                continue
            else:
                # Try integer snap
                U_int = np.round(U).tolist()
                V_int = np.round(V).tolist()
                W_int = np.round(W).tolist()
                if _verify_exact_fraction(U_int, V_int, W_int, n, m, p):
                    result = {"U": U_int, "V": V_int, "W": W_int}
                    best_rank = target_rank
                    continue
    
    if time.time() > deadline:
        return result

    return result
# EVOLVE-BLOCK-END
