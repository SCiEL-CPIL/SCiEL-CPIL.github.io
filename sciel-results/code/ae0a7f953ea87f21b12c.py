"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]
_DEADLINE = 135.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START

def _compute_max_error(U, V, W, n, m, p):
    """Compute max absolute error of the decomposition."""
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err


def _rationalize_matrix(M, tol=1e-7, max_den=200):
    """Rationalize a matrix to have integer or half-integer entries."""
    rows, cols = M.shape
    out = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            v = M[r, c]
            if abs(v) < tol:
                out[r][c] = 0
            else:
                found = False
                # Try integers first
                iv = round(v)
                if abs(v - iv) < tol:
                    out[r][c] = int(iv)
                    found = True
                else:
                    # Try halves
                    hv = round(v * 2)
                    if abs(v - hv / 2.0) < tol:
                        out[r][c] = hv / 2.0
                        found = True
                    else:
                        # Try small denominators
                        for den in range(3, max_den + 1):
                            num = round(v * den)
                            if abs(v - num / den) < tol:
                                out[r][c] = num / den
                                found = True
                                break
                        if not found:
                            out[r][c] = v  # keep as float
    return out


def _verify_rational(U, V, W, n, m, p, tol=1e-6):
    """Verify rational decomposition is exact."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t] 
                         for t in range(len(U[0])))
                if abs(val - 1.0) > tol:
                    return False
    return True


def _als_one_step(U, V, W, n, m, p):
    """One ALS iteration updating all three factors."""
    rank = U.shape[1]
    
    # Update U
    for i in range(n):
        for j in range(m):
            A = np.zeros((p, rank))
            for k in range(p):
                A[k, :] = V[j * p + k] * W[k * n + i]
            try:
                U[i * m + j] = np.linalg.lstsq(A.T, np.ones(p), rcond=None)[0]
            except:
                pass
    
    # Update V
    for j in range(m):
        for k in range(p):
            B = np.zeros((n, rank))
            for i in range(n):
                B[i, :] = U[i * m + j] * W[k * n + i]
            try:
                V[j * p + k] = np.linalg.lstsq(B.T, np.ones(n), rcond=None)[0]
            except:
                pass
    
    # Update W
    for k in range(p):
        for i in range(n):
            C = np.zeros((m, rank))
            for j in range(m):
                C[j, :] = U[i * m + j] * V[j * p + k]
            try:
                W[k * n + i] = np.linalg.lstsq(C.T, np.ones(m), rcond=None)[0]
            except:
                pass


def _polish(U, V, W, n, m, p, max_iter=200, tol=1e-12):
    """Polish with tight tolerance."""
    for _ in range(max_iter):
        err = _compute_max_error(U, V, W, n, m, p)
        if err < tol:
            break
        _als_one_step(U, V, W, n, m, p)
    return _compute_max_error(U, V, W, n, m, p)


def _try_rank_als(rank, n, m, p, time_budget=15.0, n_restarts=30):
    """Try to find rank-r decomposition via ALS."""
    start = time.time()
    
    for restart in range(n_restarts):
        if time.time() - start > time_budget:
            break
        
        rng = np.random.RandomState(42 + rank * 1000 + restart * 7919)
        
        # Different initialization strategies
        if restart % 4 == 0:
            # Random
            U = rng.randn(n * m, rank) * 0.5
            V = rng.randn(m * p, rank) * 0.5
            W = rng.randn(p * n, rank) * 0.5
        elif restart % 4 == 1:
            # Sparse-like
            U = rng.randn(n * m, rank) * 0.3
            V = rng.randn(m * p, rank) * 0.3
            W = rng.randn(p * n, rank) * 0.3
            for t in range(min(rank, n * m)):
                U[t % (n * m), t] += 1.0
            for t in range(min(rank, m * p)):
                V[t % (m * p), t] += 1.0
            for t in range(min(rank, p * n)):
                W[t % (p * n), t] += 1.0
        elif restart % 4 == 2:
            # Diagonal-dominant
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            for t in range(rank):
                U[t % (n * m), t] = 1.0
                V[t % (m * p), t] = 1.0
                W[t % (p * n), t] = 1.0
            U += rng.randn(n * m, rank) * 0.1
            V += rng.randn(m * p, rank) * 0.1
            W += rng.randn(p * n, rank) * 0.1
        else:
            # Structured for n=2: make W rows related
            U = rng.randn(n * m, rank) * 0.5
            V = rng.randn(m * p, rank) * 0.5
            W = rng.randn(p * n, rank) * 0.5
            for k in range(p):
                scale = rng.uniform(0.5, 2.0)
                W[2 * k + 1] = scale * W[2 * k] + rng.randn(rank) * 0.1
        
        # Run ALS
        for it in range(300):
            if time.time() - start > time_budget:
                break
            _als_one_step(U, V, W, n, m, p)
            
            if it % 15 == 14:
                err = _compute_max_error(U, V, W, n, m, p)
                if err < 1e-6:
                    final_err = _polish(U, V, W, n, m, p, max_iter=100, tol=1e-12)
                    if final_err < 1e-10:
                        U_r = _rationalize_matrix(U)
                        V_r = _rationalize_matrix(V)
                        W_r = _rationalize_matrix(W)
                        if _verify_rational(U_r, V_r, W_r, n, m, p, tol=1e-6):
                            return U_r, V_r, W_r
                        # Try with looser rationalization
                        U_r2 = _rationalize_matrix(U, tol=1e-5, max_den=100)
                        V_r2 = _rationalize_matrix(V, tol=1e-5, max_den=100)
                        W_r2 = _rationalize_matrix(W, tol=1e-5, max_den=100)
                        if _verify_rational(U_r2, V_r2, W_r2, n, m, p, tol=1e-5):
                            return U_r2, V_r2, W_r2
    
    return None


def _schoolbook():
    n, m, p = DIMENSIONS
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


def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    max_rank = n * m * p  # 60
    
    min_possible = max(n * m, m * p, p * n)  # 30
    
    # Try ranks from lower bound upward
    for rank in range(min_possible, max_rank + 1):
        if time.time() - start > _DEADLINE - 15:
            break
        
        remaining = _DEADLINE - 15 - (time.time() - start)
        if remaining < 5.0:
            break
        
        time_budget = min(15.0, remaining)
        
        result = _try_rank_als(rank, n, m, p, time_budget=time_budget)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return _schoolbook()

# EVOLVE-BLOCK-END
