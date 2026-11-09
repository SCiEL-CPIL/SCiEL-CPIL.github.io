import numpy as np
import time
import random

DIMENSIONS = [3, 3, 8]

def _validate(U, V, W, n=3, m=3, p=8, tol=1e-9):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if rank != V.shape[1] or rank != W.shape[1]:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _solve_for_W(U, V, n, m, p):
    R = U.shape[1]
    if R == 0:
        return np.zeros((p * n, 0))
    W = np.zeros((p * n, R))
    for i in range(n):
        for k in range(p):
            A = np.zeros((m, R))
            for j in range(m):
                A[j, :] = U[i*m+j, :] * V[j*p+k, :]
            b = np.ones(m)
            w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            W[k*n+i, :] = w
    return W

def _solve_for_V(U, W, n, m, p):
    R = U.shape[1]
    if R == 0:
        return np.zeros((m * p, 0))
    V = np.zeros((m * p, R))
    for j in range(m):
        for k in range(p):
            A = np.zeros((n, R))
            for i in range(n):
                A[i, :] = U[i*m+j, :] * W[k*n+i, :]
            b = np.ones(n)
            v, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            V[j*p+k, :] = v
    return V

def _solve_for_U(V, W, n, m, p):
    R = V.shape[1]
    if R == 0:
        return np.zeros((n * m, 0))
    U = np.zeros((n * m, R))
    for i in range(n):
        for j in range(m):
            A = np.zeros((p, R))
            for k in range(p):
                A[k, :] = V[j*p+k, :] * W[k*n+i, :]
            b = np.ones(p)
            u, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            U[i*m+j, :] = u
    return U

def _round_to_half_int(x):
    return np.round(x * 2) / 2

def _make_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
    return U, V, W

def _alt_min(U, V, W, n, m, p, max_iter=50, deadline=None):
    for _ in range(max_iter):
        if deadline and time.time() > deadline:
            break
        W = _solve_for_W(U, V, n, m, p)
        V = _solve_for_V(U, W, n, m, p)
        U = _solve_for_U(V, W, n, m, p)
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Schoolbook
    U0, V0, W0 = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    
    np.random.seed(42)
    random.seed(42)
    
    # Try target ranks from low to high
    for target_rank in range(25, best_rank):
        if time.time() > deadline - 15:
            break
        
        found = False
        # Try multiple initializations
        for attempt in range(10):
            if time.time() > deadline - 15:
                break
            
            # Different initialization strategies
            if attempt < 3:
                # Random column selection from schoolbook
                cols = np.random.choice(best_rank, target_rank, replace=False)
                U_trial = U0[:, cols].copy()
                V_trial = V0[:, cols].copy()
                W_trial = W0[:, cols].copy()
            elif attempt < 6:
                # Random initialization
                U_trial = np.random.randn(n*m, target_rank) * 0.5
                V_trial = np.random.randn(m*p, target_rank) * 0.5
                W_trial = np.random.randn(p*n, target_rank) * 0.5
            else:
                # Sparse random initialization
                U_trial = np.zeros((n*m, target_rank))
                V_trial = np.zeros((m*p, target_rank))
                W_trial = np.zeros((p*n, target_rank))
                for c in range(target_rank):
                    # Each column has a few nonzero entries
                    for _ in range(random.randint(1, 3)):
                        U_trial[random.randint(0, n*m-1), c] = random.choice([-1, 1])
                    for _ in range(random.randint(1, 3)):
                        V_trial[random.randint(0, m*p-1), c] = random.choice([-1, 1])
                    for _ in range(random.randint(1, 3)):
                        W_trial[random.randint(0, p*n-1), c] = random.choice([-1, 1])
            
            # Alternating minimization
            U_trial, V_trial, W_trial = _alt_min(U_trial, V_trial, W_trial, n, m, p, max_iter=30, deadline=deadline-15)
            
            # Round to half-integers
            U_r = _round_to_half_int(U_trial)
            V_r = _round_to_half_int(V_trial)
            W_r = _round_to_half_int(W_trial)
            
            if _validate(U_r, V_r, W_r, n, m, p):
                best_rank = target_rank
                best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                found = True
                break
        
        if found:
            # Try to go even lower
            continue
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
