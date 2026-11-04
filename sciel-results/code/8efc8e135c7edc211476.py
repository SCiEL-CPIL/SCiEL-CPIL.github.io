import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

def _build_schoolbook(n, m, p):
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
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify the tensor decomposition reconstructs matrix multiplication exactly."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    # Check zero constraints
    for a in range(n * m):
        for b in range(m * p):
            for g in range(p * n):
                i = a // m; j = a % m
                jp = b // p; k = b % p
                kn = g // n; ip = g % n
                if i != ip or j != jp or k != kn:
                    val = 0.0
                    for r in range(rank):
                        val += U[a, r] * V[b, r] * W[g, r]
                    if abs(val) > tol:
                        return False
    return True

def _round_to_half_integer(M):
    """Round matrix entries to nearest 0.5."""
    return np.round(M * 2) / 2

def _try_structured_decomposition(n, m, p, target_rank, seed=42):
    """
    Try structured sparse patterns for U, V, then solve for W.
    """
    rng = random.Random(seed)
    
    # Try different sparsity levels
    for nnz_u in [1, 2, 3]:
        for nnz_v in [1, 2, 3]:
            if target_rank < n * m * p // (nnz_u * nnz_v):
                continue
            
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            
            for a in range(n * m):
                cols = rng.sample(range(target_rank), min(nnz_u, target_rank))
                for c in cols:
                    U[a, c] = rng.choice([1, -1, 0.5, -0.5])
            
            for b in range(m * p):
                cols = rng.sample(range(target_rank), min(nnz_v, target_rank))
                for c in cols:
                    V[b, c] = rng.choice([1, -1, 0.5, -0.5])
            
            # Solve for W
            W = np.zeros((p * n, target_rank))
            solved = True
            
            for g in range(p * n):
                k = g // n
                i = g % n
                
                # Build system: for each (a,b), sum_r U[a,r]*V[b,r]*W[g,r] = T[a,b,g]
                # This is linear in W[g,:]
                num_ab = n * m * m * p
                A = np.zeros((num_ab, target_rank))
                t = np.zeros(num_ab)
                
                idx = 0
                for a in range(n * m):
                    ai = a // m; aj = a % m
                    for b in range(m * p):
                        bj = b // p; bk = b % p
                        A[idx, :] = U[a, :] * V[b, :]
                        if ai == i and aj == bj and bk == k:
                            t[idx] = 1.0
                        else:
                            t[idx] = 0.0
                        idx += 1
                
                try:
                    w, res, rk, sv = np.linalg.lstsq(A, t, rcond=None)
                    W[g, :] = w
                except:
                    solved = False
                    break
            
            if not solved:
                continue
            
            if _verify_decomposition(U, V, W, n, m, p):
                return U, V, W
    
    return None

def _als_optimize(n, m, p, initial_rank, max_time=50):
    """Alternating Least Squares for tensor decomposition."""
    start = time.time()
    deadline = start + max_time
    
    for target_rank in range(initial_rank - 1, max(20, initial_rank - 30), -1):
        if time.time() > deadline:
            break
        
        for seed in range(3):
            if time.time() > deadline:
                break
            
            np.random.seed(seed)
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            for it in range(30):
                if time.time() > deadline:
                    break
                
                # Update U
                GV = V.T @ V
                GW = W.T @ W
                A_mat = GV * GW + 1e-8 * np.eye(target_rank)
                
                for a in range(n * m):
                    bv = np.zeros(target_rank)
                    i = a // m; j = a % m
                    for k in range(p):
                        b = j * p + k
                        g = k * n + i
                        bv += V[b, :] * W[g, :]
                    try:
                        U[a, :] = np.linalg.solve(A_mat, bv)
                    except:
                        pass
                
                # Update V
                GU = U.T @ U
                GW = W.T @ W
                A_mat = GU * GW + 1e-8 * np.eye(target_rank)
                
                for b in range(m * p):
                    bv = np.zeros(target_rank)
                    j = b // p; k = b % p
                    for i in range(n):
                        a = i * m + j
                        g = k * n + i
                        bv += U[a, :] * W[g, :]
                    try:
                        V[b, :] = np.linalg.solve(A_mat, bv)
                    except:
                        pass
                
                # Update W
                GU = U.T @ U
                GV = V.T @ V
                A_mat = GU * GV + 1e-8 * np.eye(target_rank)
                
                for g in range(p * n):
                    bv = np.zeros(target_rank)
                    k = g // n; i = g % n
                    for j in range(m):
                        a = i * m + j
                        b = j * p + k
                        bv += U[a, :] * V[b, :]
                    try:
                        W[g, :] = np.linalg.solve(A_mat, bv)
                    except:
                        pass
                
                if it % 5 == 4:
                    # Check error
                    err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                a = i*m+j; b = j*p+k; g = k*n+i
                                val = np.sum(U[a,:] * V[b,:] * W[g,:])
                                err += (val - 1.0)**2
                    # Sample zero constraints
                    np.random.seed(42)
                    for _ in range(100):
                        a = np.random.randint(0, n*m)
                        b = np.random.randint(0, m*p)
                        g = np.random.randint(0, p*n)
                        i = a//m; j = a%m
                        jp = b//p; k = b%p
                        kn = g//n; ip = g%n
                        if i != ip or j != jp or k != kn:
                            val = np.sum(U[a,:] * V[b,:] * W[g,:])
                            err += val**2
                    
                    if err < 1e-4:
                        Ur = _round_to_half_integer(U)
                        Vr = _round_to_half_integer(V)
                        Wr = _round_to_half_integer(W)
                        if _verify_decomposition(Ur, Vr, Wr, n, m, p):
                            return Ur, Vr, Wr
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    best_U, best_V, best_W = _build_schoolbook(n, m, p)
    best_rank = best_U.shape[1]
    
    # Try structured decompositions
    for target_rank in range(60, best_rank, 5):
        if time.time() > deadline - 40:
            break
        for seed in [42, 123, 456]:
            if time.time() > deadline - 35:
                break
            result = _try_structured_decomposition(n, m, p, target_rank, seed=seed)
            if result is not None:
                U, V, W = result
                if U.shape[1] < best_rank:
                    best_U, best_V, best_W = U, V, W
                    best_rank = U.shape[1]
                    break
    
    # Try ALS if time remains
    if time.time() < deadline - 10:
        result = _als_optimize(n, m, p, initial_rank=best_rank, max_time=int(deadline - time.time() - 5))
        if result is not None:
            U, V, W = result
            if U.shape[1] < best_rank:
                best_U, best_V, best_W = U, V, W
                best_rank = U.shape[1]
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
