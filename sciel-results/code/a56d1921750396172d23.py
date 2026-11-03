import numpy as np
import time
import random

DIMENSIONS = [3, 4, 8]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check the cyclic bilinear identity."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _build_rank88():
    """Verified rank-88 construction."""
    n, m, p = DIMENSIONS
    rank = 88
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for j in range(m):
        for k in range(p):
            U[0 * m + j, col] = 1.0
            V[j * p + k, col] = 1.0
            W[k * n + 0, col] = 1.0
            col += 1
    sU = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
    sV = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    sW = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]
    for jb in range(2):
        js = jb * 2
        for kb in range(4):
            ks = kb * 2
            for t in range(7):
                for l in range(4):
                    if sU[t][l]:
                        li, lj = l // 2, l % 2
                        U[(1 + li) * m + (js + lj), col] += sU[t][l]
                    if sV[t][l]:
                        lj, lk = l // 2, l % 2
                        V[(js + lj) * p + (ks + lk), col] += sV[t][l]
                    if sW[t][l]:
                        li, lk = l // 2, l % 2
                        W[(ks + lk) * n + (1 + li), col] += sW[t][l]
                col += 1
    return U, V, W

def _tensor_from_decomp(U, V, W, n, m, p):
    """Compute the tensor T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]"""
    R = U.shape[1]
    T = np.zeros((n, m, p))
    for r in range(R):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    return T

def _extract_rank1_from(T, n, m, p, tol=1e-8):
    """Extract a rank-1 approximation u,v,w from tensor T using ALS."""
    idx = np.argwhere(np.abs(T) > tol)
    if len(idx) == 0:
        return None
    
    # Initialize
    u = np.random.randn(n, m)
    v = np.random.randn(m, p)
    w = np.random.randn(p, n)
    
    for _ in range(50):
        # Update u
        for i in range(n):
            for j in range(m):
                denom = v[j, :] * w[:, i]
                dd = float(np.dot(denom, denom))
                if dd > tol:
                    u[i, j] = float(np.dot(T[i, j, :], denom)) / dd
        # Update v
        for j in range(m):
            for k in range(p):
                denom = u[:, j] * w[k, :]
                dd = float(np.dot(denom, denom))
                if dd > tol:
                    v[j, k] = float(np.dot(T[:, j, k], denom)) / dd
        # Update w
        for k in range(p):
            for i in range(n):
                denom = u[i, :] * v[:, k]
                dd = float(np.dot(denom, denom))
                if dd > tol:
                    w[k, i] = float(np.dot(T[i, :, k], denom)) / dd
    
    # Check quality
    T1 = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T1[i, j, k] = u[i, j] * v[j, k] * w[k, i]
    
    err = np.linalg.norm(T - T1)
    if err < 0.1: # heuristic threshold
        return u, v, w
    return None

def _try_reduce_rank(U, V, W, n, m, p, deadline, max_reduction=5):
    """Try to reduce rank by extracting rank-1 terms and removing columns."""
    U = np.asarray(U, dtype=float).copy()
    V = np.asarray(V, dtype=float).copy()
    W = np.asarray(W, dtype=float).copy()
    
    T = _tensor_from_decomp(U, V, W, n, m, p)
    R = U.shape[1]
    
    # We try to find a set of rank-1 terms that sum to T
    # and then see if we can represent them with fewer columns.
    # This is essentially trying to find a sparser representation or
    # checking if the current rank is minimal.
    
    # A simpler heuristic: check if any column is redundant.
    # If column r is a linear combination of others, we might be able to remove it?
    # No, that doesn't reduce rank.
    
    # The real reduction comes if we can find a decomposition with fewer terms.
    # Let's try to extract rank-1 terms greedily.
    
    T_rem = T.copy()
    terms = []
    for _ in range(R):
        if time.time() > deadline:
            break
        if np.linalg.norm(T_rem) < 1e-6:
            break
            
        res = _extract_rank1_from(T_rem, n, m, p)
        if res is None:
            break
        u, v, w = res
        terms.append((u, v, w))
        
        # Subtract
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_rem[i, j, k] -= u[i, j] * v[j, k] * w[k, i]
        
        if len(terms) < R - max_reduction:
            # We have fewer terms than original rank, success!
            # Construct new U, V, W
            new_R = len(terms)
            new_U = np.zeros((n * m, new_R))
            new_V = np.zeros((m * p, new_R))
            new_W = np.zeros((p * n, new_R))
            
            for r, (u, v, w) in enumerate(terms):
                new_U[:, r] = u.flatten()
                new_V[:, r] = v.flatten()
                new_W[:, r] = w.flatten()
                
            if _verify(new_U, new_V, new_W, n, m, p):
                return new_U, new_V, new_W
            else:
                # Verification failed, maybe due to numerical error in extraction
                # Try to polish? Or just return None.
                pass
                
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140
    n, m, p = DIMENSIONS
    
    # Fallback to rank 88
    U, V, W = _build_rank88()
    
    # Attempt to find better
    if time.time() < deadline - 10:
        reduced = _try_reduce_rank(U, V, W, n, m, p, deadline)
        if reduced is not None:
            U, V, W = reduced
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

