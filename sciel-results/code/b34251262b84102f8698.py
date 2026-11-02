"""Matrix multiplication 3x3x5 rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 5]


def _verify(U, V, W, n=3, m=3, p=5, tol=1e-9):
    """Verify decomposition: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    return np.allclose(T, 1.0, atol=tol, rtol=tol)


def _schoolbook(n, m, p):
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


def _try_rank1(n, m, p):
    """Try rank-1: U[i*m+j,0]=1, V[j*p+k,0]=1, W[k*n+i,0]=1."""
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 115.0  # 5s safety margin before 120s kill

    # Try rank 1 first
    cand = _try_rank1(n, m, p)
    if _verify(cand["U"], cand["V"], cand["W"], n, m, p):
        return cand

    # Try rank 2
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r=0,1 = 1
    # Try: U[:,0]=1, V[:,0]=1, W[:,0]=0; U[:,1]=0, V[:,1]=0, W[:,1]=1 -> gives 0
    # That doesn't work. Need a proper search.
    # For rank 2: need a,b,c,d,e,f,g,h,i such that... this is complex.
    # Let's just fall back to schoolbook if rank 1 fails.

    # Actually, let's try a few small ranks with a simple heuristic
    for target_rank in range(2, min(10, n * m * p)):
        if time.time() > deadline:
            break
        # Simple attempt: try to find rank-r by random initialization + verification
        # This is very unlikely to work for small r, but let's try
        for _ in range(200):
            if time.time() > deadline:
                break
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fall back to schoolbook
    return _schoolbook(n, m, p)
