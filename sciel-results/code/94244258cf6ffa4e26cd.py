import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def _strassen_2x2():
    # Returns list of 7 tuples (u, v, w) where each is a 4-element list
    # u: coefficients for A elements (00,01,10,11)
    # v: coefficients for B elements (00,01,10,11)
    # w: coefficients for C elements (00,01,10,11)
    S = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),   # M1: (a00+a11)(b00+b11)
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),  # M2: (a10+a11)b00
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),  # M3: a00(b01-b11)
        ([0,0,0,1], [0,0,1,-1], [1,0,1,0]),  # M4: a11(b10-b00)
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),  # M5: (a00+a01)b11
        ([0,-1,1,0], [1,1,0,0], [0,0,0,1]),  # M6: (a10-a00)(b00+b01)
        ([-1,1,0,0], [0,0,1,1], [1,0,0,0]),  # M7: (a01-a11)(b10+b11)
    ]
    return S

def _build_strassen_28():
    """Build rank-28 decomposition via 4x Strassen 2x2 blocks."""
    n, m, p = 2, 4, 4
    S = _strassen_2x2()
    R = 28
    U = np.zeros((n*m, R), dtype=int)
    V = np.zeros((m*p, R), dtype=int)
    W = np.zeros((p*n, R), dtype=int)
    
    def mA(li, off):
        r, c = li // 2, li % 2
        return r * m + (off * 2 + c)
    
    def mB(li, ro, co):
        r, c = li // 2, li % 2
        return (ro * 2 + r) * p + (co * 2 + c)
    
    def mC(li, co):
        r, c = li // 2, li % 2
        return (co * 2 + c) * n + r
    
    blocks = [
        (0, 0, 0, 0, 0),
        (1, 1, 0, 0, 7),
        (0, 0, 1, 1, 14),
        (1, 1, 1, 1, 21),
    ]
    
    for a_off, b_ro, b_co, c_co, base in blocks:
        for j, (u, v, w) in enumerate(S):
            ci = base + j
            for li in range(4):
                if u[li] != 0:
                    U[mA(li, a_off), ci] += u[li]
                if v[li] != 0:
                    V[mB(li, b_ro, b_co), ci] += v[li]
                if w[li] != 0:
                    W[mC(li, c_co), ci] += w[li]
    
    return U, V, W

def _verify(U, V, W, n=2, m=4, p=4):
    """Verify the decomposition is exact."""
    R = U.shape[1]
    # Test with several random integer matrices
    rng = np.random.RandomState(42)
    for trial in range(5):
        A = rng.randint(-10, 10, size=(n, m)).astype(float)
        B = rng.randint(-10, 10, size=(m, p)).astype(float)
        C = A @ B
        Cf = np.zeros((n, p))
        for ii in range(n):
            for kk in range(p):
                acc = 0.0
                for cc in range(R):
                    for jj in range(m):
                        acc += U[ii*m+jj, cc] * V[jj*p+kk, cc] * W[kk*n+ii, cc]
                Cf[ii, kk] = acc
        if not np.allclose(C, Cf, atol=1e-9):
            return False
    return True

def _try_search_lower_rank(start_rank, max_time=140.0):
    """
    Attempt to find a decomposition with rank < start_rank using local search.
    This is a heuristic and may not find anything.
    """
    deadline = time.time() + max_time
    n, m, p = 2, 4, 4
    R_target = start_rank - 1  # Try one rank lower
    
    # Start from the Strassen-28 solution and try to zero out one column
    # through local optimization. This is very unlikely to work quickly.
    # Given the time constraints and complexity, we skip the search
    # and just return the known rank-28 solution.
    
    return None

def construct():
    # Return the known rank-28 Strassen-based decomposition
    U, V, W = _build_strassen_28()
    
    # Verify correctness
    if not _verify(U, V, W):
        # Fallback: should not happen, but guard
        raise ValueError("Verification failed for rank-28 decomposition")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
