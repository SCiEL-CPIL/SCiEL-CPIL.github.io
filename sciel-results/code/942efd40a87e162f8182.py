import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def _strassen_2x2x2():
    s = [
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(0,1):1,(1,1):-1}),
        ({(0,0):1}, {(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}),
        ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}),
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(1,0):1}),
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
    ]
    return s

def _schoolbook_1x2x2():
    s = [
        ({(0,0):1}, {(0,0):1}, {(0,0):1}),
        ({(0,1):1}, {(1,0):1}, {(0,0):1}),
        ({(0,0):1}, {(0,1):1}, {(1,0):1}),
        ({(0,1):1}, {(1,1):1}, {(1,0):1}),
    ]
    return s

def _build_schoolbook_48():
    """Build the full 48-term schoolbook decomposition for 3x4x4."""
    n, m, p = 3, 4, 4
    U = np.zeros((n*m, 48), dtype=np.float64)
    V = np.zeros((m*p, 48), dtype=np.float64)
    W = np.zeros((p*n, 48), dtype=np.float64)
    rc = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, rc] = 1.0
                V[j*p+k, rc] = 1.0
                W[k*n+i, rc] = 1.0
                rc += 1
    return U, V, W

def _find_linear_dependency(U, V, W, tol=1e-10):
    """
    Check if there's a linear dependency among the 48 schoolbook terms.
    A linear combination sum_r c_r * (u_r ⊗ v_r ⊗ w_r) = 0
    means for all (i,j,k): sum_r c_r * U[i,j,r] * V[j,k,r] * W[k,i,r] = 0
    
    This is a system of 48 equations in 48 unknowns c_r.
    If the system has a non-trivial solution, we can reduce the rank.
    """
    n, m, p = 3, 4, 4
    rank = U.shape[1]
    # Build the 48x48 matrix where entry (ijk, r) = U[i,j,r] * V[j,k,r] * W[k,i,r]
    A = np.zeros((n*m*p, rank), dtype=np.float64)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                A[idx, :] = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                idx += 1
    # Find the null space of A
    try:
        from scipy.linalg import null_space
        null = null_space(A, rcond=tol)
        if null.shape[1] > 0:
            return null
    except Exception:
        pass
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110
    n, m, p = 3, 4, 4
    strassen = _strassen_2x2x2()
    schoolbook = _schoolbook_1x2x2()
    
    # First, try to find if we can reduce the schoolbook part
    U48, V48, W48 = _build_schoolbook_48()
    null = _find_linear_dependency(U48, V48, W48)
    
    if null is not None and null.shape[1] > 0:
        # We found a dependency - use it to reduce rank
        # For now, just note it but use the standard decomposition
        # A proper implementation would project out the dependency
        pass
    
    total_rank = 4 * len(strassen) + 4 * len(schoolbook)  # 28 + 16 = 44
    U = np.zeros((n*m, total_rank), dtype=np.float64)
    V = np.zeros((m*p, total_rank), dtype=np.float64)
    W = np.zeros((p*n, total_rank), dtype=np.float64)
    rc = 0
    def _emit(i_off, j_off, k_off, terms):
        nonlocal rc
        for (ud, vd, wd) in terms:
            for (li, lj), c in ud.items():
                U[(i_off + li) * m + (j_off + lj), rc] += c
            for (lj, lk), c in vd.items():
                V[(j_off + lj) * p + (k_off + lk), rc] += c
            for (lk, li), c in wd.items():
                W[(k_off + lk) * n + (i_off + li), rc] += c
            rc += 1
    _emit(0, 0, 0, strassen)
    _emit(0, 2, 0, strassen)
    _emit(0, 0, 2, strassen)
    _emit(0, 2, 2, strassen)
    if time.time() > deadline - 5:
        return {"U": U[:, :rc].tolist(), "V": V[:, :rc].tolist(), "W": W[:, :rc].tolist()}
    _emit(2, 0, 0, schoolbook)
    _emit(2, 2, 0, schoolbook)
    _emit(2, 0, 2, schoolbook)
    _emit(2, 2, 2, schoolbook)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
