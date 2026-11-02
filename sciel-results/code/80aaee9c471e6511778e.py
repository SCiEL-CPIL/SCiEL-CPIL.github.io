import numpy as np
import time

def _build_rank14():
    """Construct the rank-14 Strassen-based decomposition."""
    n, m, p = 2, 2, 4
    rank = 14
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 7 terms for 2x2*2x2
    # A indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # B local: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # C local: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    
    terms = [
        # (U_coeffs, V_coeffs, W_coeffs) each is list of (flat_idx, coeff)
        # M1 = (a00+a11)(b00+b11)
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        # M2 = (a10+a11)b00
        ([(2,1),(3,1)], [(0,1)], [(1,1),(3,-1)]),
        # M3 = a00(b01-b11)
        ([(0,1)], [(1,1),(3,-1)], [(2,1),(3,1)]),
        # M4 = a11(b10-b00)
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(1,1)]),
        # M5 = (a00+a01)b11
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(2,1)]),
        # M6 = (a10-a00)(b00+b01)
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        # M7 = (a01-a11)(b10+b11)
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]
    
    for block in range(2):
        base = block * 7
        for t in range(7):
            r = base + t
            uc, vc, wc = terms[t]
            for (idx, val) in uc:
                U[idx, r] += val
            for (idx, val) in vc:
                j, k_loc = idx // 2, idx % 2
                k = k_loc + block * 2
                V[j * p + k, r] += val
            for (idx, val) in wc:
                k_loc, i = idx // 2, idx % 2
                k = k_loc + block * 2
                W[k * n + i, r] += val
    return U, V, W

def _verify(U, V, W, tol=1e-9):
    """Check if decomposition is exact: M[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[i,k,r] == 1"""
    n, m, p = 2, 2, 4
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_rank13(time_budget):
    """Attempt to find rank-13 decomposition via alternating minimization."""
    n, m, p = 2, 2, 4
    rank = 13
    start = time.time()
    
    # Build target tensor: M[i,j,k] = 1 for all i,j,k
    # Reshape into matrix for least squares: for fixed W, solve for U and V jointly
    # Actually, let's do: fix two, solve for third
    
    # Initialize with random values
    np.random.seed(42)
    U = np.random.randn(n*m, rank) * 0.5
    V = np.random.randn(m*p, rank) * 0.5
    W = np.random.randn(p*n, rank) * 0.5
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(2000):
        if time.time() - start > time_budget:
            break
        
        # Solve for U: fix V, W. For each (i,j), we need:
        # sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for all k
        # This is: U[i*m+j, :] @ diag-like = ones(p)
        # More precisely: for each (i,j), define vector v_{i,j} of length rank:
        #   v_{i,j}[r] = sum_k ... no.
        # 
        # The constraint is: for all (i,j,k): sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1
        #
        # Fix V and W. For a given (i,j), the equation over k is:
        #   sum_r U[i*m+j,r] * (sum over k... no, it's per k)
        #
        # Actually for fixed (i,j), we have p equations (one per k) in rank unknowns U[i*m+j,:]:
        #   For each k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        #
        # Let A_{i,j} be a p x rank matrix where A_{i,j}[k, r] = V[j*p+k, r] * W[k*n+i, r]
        # Then we need A_{i,j} @ U[i*m+j, :]^T = ones(p)
        #
        # Similarly fix U and W, solve for V: for each (j,k), m equations
        # Fix U and V, solve for W: for each (i,k), m equations
        
        # Solve U
        for i in range(n):
            for j in range(m):
                Aij = np.zeros((p, rank))
                for k in range(p):
                    Aij[k, :] = V[j*p+k, :] * W[k*n+i, :]
                # Least squares: Aij @ U_col = ones(p)
                try:
                    U[i*m+j, :], _, _, _ = np.linalg.lstsq(Aij, np.ones(p), rcond=None)
                except:
                    pass
        
        # Solve V
        for j in range(m):
            for k in range(p):
                Ajk = np.zeros((n, rank))
                for i in range(n):
                    Ajk[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    V[j*p+k, :], _, _, _ = np.linalg.lstsq(Ajk, np.ones(n), rcond=None)
                except:
                    pass
        
        # Solve W
        for i in range(n):
            for k in range(p):
                Aik = np.zeros((m, rank))
                for j in range(m):
                    Aik[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    W[k*n+i, :], _, _, _ = np.linalg.lstsq(Aik, np.ones(m), rcond=None)
                except:
                    pass
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    err += (s - 1.0)**2
        err = np.sqrt(err)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_err < 1e-10:
            return best_U, best_V, best_W
    
    if best_err < 1e-10:
        return best_U, best_V, best_W
    return None

def construct():
    # Rank 13 is not achievable with integer/half-integer coefficients for this tensor
    # (Rank 14 is the direct sum of two rank-7 Strassen decompositions).
    # We directly construct the rank-14 solution which uses integer coefficients.
    U, V, W = _build_rank14()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
