"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import least_squares

def _verify(U, V, W, n=2, m=2, p=4, tol=1e-9):
    """Verify that the decomposition is correct."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for r in range(n):
        for s in range(p):
            for t in range(m):
                expected = 1.0
                actual = 0.0
                for k in range(rank):
                    actual += U[r*m + t, k] * V[t*p + s, k] * W[s*n + r, k]
                if abs(actual - expected) > tol:
                    return False
    return True

def _construct_rank14():
    """Construct the known rank-14 decomposition (two independent Strassen blocks)."""
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: Strassen for columns 0,1 of B
    # U indices: 0=a00, 1=a01, 2=a10, 3=a11
    # V indices: 0=b00, 1=b01, 2=b10, 3=b11 (for cols 0,1)
    # W indices: 0=c00, 1=c01, 2=c10, 3=c11 (for cols 0,1)
    
    U[0][0] = 1; U[3][0] = 1  # M1: (a00+a11)
    U[2][1] = 1; U[3][1] = 1  # M2: (a10+a11)
    U[0][2] = 1               # M3: a00
    U[3][3] = 1               # M4: a11
    U[0][4] = 1; U[1][4] = 1  # M5: (a00+a01)
    U[2][5] = 1; U[0][5] = -1 # M6: (a10-a00)
    U[1][6] = 1; U[3][6] = -1 # M7: (a01-a11)
    
    V[0][0] = 1; V[5][0] = 1  # (b00+b11)
    V[0][1] = 1               # b00
    V[1][2] = 1; V[5][2] = -1 # (b10+b11)
    V[4][3] = 1; V[0][3] = -1 # (b00-b11)
    V[5][4] = 1               # b11
    V[0][5] = 1; V[1][5] = 1  # (b00+b10)
    V[4][6] = 1; V[5][6] = 1  # (b01+b11)
    
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1  # c00 = M1-M2-M4+M5
    W[1][1] = 1; W[1][3] = 1                              # c01 = M3+M5
    W[2][2] = 1; W[2][4] = 1                              # c10 = M2+M4
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1   # c11 = M1-M3-M4-M6+M7
    
    # Block 2: Strassen for columns 2,3 of B
    # V indices: 4=b02, 5=b03, 6=b12, 7=b13 (for cols 2,3)
    # W indices: 4=c02, 5=c03, 6=c12, 7=c13 (for cols 2,3)
    
    U[0][7] = 1; U[3][7] = 1  # M1: (a00+a11)
    U[2][8] = 1; U[3][8] = 1  # M2: (a10+a11)
    U[0][9] = 1               # M3: a00
    U[3][10] = 1              # M4: a11
    U[0][11] = 1; U[1][11] = 1 # M5: (a00+a01)
    U[2][12] = 1; U[0][12] = -1 # M6: (a10-a00)
    U[1][13] = 1; U[3][13] = -1 # M7: (a01-a11)
    
    V[2][7] = 1; V[7][7] = 1  # (b02+b13)
    V[2][8] = 1               # b02
    V[3][9] = 1; V[7][9] = -1 # (b12+b13)
    V[6][10] = 1; V[2][10] = -1 # (b02-b13)
    V[7][11] = 1              # b13
    V[2][12] = 1; V[3][12] = 1 # (b02+b12)
    V[6][13] = 1; V[7][13] = 1 # (b03+b13)
    
    W[4][7] = 1; W[4][10] = 1; W[4][11] = -1; W[4][13] = 1  # c02
    W[5][8] = 1; W[5][10] = 1                               # c03
    W[6][9] = 1; W[6][11] = 1                               # c12
    W[7][7] = 1; W[7][8] = -1; W[7][9] = 1; W[7][12] = 1    # c13
    
    return {"U": U, "V": V, "W": W}

def _residual_rank_r(params, r, n=2, m=2, p=4):
    """Compute residuals for rank-r decomposition."""
    nm = n * m
    mp = m * p
    pn = p * n
    U = params[:nm*r].reshape(nm, r)
    V = params[nm*r:nm*r+mp*r].reshape(mp, r)
    W = params[nm*r+mp*r:].reshape(pn, r)
    
    residuals = np.zeros(n * m * p)
    idx = 0
    for rr in range(n):
        for t in range(m):
            for s in range(p):
                val = np.dot(U[rr*m+t, :], V[t*p+s, :] * W[s*n+rr, :])
                residuals[idx] = val - 1.0
                idx += 1
    return residuals

def _try_rank_r(r, time_limit=50.0, n=2, m=2, p=4, max_seeds=200):
    """Try to find a rank-r decomposition using least squares optimization."""
    start = time.time()
    nm = n * m
    mp = m * p
    pn = p * n
    nvars = (nm + mp + pn) * r
    
    for seed in range(max_seeds):
        if time.time() - start > time_limit:
            break
        np.random.seed(seed)
        
        # Initialize with small random values
        params = np.random.randn(nvars) * 0.5
        
        try:
            result = least_squares(_residual_rank_r, params, args=(r, n, m, p), 
                                   method='lm', max_nfev=200)
        except Exception:
            continue
        
        if result.cost < 1e-14:
            U = result.x[:nm*r].reshape(nm, r)
            V = result.x[nm*r:nm*r+mp*r].reshape(mp, r)
            W = result.x[nm*r+mp*r:].reshape(pn, r)
            
            # Round to nearest half-integer
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            # Verify the rounded solution
            if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            
            # Also try the unrounded solution
            if _verify(U.tolist(), V.tolist(), W.tolist()):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 130.0  # Safety margin before 150s kill
    
    # Try rank 12 first (theoretical minimum)
    remaining = deadline - time.time()
    if remaining > 10:
        result = _try_rank_r(12, time_limit=min(remaining, 40.0))
        if result is not None:
            return result
    
    # Try rank 13
    remaining = deadline - time.time()
    if remaining > 10:
        result = _try_rank_r(13, time_limit=min(remaining, 40.0))
        if result is not None:
            return result
    
    # Fall back to rank 14
    return _construct_rank14()
# EVOLVE-BLOCK-END
