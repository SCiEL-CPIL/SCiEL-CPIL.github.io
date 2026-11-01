"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W):
    """Verify that the decomposition exactly matches the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = len(W[0])
    rng = np.random.default_rng(12345)
    A = rng.standard_normal((n, m))
    B = rng.standard_normal((m, p))
    C_expected = A @ B
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            for r in range(rank):
                lA = sum(U[i*m+j, r] * A[i, j] for j in range(m))
                lB = sum(V[j*p+k, r] * B[j, k] for j in range(m))
                C_computed[i, k] += W[k*n+i, r] * lA * lB
    return np.allclose(C_computed, C_expected, atol=1e-10)

def _check_decomposition(U, V, W, n, m, p):
    """Check if the decomposition satisfies all 12 tensor equations."""
    rank = len(W[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(val - 1) > 1e-8:
                    return False
    return True

def _try_rank_10():
    """Try to find a rank-10 decomposition via structured search."""
    n, m, p = DIMENSIONS
    rank = 10
    start_time = time.time()
    deadline = start_time + 90  # Leave time for verification and fallback
    
    # Strategy: Warm-start from rank-11, try to merge two schoolbook terms
    # by adjusting Strassen terms to absorb the third column contributions.
    
    # Base rank-11 solution
    U11 = np.zeros((n*m, 11))
    V11 = np.zeros((m*p, 11))
    W11 = np.zeros((p*n, 11))
    
    # Strassen part (terms 0-6)
    U11[0,0]=1; U11[3,0]=1; V11[0,0]=1; V11[4,0]=1
    U11[2,1]=1; U11[3,1]=1; V11[0,1]=1
    U11[0,2]=1; V11[1,2]=1; V11[4,2]=-1
    U11[3,3]=1; V11[3,3]=1; V11[0,3]=-1
    U11[0,4]=1; U11[1,4]=1; V11[4,4]=1
    U11[2,5]=1; U11[0,5]=-1; V11[0,5]=1; V11[1,5]=1
    U11[1,6]=1; U11[3,6]=-1; V11[3,6]=1; V11[4,6]=1
    
    W11[0,0]=1; W11[0,3]=1; W11[0,4]=-1; W11[0,6]=1
    W11[2,2]=1; W11[2,4]=1
    W11[1,1]=1; W11[1,3]=1
    W11[3,0]=1; W11[3,1]=-1; W11[3,2]=1; W11[3,5]=1
    
    # Schoolbook for col 2 (terms 7-10)
    U11[0,7]=1; V11[2,7]=1; W11[4,7]=1
    U11[1,8]=1; V11[5,8]=1; W11[4,8]=1
    U11[2,9]=1; V11[2,9]=1; W11[5,9]=1
    U11[3,10]=1; V11[5,10]=1; W11[5,10]=1
    
    # Try to find rank-10 by using scipy optimization
    try:
        from scipy.optimize import minimize
    except ImportError:
        return None
    
    # Parameterize: take first 10 columns of a perturbed rank-11
    # We'll optimize over small perturbations of the 10-term subset
    
    # Try different ways to reduce from 11 to 10:
    # Option: merge two of the schoolbook terms by allowing Strassen terms
    # to have non-zero entries in the column-2 positions of V.
    
    # Let's try: keep terms 0-6 (Strassen) and terms 7-9 (three of four schoolbook)
    # and allow V[2,7], V[5,7], etc. to be non-zero to absorb the missing term.
    
    # Actually, let's try a direct optimization approach:
    # Minimize sum of squared residuals over the 12 equations
    
    def objective(params):
        """params is a flat vector of size 4*10 + 6*10 + 6*10 = 160"""
        U_flat = params[0:4*10]
        V_flat = params[4*10:4*10+6*10]
        W_flat = params[4*10+6*10:4*10+6*10+6*10]
        
        U = U_flat.reshape((4, 10))
        V = V_flat.reshape((6, 10))
        W = W_flat.reshape((6, 10))
        
        residuals = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                    residuals.append(val - 1)
        return sum(x*x for x in residuals)
    
    # Initialize from rank-11 with last column zeroed and scaled
    U_init = U11[:, :10].copy()
    V_init = V11[:, :10].copy()
    W_init = W11[:, :10].copy()
    
    # Also try other initializations
    inits = []
    inits.append((U_init.flatten(), V_init.flatten(), W_init.flatten()))
    
    # Try removing different single terms and redistributing
    for skip in range(11):
        if time.time() > deadline:
            break
        U_s = np.delete(U11, skip, axis=1)
        V_s = np.delete(V11, skip, axis=1)
        W_s = np.delete(W11, skip, axis=1)
        inits.append((U_s.flatten(), V_s.flatten(), W_s.flatten()))
    
    best_result = None
    best_obj = float('inf')
    
    for init_idx, (uf, vf, wf) in enumerate(inits):
        if time.time() > deadline:
            break
        
        params0 = np.concatenate([uf, vf, wf])
        
        try:
            result = minimize(objective, params0, method='BFGS', 
                             options={'maxiter': 200, 'gtol': 1e-12})
            if result.fun < best_obj:
                best_obj = result.fun
                best_result = result.x
        except Exception:
            continue
    
    if best_result is not None and best_obj < 1e-8:
        U_flat = best_result[0:4*10]
        V_flat = best_result[4*10:4*10+6*10]
        W_flat = best_result[4*10+6*10:4*10+6*10+6*10]
        U = U_flat.reshape((4, 10))
        V = V_flat.reshape((6, 10))
        W = W_flat.reshape((6, 10))
        
        if _check_decomposition(U, V, W, n, m, p):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # First try rank 10
    result_10 = _try_rank_10()
    if result_10 is not None:
        U, V, W = result_10
        if _verify(U, V, W):
            return {"U": U, "V": V, "W": W}
    
    # Fall back to rank 11 (Strassen + schoolbook)
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen-style part for the 2x2 block (columns 0 and 1 of B)
    # M1 = (a+d)(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[4][0] = 1
    
    # M2 = (c+d)b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    
    # M3 = a(b01-b11)
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    
    # M4 = d(b10-b00)
    U[3][3] = 1
    V[3][3] = 1; V[0][3] = -1
    
    # M5 = (a+b)b11
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    
    # M6 = (c-a)(b00+b01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    
    # M7 = (b-d)(b10+b11)
    U[1][6] = 1; U[3][6] = -1
    V[3][6] = 1; V[4][6] = 1
    
    # W for Strassen outputs
    # C00 = M1 + M4 - M5 + M7
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    # C01 = M3 + M5
    W[2][2] = 1; W[2][4] = 1
    # C10 = M2 + M4
    W[1][1] = 1; W[1][3] = 1
    # C11 = M1 - M2 + M3 + M6
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    # Schoolbook part for column 2 of B
    # M8 = a * b02
    U[0][7] = 1
    V[2][7] = 1
    
    # M9 = b * b12
    U[1][8] = 1
    V[5][8] = 1
    
    # M10 = c * b02
    U[2][9] = 1
    V[2][9] = 1
    
    # M11 = d * b12
    U[3][10] = 1
    V[5][10] = 1
    
    # W for Schoolbook outputs
    # C02 = M8 + M9
    W[4][7] = 1; W[4][8] = 1
    # C12 = M10 + M11
    W[5][9] = 1; W[5][10] = 1
    
    # Verify the decomposition
    if not _verify(U, V, W):
        # Fallback to schoolbook if verification fails (should not happen)
        rank = 12
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
