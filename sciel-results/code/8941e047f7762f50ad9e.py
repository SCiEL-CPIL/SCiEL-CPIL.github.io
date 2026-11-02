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

def _residual_norm(U, V, W):
    """Compute the residual norm of the tensor decomposition."""
    n, m, p = DIMENSIONS
    rank = len(W[0])
    residual = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                residual += (val - 1.0) ** 2
    return np.sqrt(residual)

def _try_rank_10_structured():
    """
    Try to find a rank-10 decomposition using structured parameterization.
    
    Strategy: Start from the rank-11 solution and try to merge two of the schoolbook
    terms for column 2 with one of the Strassen terms by introducing a shared factor.
    
    Key idea: In the rank-11 solution, terms 8-11 are schoolbook for column 2:
      M8: A[0,0]*B[0,2] -> C[0,2]
      M9: A[0,1]*B[1,2] -> C[0,2]  
      M10: A[1,0]*B[0,2] -> C[1,2]
      M11: A[1,1]*B[1,2] -> C[1,2]
    
    We can try to express these using fewer terms by sharing linear combinations.
    For example, use (A[0,0]+A[1,0])*B[0,2] and (A[0,1]+A[1,1])*B[1,2]
    but then we need to separate the contributions to C[0,2] and C[1,2].
    
    Alternative: Use a Strassen-like trick on the 2x2x3 problem directly.
    """
    n, m, p = DIMENSIONS
    rank = 10
    
    # Parameterized approach: 
    # Terms 0-6: Strassen for columns 0,1 (same as rank-11)
    # Terms 7-9: Three new terms for column 2, using linear combinations
    
    # Let's try: 
    # M7: (a+c) * b02 -> contributes to C[0,2] and C[1,2]
    # M8: (b+d) * b12 -> contributes to C[0,2] and C[1,2]
    # M9: (a-b) * (b02-b12) -> correction term
    
    # But this needs to give exactly:
    # C[0,2] = a*b02 + b*b12
    # C[1,2] = c*b02 + d*b12
    
    # Let's set up:
    # M7: U[0]=1, U[2]=1, V[2]=1, W[4]=x, W[5]=y  -> (a+c)*b02
    # M8: U[1]=1, U[3]=1, V[5]=1, W[4]=z, W[5]=w  -> (b+d)*b12
    # M9: U[0]=1, U[1]=-1, V[2]=1, V[5]=-1, W[4]=p, W[5]=q -> (a-b)*(b02-b12)
    
    # C[0,2] = x*(a+c)*b02 + z*(b+d)*b12 + p*(a-b)*(b02-b12)
    #         = (x + p)*a*b02 + x*c*b02 + z*b*b12 + z*d*b12 - p*a*b12 + p*b*b12
    #         = a*b02*(x+p) + c*b02*x + b*b12*(z+p) + d*b12*z - a*b12*p
    # Need: a*b02 + b*b12
    # So: x+p = 1, x = 0, z+p = 1, z = 0, -p = 0 -> p=0, x=1, z=1
    # But then c*b02*1 + d*b12*1 remain, which is wrong.
    
    # This doesn't work directly. Let's try a different parameterization.
    
    # Try: 
    # M7: a*b02 (U[0]=1, V[2]=1, W[4]=1)
    # M8: b*b12 (U[1]=1, V[5]=1, W[4]=1)
    # M9: c*b02 + d*b12 (U[2]=1, U[3]=1, V[2]=1, V[5]=1, W[5]=1)
    #     This gives (c+d)*b02 + (c+d)*b12, not what we want.
    
    # The issue is that we need to separate the contributions.
    
    # Let's try a completely different approach: use optimization.
    start_time = time.time()
    deadline = start_time + 60  # Leave time for other methods
    
    best_residual = np.inf
    best_params = None
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        rng = np.random.default_rng(attempt * 7919 + 42)
        
        # Initialize with the rank-11 structure, then perturb
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Copy Strassen terms 0-6
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[4,0] = 1
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        U[0,2] = 1
        V[1,2] = 1; V[4,2] = -1
        U[3,3] = 1
        V[3,3] = 1; V[0,3] = -1
        U[0,4] = 1; U[1,4] = 1
        V[4,4] = 1
        U[2,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[1,5] = 1
        U[1,6] = 1; U[3,6] = -1
        V[3,6] = 1; V[4,6] = 1
        
        W[0,0] = 1; W[0,3] = 1; W[0,4] = -1; W[0,6] = 1
        W[2,2] = 1; W[2,4] = 1
        W[1,1] = 1; W[1,3] = 1
        W[3,0] = 1; W[3,1] = -1; W[3,2] = 1; W[3,5] = 1
        
        # Now try different parameterizations for terms 7-9
        # Option 1: Perturb the schoolbook terms
        if attempt % 4 == 0:
            # Start from schoolbook and add small perturbations
            U[0,7] = 1 + rng.normal(0, 0.1)
            V[2,7] = 1 + rng.normal(0, 0.1)
            W[4,7] = 1 + rng.normal(0, 0.1)
            
            U[1,8] = 1 + rng.normal(0, 0.1)
            V[5,8] = 1 + rng.normal(0, 0.1)
            W[4,8] = 1 + rng.normal(0, 0.1)
            
            U[2,9] = 1 + rng.normal(0, 0.1)
            V[2,9] = 1 + rng.normal(0, 0.1)
            W[5,9] = 1 + rng.normal(0, 0.1)
            
            U[3,9] = rng.normal(0, 0.1)
            V[5,9] += rng.normal(0, 0.1)
            W[4,9] = rng.normal(0, 0.1)
            W[5,9] += rng.normal(0, 0.1)
            
        # Option 2: Use linear combinations
        elif attempt % 4 == 1:
            # M7: (a + alpha*c) * b02
            U[0,7] = 1
            U[2,7] = rng.normal(0, 1)
            V[2,7] = 1
            W[4,7] = 1
            W[5,7] = rng.normal(0, 0.5)
            
            # M8: (b + beta*d) * b12
            U[1,8] = 1
            U[3,8] = rng.normal(0, 1)
            V[5,8] = 1
            W[4,8] = 1
            W[5,8] = rng.normal(0, 0.5)
            
            # M9: correction
            U[0,9] = rng.normal(0, 1)
            U[1,9] = rng.normal(0, 1)
            U[2,9] = rng.normal(0, 1)
            U[3,9] = rng.normal(0, 1)
            V[2,9] = rng.normal(0, 1)
            V[5,9] = rng.normal(0, 1)
            W[4,9] = rng.normal(0, 1)
            W[5,9] = rng.normal(0, 1)
            
        # Option 3: Fully random sparse
        elif attempt % 4 == 2:
            for r in range(7, 10):
                for idx in rng.choice(n*m, size=rng.integers(1, 3), replace=False):
                    U[idx, r] = rng.normal(0, 1)
                for idx in rng.choice(m*p, size=rng.integers(1, 3), replace=False):
                    V[idx, r] = rng.normal(0, 1)
                for idx in rng.choice(p*n, size=rng.integers(1, 3), replace=False):
                    W[idx, r] = rng.normal(0, 1)
            
        # Option 4: Integer-based search
        else:
            for r in range(7, 10):
                for idx in rng.choice(n*m, size=rng.integers(1, 3), replace=False):
                    U[idx, r] = rng.choice([-2, -1, 1, 2])
                for idx in rng.choice(m*p, size=rng.integers(1, 3), replace=False):
                    V[idx, r] = rng.choice([-2, -1, 1, 2])
                for idx in rng.choice(p*n, size=rng.integers(1, 3), replace=False):
                    W[idx, r] = rng.choice([-2, -1, 1, 2])
        
        # Compute residual
        res = _residual_norm(U, V, W)
        if res < best_residual:
            best_residual = res
            best_params = (U.copy(), V.copy(), W.copy())
        
        # If residual is very small, try to polish
        if res < 1e-6:
            # Try to find exact solution by rounding
            U_round = np.round(U)
            V_round = np.round(V)
            W_round = np.round(W)
            res_round = _residual_norm(U_round, V_round, W_round)
            if res_round < 1e-10:
                return U_round.tolist(), V_round.tolist(), W_round.tolist()
    
    return None

def _try_rank_10_homotopy():
    """
    Try homotopy continuation from the 2x2x4 tensor (rank 7) to 2x2x3.
    
    The 2x2x4 multiplication tensor has rank 7 (Strassen).
    We can view 2x2x3 as a specialization where one column is zero.
    This might allow us to find a rank-10 solution by perturbing the 2x2x4
    decomposition and then projecting.
    
    However, this is complex. Instead, we'll try a direct numerical approach.
    """
    n, m, p = DIMENSIONS
    rank = 10
    
    start_time = time.time()
    deadline = start_time + 40
    
    # Use scipy's least_squares if available
    try:
        from scipy.optimize import least_squares
    except ImportError:
        return None
    
    def _flatten(U, V, W):
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def _unflatten(x):
        nU = n*m*rank
        nV = m*p*rank
        U = x[:nU].reshape(n*m, rank)
        V = x[nU:nU+nV].reshape(m*p, rank)
        W = x[nU+nV:].reshape(p*n, rank)
        return U, V, W
    
    def _residual_vec(x):
        U, V, W = _unflatten(x)
        res = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                    res.append(val - 1.0)
        return np.array(res)
    
    best_residual = np.inf
    best_params = None
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        rng = np.random.default_rng(attempt * 104729 + 42)
        
        # Initialize near the rank-11 solution
        x0 = np.zeros(n*m*rank + m*p*rank + p*n*rank)
        
        # Set up Strassen terms
        U_init = np.zeros((n*m, rank))
        V_init = np.zeros((m*p, rank))
        W_init = np.zeros((p*n, rank))
        
        U_init[0,0] = 1; U_init[3,0] = 1
        V_init[0,0] = 1; V_init[4,0] = 1
        U_init[2,1] = 1; U_init[3,1] = 1
        V_init[0,1] = 1
        U_init[0,2] = 1
        V_init[1,2] = 1; V_init[4,2] = -1
        U_init[3,3] = 1
        V_init[3,3] = 1; V_init[0,3] = -1
        U_init[0,4] = 1; U_init[1,4] = 1
        V_init[4,4] = 1
        U_init[2,5] = 1; U_init[0,5] = -1
        V_init[0,5] = 1; V_init[1,5] = 1
        U_init[1,6] = 1; U_init[3,6] = -1
        V_init[3,6] = 1; V_init[4,6] = 1
        
        W_init[0,0] = 1; W_init[0,3] = 1; W_init[0,4] = -1; W_init[0,6] = 1
        W_init[2,2] = 1; W_init[2,4] = 1
        W_init[1,1] = 1; W_init[1,3] = 1
        W_init[3,0] = 1; W_init[3,1] = -1; W_init[3,2] = 1; W_init[3,5] = 1
        
        # Random initialization for terms 7-9
        for r in range(7, 10):
            U_init[:, r] = rng.normal(0, 0.5, n*m)
            V_init[:, r] = rng.normal(0, 0.5, m*p)
            W_init[:, r] = rng.normal(0, 0.5, p*n)
        
        x0 = _flatten(U_init, V_init, W_init)
        
        try:
            result = least_squares(_residual_vec, x0, method='lm', max_nfev=1000)
            res = np.sqrt(np.sum(result.fun**2))
            
            if res < best_residual:
                best_residual = res
                U, V, W = _unflatten(result.x)
                best_params = (U.copy(), V.copy(), W.copy())
            
            if res < 1e-8:
                # Try to round
                U_round = np.round(U)
                V_round = np.round(V)
                W_round = np.round(W)
                res_round = _residual_norm(U_round, V_round, W_round)
                if res_round < 1e-10:
                    return U_round.tolist(), V_round.tolist(), W_round.tolist()
                
                # Try half-integer rounding
                U_half = np.round(U * 2) / 2
                V_half = np.round(V * 2) / 2
                W_half = np.round(W * 2) / 2
                res_half = _residual_norm(U_half, V_half, W_half)
                if res_half < 1e-10:
                    return U_half.tolist(), V_half.tolist(), W_half.tolist()
                    
        except Exception:
            pass
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Try structured rank-10 search
    result_10 = _try_rank_10_structured()
    if result_10 is not None:
        U, V, W = result_10
        if _verify(U, V, W):
            return {"U": U, "V": V, "W": W}
    
    # Try homotopy-based rank-10 search
    if time.time() < deadline - 20:
        result_10h = _try_rank_10_homotopy()
        if result_10h is not None:
            U, V, W = result_10h
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
