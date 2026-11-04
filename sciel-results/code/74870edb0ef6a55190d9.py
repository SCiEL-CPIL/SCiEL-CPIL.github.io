"""Program for 4x4x5 matrix multiplication tensor decomposition."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _winograd_f4_4x4x4():
    """
    Winograd F(4) decomposition for 4x4x4 matrix multiplication.
    Returns 16 rank-1 terms as (A_dict, B_dict, C_dict).
    
    Uses the real Winograd F(4) algorithm based on the cyclic group Z_4.
    Transform matrices:
      C = TC @ ((TA @ A) * (B @ TB)) @ TC.T
    where * denotes elementwise (Hadamard) product.
    
    Each rank-1 term (a,b) for a,b in {0,1,2,3}:
      A part: TA[a, i] for each i (row a of TA, applied to columns of A)
      B part: TB[k, b] for each k (column b of TB, applied to rows of B)  
      C part: TC[i, a] * TC[k, b] for each (i,k)
    """
    import numpy as np
    
    TA = np.array([
        [1,  1,  1,  1],
        [1, -1,  1, -1],
        [1,  1, -1, -1],
        [1, -1, -1,  1],
    ], dtype=float)
    
    TB = np.array([
        [1,  1,  1,  1],
        [1, -1, -1,  1],
        [1,  1, -1, -1],
        [1, -1,  1, -1],
    ], dtype=float)
    
    TC = np.array([
        [1,  1,  1,  1],
        [1, -1,  1, -1],
        [1,  1, -1, -1],
        [1, -1, -1,  1],
    ], dtype=float)
    
    # Verify the transform identity
    A_test = np.random.rand(4,4)
    B_test = np.random.rand(4,4)
    C_expected = A_test @ B_test
    C_wino = TC @ ((TA @ A_test) * (B_test @ TB)) @ TC.T
    if not np.allclose(C_wino, C_expected, atol=1e-10):
        # Try alternative: C = TC @ (TA @ A) * (TB @ B) @ TC.T
        C_wino2 = TC @ ((TA @ A_test) * (TB @ B_test)) @ TC.T
        if not np.allclose(C_wino2, C_expected, atol=1e-10):
            # Try: C = TC @ (TA @ A * B @ TB) @ TC.T with different association
            C_wino3 = TC @ ((TA @ A_test) * (B_test @ TB).T) @ TC.T
            if not np.allclose(C_wino3, C_expected, atol=1e-10):
                # All failed, return None to signal fallback
                return None
    
    # Build the 16 rank-1 terms
    terms = []
    for a in range(4):
        for b in range(4):
            A_dict = {}
            B_dict = {}
            C_dict = {}
            
            # A part: (TA @ A)[a, j] = sum_i TA[a,i] * A[i,j]
            # So for rank-1 term (a,b), A[i,j] coefficient is TA[a,i]
            for i in range(4):
                for j in range(4):
                    coeff = TA[a, i]
                    if coeff != 0:
                        A_dict[(i, j)] = A_dict.get((i, j), 0) + coeff
            
            # B part: (B @ TB)[i, b] = sum_k B[i,k] * TB[k,b]
            # So for rank-1 term (a,b), B[i,k] coefficient is TB[k,b]
            for i in range(4):
                for k in range(4):
                    coeff = TB[k, b]
                    if coeff != 0:
                        B_dict[(i, k)] = B_dict.get((i, k), 0) + coeff
            
            # C part: C[i,k] gets contribution TC[i,a] * M[a,b] * TC[k,b]
            # where M[a,b] is the scalar product. So C[i,k] coefficient is TC[i,a]*TC[k,b]
            for i in range(4):
                for k in range(4):
                    coeff = TC[i, a] * TC[k, b]
                    if coeff != 0:
                        C_dict[(k, i)] = C_dict.get((k, i), 0) + coeff  # W index is k*n+i
            
            terms.append((A_dict, B_dict, C_dict))
    
    return terms

def _build_4x4x4_strassen():
    """Fallback: Strassen 4x4x4 with 49 rank-1 terms."""
    terms = []
    # 7x7 = 49 terms from composing two 2x2x2 Strassen
    s2 = [
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1}),
        ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1}),
        ({(1,1):1}, {(1,0):1,(0,0):-1}, {(1,0):1}),
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,1):1}),
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
    ]
    for oA, oB, oC in s2:
        for iA, iB, iC in s2:
            A = {}
            B = {}
            C = {}
            for (bI, bJ), oa in oA.items():
                for (li, lj), ia in iA.items():
                    gi, gj = bI*2+li, bJ*2+lj
                    A[(gi,gj)] = A.get((gi,gj),0) + oa*ia
            for (bJ, bK), ob in oB.items():
                for (lj, lk), ib in iB.items():
                    gj, gk = bJ*2+lj, bK*2+lk
                    B[(gj,gk)] = B.get((gj,gk),0) + ob*ib
            for (bK, bI), oc in oC.items():
                for (lk, li), ic in iC.items():
                    gk, gi = bK*2+lk, bI*2+li
                    C[(gk,gi)] = C.get((gk,gi),0) + oc*ic
            terms.append((A,B,C))
    return terms

def _decompose_terms(terms, n, m, p):
    """Build U, V, W from list of (A_dict, B_dict, C_dict) terms."""
    R = len(terms)
    U = [[0]*R for _ in range(n*m)]
    V = [[0]*R for _ in range(m*p)]
    W = [[0]*R for _ in range(p*n)]
    for c, (A, B, C) in enumerate(terms):
        for (i, j), val in A.items():
            if 0 <= i < n and 0 <= j < m:
                U[i*m+j][c] += val
        for (j, k), val in B.items():
            if 0 <= j < m and 0 <= k < p:
                V[j*p+k][c] += val
        for (k, i), val in C.items():
            if 0 <= k < p and 0 <= i < n:
                W[k*n+i][c] += val
    return U, V, W

def _schoolbook(n, m, p):
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, rank):
    """Verify the decomposition is correct."""
    import numpy as np
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_expected = A @ B
    Ua = np.array(U)
    Va = np.array(V)
    Wa = np.array(W)
    Cdec = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            s = 0.0
            for j in range(m):
                for c in range(rank):
                    s += Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
            Cdec[i, k] = s
    return np.allclose(Cdec, C_expected, atol=1e-9)

def construct():
    n, m, p = DIMENSIONS
    
    # Try Winograd F(4) for 4x4x4 (rank 16) + schoolbook for 5th column (rank 16) = rank 32
    try:
        terms = _winograd_f4_4x4x4()
        if terms is not None:
            U4, V4, W4 = _decompose_terms(terms, n, m, 4)
            R1 = len(terms)
            R2 = 16  # 4*4 terms for k=4
            rank = R1 + R2
            
            U = [row[:] for row in U4]
            V = [row[:] for row in V4]
            W = [row[:] for row in W4]
            for _ in range(R2):
                for row in U: row.append(0)
                for row in V: row.append(0)
                for row in W: row.append(0)
            
            # Add 16 schoolbook terms for the 5th column (k=4)
            t = 0
            for i in range(n):
                for j in range(m):
                    U[i*m+j][R1+t] = 1
                    V[j*p+4][R1+t] = 1
                    W[4*n+i][R1+t] = 1
                    t += 1
            
            if _verify(U, V, W, n, m, p, rank):
                return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback: Strassen 4x4x4 (rank 49) + schoolbook 5th column (rank 16) = rank 65
    try:
        terms = _build_4x4x4_strassen()
        U4, V4, W4 = _decompose_terms(terms, n, m, 4)
        R1 = len(terms)
        R2 = 16
        rank = R1 + R2
        
        U = [row[:] for row in U4]
        V = [row[:] for row in V4]
        W = [row[:] for row in W4]
        for _ in range(R2):
            for row in U: row.append(0)
            for row in V: row.append(0)
            for row in W: row.append(0)
        
        t = 0
        for i in range(n):
            for j in range(m):
                U[i*m+j][R1+t] = 1
                V[j*p+4][R1+t] = 1
                W[4*n+i][R1+t] = 1
                t += 1
        
        if _verify(U, V, W, n, m, p, rank):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Final fallback: full schoolbook rank 80
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
