import time
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    def strassen_decomp(row_off_a, col_off_a, row_off_b, col_off_b, U_t, V_t, W_t):
        """Add Strassen decomposition for 2x2*2x2 block at given offsets."""
        a = (row_off_a, col_off_a)
        b = (row_off_a, col_off_a + 1)
        c = (row_off_a + 1, col_off_a)
        d = (row_off_a + 1, col_off_a + 1)
        e = (row_off_b, col_off_b)
        f = (row_off_b, col_off_b + 1)
        g = (row_off_b + 1, col_off_b)
        h = (row_off_b + 1, col_off_b + 1)
        
        C00 = (row_off_a, col_off_b)
        C01 = (row_off_a, col_off_b + 1)
        C10 = (row_off_a + 1, col_off_b)
        C11 = (row_off_a + 1, col_off_b + 1)
        
        # m1 = (a+d)(e+h)
        U_t.append([(a, 1), (d, 1)])
        V_t.append([(e, 1), (h, 1)])
        W_t.append([(C00, 1), (C11, 1)])
        
        # m2 = (c+d)*e
        U_t.append([(c, 1), (d, 1)])
        V_t.append([(e, 1)])
        W_t.append([(C10, 1), (C11, -1)])
        
        # m3 = a*(f-h)
        U_t.append([(a, 1)])
        V_t.append([(f, 1), (h, -1)])
        W_t.append([(C01, 1), (C11, 1)])
        
        # m4 = d*(g-e)
        U_t.append([(d, 1)])
        V_t.append([(g, 1), (e, -1)])
        W_t.append([(C00, 1), (C10, 1)])
        
        # m5 = (a+b)*h
        U_t.append([(a, 1), (b, 1)])
        V_t.append([(h, 1)])
        W_t.append([(C00, -1), (C01, 1)])
        
        # m6 = (c-a)*(e+f)
        U_t.append([(c, 1), (a, -1)])
        V_t.append([(e, 1), (f, 1)])
        W_t.append([(C11, 1)])
        
        # m7 = (b-d)*(g+h)
        U_t.append([(b, 1), (d, -1)])
        V_t.append([(g, 1), (h, 1)])
        W_t.append([(C00, 1)])
    
    def direct_term(i, j, k, U_t, V_t, W_t):
        """Add a direct term A[i,j]*B[j,k] to C[i,k]."""
        U_t.append([((i, j), 1)])
        V_t.append([((j, k), 1)])
        W_t.append([((i, k), 1)])
    
    def term_to_vector(U_t, V_t, W_t, idx, n, m, p):
        """Convert a single term to a vector in the 72-dim tensor space."""
        vec = np.zeros(n * m * p)
        for (i, j), cu in U_t[idx]:
            for (j2, k), cv in V_t[idx]:
                if j == j2:
                    for (i2, k2), cw in W_t[idx]:
                        if i == i2 and k == k2:
                            vec[i * m * p + j * p + k] = cu * cv * cw
        return vec
    
    def try_reduce_rank(U_t, V_t, W_t, n, m, p, max_time):
        """Try to reduce rank by finding linear dependencies."""
        start = time.time()
        R = len(W_t)
        
        # Build matrix where each column is a term vector
        M = np.zeros((n * m * p, R))
        for idx in range(R):
            M[:, idx] = term_to_vector(U_t, V_t, W_t, idx, n, m, p)
        
        # Find rank via SVD
        try:
            U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
        except:
            return U_t, V_t, W_t, R
        
        # Determine numerical rank
        tol = max(M.shape) * np.finfo(float).eps * s[0] if len(s) > 0 else 0
        num_rank = np.sum(s > tol)
        
        if num_rank >= R:
            return U_t, V_t, W_t, R
        
        # The terms are linearly dependent. Try to find a reduced set.
        # Use the top num_rank right singular vectors as a basis.
        # Express each original term as a combination of the basis.
        # But this doesn't directly give us fewer rank-1 terms.
        
        # Alternative: greedily select independent terms and express others as combinations.
        # This is complex; for now, just return the original if no clear reduction.
        return U_t, V_t, W_t, R
    
    # Build the 68-term Strassen-based decomposition
    U_t = []
    V_t = []
    W_t = []
    
    # Strassen blocks for A[0:2, 0:2] * B[0:2, c:c+2] for c in 0, 2, 4, 6
    for c in range(0, 8, 2):
        strassen_decomp(0, 0, 0, c, U_t, V_t, W_t)
        
    # Direct terms for remaining A indices
    # A[0:2, 2] * B[2, 0:8]
    for i in range(2):
        for k in range(8):
            direct_term(i, 2, k, U_t, V_t, W_t)
            
    # A[2, 0:2] * B[0:2, 0:8]
    for j in range(2):
        for k in range(8):
            direct_term(2, j, k, U_t, V_t, W_t)
            
    # A[2, 2] * B[2, 0:8]
    for k in range(8):
        direct_term(2, 2, k, U_t, V_t, W_t)
        
    rank = len(W_t)
    
    # Try to reduce rank
    if time.time() - start_time < DEADLINE - 30:
        U_t, V_t, W_t, rank = try_reduce_rank(U_t, V_t, W_t, n, m, p, DEADLINE - time.time())
    
    # Build matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for idx in range(rank):
        for (i, j), coeff in U_t[idx]:
            U[i * m + j][idx] = coeff
        for (j, k), coeff in V_t[idx]:
            V[j * p + k][idx] = coeff
        for (i, k), coeff in W_t[idx]:
            W[k * n + i][idx] = coeff
            
    # Convert to lists of floats/ints
    def to_nice_list(mat):
        res = []
        for row in mat:
            nrow = []
            for v in row:
                if abs(v - round(v*2)/2) < 1e-9:
                    nv = round(v*2)/2
                    if abs(nv - round(nv)) < 1e-9:
                        nrow.append(int(round(nv)))
                    else:
                        nrow.append(nv)
                else:
                    nrow.append(v)
            res.append(nrow)
        return res
        
    return {"U": to_nice_list(U), "V": to_nice_list(V), "W": to_nice_list(W)}
# EVOLVE-BLOCK-END
