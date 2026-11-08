import numpy as np
import time
from itertools import product

_START_TIME = time.time()
DEADLINE_SECONDS = 130

def _elapsed():
    return time.time() - _START_TIME

def _time_remaining():
    return DEADLINE_SECONDS - _elapsed()

def _find_rank10_2x2x3():
    """
    Find a rank-10 decomposition for 2x2x3 matrix multiplication.
    A is 2x2, B is 2x3, C is 2x3 (C = A*B).
    Returns (U, V, W) with shapes (4,10), (6,10), (6,10) or None.
    """
    n, m, p = 2, 2, 3
    nA, nB, nW = n*m, m*p, p*n  # 4, 6, 6
    r = 10
    
    # We need to find U (4xr), V (6xr), W (6xr) such that
    # for all (i,j,k): sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] = 1
    #
    # This is 12 equations in 10*4 + 10*6 + 10*6 = 160 unknowns.
    # Let's try a structured approach.
    #
    # Strategy: Use Strassen for a 2x2x2 sub-block (columns 0,1 of B) and 
    # find 3 more terms that handle column 2 while reusing some structure.
    #
    # Actually, let me try a numerical approach with random initialization
    # and gradient descent, or use the fact that we can fix some variables.
    #
    # Let me try: fix W to be identity-like and solve for U, V.
    # Or better: use the known rank-7 Strassen for 2x2x2 and add 3 terms.
    #
    # For 2x2x2 Strassen (B has 2 columns):
    # M1=(a+d)(e+h), M2=(c+d)e, M3=a(f-h), M4=d(f-e), M5=(a+b)h, M6=(c-a)(e+f), M7=(b-d)(h+i)
    # Wait, standard Strassen for A=[[a,b],[c,d]] * B=[[e,f],[g,h]]:
    # M1=(a+d)(e+h), M2=(c+d)e, M3=a(f-h), M4=d(g-e), M5=(a+b)h, M6=(c-a)(e+f), M7=(b-d)(g+h)
    # C00=M1+M4-M5+M7, C01=M3+M1, C10=M2+M6, C11=M1-M2+M3+M5
    #
    # For 2x2x3: B = [[e,f,g],[h,i,j]] (2 rows, 3 columns)
    # C = A*B = [[a*e+b*h, a*f+b*i, a*g+b*j], [c*e+d*h, c*f+d*i, c*g+d*j]]
    #
    # Use Strassen on first 2 columns: B12 = [[e,f],[h,i]]
    # M1=(a+d)(e+h), M2=(c+d)e, M3=a(f-h), M4=d(i-e), M5=(a+b)h, M6=(c-a)(e+f), M7=(b-d)(h+i)
    # C00=M1+M4-M5+M7, C01=M3+M1, C10=M2+M6, C11=M1-M2+M3+M5
    #
    # For column 2: C02=a*g+b*j, C12=c*g+d*j
    # Need to express these with 3 more products.
    #
    # Let's try:
    # N1 = (a+b)*g  => a*g + b*g
    # N2 = (c+d)*j  => c*j + d*j
    # N3 = (b-d)*g  => b*g - d*g  ... not directly helpful
    #
    # Alternative:
    # N1 = a*g
    # N2 = b*j  
    # N3 = (c+d)*g - d*g + d*j = c*g + d*j  ... that's just C12
    #
    # Let me try:
    # N1 = (a+c)*g = a*g + c*g
    # N2 = (b+d)*j = b*j + d*j
    # N3 = (b-d)*(g-j) = b*g - b*j - d*g + d*j
    #
    # C02 = a*g + b*j = ?
    # C12 = c*g + d*j = ?
    #
    # From N1: a*g = N1 - c*g
    # From N2: b*j = N2 - d*j
    # C02 = N1 - c*g + N2 - d*j = N1 + N2 - (c*g + d*j) = N1 + N2 - C12
    # So if I can get C12, I get C02 for free.
    #
    # C12 = c*g + d*j
    # From N1: c*g = N1 - a*g
    # From N2: d*j = N2 - b*j
    # C12 = N1 - a*g + N2 - b*j = N1 + N2 - (a*g + b*j) = N1 + N2 - C02
    # Circular!
    #
    # Let me try different N terms:
    # N1 = a*g
    # N2 = (b+d)*j = b*j + d*j
    # N3 = (c-a)*g = c*g - a*g
    #
    # C02 = a*g + b*j = N1 + (N2 - d*j) = N1 + N2 - d*j
    # C12 = c*g + d*j = (N3 + a*g) + d*j = N3 + N1 + d*j
    #
    # Still have d*j. Let me add:
    # N4 = d*j
    # Then C02 = N1 + N2 - N4, C12 = N3 + N1 + N4
    # That's 4 extra terms, total 11. Not good.
    #
    # Let me try yet another approach. What if I share terms between the Strassen part
    # and the extra column?
    #
    # Actually, let me just try to numerically find a rank-10 solution.
    # I'll set up the equations and solve.
    
    # The tensor T has entries T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,1,2}
    # We want: T = sum_l u_l ⊗ v_l ⊗ w_l where u_l is in R^4, v_l in R^6, w_l in R^6
    
    # Let's try a random initialization and see if we can solve.
    # Actually, for small systems, we can use the following trick:
    # Fix one set of vectors and solve for the others.
    
    # Let's try fixing W to be a 6x10 matrix with some structure.
    # Actually, let me just try random search with verification.
    
    np.random.seed(42)
    for attempt in range(1000):
        # Random initialization
        U = np.random.randn(nA, r)
        V = np.random.randn(nB, r)
        W = np.random.randn(nW, r)
        
        # Check if this decomposition works
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for l in range(r):
                        val += U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l]
                    max_err = max(max_err, abs(val - 1.0))
        
        if max_err < 1e-6:
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    n, m, p = 2, 4, 7
    nA = n * m   # 8
    nB = m * p   # 28
    nW = p * n   # 14

    def build_rank50():
        """Strassen 2x2x2 blocks + schoolbook 2x2x1, total rank 50."""
        rank_max = 50
        U = [[0] * rank_max for _ in range(nA)]
        V = [[0] * rank_max for _ in range(nB)]
        W = [[0] * rank_max for _ in range(nW)]
        col = 0

        def add_strassen(i0, j0, k0):
            nonlocal col
            a_idx = i0 * m + j0
            b_idx = i0 * m + (j0 + 1)
            c_idx = (i0 + 1) * m + j0
            d_idx = (i0 + 1) * m + (j0 + 1)
            e_idx = j0 * p + k0
            f_idx = j0 * p + (k0 + 1)
            g_idx = (j0 + 1) * p + k0
            h_idx = (j0 + 1) * p + (k0 + 1)
            c00 = k0 * n + i0
            c01 = (k0 + 1) * n + i0
            c10 = k0 * n + (i0 + 1)
            c11 = (k0 + 1) * n + (i0 + 1)

            U[a_idx][col] += 1; U[d_idx][col] += 1
            V[e_idx][col] += 1; V[h_idx][col] += 1
            W[c00][col] += 1; W[c11][col] += 1
            col += 1

            U[c_idx][col] += 1; U[d_idx][col] += 1
            V[e_idx][col] += 1
            W[c10][col] += 1; W[c11][col] -= 1
            col += 1

            U[a_idx][col] += 1
            V[f_idx][col] += 1; V[h_idx][col] -= 1
            W[c01][col] += 1; W[c11][col] += 1
            col += 1

            U[d_idx][col] += 1
            V[g_idx][col] += 1; V[e_idx][col] -= 1
            W[c00][col] += 1; W[c10][col] += 1
            col += 1

            U[a_idx][col] += 1; U[b_idx][col] += 1
            V[h_idx][col] += 1
            W[c00][col] -= 1; W[c01][col] += 1
            col += 1

            U[c_idx][col] += 1; U[a_idx][col] -= 1
            V[e_idx][col] += 1; V[f_idx][col] += 1
            W[c11][col] += 1
            col += 1

            U[b_idx][col] += 1; U[d_idx][col] -= 1
            V[g_idx][col] += 1; V[h_idx][col] += 1
            W[c00][col] += 1
            col += 1

        def add_schoolbook_221(i0, j0, k0):
            nonlocal col
            a_idx = i0 * m + j0
            b_idx = i0 * m + (j0 + 1)
            c_idx = (i0 + 1) * m + j0
            d_idx = (i0 + 1) * m + (j0 + 1)
            e_idx = j0 * p + k0
            g_idx = (j0 + 1) * p + k0
            c00 = k0 * n + i0
            c10 = k0 * n + (i0 + 1)

            U[a_idx][col] += 1; V[e_idx][col] += 1; W[c00][col] += 1; col += 1
            U[b_idx][col] += 1; V[g_idx][col] += 1; W[c00][col] += 1; col += 1
            U[c_idx][col] += 1; V[e_idx][col] += 1; W[c10][col] += 1; col += 1
            U[d_idx][col] += 1; V[g_idx][col] += 1; W[c10][col] += 1; col += 1

        for j_start in [0, 2]:
            for k_start in [0, 2, 4]:
                add_strassen(0, j_start, k_start)
            add_schoolbook_221(0, j_start, 6)

        return U[:col], V[:col], W[:col], col

    def verify(U, V, W, tol=1e-9):
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    ai = i * m + j
                    bi = j * p + k
                    wi = k * n + i
                    for l in range(r):
                        val += U[ai][l] * V[bi][l] * W[wi][l]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def build_rank48():
        """
        Use rank-10 decomposition for 2x2x3 blocks.
        2x2x7 = 2x2x3 (rank 10) + 2x2x3 (rank 10) + 2x2x1 (rank 4) = 24
        Two such blocks (j=0,1 and j=2,3): total rank 48.
        
        Rank-10 for 2x2x3:
        A = [[a,b],[c,d]], B = [[e,f,g],[h,i,j]], C = [[C00,C01,C02],[C10,C11,C12]]
        
        We use Strassen for the first 2 columns (rank 7) + 3 extra for column 2.
        
        Strassen products for columns 0,1 (e,f / h,i):
        P1 = (a+d)(e+h)
        P2 = (c+d)e  
        P3 = a(f-h)
        P4 = d(g-e)  -- wait, g is column 2!
        
        Let me redo. For 2x2x2 with B having columns [e,f;h,i]:
        P1 = (a+d)(e+h)
        P2 = (c+d)e
        P3 = a(f-h)
        P4 = d(f-e)  -- using f and e from first two cols
        P5 = (a+b)h
        P6 = (c-a)(e+f)
        P7 = (b-d)(h+i)
        
        C00 = P1 + P4 - P5 + P7  -- wait, standard Strassen:
        C00 = P1 + P4 - P5 + P7  -- hmm let me use standard formulas
        
        Standard Strassen for 2x2x2 (A*B where B is 2x2):
        M1=(a+d)(e+h), M2=(c+d)e, M3=a(f-h), M4=d(g-e), M5=(a+b)h, M6=(c-a)(e+f), M7=(b-d)(g+h)
        where B=[[e,f],[g,h]]
        
        C00=M1+M4-M5+M7, C01=M3+M1, C10=M2+M6, C11=M1-M2+M3+M5
        
        For our 2x2x3, B has 3 columns. Let B = [[e,f,g],[h,i,j]].
        Use Strassen on first two columns: B12 = [[e,f],[h,i]]
        M1=(a+d)(e+h), M2=(c+d)e, M3=a(f-h), M4=d(i-e), M5=(a+b)h, M6=(c-a)(e+f), M7=(b-d)(h+i)
        
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M1
        C10 = M2 + M6
        C11 = M1 - M2 + M3 + M5
        
        For column 2: C02 = a*g + b*j, C12 = c*g + d*j
        Need 3 more products to get these. 
        N1 = (a+b)*g  → a*g + b*g
        N2 = (c+d)*j  → c*j + d*j  
        N3 = (b-d)*g  → b*g - d*g  ... hmm
        
        Actually: C02 = a*g + b*j, C12 = c*g + d*j
        Let N1 = (a+c)*g = a*g + c*g
        Let N2 = (b+d)*j = b*j + d*j
        Let N3 = (b-c)*g = b*g - c*g  ... not helpful directly
        
        Better: 
        N1 = a*g
        N2 = b*j  
        N3 = (c+d)*g = c*g + d*g
        Then C02 = N1 + N2, C12 = N3 - d*g + d*j... no.
        
        Let me try:
        N1 = (a+b)*g = a*g + b*g
        N2 = (c+d)*j = c*j + d*j
        N3 = (b-d)*g = b*g - d*g
        C02 = a*g + b*j = N1 - b*g + b*j = N1 - (N3 + d*g) + b*j ... still messy.
        
        Alternative:
        N1 = a*g
        N2 = (b+d)*j = b*j + d*j
        N3 = (c-a)*g = c*g - a*g
        C02 = N1 + b*j = N1 + N2 - d*j ... no.
        
        OK, simplest: just use 4 schoolbook for column 2:
        N1 = a*g, N2 = b*j, N3 = c*g, N4 = d*j
        C02 = N1 + N2, C12 = N3 + N4
        Total rank = 7 + 4 = 11. That gives 2*11 = 22 per... no, 2*(11+11+4) = 52. Worse!
        
        I need a true rank-10 construction. Let me think differently.
        
        The key insight for rank 10: we can share some products between the Strassen part and the extra column.
        
        Actually, I recall that the bilinear complexity of M_{2,2,3} is 10, but constructing it explicitly is non-trivial. Let me try a numerical approach.
        """
        # Try to find a rank-10 decomposition numerically
        # The tensor has 2*2*3 = 12 entries
        # We need U (4 x 10), V (6 x 10), W (6 x 10) such that
        # sum_l U[i*2+j, l] * V[j*3+k, l] * W[k*2+i, l] = 1 for all (i,j,k)
        
        # This is a system of 12 equations in 10*4 + 10*6 + 10*6 = 160 unknowns
        # Very underdetermined. Let's try to find one.
        
        # Strategy: parameterize and solve
        # Actually, let me just try the 7+4=11 approach first and see if I can merge.
        
        # For now, return None to indicate this approach failed
        return None

    def build_rank48_v2():
        """
        Alternative approach using different block structure.
        """
        return None

    def try_reduce_rank(U, V, W, max_iter=200, tol=1e-8):
        """
        Try to reduce rank by finding null vectors in the tensor space.
        Uses SVD to identify linear dependencies among the rank-1 terms.
        """
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        r = U_np.shape[1]
        
        def tensor_vector(U_l, V_l, W_l):
            vec = np.zeros(n * m * p)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        vec[i * m * p + j * p + k] = U_l[i * m + j] * V_l[j * p + k] * W_l[k * n + i]
            return vec
        
        # Build the matrix where each column is a rank-1 tensor vector
        Tmat = np.zeros((n * m * p, r))
        for l in range(r):
            Tmat[:, l] = tensor_vector(U_np[:, l], V_np[:, l], W_np[:, l])
        
        # Check for linear dependencies
        rank_Tmat = np.linalg.matrix_rank(Tmat, tol=tol)
        if rank_Tmat >= r:
            return U, V, W, r
        
        # Find null space vectors
        U_sv, s, Vt = np.linalg.svd(Tmat, full_matrices=True)
        null_dim = r - rank_Tmat
        
        # Try to use null vectors to eliminate columns
        # A null vector z means sum_l z[l] * T_l = 0
        # This doesn't directly give us a way to reduce rank, but it indicates
        # that some combination of the rank-1 terms cancels out.
        
        # For now, just trim zero columns
        keep = []
        for l in range(r):
            if not (all(U[i][l] == 0 for i in range(nA)) and
                    all(V[i][l] == 0 for i in range(nB)) and
                    all(W[i][l] == 0 for i in range(nW))):
                keep.append(l)
        
        if len(keep) < r:
            U = [[U[i][l] for l in keep] for i in range(nA)]
            V = [[V[i][l] for l in keep] for i in range(nB)]
            W = [[W[i][l] for l in keep] for i in range(nW)]
            r = len(keep)
        
        return U, V, W, r

    # First try to build rank-48 solution
    result48 = build_rank48()
    if result48 is not None:
        U, V, W, rank = result48
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
    
    # Build the rank-50 solution
    U, V, W, rank = build_rank50()
    
    # Verify
    if not verify(U, V, W):
        # Fallback: schoolbook rank 56
        U = [[0] * (nA * nW) for _ in range(nA)]
        V = [[0] * (nA * nW) for _ in range(nB)]
        W = [[0] * (nA * nW) for _ in range(nW)]
        col = 0
        for j in range(m):
            for i in range(n):
                for k in range(p):
                    ai = i * m + j
                    bi = j * p + k
                    wi = k * n + i
                    U[ai][col] = 1
                    V[bi][col] = 1
                    W[wi][col] = 1
                    col += 1
        U = [row[:col] for row in U]
        V = [row[:col] for row in V]
        W = [row[:col] for row in W]
        rank = col
        return {"U": U, "V": V, "W": W}

    # Try to reduce rank
    U, V, W, rank = try_reduce_rank(U, V, W)

    return {"U": U, "V": V, "W": W}
