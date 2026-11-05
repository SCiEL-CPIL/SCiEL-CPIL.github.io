"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=3, m=5, p=7
    
    # We use a hybrid decomposition:
    # - For the 2x2x2 sub-block (rows 0-1, cols 0-1, out 0-1), use Strassen (rank 7)
    # - For all other terms, use trivial decomposition (rank = 1 each)
    
    # Trivial decomposition for all (i, j, k)
    U_triv = []
    V_triv = []
    W_triv = []
    
    # We'll build the full set of terms, then replace the 2x2x2 block with Strassen
    # First, identify which terms are in the 2x2x2 block
    # i in {0,1}, j in {0,1}, k in {0,1}
    
    # Start with all trivial terms
    all_terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                all_terms.append((i, j, k))
    
    # Separate into block terms and non-block terms
    block_terms = [(i, j, k) for (i, j, k) in all_terms if i < 2 and j < 2 and k < 2]
    non_block_terms = [(i, j, k) for (i, j, k) in all_terms if not (i < 2 and j < 2 and k < 2)]
    
    # Non-block terms: trivial decomposition
    U = []
    V = []
    W = []
    
    for (i, j, k) in non_block_terms:
        U.append([1 if (ii == i and jj == j) else 0 for ii in range(n) for jj in range(m)])
        V.append([1 if (jj == j and kk == k) else 0 for jj in range(m) for kk in range(p)])
        W.append([1 if (kk == k and ii == i) else 0 for kk in range(p) for ii in range(n)])
    
    # Block terms: Strassen decomposition for 2x2x2
    # Strassen for C = AB where A is 2x2, B is 2x2
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-b)
    # m5 = (a+b)h
    # m6 = (c-d)(g+h)
    # m7 = (a-c)(e+f)
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 - m4
    # C11 = m1 - m3 + m6
    
    # In our notation:
    # A = [[a, b], [c, d]] where a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # B = [[e, f], [g, h]] where e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    # C = [[C00, C01], [C10, C11]]
    
    # U vectors are for A (size n*m = 15, but we only care about indices for i<2, j<2)
    # V vectors are for B (size m*p = 35, but we only care about indices for j<2, k<2)
    # W vectors are for C (size p*n = 21, but we only care about indices for k<2, i<2)
    
    # Index mapping:
    # A index: i*m + j = i*5 + j
    # B index: j*p + k = j*7 + k
    # C index: k*n + i = k*3 + i
    
    # Strassen terms for the 2x2x2 block
    strassen_terms = [
        # (U_vec, V_vec, W_vec)
        # m1 = (a+d)(e+h)
        ([1,0,1,0], [1,0,0,1], [1,0,0,1]),
        # m2 = (c+d)e
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
        # m3 = a(f-h)
        ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),
        # m4 = d(g-b)
        ([0,0,0,1], [0,0,1,-1], [1,0,0,0]),
        # m5 = (a+b)h
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        # m6 = (c-d)(g+h)
        ([0,0,1,-1], [0,0,1,1], [0,0,0,1]),
        # m7 = (a-c)(e+f)
        ([1,0,-1,0], [1,1,0,0], [1,0,0,1]),
    ]
    
    for (u4, v4, w4) in strassen_terms:
        # Expand u4 to full U vector (size n*m = 15)
        U_full = [0] * (n * m)
        for idx, val in enumerate(u4):
            i = idx // 2
            j = idx % 2
            if 0 <= i < n and 0 <= j < m:
                U_full[i * m + j] = val
        U.append(U_full)
        
        # Expand v4 to full V vector (size m*p = 35)
        V_full = [0] * (m * p)
        for idx, val in enumerate(v4):
            j = idx // 2
            k = idx % 2
            if 0 <= j < m and 0 <= k < p:
                V_full[j * p + k] = val
        V.append(V_full)
        
        # Expand w4 to full W vector (size p*n = 21)
        W_full = [0] * (p * n)
        for idx, val in enumerate(w4):
            k = idx // 2
            i = idx % 2
            if 0 <= k < p and 0 <= i < n:
                W_full[k * n + i] = val
        W.append(W_full)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
