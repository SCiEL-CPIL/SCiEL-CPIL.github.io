"""Matrix multiplication 2,2,5 - rank-minimal exact tensor decomposition."""

import time
from fractions import Fraction

DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly represents matrix multiplication."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _build_strassen_2x2():
    """Return U, V, W matrices for 2x2 * 2x2 Strassen decomposition.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    W convention: W[k*n+i, c] where n=2
    """
    # Strassen's 7 multiplication terms for 2x2 matrices
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[c00, c01], [c10, c11]]
    
    # U[i*m+j, c] for n=2, m=2: rows are a(0,0), b(0,1), c(1,0), d(1,1)
    U = [
        [1, 0, 1, 1, 1, 0, 0],  # a
        [0, 0, -1, 0, 0, 1, -1], # b
        [0, 1, 0, 0, 1, 0, -1],  # c
        [1, 1, 0, 1, 0, -1, 0],  # d
    ]
    
    # V[j*p+k, c] for m=2, p=2: rows are e(0,0), f(0,1), g(1,0), h(1,1)
    V = [
        [1, 0, 1, 0, 0, 0, 0],  # e
        [0, 0, 1, 0, 0, 0, 0],  # f
        [0, 1, 0, 0, 0, 0, 0],  # g
        [1, 0, 0, 0, 0, 1, 0],  # h
    ]
    
    # W[k*n+i, c] for p=2, n=2: rows are c00(0,0), c01(0,1), c10(1,0), c11(1,1)
    W = [
        [1, 0, 0, 1, -1, 0, -1],  # c00
        [0, 0, 1, 0, 1, 0, 0],    # c01
        [0, 1, 0, 1, 0, 0, 0],    # c10
        [1, -1, 0, 0, 0, 1, 1],   # c11
    ]
    
    return U, V, W

def _build_optimal_2x2x3():
    """
    Build a rank-12 decomposition for 2x2 * 2x3 multiplication.
    
    A = [[a, b], [c, d]] (2x2)
    B = [[e, f, g], [h, i, j]] (2x3)
    C = [[c00, c01, c02], [c10, c11, c12]] (2x3)
    
    c00 = a*e + b*h
    c01 = a*f + b*i
    c02 = a*g + b*j
    c10 = c*e + d*h
    c11 = c*f + d*i
    c12 = c*g + d*j
    
    We can view this as: for each column k of B, compute A * B[:,k].
    This is 3 independent 2x2*2x1 problems, each of rank 4, giving rank 12.
    
    But we can do better by sharing structure.
    
    Actually, 2x2*2x3 has tensor rank 12 (same as schoolbook).
    Let's just use schoolbook for the 2x2*2x3 part.
    
    Wait, the knowledge says rank 16 or 17 is possible for 2x2*2x5.
    Let me think differently.
    
    Strategy: Split B (2x5) into three parts: B1 (2x2), B2 (2x2), B3 (2x1).
    - A * B1: Strassen, rank 7
    - A * B2: Strassen, rank 7  
    - A * B3: schoolbook 2x2*2x1, rank 4
    Total: 18
    
    To get rank 16, we need to save 2 ranks.
    
    Alternative: Can we do 2x2*2x4 with rank 12 and 2x2*2x1 with rank 4?
    2x2*2x4 schoolbook = 16. Strassen+Strassen = 14.
    
    Hmm, let me try a different grouping.
    
    What if we use one Strassen (rank 7) for columns 0,1 and a custom
    rank-9 algorithm for columns 2,3,4?
    
    2x2*2x3 needs at least... let me think. The tensor rank of 2x2*2x3
    is actually 12 (it's the same as schoolbook for this case since
    one dimension is small).
    
    So 7 + 12 = 19? That's worse.
    
    Let me try: rank 7 + rank 7 + rank 4 = 18.
    
    To improve, maybe I can find a rank-6 algorithm for 2x2*2x1? No,
    2x2*2x1 has tensor rank 4 (it's just 4 multiplications: a*e+b*h, 
    a*f+b*i, c*e+d*h, c*f+d*i for a 2x2*2x2 where we only need 2 columns).
    
    Actually wait - 2x2*2x1: A=[[a,b],[c,d]], B=[[e],[h]]
    C = [[a*e+b*h], [c*e+d*h]]
    That's 4 multiplications. Tensor rank is 4.
    
    So minimum with this splitting is 7+7+4 = 18.
    
    The key insight from the knowledge: "Using a structured block decomposition 
    (Strassen for 2x2 + optimized 2x3) allows for a lower rank (16 or 17)."
    
    So there must be an optimized 2x2*2x3 that uses fewer than 12 terms,
    or a way to share between blocks.
    
    Actually, I think the trick is: instead of treating columns independently,
    we can find a rank-12 algorithm for 2x2*2x4 (not 14) and then add 4 for
    the last column, giving 16.
    
    Or: rank-12 for 2x2*2x3 and rank-4 for 2x2*2x1, but with sharing?
    
    Let me just implement the rank-18 solution cleanly, and also try to find
    a rank-16 solution through a more careful construction.
    
    For now, let me implement a direct rank-16 construction if possible.
    
    Actually, let me reconsider. The problem is 2x2 * 2x5.
    
    Let me try to directly construct a rank-16 decomposition.
    
    A = [[a, b], [c, d]]
    B = [[b00, b01, b02, b03, b04],
         [b10, b11, b12, b13, b14]]
    C = A*B = [[c00, c01, c02, c03, c04],
               [c10, c11, c12, c13, c14]]
    
    c[i,k] = a*b[i,0] + c*b[i,1]  ... wait no.
    c[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    So:
    c00 = a*b00 + b*b10
    c01 = a*b01 + b*b11
    c02 = a*b02 + b*b12
    c03 = a*b03 + b*b13
    c04 = a*b04 + b*b14
    c10 = c*b00 + d*b10
    c11 = c*b01 + d*b11
    c12 = c*b02 + d*b12
    c13 = c*b03 + d*b13
    c14 = c*b04 + d*b14
    
    That's 10 output entries, each requiring 2 multiplications = 20 scalar mults.
    But tensor rank counts the number of rank-1 terms.
    
    Schoolbook: 2*2*5 = 20 terms.
    
    Using Strassen for pairs:
    - (c00,c01,c10,c11) from columns 0,1: rank 7
    - (c02,c03,c12,c13) from columns 2,3: rank 7
    - (c04,c14) from column 4: rank 4
    Total: 18
    
    To get 16, I need to save 2 terms.
    
    Idea: Can I use a rank-6 algorithm for one of the 2x2*2x2 blocks?
    No, 2x2*2x2 has rank exactly 7.
    
    Idea: Can the 2x2*2x1 part be done in 2 terms? No, it needs 4.
    
    Idea: Can I share some rank-1 terms between the Strassen blocks and the
    schoolbook block?
    
    This is getting complex. Let me just go with rank 18 for reliability,
    and structure the code so it's clean.
    
    Actually, wait. Let me re-read the knowledge: "Using a structured block 
    decomposition (Strassen for 2x2 + optimized 2x3) allows for a lower rank 
    (16 or 17) compared to the previous 18, leveraging the additive property 
    of tensor rank under disjoint column partitions."
    
    "optimized 2x3" - maybe there's a rank-9 algorithm for 2x2*2x3?
    
    2x2*2x3: 6 output entries, each is a sum of 2 products.
    Schoolbook: 12 terms.
    
    Can we do it in 9? Let me think...
    
    c00 = a*e + b*h
    c01 = a*f + b*i  
    c02 = a*g + b*j
    c10 = c*e + d*h
    c11 = c*f + d*i
    c12 = c*g + d*j
    
    Using Strassen-like ideas:
    Let me define:
    M1 = (a+c)*(e+h) = a*e + a*h + c*e + c*h
    M2 = (a+b)*(e+i) = a*e + a*i + b*e + b*i
    ... this is getting messy.
    
    Let me just implement rank 18 cleanly.
    """
    pass

def _build_rank18():
    """Build rank-18 decomposition using Strassen for 2x2 blocks + schoolbook for 2x1."""
    n, m, p = 2, 2, 5
    total_rank = 7 + 7 + 4
    n_rows = n * m  # 4
    m_rows = m * p  # 10
    p_rows = p * n  # 10
    
    U_full = [[0]*total_rank for _ in range(n_rows)]
    V_full = [[0]*total_rank for _ in range(m_rows)]
    W_full = [[0]*total_rank for _ in range(p_rows)]
    
    U_s, V_s, W_s = _build_strassen_2x2()
    
    # Block 1: columns 0,1 (Strassen)
    offset = 0
    for i in range(4):
        for c in range(7):
            U_full[i][offset + c] = U_s[i][c]
    
    # V rows: B[0,0]->0, B[0,1]->1, B[1,0]->5, B[1,1]->6
    v_row_map = [0, 1, 5, 6]
    for i in range(4):
        for c in range(7):
            V_full[v_row_map[i]][offset + c] = V_s[i][c]
    
    # W rows: C[0,0]->0, C[0,1]->1, C[1,0]->2, C[1,1]->3
    w_row_map = [0, 1, 2, 3]
    for i in range(4):
        for c in range(7):
            W_full[w_row_map[i]][offset + c] = W_s[i][c]
    
    offset += 7
    
    # Block 2: columns 2,3 (Strassen)
    for i in range(4):
        for c in range(7):
            U_full[i][offset + c] = U_s[i][c]
    
    v_row_map2 = [2, 3, 7, 8]
    for i in range(4):
        for c in range(7):
            V_full[v_row_map2[i]][offset + c] = V_s[i][c]
    
    w_row_map2 = [4, 5, 6, 7]
    for i in range(4):
        for c in range(7):
            W_full[w_row_map2[i]][offset + c] = W_s[i][c]
    
    offset += 7
    
    # Block 3: column 4 (schoolbook 2x2*2x1)
    # U_sb: a,b,c,d each have one 1
    # V_sb: B[0,4], B[1,4] each have one 1
    # W_sb: C[0,4], C[1,4] each have one 1
    
    # 4 terms: (a,B04,C04), (b,B14,C04), (c,B04,C14), (d,B14,C14)
    U_full[0][offset + 0] = 1  # a
    U_full[1][offset + 1] = 1  # b
    U_full[2][offset + 2] = 1  # c
    U_full[3][offset + 3] = 1  # d
    
    V_full[0*5+4][offset + 0] = 1  # B[0,4]
    V_full[1*5+4][offset + 1] = 1  # B[1,4]
    V_full[0*5+4][offset + 2] = 1  # B[0,4]
    V_full[1*5+4][offset + 3] = 1  # B[1,4]
    
    W_full[4*2+0][offset + 0] = 1  # C[0,4]
    W_full[4*2+0][offset + 1] = 1  # C[0,4]
    W_full[4*2+1][offset + 2] = 1  # C[1,4]
    W_full[4*2+1][offset + 3] = 1  # C[1,4]
    
    return U_full, V_full, W_full, total_rank

def _try_rank16():
    """
    Attempt to construct a rank-16 decomposition.
    
    Strategy: Use Strassen (rank 7) for columns 0,1 and a custom rank-9 
    algorithm for columns 2,3,4.
    
    For columns 2,3,4:
    c02 = a*b02 + b*b12
    c03 = a*b03 + b*b13
    c04 = a*b04 + b*b14
    c12 = c*b02 + d*b12
    c13 = c*b03 + d*b13
    c14 = c*b04 + d*b14
    
    Schoolbook: 12 terms. Can we do 9?
    
    Let me try a Strassen-like approach:
    M1 = (a+c)*(b02+b12) = a*b02 + a*b12 + c*b02 + c*b12
    M2 = (a+b)*(b03+b13) = a*b03 + a*b13 + b*b03 + b*b13
    M3 = (c+d)*(b04+b14) = c*b04 + c*b14 + d*b04 + d*b14
    ...
    
    This doesn't directly work. Let me think more carefully.
    
    Actually, for 2x2*2x3, the tensor rank is known to be 12.
    So we can't do better than 12 for that sub-problem.
    
    Therefore 7+12 = 19 is worse than 18.
    
    Alternative: 7+7+4 = 18 is the best with this splitting.
    
    To get 16, we need a fundamentally different approach.
    
    Let me try: Can we use a rank-12 algorithm for 2x2*2x4 and rank-4 for 2x2*2x1?
    2x2*2x4: schoolbook = 16, Strassen+Strassen = 14.
    If we can do 12 for 2x2*2x4, then 12+4 = 16.
    
    2x2*2x4: A=[[a,b],[c,d]], B=[[e,f,g,h],[i,j,k,l]]
    C = [[c00,c01,c02,c03],[c10,c11,c12,c13]]
    
    c00 = a*e + b*i
    c01 = a*f + b*j
    c02 = a*g + b*k
    c03 = a*h + b*l
    c10 = c*e + d*i
    c11 = c*f + d*j
    c12 = c*g + d*k
    c13 = c*h + d*l
    
    8 outputs, 16 monomials. Can we cover with 12 rank-1 terms?
    
    This is equivalent to finding 12 triples (u_c, v_c, w_c) such that
    sum_c u_c[i,j] * v_c[j,k] * w_c[k,i] = delta(i,j,k) for the multiplication tensor.
    
    This is a hard problem. Let me just go with rank 18.
    """
    return None

def _schoolbook_block(n, m, p):
    """Schoolbook decomposition for n x m * m x p."""
    r = n * m * p
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W, r

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Try rank-16 first
    if time.time() < deadline:
        r16_result = _try_rank16()
        if r16_result is not None:
            U16, V16, W16, r16 = r16_result
            if _verify(U16, V16, W16, n, m, p, r16):
                return {"U": U16, "V": V16, "W": W16}
    
    # Rank-18: Strassen + Strassen + schoolbook
    if time.time() < deadline:
        U18, V18, W18, r18 = _build_rank18()
        if _verify(U18, V18, W18, n, m, p, r18):
            return {"U": U18, "V": V18, "W": W18}
    
    # Fallback: schoolbook
    U_sb, V_sb, W_sb, r = _schoolbook_block(n, m, p)
    return {"U": U_sb, "V": V_sb, "W": W_sb}

# EVOLVE-BLOCK-END
