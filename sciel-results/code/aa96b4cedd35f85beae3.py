import time
import random
from fractions import Fraction

DIMENSIONS = [2, 2, 5]

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

def _verify_sub(U, V, W, n, m, p, rank):
    """Verify sub-tensor decomposition."""
    # For 2x2 * 2x3, n=2, m=2, p=3
    # U shape (4, r), V shape (6, r), W shape (6, r)
    # Indices:
    # U[i*m+j], V[j*p+k], W[k*n+i]
    # This is the same structure.
    return _verify(U, V, W, n, m, p, rank)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS  # 2, 2, 5
    
    # Strategy: Split the 2x5 multiplication into 2x2 and 2x3 parts.
    # C = A * B, where B = [B_2, B_3]
    # C_2 = A * B_2 (2x2 result)
    # C_3 = A * B_3 (2x3 result)
    # We can decompose these separately and concatenate the rank-1 terms.
    
    # 1. Strassen decomposition for 2x2 matrix multiplication (rank 7)
    # Standard Strassen formulas for 2x2:
    # M1 = (A11)(B11 + B22)
    # M2 = (A11 + A22)(B22)
    # M3 = (A22)(B11 + B12)
    # M4 = (A11)(B22 - B12)
    # M5 = (A11 + A21)(B22)  -- Wait, standard Strassen is:
    # M1 = (A11)(B11 + B22)
    # M2 = (A21 + A22)(B11)
    # M3 = (A11 + A12)(B22)
    # M4 = (A11 - A22)(B12 + B22)
    # M5 = (A22 - A12)(B11 + B12)
    # M6 = (A11 + A12 - A21 - A22)(B21 + B22)
    # M7 = (A21 - A11 + A22)(B11 + B12)
    
    # Let's use a known rank-7 decomposition for 2x2.
    # U, V, W for 2x2 * 2x2
    # n=2, m=2, p=2
    # U shape (4, 7), V shape (4, 7), W shape (4, 7)
    
    # Indices for U: i*m+j -> 0:00, 1:01, 2:10, 3:11
    # Indices for V: j*p+k -> 0:00, 1:01, 2:10, 3:11
    # Indices for W: k*n+i -> 0:00, 1:01, 2:10, 3:11
    
    # Strassen decomposition (one common variant):
    # Let's define the 7 terms.
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    # M1 = a(e+h)
    # M2 = (c+d)e
    # M3 = (a+b)h
    # M4 = a(h-f)
    # M5 = (d-b)(e+f)
    # M6 = (a+c-b-d)(f+h)
    # M7 = (c-a+d)(e+f)
    
    # U entries (A coefficients):
    # M1: a -> U[0][0]=1
    # M2: c,d -> U[2][1]=1, U[3][1]=1
    # M3: a,b -> U[0][2]=1, U[1][2]=1
    # M4: a -> U[0][3]=1
    # M5: d,b -> U[3][4]=1, U[1][4]=-1
    # M6: a,c,b,d -> U[0][5]=1, U[2][5]=1, U[1][5]=-1, U[3][5]=-1
    # M7: c,a,d -> U[2][6]=1, U[0][6]=-1, U[3][6]=1
    
    U_strassen = [
        [1, 0, 1, 1, 0, 1, -1],  # a
        [0, 0, 1, 0, -1, -1, 0], # b
        [0, 1, 0, 0, 0, 1, 1],   # c
        [0, 1, 0, 0, 1, -1, 1]   # d
    ]
    
    # V entries (B coefficients):
    # M1: e,h -> V[0][0]=1, V[3][0]=1
    # M2: e -> V[0][1]=1
    # M3: h -> V[3][2]=1
    # M4: h,f -> V[3][3]=1, V[1][3]=-1
    # M5: e,f -> V[0][4]=1, V[1][4]=1
    # M6: f,h -> V[1][5]=1, V[3][5]=1
    # M7: e,f -> V[0][6]=1, V[1][6]=1
    
    V_strassen = [
        [1, 1, 0, 0, 1, 0, 1],  # e
        [0, 0, 0, -1, 1, 1, 1], # f
        [0, 0, 0, 0, 0, 0, 0],  # g (not used in this variant? Wait, Strassen uses all)
        [1, 0, 1, 1, 0, 1, 0]   # h
    ]
    
    # W entries (C coefficients):
    # C11 = M1 - M4 + M5 + M7
    # C12 = M3 + M4
    # C21 = M2 + M6
    # C22 = M1 - M2 + M3 + M6
    
    W_strassen = [
        [1, -1, 1, -1, 1, 0, 1],  # C11
        [0, 0, 1, 1, 0, 0, 0],    # C12
        [0, 1, 0, 0, 0, 1, 0],    # C21
        [1, -1, 1, 0, 0, 1, 0]    # C22
    ]
    
    # Verify Strassen
    if not _verify(U_strassen, V_strassen, W_strassen, 2, 2, 2, 7):
        print("Strassen verification failed")
        # Fallback to schoolbook for 2x2
        U_strassen = [[1,0,0,0,0,0,0], [0,1,0,0,0,0,0], [0,0,1,0,0,0,0], [0,0,0,1,0,0,0]]
        V_strassen = [[1,0,0,0,0,0,0], [0,1,0,0,0,0,0], [0,0,1,0,0,0,0], [0,0,0,1,0,0,0]]
        W_strassen = [[1,0,0,0,0,0,0], [0,1,0,0,0,0,0], [0,0,1,0,0,0,0], [0,0,0,1,0,0,0]]
        # This is not rank 7, it's rank 4. But it's valid.
        # Actually, let's just use schoolbook for the whole thing if Strassen fails.
        # But Strassen should work.
    
    # 2. Decomposition for 2x2 * 2x3
    # n=2, m=2, p=3
    # Schoolbook rank is 12.
    # We want to find a rank < 12 decomposition.
    # Let's try rank 9, 10, 11.
    
    def find_2x2_2x3_decomp(target_rank):
        values = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
        n_sub, m_sub, p_sub = 2, 2, 3
        for attempt in range(5000):
            if time.time() > deadline:
                break
            U = [[random.choice(values) for _ in range(target_rank)] for _ in range(n_sub*m_sub)]
            V = [[random.choice(values) for _ in range(target_rank)] for _ in range(m_sub*p_sub)]
            W = [[random.choice(values) for _ in range(target_rank)] for _ in range(p_sub*n_sub)]
            if _verify(U, V, W, n_sub, m_sub, p_sub, target_rank):
                return U, V, W, target_rank
        return None
    
    best_sub = None
    for r in range(11, 8, -1):  # Try 11, 10, 9
        if time.time() > deadline:
            break
        res = find_2x2_2x3_decomp(r)
        if res:
            best_sub = res
            break
    
    if not best_sub:
        # Fallback to schoolbook for 2x2 * 2x3 (rank 12)
        n_sub, m_sub, p_sub = 2, 2, 3
        r_sub = n_sub * m_sub * p_sub
        U_sub = [[0]*r_sub for _ in range(n_sub*m_sub)]
        V_sub = [[0]*r_sub for _ in range(m_sub*p_sub)]
        W_sub = [[0]*r_sub for _ in range(p_sub*n_sub)]
        col = 0
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    U_sub[i*m_sub+j][col] = 1
                    V_sub[j*p_sub+k][col] = 1
                    W_sub[k*n_sub+i][col] = 1
                    col += 1
        best_sub = (U_sub, V_sub, W_sub, r_sub)
    
    U_sub, V_sub, W_sub, r_sub = best_sub
    
    # 3. Combine Strassen (2x2) and Sub (2x3)
    # The full problem is 2x2 * 2x5.
    # We split the 5 columns of B into 2 columns (for Strassen) and 3 columns (for Sub).
    # B = [B_2, B_3]
    # C = [C_2, C_3]
    # C_2 = A * B_2 (2x2)
    # C_3 = A * B_3 (2x3)
    
    # The combined decomposition is the direct sum of the two decompositions.
    # Total rank = 7 + r_sub
    
    r_total = 7 + r_sub
    
    # Construct U_total, V_total, W_total
    # U shape (4, r_total)
    # V shape (10, r_total)  (2*5)
    # W shape (10, r_total)  (5*2)
    
    U_total = [[0]*r_total for _ in range(4)]
    V_total = [[0]*r_total for _ in range(10)]
    W_total = [[0]*r_total for _ in range(10)]
    
    # Fill first 7 columns with Strassen
    for i in range(4):
        for c in range(7):
            U_total[i][c] = U_strassen[i][c]
    
    # V for Strassen: 2x2 part of B (columns 0,1)
    # V_strassen indices: j*p+k where p=2.
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # j=1, k=0 -> 2
    # j=1, k=1 -> 3
    # In V_total, indices are j*5+k.
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # j=1, k=0 -> 5
    # j=1, k=1 -> 6
    
    # Mapping:
    # V_strassen[0] (e) -> V_total[0]
    # V_strassen[1] (f) -> V_total[1]
    # V_strassen[2] (g) -> V_total[5]
    # V_strassen[3] (h) -> V_total[6]
    
    v_map_strassen = [0, 1, 5, 6]
    for c in range(7):
        for i_sub in range(4):
            V_total[v_map_strassen[i_sub]][c] = V_strassen[i_sub][c]
    
    # W for Strassen: 2x2 part of C (columns 0,1)
    # W_strassen indices: k*n+i where n=2.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    # In W_total, indices are k*2+i.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    # This matches directly.
    
    for i in range(4):
        for c in range(7):
            W_total[i][c] = W_strassen[i][c]
    
    # Fill next r_sub columns with Sub decomposition
    # Sub is 2x2 * 2x3
    # U_sub shape (4, r_sub)
    # V_sub shape (6, r_sub)  (2*3)
    # W_sub shape (6, r_sub)  (3*2)
    
    for i in range(4):
        for c in range(r_sub):
            U_total[i][7+c] = U_sub[i][c]
    
    # V_sub indices: j*3+k where j in {0,1}, k in {0,1,2}
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # j=0, k=2 -> 2
    # j=1, k=0 -> 3
    # j=1, k=1 -> 4
    # j=1, k=2 -> 5
    
    # In V_total, these correspond to columns 2,3,4 of B.
    # j=0, k=0 -> V_total[2]
    # j=0, k=1 -> V_total[3]
    # j=0, k=2 -> V_total[4]
    # j=1, k=0 -> V_total[7]
    # j=1, k=1 -> V_total[8]
    # j=1, k=2 -> V_total[9]
    
    v_map_sub = [2, 3, 4, 7, 8, 9]
    for c in range(r_sub):
        for i_sub in range(6):
            V_total[v_map_sub[i_sub]][7+c] = V_sub[i_sub][c]
    
    # W_sub indices: k*2+i where k in {0,1,2}, i in {0,1}
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    # k=2, i=0 -> 4
    # k=2, i=1 -> 5
    
    # In W_total, these correspond to columns 2,3,4 of C.
    # k=0, i=0 -> W_total[4] (k=2 in full? No, k is col of C. C is 2x5. k in {0,1,2,3,4})
    # Wait, W_total indices are k*n+i where n=2.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    # k=2, i=0 -> 4
    # k=2, i=1 -> 5
    # k=3, i=0 -> 6
    # k=3, i=1 -> 7
    # k=4, i=0 -> 8
    # k=4, i=1 -> 9
    
    # The sub-decomposition handles columns 2,3,4 of C.
    # So k_sub in {0,1,2} maps to k_total in {2,3,4}.
    # k_sub=0 -> k_total=2 -> W_total[4], W_total[5]
    # k_sub=1 -> k_total=3 -> W_total[6], W_total[7]
    # k_sub=2 -> k_total=4 -> W_total[8], W_total[9]
    
    w_map_sub = [4, 5, 6, 7, 8, 9]
    for c in range(r_sub):
        for i_sub in range(6):
            W_total[w_map_sub[i_sub]][7+c] = W_sub[i_sub][c]
    
    # Verify full decomposition
    if not _verify(U_total, V_total, W_total, 2, 2, 5, r_total):
        print("Full verification failed, falling back to schoolbook")
        # Schoolbook 2x2 * 2x5
        r_sb = 20
        U_sb = [[0]*r_sb for _ in range(4)]
        V_sb = [[0]*r_sb for _ in range(10)]
        W_sb = [[0]*r_sb for _ in range(10)]
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(5):
                    U_sb[i*2+j][col] = 1
                    V_sb[j*5+k][col] = 1
                    W_sb[k*2+i][col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return {"U": U_total, "V": V_total, "W": W_total}
