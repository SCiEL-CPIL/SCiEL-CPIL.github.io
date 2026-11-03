"""Matrix multiplication 4x4x6 - structured decomposition with dependency elimination."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def _strassen_4x4_terms():
    """Return the 7 Strassen rank-1 terms for 2x2x2, then embed into 4x4x4.
    Each term is (u_vec, v_vec, w_vec) where u is length 4 (2x2 row-major),
    v length 4, w length 4 (w index = k*2 + i per trace convention W[k*n+i]).
    Actually for 2x2x2: U index i*m+j, V index j*p+k, W index k*n+i with n=m=p=2.
    We'll build 4x4x4 by tiling: use Kronecker of two Strassen 2x2 decompositions.
    That gives 7*7=49 terms for 4x4x4."""
    # Strassen 2x2: 7 terms. For each term t, define U (2x2), V (2x2), W (2x2)
    # such that the product tensor equals sum_t U_t \otimes V_t \otimes W_t
    # with W indexed as k*n+i (trace convention).
    # Standard Strassen:
    # M1 = (A11 - A22)(B21 + B22)
    # M2 = (A11 + A22)B22
    # M3 = A11(B11 - B22)
    # M4 = A22(B11 + B21)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # We need to express each as sum of rank-1 (u,v,w) where C = sum u \otimes (v \otimes w)^T appropriately.
    # The tensor T[i,j,k] = delta such that C[i,k] = sum_j A[i,j] B[j,k].
    # Rank-1 term: contribution to C[i,k] = u[i] * v[j] * w[k] summed over j.
    # So each term is (u of length 2, v of length 2, w of length 2) and contributes
    # u[i] * v[j] * w[k] to C[i,k] for the (i,j,k) triple.
    # Strassen terms (each is a product of a linear combo of A and linear combo of B):
    # Term structure: C += (sum_i u[i] * (linear A combo at i)) ... 
    # Let me define each Strassen M_t as: M_t = (a_t^T A) * (B b_t) where a_t, b_t are vectors?
    # No. M1 = (A11 - A22)(B21 + B22). This is scalar product of two linear combos.
    # It contributes M1 to C11 and M1 to C22 (with signs).
    # So for M1: u vector (for A side) and the A linear combo is (1,0,0,-1) in (A11,A12,A21,A22) order.
    # B linear combo is (0,0,1,1) in (B11,B12,B21,B22) order.
    # C contribution: +1 to C11, +1 to C22. So w vector (C11,C22) = (1,1) but we need per (i,k).
    # Let me just define the 7 terms explicitly as (u4, v4, w4) for 4x4x4 after Kronecker.
    # For 2x2x2, each term: u (len2, A row), v (len2, B col), w (len2, C: w[k*n+i] so w[0]=C11(i=1,k=1)? 
    # W index k*n+i, n=2: k=0,i=0 ->0 (C11), k=0,i=1->1(C12), k=1,i=0->2(C21), k=1,i=1->3(C22).
    # C[i,k] with i row, k col. C11 = i0k0 -> W[0]. C12=i0k1->W[1]. C21=i1k0->W[2]. C22=i1k1->W[3].
    # So w vector indexed [W0,W1,W2,W3] = [C11,C12,C21,C22].
    # For a rank-1 term, contribution to C[i,k] = u[i]*v[j]*w[k] summed over j (v is over j, the shared index).
    # Wait, v should be over the middle index j (A col / B row). Let me restate:
    # C[i,k] = sum_j A[i,j] B[j,k]. Rank-1: u[i]*v[j]*w[k]. Sum over j.
    # For Strassen M1 = (A11 - A22)(B21 + B22):
    #   A part: A11 - A22 = sum_j (A[i,j]) with coefficients: for i=0: j=0 coeff 1; i=1: j=1 coeff -1. So u? 
    #   Actually A11 is A[0,0], A22 is A[1,1]. The linear combo L_A = A11 - A22.
    #   B21 = B[1,0], B22 = B[1,1]. L_B = B21 + B22.
    #   M1 = L_A * L_B.
    #   This contributes to C: C11 += M1, C22 += M1.
    #   As rank-1 in the (i,j,k) tensor: we need u[i]*v[j]*w[k] such that sum_j u[i]v[j]w[k] = (L_A)(L_B) * (delta_{i,0}delta_{k,0} + delta_{i,1}delta_{k,1}).
    #   This is not a single rank-1 in general because L_A and L_B each involve sums.
    #   L_A = A[0,0] - A[1,1]. L_B = B[1,0] + B[1,1].
    #   M1 = (A[0,0]-A[1,1])(B[1,0]+B[1,1]).
    #   Contribution to C[0,0]: M1. Contribution to C[1,1]: M1.
    #   C[0,0] = sum_j A[0,j]B[j,0]. The M1 term adds (A[0,0]-A[1,1])(B[1,0]+B[1,1]) to C[0,0].
    #   This is a product of a linear form in A and a linear form in B, times a delta in C.
    #   As a tensor term in the full A,B,C space, this is rank-1: 
    #     a_vector (over A indices (i,j)): a[0,0]=1, a[1,1]=-1, rest 0.
    #     b_vector (over B indices (j,k)): b[1,0]=1, b[1,1]=1, rest 0.
    #     c_vector (over C indices (i,k)): c[0,0]=1, c[1,1]=1, rest 0.
    #   Term = a \otimes b \otimes c. This is one rank-1 term!
    #   So each Strassen M_t gives ONE rank-1 term with these three vectors.
    #   7 terms for 2x2x2.
    # For 4x4x4 via Kronecker: each 4x4x4 term = Kronecker of two 2x2x2 terms.
    #   a4 = kron(a1, a2), b4 = kron(b1, b2), c4 = kron(c1, c2). 49 terms.
    # Let me define the 7 terms for 2x2x2.
    terms2 = []
    # M1: a=(A11 - A22), b=(B21 + B22), c=(C11 + C22)
    a1 = np.zeros(4); a1[0]=1; a1[3]=-1   # A[0,0]=1, A[1,1]=-1
    b1 = np.zeros(4); b1[2]=1; b1[3]=1    # B[1,0]=1, B[1,1]=1
    c1 = np.zeros(4); c1[0]=1; c1[3]=1    # C[0,0]=1, C[1,1]=1
    terms2.append((a1,b1,c1))
    # M2: (A11 + A22)B22 -> a=(A11+A22), b=(B22), c=(C21 + C22)
    a2 = np.zeros(4); a2[0]=1; a2[3]=1
    b2 = np.zeros(4); b2[3]=1
    c2 = np.zeros(4); c2[2]=1; c2[3]=1
    terms2.append((a2,b2,c2))
    # M3: A11(B11 - B22) -> a=(A11), b=(B11 - B22), c=(C12 + C22)
    a3 = np.zeros(4); a3[0]=1
    b3 = np.zeros(4); b3[0]=1; b3[3]=-1
    c3 = np.zeros(4); c3[1]=1; c3[3]=1
    terms2.append((a3,b3,c3))
    # M4: A22(B11 + B21) -> a=(A22), b=(B11+B21), c=(C11 + C21)
    a4 = np.zeros(4); a4[3]=1
    b4 = np.zeros(4); b4[0]=1; b4[2]=1
    c4 = np.zeros(4); c4[0]=1; c4[2]=1
    terms2.append((a4,b4,c4))
    # M5: (A11 + A12)B22 -> a=(A11+A12), b=(B22), c=(C11 - C21)  [C11 += M5, C21 -= M5]
    a5 = np.zeros(4); a5[0]=1; a5[1]=1
    b5 = np.zeros(4); b5[3]=1
    c5 = np.zeros(4); c5[0]=1; c5[2]=-1
    terms2.append((a5,b5,c5))
    # M6: (A21 - A11)(B11 + B12) -> a=(A21 - A11), b=(B11+B12), c=(C22 + C21?) 
    # C22 += M6, and C? M6 appears only in C22. Let me check: C22 = M1 - M2 + M3 + M6. Yes only C22.
    a6 = np.zeros(4); a6[2]=1; a6[0]=-1
    b6 = np.zeros(4); b6[0]=1; b6[1]=1
    c6 = np.zeros(4); c6[3]=1
    terms2.append((a6,b6,c6))
    # M7: (A12 - A22)(B21 + B22) -> a=(A12 - A22), b=(B21+B22), c=(C11 + C12)
    # C11 += M7, C12 += M7.
    a7 = np.zeros(4); a7[1]=1; a7[3]=-1
    b7 = np.zeros(4); b7[2]=1; b7[3]=1
    c7 = np.zeros(4); c7[0]=1; c7[1]=1
    terms2.append((a7,b7,c7))
    return terms2

def _build_4x4_terms():
    terms2 = _strassen_4x4_terms()
    terms4 = []
    for t1 in terms2:
        for t2 in terms2:
            a = np.kron(t1[0], t2[0])
            b = np.kron(t1[1], t2[1])
            c = np.kron(t1[2], t2[2])
            terms4.append((a,b,c))
    return terms4

def _build_4x4x6_terms():
    """Build 4x4x6 terms: use 49 terms for the 4x4x4 part, and for the 6th column of B / row of C,
    we need to extend. Actually 4x4x6 means A is 4x4, B is 4x6, C is 4x6.
    The tensor is T[i,j,k] with i in 0..3 (A row), j in 0..3 (A col / B row), k in 0..5 (B col / C col).
    C[i,k] = sum_j A[i,j] B[j,k].
    We can decompose B and C columns: for k in 0..3 use the 4x4x4 Strassen terms, and for k in 4,5 
    we need additional terms. 
    Alternative: treat as 4x4x4 (for first 4 cols) plus 4x4x2 (for last 2 cols).
    For the 4x4x2 part: C[i,4] and C[i,5] = sum_j A[i,j] B[j,4], A[i,j]B[j,5].
    This is like multiplying A (4x4) by a 4x2 matrix. We can use a simple decomposition:
    for each (i,j,k) with k in {4,5}: term with a=e_i (A row i), b=e_j (B row j), c=e_k (C col k).
    That's 4*4*2 = 32 terms. Total 49+32 = 81. Not great.
    Better: use Strassen for the 4x4x4 and a smarter 4x4x2.
    Actually, we can view 4x4x6 as: split B into B1 (4x4) and B2 (4x2). C = A*B1 + A*B2.
    A*B1: 49 terms (Strassen). A*B2: we need rank for 4x4x2.
    For 4x4x2, minimal rank is at most 4*4*2=32 but can be less. 
    Actually, A*B2 where B2 is 4x2: C2[i,k] = sum_j A[i,j] B2[j,k] for k=0,1.
    We can factor: C2 = A * B2. This is a 4x4 times 4x2. 
    Rank of this tensor: we can use the fact that B2 has only 2 columns.
    A simple bound: for each column k of B2, C[:,k] = A * B2[:,k]. 
    We can write C[:,k] = sum_j A[:,j] * B2[j,k]. 
    As tensor terms: for each j and k, term = (e_j in A-col) \otimes (e_j in B-row) \otimes (e_k in C-col)? 
    No. Let me think: C[i,k] = sum_j A[i,j] B2[j,k].
    Rank-1 term: u[i] * v[j] * w[k]. Sum over j.
    For a fixed k, C[:,k] = A * B2[:,k]. This is a matrix product of 4x4 and 4x1.
    We can decompose A * (vector) as sum of rank-1. But A is a variable matrix.
    Actually the tensor for 4x4x2: T[i,j,k] = delta_{i,?}... no.
    T[i,j,k] is the coefficient such that C[i,k] = sum_j A[i,j] B[j,k].
    For the full tensor, T[i,j,k] = 1 if we consider the identity, but that's the schoolbook.
    The tensor is T[i,j,k] = delta (it's the multiplication tensor, T[i,j,k]=1 for all? No).
    The multiplication tensor for m*n*p is T[i,j,k] where C[i,k] = sum_j A[i,j]B[j,k], so T[i,j,k] = 1 for all i,j,k (it's the all-ones in the sense that each (i,j,k) triple contributes). 
    Wait no. The tensor T has T[i,j,k] as the coefficient of A[i,j]*B[j,k] in C[i,k]. Since C[i,k] = sum_j A[i,j] B[j,k], the coefficient of A[i,j]*B[j,k] in C[i,k] is 1 if the indices match, else 0. So T[i,j,k] = 1 for all i,j,k (each triple (i,j,k) corresponds to A[i,j]*B[j,k] contributing to C[i,k] with coeff 1). Yes! The multiplication tensor is the all-ones tensor of shape (n*m, m*p, p*n) in the flattened sense, but structured: T[i,j,k]=1.
    So for 4x4x6, T[i,j,k]=1 for i in 0..3, j in 0..3, k in 0..5. Total 4*4*6=96 entries, all 1.
    A rank-1 decomposition: T = sum_t u_t \otimes v_t \otimes w_t where u_t is length 16 (A), v_t length 24 (B), w_t length 24 (C).
    u_t[i*m+j] = u_t[i*4+j], v_t[j*p+k] = v_t[j*6+k], w_t[k*n+i] = w_t[k*4+i].
    We need sum_t u_t[i*4+j] * v_t[j*6+k] * w_t[k*4+i] = 1 for all i,j,k.
    This is a system of 96 equations.
    For the 4x4x4 part (k in 0..3), we have 49 Strassen terms that satisfy sum u[i*4+j]v[j*4+k]w[k*4+i] = 1 for k in 0..3.
    But our B and C have p=6, so v is length 24, w is length 24. The Strassen terms for 4x4x4 have v length 16, w length 16. We need to extend them to length 24 by adding zeros for k=4,5.
    So the 49 Strassen terms, when embedded (v padded to 24, w padded to 24), satisfy the equations for k in 0..3, and contribute 0 for k=4,5.
    Then we need additional terms to satisfy the 32 equations for k=4,5 (i in 0..3, j in 0..3, k in 4,5: 4*4*2=32 equations).
    The additional terms must have u, v, w such that they don't disturb the k=0..3 equations (or we allow them to and re-solve, but that's complex).
    Simplest: additional terms that are zero for k=0..3 in v and w, so they only affect k=4,5.
    For k=4,5: we need sum_t u_t[i*4+j] v_t[j*6+k] w_t[k*4+i] = 1.
    If we make v_t nonzero only at k=4 or 5, and w_t nonzero only at k=4 or 5, then these terms only affect k=4,5.
    For a term with v nonzero only at (j,4): v[j*6+4] = a, w[4*4+i] = b, u[i*4+j] = c.
    Contribution to C[i,4] = c*a*b (for the specific i,j). But we need sum over j.
    Let's just use schoolbook for the k=4,5 part: for each i,j,k in {4,5}: term with u[i*4+j]=1, v[j*6+k]=1, w[k*4+i]=1. That's 32 terms.
    Total 49+32 = 81 terms.
    Can we do better for the 4x4x2 part? The 4x4x2 tensor (i in 0..3, j in 0..3, k in 0..1) has T=1 for all 32 entries.
    We want rank of this 4x4x2 tensor. 
    Note: 4x4x2 tensor: C[i,k] = sum_j A[i,j] B[j,k] for k=0,1.
    This is equivalent to two matrix-vector products: C[:,0] = A*B[:,0], C[:,1] = A*B[:,1].
    We can write C = A * B2 where B2 is 4x2.
    Rank of 4x4x2 multiplication tensor: I recall it's at most 4*4 + 4*2 = 24? Or use the fact that it's a "thin" matrix multiplication.
    Actually, a known bound: rank of n x m x p <= n*m + m*p + n*p - ... no.
    Let me just use a constructive approach: 
    C[i,k] = sum_j A[i,j] B[j,k].
    Write B[j,k] for k=0,1. 
    C[i,0] = sum_j A[i,j] B[j,0], C[i,1] = sum_j A[i,j] B[j,1].
    We can introduce auxiliary variables. 
    For each j, define S_j = A[:,j] (a column of A, length 4). Then C[:,k] = sum_j S_j * B[j,k].
    This is a linear combination of the columns of A, with coefficients B[j,k].
    As a tensor, this is sum over j of (S_j) \otimes (e_j in B) \otimes (B[:,k] in C)? Not quite rank-1.
    Let me just use the schoolbook 32 terms for now, total 81, and then try to reduce.
    Actually, let me reconsider. Maybe I should just build the 81-term decomposition and then run dependency elimination to reduce it.
    """
    terms4 = _build_4x4_terms()  # 49 terms, each (a16, b16, c16)
    terms = []
    for (a,b,c) in terms4:
        a24 = np.zeros(16)  # A is 4x4, so length 16
        a24[:] = a
        b24 = np.zeros(24)  # B is 4x6, length 24
        b24[:16] = b  # b is length 16 (4x4), place in first 16 (k=0..3)
        c24 = np.zeros(24)  # C is 4x6, length 24
        c24[:16] = c  # c is length 16, place in first 16 (k=0..3)
        terms.append((a24, b24, c24))
    # Add schoolbook terms for k=4,5
    for i in range(4):
        for j in range(4):
            for k in [4,5]:
                a = np.zeros(16); a[i*4+j] = 1
                b = np.zeros(24); b[j*6+k] = 1
                c = np.zeros(24); c[k*4+i] = 1
                terms.append((a,b,c))
    return terms

def _verify_decomposition(terms, n=4, m=4, p=6, tol=1e-9):
    """Check that sum_t u_t[i*m+j] * v_t[j*p+k] * w_t[k*n+i] == 1 for all i,j,k."""
    n, m, p = 4, 4, 6
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for (u,v,w) in terms:
                    s += u[i*m+j] * v[j*p+k] * w[k*n+i]
                if abs(s - 1.0) > tol:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    return ok

def _build_decomposition_from_terms(terms):
    rank = len(terms)
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(24)]
    W = [[0]*rank for _ in range(24)]
    for t, (u,v,w) in enumerate(terms):
        for idx in range(16):
            if u[idx] != 0:
                U[idx][t] = float(u[idx])
        for idx in range(24):
            if v[idx] != 0:
                V[idx][t] = float(v[idx])
            if w[idx] != 0:
                W[idx][t] = float(w[idx])
    return {"U": U, "V": V, "W": W}

def _try_reduce_rank(terms, deadline):
    """Try to reduce rank by finding linear dependencies.
    Represent each term as a vector in R^{16*24*24}. Check if any term is in the span of others.
    If so, remove it. Repeat until no more can be removed or deadline."""
    start = time.time()
    n_terms = len(terms)
    # Build the big matrix: each term is a row of length 16*24*24 = 9216
    # That's too big. Instead, use the 96 equations: each term defines a vector in R^{96} (its contribution to each (i,j,k)).
    # Two terms are "equivalent" if their contribution vectors are proportional.
    # Actually, we want to find if sum_t alpha_t * vec_t = 0 for some alpha, and then remove one term.
    # Build matrix M of shape (96, n_terms) where M[i,j,k, t] = u_t[i*m+j]*v_t[j*p+k]*w_t[k*n+i].
    # We need M * 1 = 1 (the all-ones vector). We want to find the minimal number of terms.
    # This is: find alpha in R^{n_terms} such that M * alpha = 1, minimizing sum |alpha_t| > 0 (i.e., number of nonzero alpha).
    # That's a sparse recovery / L0 problem, hard.
    # Simpler: iteratively check if removing term t still allows a solution.
    # i.e., is 1 in the column space of M without column t?
    # Check: rank(M) == rank(M without col t) and 1 is in the span.
    # Let me build M as a 96 x n_terms matrix.
    n_eq = 16*24  # 384? No, n*m*p = 4*4*6 = 96 equations.
    n_eq = 96
    M = np.zeros((n_eq, n_terms))
    eq_idx = 0
    for i in range(4):
        for j in range(4):
            for k in range(6):
                for t in range(n_terms):
                    M[eq_idx, t] = terms[t][0][i*4+j] * terms[t][1][j*6+k] * terms[t][2][k*4+i]
                eq_idx += 1
    target = np.ones(n_eq)
    # Check initial validity
    if not np.allclose(M @ np.ones(n_terms), target, atol=1e-9):
        return terms  # invalid, return as is
    # Iteratively try to remove terms
    active = list(range(n_terms))
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for t in list(active):
            if len(active) <= 1:
                break
            # Try removing t
            remaining = [x for x in active if x != t]
            M_sub = M[:, remaining]
            # Check if target is in the column space of M_sub
            # i.e., does M_sub * alpha = target have a solution?
            # Check rank(M_sub) == rank([M_sub | target])
            r1 = np.linalg.matrix_rank(M_sub)
            r2 = np.linalg.matrix_rank(np.column_stack([M_sub, target]))
            if r1 == r2:
                # Check if there's a solution with all alpha nonzero? No, we just need existence.
                # But we also need the solution to give exactly 1, not just be in the span.
                # If target is in the column space, there exists alpha such that M_sub @ alpha = target.
                # That's sufficient! We can remove t.
                active = remaining
                improved = True
                break
    return [terms[t] for t in active]

def construct():
    deadline = time.time() + 130  # safety margin
    start = time.time()
    # Build initial 81-term decomposition
    terms = _build_4x4x6_terms()
    # Verify
    if not _verify_decomposition(terms):
        # Fallback to schoolbook
        rank = 4*4*6
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(24)]
        W = [[0]*rank for _ in range(24)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(6):
                    U[i*4+j][col] = 1
                    V[j*6+k][col] = 1
                    W[k*4+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    # Try to reduce rank
    if time.time() < deadline - 20:
        terms = _try_reduce_rank(terms, deadline)
    # Final verification
    if not _verify_decomposition(terms):
        # Fallback
        rank = 4*4*6
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(24)]
        W = [[0]*rank for _ in range(24)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(6):
                    U[i*4+j][col] = 1
                    V[j*6+k][col] = 1
                    W[k*4+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    return _build_decomposition_from_terms(terms)
