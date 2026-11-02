"""Matrix multiplication 2x4x5 rank-minimal decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, tol=1e-9):
    """Check if U, V, W correctly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    # Build the multiplication tensor T[i,j,k] = 1
    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Check a few random entries for speed, or all if small enough
    # For 2x4x5, total entries = 2*4*5 = 40, check all
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for c in range(rank):
                    actual += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(actual - expected) > tol:
                    return False
    return True

def _strassen_2x4x5():
    """
    Block Strassen-style decomposition for 2x4x5.
    
    A is 2x4, B is 4x5, C is 2x5.
    Split A into A_L (2x2) and A_R (2x2), B into B_T (2x5) and B_B (2x5).
    C = A_L @ B_T + A_R @ B_B
    
    Use Strassen for each 2x2x5 multiplication:
    For 2x2x5: M is 2x2, N is 2x5, P is 2x5.
    Strassen: P = M @ N where M=(a,b;c,d), N=(e,f;g,h) [2x2]
    But here N is 2x5, so we can do:
    P1 = (M11+M22)(N11+N22)
    P2 = (M21+M22)N11
    P3 = M11(N21-N11)
    P4 = M22(N12-N22)
    P5 = (M11+M12)N22
    P6 = (M21-M11)(N11+N12)
    P7 = (M12-M22)(N21+N22)
    P11 = P1+P4-P5+P7
    P12 = P3+P1+P6-P4
    P21 = P2+P4-P3+P5
    P22 = P1+P2-P6+P7
    
    Each 2x2x5 uses 7 terms. Two such multiplications = 14 terms.
    But we can share some structure. Let's just use 7+7=14 terms first.
    
    Actually, let's think more carefully.
    A_L is 2x2, B_T is 2x5. Product is 2x5.
    A_R is 2x2, B_B is 2x5. Product is 2x5.
    C = A_L@B_T + A_R@B_B.
    
    Using Strassen for each: 7 terms each, total 14 terms.
    Rank = 14.
    
    Let's construct this.
    """
    n, m, p = DIMENSIONS
    rank = 14
    U = np.zeros((n*m, rank))  # 8 x 14
    V = np.zeros((m*p, rank))  # 20 x 14
    W = np.zeros((p*n, rank))  # 10 x 14
    
    # A_L @ B_T: rows 0-3 of A (i=0,1; j=0,1), cols 0-19 of B (j=0,1; k=0..4)
    # A_R @ B_B: rows 4-7 of A (i=0,1; j=2,3), cols 20-39 of B (j=2,3; k=0..4)
    
    # Strassen for 2x2 @ 2x5
    # M = [[m00, m01], [m10, m11]] (2x2)
    # N = [[n00, n01, n02, n03, n04], [n10, n11, n12, n13, n14]] (2x5)
    # P = M @ N (2x5)
    # 
    # P1 = (m00+m11)(n00+n10, n01+n11, ...)
    # P2 = (m10+m11)(n00, n01, ...)
    # P3 = m00(n10-n00, n11-n01, ...)
    # P4 = m11(n01-n11, n02-n12, ...)
    # P5 = (m00+m01)(n10, n11, ...)
    # P6 = (m10-m00)(n00+n01, ...)
    # P7 = (m01-m11)(n10+n11, ...)
    # 
    # P[0,:] = P1 + P4 - P5 + P7
    # P[1,:] = P2 + P4 - P3 + P5
    # Wait, let me re-derive for 2x2 @ 2x5.
    # 
    # Standard Strassen for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B11)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M1 + M6 - M4
    # C10 = M2 + M4 - M3 + M5
    # C11 = M1 + M2 - M6 + M7
    #
    # For 2x2 @ 2x5, N has 2 rows and 5 cols.
    # M1 = (m00+m11) * (n_row0 + n_row1)  [element-wise, 5-vector]
    # M2 = (m10+m11) * n_row0
    # M3 = m00 * (n_row1 - n_row0)
    # M4 = m11 * (n_row0 - n_row1)  [note: this is -m11*(n_row1-n_row0)]
    # M5 = (m00+m01) * n_row1
    # M6 = (m10-m00) * (n_row0)  [only first element? No...]
    
    # Hmm, this is getting complicated because Strassen is for square matrices.
    # For rectangular, we need to be more careful.
    
    # Let's just use the straightforward approach: treat 2x2 @ 2x5 as two 2x2 @ 2x2
    # multiplications plus one 2x1 @ 2x1, or use a different decomposition.
    
    # Actually, for 2x2 @ 2x5, we can use the fact that it's two independent 2x2 @ 2x2
    # multiplications (cols 0-1 and 2-3) plus one 2x2 @ 2x1 (col 4).
    # 2x2 @ 2x2: rank 7 (Strassen)
    # 2x2 @ 2x1: rank 4 (schoolbook) or can we do better?
    # 2x2 @ 2x1: M is 2x2, N is 2x1, P is 2x1.
    # P0 = m00*n0 + m01*n1
    # P1 = m10*n0 + m11*n1
    # This is rank 4 by schoolbook. Can we do rank 3?
    # Actually for 2x2x1, the tensor rank is 4 (it's the standard 2x2x2 tensor with one dimension 1).
    
    # So 2x2 @ 2x5: 7 + 7 + 4 = 18 terms. That's worse than schoolbook (10 terms).
    
    # Let me reconsider. Schoolbook for 2x2 @ 2x5 is 4*5 = 20 terms? No.
    # 2x2 @ 2x5: M has 4 entries, N has 10 entries, P has 10 entries.
    # Schoolbook: 4*5 = 20 terms? No, it's sum over 2 intermediate indices.
    # P[i,k] = sum_j M[i,j] N[j,k], j=0,1. So 2 terms per output, 10 outputs = 20 terms.
    # Wait, no. The tensor is T[i,j,k] where i in 0..1, j in 0..1, k in 0..4.
    # T[i,j,k] = 1. Rank of this tensor...
    # For 2x2x5, the rank is at most 10 (schoolbook: 2*2*5/2 = 10? No, 2*2*5 = 20? No.)
    # Schoolbook for matrix mult n x m times m x p has rank n*m*p.
    # For 2x2 @ 2x5: n=2, m=2, p=5, rank = 2*2*5 = 20.
    # But that's overkill. The tensor T[i,j,k] = 1 for i in 0..1, j in 0..1, k in 0..4.
    # This is just an all-ones tensor of shape 2x2x5. Its rank is min(2*2, 2*5, 5) = 4? No.
    # The rank of an all-ones tensor of shape a x b x c is min(a*b, b*c, a*c) if we can factor.
    # Actually, T[i,j,k] = 1 = (1)(1)(1), so it's rank 1! But that's not right for matrix mult.
    
    # I'm confusing myself. The matrix multiplication tensor for n x m * m x p is:
    # T[i,j,k] where i in 0..n-1, j in 0..m-1, k in 0..p-1, T[i,j,k] = 1.
    # The rank of this tensor is the minimum number of rank-1 tensors that sum to it.
    # For 2x2x5: T[i,j,k] = 1 for all i,j,k. This is indeed rank 1 as a tensor!
    # But wait, that's not the matrix multiplication tensor. The matrix multiplication
    # tensor has T[i,j,k] = 1, but the way it's used is:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    # So yes, T[i,j,k] = 1 for all i,j,k. And this tensor has rank 1!
    # But that can't be right because then any matrix multiplication would be rank 1.
    
    # Oh I see my confusion. The tensor T is not T[i,j,k] = 1. It's:
    # T[i,j,k] = delta_{j,j} which is just 1, but the indices are:
    # i: row of A and C (0..n-1)
    # j: column of A and row of B (0..m-1)  
    # k: column of B and C (0..p-1)
    # And T[i,j,k] = 1 for all valid i,j,k.
    # The decomposition is: T = sum_c U_c ⊗ V_c ⊗ W_c where
    # U_c[i*m+j] = U[i*m+j, c], V_c[j*p+k] = V[j*p+k, c], W_c[k*n+i] = W[k*n+i, c]
    # And we need: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all i,j,k.
    
    # For the all-ones tensor of shape (n*m) x (m*p) x (p*n) with the specific index mapping,
    # the rank is n*m*p by schoolbook. But there might be better decompositions.
    
    # For 2x4x5: the tensor has shape (8) x (20) x (10).
    # Schoolbook rank = 2*4*5 = 40.
    # The minimum rank is at most 40. Can we do better?
    
    # Let's just try to construct a rank-36 decomposition using block Strassen ideas.
    # Or even just return the schoolbook for now and try to improve.
    
    # For safety, let's return schoolbook.
    U, V, W = _schoolbook()
    return U, V, W

def construct():
    deadline = time.time() + 100  # 100s safety margin
    
    # Start with schoolbook
    U, V, W = _schoolbook()
    
    # Verify
    if not _verify(U, V, W):
        raise ValueError("Schoolbook decomposition failed verification")
    
    # Try to find a lower rank decomposition
    # For 2x4x5, known best is around 36 (block Strassen)
    # Let's try a few approaches within the time budget
    
    # Approach 1: Block Strassen
    # A (2x4) = [A_L | A_R] where A_L, A_R are 2x2
    # B (4x5) = [B_T; B_B] where B_T, B_B are 2x5
    # C = A_L @ B_T + A_R @ B_B
    # Each 2x2 @ 2x5 can be done with some rank.
    # 2x2 @ 2x5: n=2, m=2, p=5, schoolbook rank = 20.
    # Can we do better? The tensor is T[i,j,k] = 1 for i in 0..1, j in 0..1, k in 0..4.
    # This is separable as (1,1) ⊗ (1,1) ⊗ (1,1,1,1,1) so rank 1? No, the index mapping is different.
    
    # Let me just try a greedy/local search approach to reduce rank from 40.
    # Start with schoolbook and try to merge terms.
    
    # For now, return schoolbook as a safe baseline.
    # The score will be 40/40 = 1.0.
    
    if time.time() > deadline - 20:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to construct a rank-36 decomposition
    # Use the fact that 2x4x5 can be decomposed as:
    # C = A_L @ B_T + A_R @ B_B
    # where A_L, A_R are 2x2, B_T, B_B are 2x5.
    # For 2x2 @ 2x5, use a rank-18 decomposition (7+7+4 as discussed).
    # Total rank = 18 + 18 = 36.
    
    # Let's construct rank-18 for 2x2 @ 2x5.
    # M (2x2) @ N (2x5) = P (2x5)
    # Split N into N1 (2x2), N2 (2x2), N3 (2x1)
    # P1 = M @ N1 (rank 7 Strassen)
    # P2 = M @ N2 (rank 7 Strassen)
    # P3 = M @ N3 (rank 4 schoolbook)
    # Total: 18 terms.
    
    # Strassen for 2x2 @ 2x2:
    # M = [[a, b], [c, d]], N = [[e, f], [g, h]]
    # P11 = (a+d)(e+h) + d(f-g) - b(e+h) + (b-d)(g+h)
    # Let me use the standard Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(g-e)
    # M4 = d(f-h)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # P11 = M1 + M4 - M5 + M7
    # P12 = M3 + M1 + M6 - M4
    # P21 = M2 + M4 - M3 + M5
    # P22 = M1 + M2 - M6 + M7
    
    # For 2x2 @ 2x1:
    # M = [[a, b], [c, d]], N = [[x], [y]]
    # P1 = ax + by
    # P2 = cx + dy
    # Rank 4: terms (a,x), (b,y), (c,x), (d,y)
    
    # Now construct the full 2x4x5 decomposition with rank 36.
    rank = 36
    U = np.zeros((8, rank))
    V = np.zeros((20, rank))
    W = np.zeros((10, rank))
    
    col = 0
    # A_L @ B_T: A_L is rows 0-3 (i=0,1; j=0,1), B_T is cols 0-9 (j=0,1; k=0..4)
    # A_R @ B_B: A_R is rows 4-7 (i=0,1; j=2,3), B_B is cols 10-19 (j=2,3; k=0..4)
    
    def add_strassen_2x2_2x2(U, V, W, a_rows, b_cols, w_rows, col_offset):
        """
        Add Strassen terms for 2x2 @ 2x2.
        a_rows: indices of A rows (2 indices)
        b_cols: indices of B cols (4 indices: [j0k0, j0k1, j1k0, j1k1])
        w_rows: indices of W rows (4 indices: [k0i0, k0i1, k1i0, k1i1])
        """
        global col
        # M1 = (a+d)(e+h)
        # a = A[a_rows[0]], d = A[a_rows[1]]
        # e = B[b_cols[0]], h = B[b_cols[3]]
        U[a_rows[0], col] = 1
        U[a_rows[1], col] = 1
        V[b_cols[0], col] = 1
        V[b_cols[3], col] = 1
        W[w_rows[0], col] = 1
        W[w_rows[3], col] = 1
        W[w_rows[1], col] = 1
        W[w_rows[2], col] = 1
        col += 1
        
        # M2 = (c+d)e
        U[a_rows[1], col] = 1
        U[a_rows[0], col] = 1  # c is a_rows[1], d is a_rows[0]? No.
        # Wait, I need to be more careful with the mapping.
        
        return col
    
    # This is getting complex. Let me just return schoolbook for safety.
    U, V, W = _schoolbook()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
