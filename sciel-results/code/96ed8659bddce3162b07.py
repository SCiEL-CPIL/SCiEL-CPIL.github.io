"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction
from itertools import combinations
from collections import defaultdict

import numpy as np

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _verify_exact(U, V, W):
    """Verify exact decomposition using Fraction arithmetic."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(V[0]) != rank or len(W[0]) != rank:
        return False
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += U_frac[i * m + j][c] * V_frac[j * p + k][c] * W_frac[k * n + i][c]
                if s != 1:
                    return False
    return True

def _strassen_2x2_2x2():
    """Return rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2.
    Returns U (4x7), V (4x7), W (4x7) where rows are indexed by (i,j), (j,k), (k,i) respectively.
    """
    # Strassen's 7 products for 2x2 matrix multiplication
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-d)(f+g)
    # M7 = (b-d)(e+g)
    # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    
    # U rows: (i,j) for i in {0,1}, j in {0,1} -> A[i,j]
    # V rows: (j,k) for j in {0,1}, k in {0,1} -> B[j,k]
    # W rows: (k,i) for k in {0,1}, i in {0,1} -> C[k,i]
    
    # U: 4x7
    # Row 0: A[0,0]=a, Row 1: A[0,1]=b, Row 2: A[1,0]=c, Row 3: A[1,1]=d
    U = [
        [1, 0, 1, 0, 1, 0, 0],   # a: M1, M3, M5
        [0, 0, 0, 0, 1, 1, 1],   # b: M5, M6, M7
        [0, 1, 0, 0, 0, 1, -1],  # c: M2, M6, M7
        [1, 1, 0, 1, 0, -1, -1]  # d: M1, M2, M4, M6, M7
    ]
    
    # V: 4x7
    # Row 0: B[0,0]=e, Row 1: B[0,1]=f, Row 2: B[1,0]=g, Row 3: B[1,1]=h
    V = [
        [1, 1, 0, -1, 0, 0, 1],  # e: M1, M2, M4, M7
        [0, 0, 1, 0, 0, 1, 0],   # f: M3, M6
        [0, 0, 0, 1, 0, 1, 1],   # g: M4, M6, M7
        [1, 0, -1, 0, 1, 0, 0]   # h: M1, M3, M5
    ]
    
    # W: 4x7
    # Row 0: C[0,0], Row 1: C[0,1], Row 2: C[1,0], Row 3: C[1,1]
    W = [
        [1, 0, 0, 1, -1, 0, 1],  # C[0,0]: M1, M4, -M5, M7
        [0, 0, 1, 0, 1, 0, 0],   # C[0,1]: M3, M5
        [0, 1, 0, 1, 0, 0, 0],   # C[1,0]: M2, M4
        [1, -1, 1, 0, 0, 1, 0]   # C[1,1]: M1, -M2, M3, M6
    ]
    
    return U, V, W

def _verify_decomp(n, m, p, U, V, W):
    """Verify that U, V, W decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    U_r = U_np.reshape(n, m, rank)
    V_r = V_np.reshape(m, p, rank)
    W_r = W_np.reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_r[i, j, :], V_r[j, k, :] * W_r[k, i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _hybrid_strassen_schoolbook():
    """
    Build a hybrid decomposition using Strassen for 2x2*2x2 blocks and schoolbook for the rest.
    
    A is 2x3, B is 3x7, C is 2x7.
    
    Split B into 3 columns groups: B = [B1(3x2), B2(3x2), B3(3x3)]
    Actually, let's think differently.
    
    Split A into columns: A = [A1(2x1), A2(2x1), A3(2x1)]
    Split B into rows: B = [B1(1x7); B2(1x7); B3(1x7)]
    
    C = A1*B1 + A2*B2 + A3*B3
    
    For each term Ai*Bi, we have a 2x1 * 1x7 = 2x7 product.
    Each such product has rank 2 (outer product of 2-vector and 7-vector... no, it's rank 1 per output element).
    
    Actually, let me think about this more carefully.
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    For the 2x2*2x2 Strassen trick, we can group columns of A and rows of B.
    
    Let's split the 3 columns of A into groups: {col0, col1} and {col2}
    Let's split the 3 rows of B into groups: {row0, row1} and {row2}
    
    C = A[:,0:2] * B[0:2,:] + A[:,2:3] * B[2:3,:]
    
    First term: 2x2 * 2x7. We can further split the 7 columns of B into {0,1}, {2,3}, {4,5}, {6}
    So B[0:2,:] = [B11(2x2), B12(2x2), B13(2x2), B14(2x1)]
    
    For each 2x2 * 2x2 block, use Strassen (rank 7)
    For each 2x2 * 2x1 block, use schoolbook (rank 2)
    
    Second term: 2x1 * 1x7. This is rank 2 (each output C[i,k] = A[i,2]*B[2,k]).
    
    Total: 3*7 + 3*2 + 2 = 21 + 6 + 2 = 29? That seems too good. Let me recheck.
    
    Wait, for 2x2 * 2x1: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]. This is 2 outputs * 1 column = 2 products, rank 2.
    
    So total = 3*7 (Strassen for three 2x2*2x2) + 3*2 (schoolbook for three 2x2*2x1) + 2 (for 2x1*1x7)
    = 21 + 6 + 2 = 29
    
    Hmm, but we need to be careful about the column splitting.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=7
    
    # Split A's columns into: group1 = {0,1}, group2 = {2}
    # Split B's rows into: group1 = {0,1}, group2 = {2}
    # Split B's columns into: {0,1}, {2,3}, {4,5}, {6}
    
    # Term 1: A[:, 0:2] (2x2) * B[0:2, :] (2x7)
    #   Sub-term 1a: A[:, 0:2] * B[0:2, 0:2] (2x2 * 2x2) -> Strassen, rank 7
    #   Sub-term 1b: A[:, 0:2] * B[0:2, 2:4] (2x2 * 2x2) -> Strassen, rank 7
    #   Sub-term 1c: A[:, 0:2] * B[0:2, 4:6] (2x2 * 2x2) -> Strassen, rank 7
    #   Sub-term 1d: A[:, 0:2] * B[0:2, 6:7] (2x2 * 2x1) -> schoolbook, rank 2
    
    # Term 2: A[:, 2:3] (2x1) * B[2:3, :] (1x7)
    #   Sub-term 2a: A[:, 2:3] * B[2:3, 0:7] (2x1 * 1x7) -> schoolbook, rank 2
    
    # Total rank: 7+7+7+2+2 = 25
    
    rank = 25
    U = [[0] * rank for _ in range(n * m)]  # 6 x 25
    V = [[0] * rank for _ in range(m * p)]  # 21 x 25
    W = [[0] * rank for _ in range(p * n)]  # 14 x 25
    
    col_offset = 0
    
    # Helper to add a Strassen block
    def add_strassen(a_row_offsets, b_row_offsets, c_col_offsets, offset):
        """
        Add Strassen decomposition for A[a_row_offsets, :] * B[:, b_row_offsets] contributing to C[:, c_col_offsets]
        a_row_offsets: list of 2 row indices in A (for the 2x2 block)
        b_row_offsets: list of 2 row indices in B
        c_col_offsets: list of 2 col indices in C
        """
        Su, Sv, Sw = _strassen_2x2_2x2()
        # A rows: a_row_offsets[0], a_row_offsets[1] -> indices a_row_offsets[0]*m + j... 
        # Wait, U is indexed by (i, j) where i is row of A, j is col of A.
        # For the 2x2 block A[a0:a0+2, b0:b0+2], the U rows are (a0, b0), (a0, b0+1), (a0+1, b0), (a0+1, b0+1)
        # But in our case, A's columns are the "j" index and B's rows are also "j".
        
        # Actually, let me re-examine. The matrix multiplication is:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # U is indexed by (i,j), V by (j,k), W by (k,i)
        
        # For a 2x2 * 2x2 block: A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        # The U rows involved: (i0, j0), (i0, j0+1), (i0+1, j0), (i0+1, j0+1)
        # The V rows involved: (j0, k0), (j0, k0+1), (j0+1, k0), (j0+1, k0+1)
        # The W rows involved: (k0, i0), (k0, i0+1), (k0+1, i0), (k0+1, i0+1)
        
        a_rows = a_row_offsets  # 2 row indices of A
        b_cols = b_row_offsets  # 2 col indices of A = 2 row indices of B
        c_cols = c_col_offsets  # 2 col indices of C
        
        # U indices: (a_rows[0], b_cols[0]), (a_rows[0], b_cols[1]), (a_rows[1], b_cols[0]), (a_rows[1], b_cols[1])
        u_idx = [
            a_rows[0] * m + b_cols[0],
            a_rows[0] * m + b_cols[1],
            a_rows[1] * m + b_cols[0],
            a_rows[1] * m + b_cols[1]
        ]
        # V indices: (b_cols[0], c_cols[0]), (b_cols[0], c_cols[1]), (b_cols[1], c_cols[0]), (b_cols[1], c_cols[1])
        v_idx = [
            b_cols[0] * p + c_cols[0],
            b_cols[0] * p + c_cols[1],
            b_cols[1] * p + c_cols[0],
            b_cols[1] * p + c_cols[1]
        ]
        # W indices: (c_cols[0], a_rows[0]), (c_cols[0], a_rows[1]), (c_cols[1], a_rows[0]), (c_cols[1], a_rows[1])
        w_idx = [
            c_cols[0] * n + a_rows[0],
            c_cols[0] * n + a_rows[1],
            c_cols[1] * n + a_rows[0],
            c_cols[1] * n + a_rows[1]
        ]
        
        for c in range(7):
            for r in range(4):
                U[u_idx[r]][offset + c] += Su[r][c]
                V[v_idx[r]][offset + c] += Sv[r][c]
                W[w_idx[r]][offset + c] += Sw[r][c]
        
        return 7
    
    def add_schoolbook_2x2_2x1(a_rows, b_cols, c_cols, offset):
        """
        Add schoolbook for A[a_rows, b_cols] (2x2) * B[b_cols, c_cols] (2x1) -> C[a_rows, c_cols] (2x1)
        C[i, k] = A[i, j0]*B[j0, k] + A[i, j1]*B[j1, k]
        """
        a0, a1 = a_rows
        j0, j1 = b_cols
        k0 = c_cols[0]
        
        # Term 1: A[a0,j0]*B[j0,k0] -> C[a0,k0], A[a1,j0]*B[j0,k0] -> C[a1,k0]
        U[a0 * m + j0][offset] = 1
        V[j0 * p + k0][offset] = 1
        W[k0 * n + a0][offset] = 1
        
        U[a1 * m + j0][offset] = 1
        V[j0 * p + k0][offset] = 1
        W[k0 * n + a1][offset] = 1
        
        # Term 2: A[a0,j1]*B[j1,k0] -> C[a0,k0], A[a1,j1]*B[j1,k0] -> C[a1,k0]
        U[a0 * m + j1][offset + 1] = 1
        V[j1 * p + k0][offset + 1] = 1
        W[k0 * n + a0][offset + 1] = 1
        
        U[a1 * m + j1][offset + 1] = 1
        V[j1 * p + k0][offset + 1] = 1
        W[k0 * n + a1][offset + 1] = 1
        
        return 2
    
    def add_schoolbook_2x1_1x7(a_row, b_col, c_cols, offset):
        """
        Add schoolbook for A[a_row, b_col] (2x1) * B[b_col, c_cols] (1x7) -> C[a_row, c_cols] (2x7)
        C[i, k] = A[i, b_col] * B[b_col, k]
        """
        a0 = a_row
        j = b_col
        for idx, k in enumerate(c_cols):
            U[a0 * m + j][offset + idx] = 1
            V[j * p + k][offset + idx] = 1
            W[k * n + a0][offset + idx] = 1
        return len(c_cols)
    
    # Term 1a: A[0:2, 0:2] * B[0:2, 0:2]
    col_offset += add_strassen([0, 1], [0, 1], [0, 1], col_offset)
    
    # Term 1b: A[0:2, 0:2] * B[0:2, 2:4]
    col_offset += add_strassen([0, 1], [0, 1], [2, 3], col_offset)
    
    # Term 1c: A[0:2, 0:2] * B[0:2, 4:6]
    col_offset += add_strassen([0, 1], [0, 1], [4, 5], col_offset)
    
    # Term 1d: A[0:2, 0:2] * B[0:2, 6:7]
    col_offset += add_schoolbook_2x2_2x1([0, 1], [0, 1], [6], col_offset)
    
    # Term 2: A[0:2, 2:3] * B[2:3, 0:7]
    col_offset += add_schoolbook_2x1_1x7(0, 2, list(range(7)), col_offset)
    
    # Wait, that's only for A[0,2]. We also need A[1,2].
    # Let me fix this.
    
    # Actually, let me redo term 2 properly.
    # Term 2: A[:, 2:3] (2x1) * B[2:3, :] (1x7)
    # C[i, k] += A[i, 2] * B[2, k] for i in {0,1}, k in 0..6
    # This is 2*7 = 14 terms, but rank is only 2 because it's an outer product structure.
    # Actually no, each (i,k) pair needs its own term in schoolbook.
    # C[0,k] = A[0,2]*B[2,k] and C[1,k] = A[1,2]*B[2,k]
    # These are independent, so we need 2*7 = 14 terms? No...
    
    # Actually, for a fixed i and k: C[i,k] += A[i,2]*B[2,k]
    # This is one product per (i,k) pair. So 2*7 = 14 terms.
    # But wait, that's schoolbook rank for a 2x1 * 1x7 matrix mult, which is 2*1*7 = 14.
    
    # Hmm, that's a lot. Let me reconsider.
    
    # Actually, the "rank" of a 2x1 * 1x7 matrix multiplication tensor is 2*7 = 14 in schoolbook.
    # But can we do better? The tensor is T[i,k] = A[i,2]*B[2,k].
    # This is an outer product of a 2-vector and a 7-vector, so it has rank 1 as a matrix!
    # But as a tensor for matrix multiplication, we need:
    # sum_c U[i, c] * V[c] * W[k, c] = A[i,2]*B[2,k]
    # We can set U[i,c] = A[i,2], V[c] = 1, W[k,c] = B[2,k] for a single c? No, that doesn't work
    # because U, V, W are fixed coefficient matrices, not depending on the input values.
    
    # The tensor entries are: for each (i, j, k), the coefficient is 1 if j=2 (the shared index).
    # T[i, j, k] = 1 if j=2, else 0.
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    # For j=2: sum_c U[i*m+2, c] * V[2*p+k, c] * W[k*n+i, c] = 1 for all i,k
    # For j!=2: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 0 for all i,k
    
    # The non-zero part is a 2x7 all-ones tensor, which has tensor rank 2*7=14 in general.
    # Actually, the all-ones tensor of size 2x7 has rank min(2,7)*max(2,7)... no.
    # The tensor rank of the all-ones tensor of size a x b is a*b (it's the sum of a*b rank-1 tensors).
    # Wait, actually the all-ones tensor T[i,k] = 1 for all i,k can be written as:
    # T[i,k] = sum_{c=1}^{a*b} e_i^{(c)} * f_k^{(c)} where...
    # Actually, T[i,k] = 1 = (1)(1) for all i,k. So it's a rank-1 tensor!
    # T[i,k] = u_i * v_k where u_i = 1, v_k = 1.
    # So the tensor rank is 1!
    
    # But in our factorization, we have the constraint that the factors are structured:
    # U is indexed by (i,j), V by (j,k), W by (k,i).
    # For the j=2 slice, we need:
    # sum_c U[i,2,c] * V[2,k,c] * W[k,i,c] = 1 for all i,k
    
    # If we use one column c: U[i,2,c] = 1, V[2,k,c] = 1, W[k,i,c] = 1 for all i,k.
    # Then sum_c = 1*1*1 = 1. 
    # But we also need for j!=2: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 0.
    # If we only use this one column, then for j!=2, U[i,j,c]=0, so the sum is 0. Good.
    
    # So actually, the 2x1*1x7 part can be done with rank 1!
    # Similarly, let me recheck the 2x2*2x1 part.
    
    # For A[0:2, 0:2] * B[0:2, 6:7]:
    # C[i, 6] += A[i,0]*B[0,6] + A[i,1]*B[1,6]
    # Tensor: T[i,j,6] = 1 for j in {0,1}, i in {0,1}
    # We need: sum_c U[i,j,c] * V[j,6,c] * W[6,i,c] = 1 for i in {0,1}, j in {0,1}
    # And 0 for other (i,j) combinations (but those are handled by other terms).
    
    # This is a 2x2 all-ones tensor (in i,j space, with k=6 fixed).
    # T[i,j] = 1 for all i,j in {0,1}. This is rank 1 as a matrix!
    # So we can use one column: U[i,j,c] = 1, V[j,6,c] = 1, W[6,i,c] = 1.
    # Then sum = 1*1*1 = 1. 
    
    # Wait, but V[j,6,c] needs to be the same for both j=0 and j=1? No, V is indexed by (j,k).
    # V[0,6,c] and V[1,6,c] can be different.
    # We need: U[i,0,c] * V[0,6,c] * W[6,i,c] + U[i,1,c] * V[1,6,c] * W[6,i,c] = 1 for all i.
    # If we use one column: U[i,0,c]=1, U[i,1,c]=1, V[0,6,c]=1, V[1,6,c]=1, W[6,i,c]=1/2.
    # Then sum = 1*1*(1/2) + 1*1*(1/2) = 1. But we need integer/half-integer entries, so 1/2 is fine!
    
    # Actually wait, let me re-read the problem. "integer/half-integer entries" - so Fractions are fine.
    
    # So the 2x2*2x1 part can be done with rank 1!
    # And the 2x1*1x7 part can be done with rank 1!
    
    # Let me rebuild with this insight.
    
    # Actually, I realize I need to be more careful. Let me just build the whole thing fresh.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    
    # For each combination of A-column-group and B-row-group and B-column-group:
    # Group A columns: G1 = {0,1}, G2 = {2}
    # Group B rows: G1 = {0,1}, G2 = {2}
    # Group B columns: {0,1}, {2,3}, {4,5}, {6}
    
    # For A-col-group {0,1} and B-row-group {0,1}:
    #   For each B-col-group of size 2: use Strassen (rank 7)
    #   For B-col-group {6}: use rank-1 trick
    # For A-col-group {2} and B-row-group {2}:
    #   For each B-col-group: use rank-1 trick (since it's 1x1 * 1xK, tensor is all-ones in (i,k) space... 
    #   wait, no. A[i,2]*B[2,k] for i in {0,1}, k in col_group.
    #   The tensor in (i, k) space is all-ones, which is rank 1.
    #   So for each B-col-group, we need rank 1.
    
    # Total: 3*7 + 1 + 4*1 = 21 + 1 + 4 = 26
    
    # Let me adjust rank
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Strassen blocks for A[0:2, 0:2] * B[0:2, c0:c0+2] for c0 in {0, 2, 4}
    for c0 in [0, 2, 4]:
        col_offset += add_strassen([0, 1], [0, 1], [c0, c0+1], col_offset)
    
    # Rank-1 for A[0:2, 0:2] * B[0:2, 6:7]
    # C[i,6] += A[i,0]*B[0,6] + A[i,1]*B[1,6]
    # Use: U[i,j,c]=1 for i in {0,1}, j in {0,1}; V[j,6,c]=1 for j in {0,1}; W[6,i,c]=1/2 for i in {0,1}
    for i in [0, 1]:
        for j in [0, 1]:
            U[i * m + j][col_offset] = 1
        V[j * p + 6][col_offset] = 1
    for i in [0, 1]:
        W[6 * n + i][col_offset] = Fraction(1, 2)
    col_offset += 1
    
    # Rank-1 for A[0:2, 2:3] * B[2:3, c0:c0+2] for c0 in {0, 2, 4}
    # C[i, k] += A[i,2]*B[2,k] for i in {0,1}, k in {c0, c0+1}
    # Tensor in (i,k) space is all-ones, rank 1.
    # Use: U[i,2,c]=1, V[2,k,c]=1, W[k,i,c]=1
    for c0 in [0, 2, 4]:
        for i in [0, 1]:
            U[i * m + 2][col_offset] = 1
        for k in [c0, c0+1]:
            V[2 * p + k][col_offset] = 1
            for i in [0, 1]:
                W[k * n + i][col_offset] = 1
        col_offset += 1
    
    # Rank-1 for A[0:2, 2:3] * B[2:3, 6:7]
    for i in [0, 1]:
        U[i * m + 2][col_offset] = 1
    V[2 * p + 6][col_offset] = 1
    for i in [0, 1]:
        W[6 * n + i][col_offset] = 1
    col_offset += 1
    
    # Verify
    if not _verify_decomp(n, m, p, U, V, W):
        # Fallback to schoolbook
        return _schoolbook()
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Try hybrid Strassen/schoolbook
    hybrid = _hybrid_strassen_schoolbook()
    hybrid_rank = len(hybrid["U"][0])
    
    # Verify hybrid
    if _verify_decomp(DIMENSIONS[0], DIMENSIONS[1], DIMENSIONS[2], hybrid["U"], hybrid["V"], hybrid["W"]):
        return hybrid
    
    # Fallback to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END

