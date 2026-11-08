"""Matrix multiplication 3x3x4 tensor decomposition - alternative approach."""
import numpy as np

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS  # n=3 rows of A, m=3 cols of A/rows of B, p=4 cols of B
    # Output C is n x p = 3 x 4
    # A is n x m = 3 x 3, B is m x p = 3 x 4
    
    # Use a systematic approach: try to find a low-rank decomposition
    # by exploiting structure. We'll use a different block partitioning.
    
    # Partition: A into 2x1 | 2x2, B into 1x2 | 2x2
    # A = [[a00, a01, a02],
    #      [a10, a11, a12],
    #      [a20, a21, a22]]
    # Let A_top = A[0:2, :] (2x3), A_bot = A[2, :] (1x3)
    # Let B_left = B[:, 0:2] (3x2), B_right = B[:, 2:4] (3x2)
    
    # C = A * B = [A_top * B_left + A_top * B_right; A_bot * B_left + A_bot * B_right]
    
    # For A_top (2x3) * B_left (3x2): use rank-7 Strassen-like or direct
    # For A_top (2x3) * B_right (3x2): similar
    # For A_bot (1x3) * B_left (3x2): rank at most 6, but can do better
    # For A_bot (1x3) * B_right (3x2): similar
    
    # Let's use the following strategy:
    # 1. Compute S = A_top * B (2x3 * 3x4 = 2x4) using efficient method
    # 2. Compute T = A_bot * B (1x3 * 3x4 = 1x4)
    
    # For 2x3 * 3x4: we can partition B into B1 (3x2) and B2 (3x2)
    # A_top * B1: 2x3 * 3x2. Use 6 direct products or find smarter way.
    # Actually 2x3*3x2 has rank at most 6, and it's known to be 6.
    
    # Let me just use a clean direct approach with some sharing.
    
    # I'll construct the decomposition explicitly with rank 33 if possible, else 34.
    
    rank = 33
    U = [[0] * rank for _ in range(n * m)]  # 9 x 33
    V = [[0] * rank for _ in range(m * p)]  # 12 x 33
    W = [[0] * rank for _ in range(p * n)]  # 12 x 33
    
    col = 0
    
    # Direct term for each (i,j,k) where C[i,k] += A[i,j] * B[j,k]
    # But we'll try to merge some.
    
    # Strategy: For the top 2 rows of A (i=0,1) and all j,k, use Strassen-like on 2x3*3x4
    # For bottom row of A (i=2), use direct products but try to share with above.
    
    # Let me use a different approach: group by B columns.
    # For each pair of B columns (k, k+1) for k=0,2:
    #   We compute A * B[:, k:k+2] which is 3x2.
    #   This can be done with rank at most... let's use 2x3*3x2 for top part and 1x3*3x2 for bottom.
    
    # Actually, let me just implement a verified rank-34 solution with a different internal structure,
    # then try to see if I can drop one term.
    
    # Revert to safe approach: implement the known decomposition carefully.
    
    # I'll use the 2+1 partition on both A and B as in the original, but implement from scratch.
    
    # A partition: A11 = A[0:2, 0:2] (2x2), A12 = A[0:2, 2] (2x1), A21 = A[2, 0:2] (1x2), A22 = A[2,2] (1x1)
    # B partition: B11 = B[0:2, 0:2] (2x2), B12 = B[0:2, 2:4] (2x2), B21 = B[2, 0:2] (1x2), B22 = B[2, 2:4] (1x2)
    
    # Indices:
    # A[i,j] -> index i*m + j = i*3 + j
    # B[j,k] -> index j*p + k = j*4 + k
    # C[i,k] -> index i*p + k = i*4 + k
    
    # A11: A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
    # A12: A[0,2]=2, A[1,2]=5
    # A21: A[2,0]=6, A[2,1]=7
    # A22: A[2,2]=8
    
    # B11: B[0,0]=0, B[0,1]=1, B[1,0]=4, B[1,1]=5
    # B12: B[0,2]=2, B[0,3]=3, B[1,2]=6, B[1,3]=7
    # B21: B[2,0]=8, B[2,1]=9
    # B22: B[2,2]=10, B[2,3]=11
    
    # C11: C[0,0]=0, C[0,1]=1, C[1,0]=4, C[1,1]=5
    # C12: C[0,2]=2, C[0,3]=3, C[1,2]=6, C[1,3]=7
    # C21: C[2,0]=8, C[2,1]=9
    # C22: C[2,2]=10, C[2,3]=11
    
    # Wait, C index is i*4+k:
    # C[0,0]=0, C[0,1]=1, C[0,2]=2, C[0,3]=3
    # C[1,0]=4, C[1,1]=5, C[1,2]=6, C[1,3]=7
    # C[2,0]=8, C[2,1]=9, C[2,2]=10, C[2,3]=11
    
    # Let me redo with correct indices.
    
    # A11*B11 -> C11 (top-left 2x2 of C)
    # Use Strassen for 2x2*2x2: 7 products
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    # C11[0,0] = M1+M4-M5+M7
    # C11[0,1] = M3+M5
    # C11[1,0] = M2+M4
    # C11[1,1] = M1-M2+M3+M6
    
    # a00=A[0,0]=0, a01=A[0,1]=1, a10=A[1,0]=3, a11=A[1,1]=4
    # b00=B[0,0]=0, b01=B[0,1]=1, b10=B[1,0]=4, b11=B[1,1]=5
    # C11[0,0]=C[0,0]=0, C11[0,1]=C[0,1]=1, C11[1,0]=C[1,0]=4, C11[1,1]=C[1,1]=5
    
    # A11*B12 -> C12 (top-right 2x2 of C)
    # b00=B[0,2]=2, b01=B[0,3]=3, b10=B[1,2]=6, b11=B[1,3]=7
    # C12[0,0]=C[0,2]=2, C12[0,1]=C[0,3]=3, C12[1,0]=C[1,2]=6, C12[1,1]=C[1,3]=7
    
    # A12*B11 -> C11 contribution
    # A12 = [a02, a12] = [A[0,2]=2, A[1,2]=5] (2x1)
    # B11 = [[b00,b01],[b10,b11]] (2x2)
    # C11 += A12 * B11: C11[0,0] += a02*b00, C11[0,1] += a02*b01, C11[1,0] += a12*b10, C11[1,1] += a12*b11
    # This is 4 products.
    
    # A12*B12 -> C12 contribution
    # C12[0,0] += a02*b00, C12[0,1] += a02*b01, C12[1,0] += a12*b10, C12[1,1] += a12*b11
    # 4 products.
    
    # A21*B11 -> C21 contribution
    # A21 = [a20, a21] = [A[2,0]=6, A[2,1]=7] (1x2)
    # B11 (2x2)
    # C21[0,0] = C[2,0]=8 += a20*b00 + a21*b10
    # C21[0,1] = C[2,1]=9 += a20*b01 + a21*b11
    # 4 products.
    
    # A21*B12 -> C22 contribution
    # C22[0,0] = C[2,2]=10 += a20*b00 + a21*b10
    # C22[0,1] = C[2,3]=11 += a20*b01 + a21*b11
    # 4 products.
    
    # A22*B21 -> C21 contribution
    # A22 = a22 = A[2,2]=8 (1x1)
    # B21 = [b20, b21] = [B[2,0]=8, B[2,1]=9] (1x2)
    # C21[0,0] += a22*b20, C21[0,1] += a22*b21
    # 2 products.
    
    # A22*B22 -> C22 contribution
    # C22[0,0] += a22*b20, C22[0,1] += a22*b21
    # Wait, B22 = [b20, b21] = [B[2,2]=10, B[2,3]=11]
    # C22[0,0] += a22*b20, C22[0,1] += a22*b21
    # 2 products.
    
    # Total: 7+7+4+4+4+4+2+2 = 34. Same as before.
    
    # To get to 33, I need to find one more merge. Let me check if any of the "trivial" 4-product blocks
    # can be done in 3 products. A 2x1 * 2x2 multiplication: can it be done in 3 products?
    # (x1, x2) * [[a,b],[c,d]] = (x1*a+x2*c, x1*b+x2*d). This is a bilinear map from C^2 x C^2 to C^2.
    # The rank of this tensor is 4 (it's the standard rank-4 tensor). So no.
    
    # What about combining A12*B11 and A22*B21? They both contribute to C21.
    # C21[0,0] = A[0,2]*B[0,0] + A[1,2]*B[1,0] + A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    # = (A[0,2]+A[2,0])*B[0,0] + (A[1,2]+A[2,1])*B[1,0] + A[2,2]*B[2,0]
    # That's 3 products! Let's verify:
    # (A[0,2]+A[2,0])*B[0,0] + (A[1,2]+A[2,1])*B[1,0] + A[2,2]*B[2,0]
    # = A[0,2]*B[0,0] + A[2,0]*B[0,0] + A[1,2]*B[1,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    # Yes! That's exactly the sum.
    
    # Similarly for C21[0,1]:
    # = A[0,2]*B[0,1] + A[1,2]*B[1,1] + A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
    # = (A[0,2]+A[2,0])*B[0,1] + (A[1,2]+A[2,1])*B[1,1] + A[2,2]*B[2,1]
    # 3 products.
    
    # So for C21, instead of 4+2=6 products, we can do 3+3=6. No gain.
    
    # But wait, the (A[0,2]+A[2,0]) term is shared between C21[0,0] and C21[0,1]!
    # In the tensor decomposition, if we use (A[0,2]+A[2,0]) as a shared U factor, we can save.
    
    # Let me reconsider. The issue is that in the tensor decomposition, each "product" is a rank-1 term
    # in the tensor. If two output entries share the same combination of A and B factors, we can potentially
    # merge them.
    
    # Actually, in the tensor model, each rank-1 term contributes to one (or more) output entries via
    # the W matrix. The W matrix can have multiple non-zero entries in a column, meaning one product
    # contributes to multiple output entries.
    
    # So if I have a term (A[0,2]+A[2,0]) * B[0,0], this is one rank-1 term that can contribute to
    # C21[0,0] via W. But it only contributes to one output entry here.
    
    # The real saving comes from sharing the A-side factor. If I use S = A[0,2]+A[2,0] as a shared
    # linear combination, then S*B[0,0] contributes to C21[0,0], and S*B[0,1] contributes to C21[0,1].
    # But these are still two separate rank-1 terms because B[0,0] and B[0,1] are different.
    
    # I think the rank 34 is likely optimal or close to optimal. Let me just implement a clean
    # rank-34 solution and ensure correctness.
    
    # Let me reset and implement carefully.
    
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def set_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
        col += 1
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
    
    # Strassen for A11*B11 -> C11
    # M1 = (a00+a11)*(b00+b11)
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[5][col] = 1
    W[0][col] = 1; W[5][col] = 1
    col += 1
    
    # M2 = (a10+a11)*b00
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[4][col] = 1; W[5][col] = -1
    col += 1
    
    # M3 = a00*(b01-b11)
    U[0][col] = 1
    V[1][col] = 1; V[5][col] = -1
    W[1][col] = 1; W[5][col] = 1
    col += 1
    
    # M4 = a11*(b10-b00)
    U[4][col] = 1
    V[4][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    # M5 = (a00+a01)*b11
    U[0][col] = 1; U[1][col] = 1
    V[5][col] = 1
    W[0][col] = -1; W[1][col] = 1
    col += 1
    
    # M6 = (a10-a00)*(b00+b01)
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[5][col] = 1
    col += 1
    
    # M7 = (a01-a11)*(b10+b11)
    U[1][col] = 1; U[4][col] = -1
    V[4][col] = 1; V[5][col] = 1
    W[0][col] = 1
    col += 1
    
    # A12*B11 -> C11
    # a02*B[0,0] -> C[0,0]
    U[2][col] = 1; V[0][col] = 1; W[0][col] = 1; col += 1
    # a02*B[0,1] -> C[0,1]
    U[2][col] = 1; V[1][col] = 1; W[1][col] = 1; col += 1
    # a12*B[1,0] -> C[1,0]
    U[5][col] = 1; V[4][col] = 1; W[4][col] = 1; col += 1
    # a12*B[1,1] -> C[1,1]
    U[5][col] = 1; V[5][col] = 1; W[5][col] = 1; col += 1
    
    # Strassen for A11*B12 -> C12
    # b00=B[0,2]=2, b01=B[0,3]=3, b10=B[1,2]=6, b11=B[1,3]=7
    # M1 = (a00+a11)*(b00+b11)
    U[0][col] = 1; U[4][col] = 1
    V[2][col] = 1; V[7][col] = 1
    W[2][col] = 1; W[7][col] = 1
    col += 1
    
    # M2 = (a10+a11)*b00
    U[3][col] = 1; U[4][col] = 1
    V[2][col] = 1
    W[6][col] = 1; W[7][col] = -1
    col += 1
    
    # M3 = a00*(b01-b11)
    U[0][col] = 1
    V[3][col] = 1; V[7][col] = -1
    W[3][col] = 1; W[7][col] = 1
    col += 1
    
    # M4 = a11*(b10-b00)
    U[4][col] = 1
    V[6][col] = 1; V[2][col] = -1
    W[2][col] = 1; W[6][col] = 1
    col += 1
    
    # M5 = (a00+a01)*b11
    U[0][col] = 1; U[1][col] = 1
    V[7][col] = 1
    W[2][col] = -1; W[3][col] = 1
    col += 1
    
    # M6 = (a10-a00)*(b00+b01)
    U[3][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[7][col] = 1
    col += 1
    
    # M7 = (a01-a11)*(b10+b11)
    U[1][col] = 1; U[4][col] = -1
    V[6][col] = 1; V[7][col] = 1
    W[2][col] = 1
    col += 1
    
    # A12*B12 -> C12
    U[2][col] = 1; V[2][col] = 1; W[2][col] = 1; col += 1
    U[2][col] = 1; V[3][col] = 1; W[3][col] = 1; col += 1
    U[5][col] = 1; V[6][col] = 1; W[6][col] = 1; col += 1
    U[5][col] = 1; V[7][col] = 1; W[7][col] = 1; col += 1
    
    # A21*B11 -> C21
    # C[2,0]=8: A[2,0]*B[0,0] + A[2,1]*B[1,0]
    # C[2,1]=9: A[2,0]*B[0,1] + A[2,1]*B[1,1]
    U[6][col] = 1; V[0][col] = 1; W[8][col] = 1; col += 1
    U[6][col] = 1; V[1][col] = 1; W[9][col] = 1; col += 1
    U[7][col] = 1; V[4][col] = 1; W[8][col] = 1; col += 1
    U[7][col] = 1; V[5][col] = 1; W[9][col] = 1; col += 1
    
    # A21*B12 -> C22
    # C[2,2]=10: A[2,0]*B[0,2] + A[2,1]*B[1,2]
    # C[2,3]=11: A[2,0]*B[0,3] + A[2,1]*B[1,3]
    U[6][col] = 1; V[2][col] = 1; W[10][col] = 1; col += 1
    U[6][col] = 1; V[3][col] = 1; W[11][col] = 1; col += 1
    U[7][col] = 1; V[6][col] = 1; W[10][col] = 1; col += 1
    U[7][col] = 1; V[7][col] = 1; W[11][col] = 1; col += 1
    
    # A22*B21 -> C21
    U[8][col] = 1; V[8][col] = 1; W[8][col] = 1; col += 1
    U[8][col] = 1; V[9][col] = 1; W[9][col] = 1; col += 1
    
    # A22*B22 -> C22
    U[8][col] = 1; V[10][col] = 1; W[10][col] = 1; col += 1
    U[8][col] = 1; V[11][col] = 1; W[11][col] = 1; col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}

