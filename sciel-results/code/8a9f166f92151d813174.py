"""Matrix multiplication 2x3 by 3x4: rank-minimal decomposition via Strassen on 2x2 blocks."""

def construct():
    n, m, p = 2, 3, 4  # A is n x m, B is m x p, C is n x p
    
    # We partition m = 3 into 1+2.
    # A = [A1 | A2] where A1 is 2x1, A2 is 2x2
    # B = [B1; B2] where B1 is 1x4, B2 is 2x4
    # C = A1*B1 + A2*B2
    
    # For A2*B2, we partition p=4 into 2+2.
    # B2 = [B21 | B22] where B21 is 2x2, B22 is 2x2
    # A2*B2 = A2*B21 + A2*B22
    
    # So C = A1*B1 + A2*B21 + A2*B22
    
    # Term 1: A1 (2x1) * B1 (1x4) = 2x4 matrix
    # This is outer product. Rank = 2*1*4 = 8? No.
    # A1 = [a11; a21], B1 = [b11 b12 b13 b14]
    # Result = [a11*b11 a11*b12 a11*b13 a11*b14; a21*b11 a21*b12 a21*b13 a21*b14]
    # This can be done with 8 terms (one for each element) or can we do better?
    # Actually, this is a rank-8 tensor (outer product of vectors).
    # Wait, the tensor for (2x1)*(1x4) has rank 8.
    # But we can use Strassen-like tricks? No, it's just outer product.
    # Actually, for multiplying a column vector by a row vector, the rank is mn = 2*4 = 8.
    # There's no known lower rank for this specific case since it's bilinear but the structure is fixed.
    
    # Term 2: A2 (2x2) * B21 (2x2) -> 2x2 result. Rank 7 (Strassen).
    # Term 3: A2 (2x2) * B22 (2x2) -> 2x2 result. Rank 7 (Strassen).
    
    # Total rank = 8 + 7 + 7 = 22.
    
    # However, we need to be careful about the W index convention: W[k*n+i, r]
    # Let's build the U, V, W matrices.
    
    # U has shape (n*m, rank) = (6, 22)
    # V has shape (m*p, rank) = (12, 22)
    # W has shape (p*n, rank) = (8, 22)
    
    U = [[0.0] * 22 for _ in range(6)]
    V = [[0.0] * 22 for _ in range(12)]
    W = [[0.0] * 22 for _ in range(8)]
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val, col):
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
    
    # Indices for U: i*m + j, where i in [0,1], j in [0,1,2]
    # i=0: j=0->0, j=1->1, j=2->2
    # i=1: j=0->3, j=1->4, j=2->5
    
    # Indices for V: j*p + k, where j in [0,1,2], k in [0,1,2,3]
    # j=0: k=0->0, k=1->1, k=2->2, k=3->3
    # j=1: k=0->4, k=1->5, k=2->6, k=3->7
    # j=2: k=0->8, k=1->9, k=2->10, k=3->11
    
    # Indices for W: k*n + i, where k in [0,1,2,3], i in [0,1]
    # k=0: i=0->0, i=1->1
    # k=1: i=0->2, i=1->3
    # k=2: i=0->4, i=1->5
    # k=3: i=0->6, i=1->7
    
    col = 0
    
    # --- Term 1: A1 * B1 ---
    # A1 is column j=0 of A. A2 is columns j=1,2 of A.
    # B1 is row j=0 of B. B2 is rows j=1,2 of B.
    
    # A1 * B1 contributes to all C[i][k] via A[0][0]*B[0][k] and A[1][0]*B[0][k]
    # A[0][0] is U index 0. B[0][k] is V index k.
    # For each k in 0..3:
    #   C[0][k] gets A[0][0]*B[0][k] -> U[0], V[k], W[k*2+0]
    #   C[1][k] gets A[1][0]*B[0][k] -> U[3], V[k], W[k*2+1]
    
    for k in range(4):
        # C[0][k] term
        add_term(0, k, k*2+0, 1.0, 1.0, 1.0, col)
        col += 1
        # C[1][k] term
        add_term(3, k, k*2+1, 1.0, 1.0, 1.0, col)
        col += 1
    
    # --- Term 2: A2 * B21 ---
    # A2 is A[:, 1:3] (2x2), B21 is B[1:3, 0:2] (2x2)
    # Result is 2x2, contributing to C[:, 0:2]
    # Let's use Strassen for 2x2 multiplication.
    # Let A2 = [[a11, a12], [a21, a22]] corresponding to A[0,1], A[0,2], A[1,1], A[1,2]
    # Let B21 = [[b11, b12], [b21, b22]] corresponding to B[1,0], B[1,1], B[2,0], B[2,1]
    
    # Strassen for C = A2 * B21:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b21-b22)
    # M4 = a22(b12-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # C11 = M1 + M4 - M2 - M3
    # C12 = M5 + M3
    # C21 = M2 + M4
    # C22 = M1 - M5 + M6 + M7
    
    # We need to decompose each M into rank-1 terms.
    # Since each M is a product of sums of A elements and sums of B elements,
    # we can express each M as a single rank-1 term if we expand the sums?
    # No, (a11+a22)(b11+b22) = a11b11 + a11b22 + a22b11 + a22b22. That's 4 rank-1 terms.
    # But Strassen's trick is to share these. The rank of the whole 2x2x2x2 tensor is 7.
    # We need to find the 7 rank-1 terms that sum to the multiplication tensor.
    
    # This is tricky to derive from scratch. Let's use a known decomposition.
    # Actually, we can just use the schoolbook for 2x2 which is rank 4.
    # If we use schoolbook for both 2x2 blocks, rank = 8 + 4 + 4 = 16.
    # That's much better than 22!
    
    # Let's restart with schoolbook for the 2x2 parts.
    # Total rank = 8 (for A1*B1) + 4 (for A2*B21) + 4 (for A2*B22) = 16.
    
    # But wait, can we do A1*B1 with less than 8?
    # A1 is 2x1, B1 is 1x4. The product is 2x4.
    # The tensor is sum_{i,k} A[i,0] B[0,k] E[i,k].
    # This is a sum of 8 rank-1 tensors.
    # Is it possible to do it with fewer?
    # The border rank of this tensor is 8. The rank is 8.
    # So 16 is a good target.
    
    # Let's build rank 16.
    col = 0
    
    # Reset U, V, W
    U = [[0.0] * 16 for _ in range(6)]
    V = [[0.0] * 16 for _ in range(12)]
    W = [[0.0] * 16 for _ in range(8)]
    
    # Term 1: A1 * B1 (8 terms)
    for k in range(4):
        # C[0][k] = A[0][0] * B[0][k]
        add_term(0, k, k*2+0, 1.0, 1.0, 1.0, col)
        col += 1
        # C[1][k] = A[1][0] * B[0][k]
        add_term(3, k, k*2+1, 1.0, 1.0, 1.0, col)
        col += 1
    
    # Term 2: A2 * B21 (4 terms, schoolbook)
    # A2 = A[:, 1:3], B21 = B[1:3, 0:2]
    # A2[i][j] = A[i][j+1], B21[j][k] = B[j+1][k]
    # C[i][k] += sum_{j=0,1} A[i][j+1] B[j+1][k]
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                # A index: i*m + (j+1)
                # V index: (j+1)*p + k
                # W index: k*n + i
                u_idx = i * 3 + (j + 1)
                v_idx = (j + 1) * 4 + k
                w_idx = k * 2 + i
                add_term(u_idx, v_idx, w_idx, 1.0, 1.0, 1.0, col)
                col += 1
    
    # Term 3: A2 * B22 (4 terms, schoolbook)
    # B22 = B[1:3, 2:4]
    # C[i][k+2] += sum_{j=0,1} A[i][j+1] B[j+1][k+2]
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                # A index: i*m + (j+1)
                # V index: (j+1)*p + (k+2)
                # W index: (k+2)*n + i
                u_idx = i * 3 + (j + 1)
                v_idx = (j + 1) * 4 + (k + 2)
                w_idx = (k + 2) * 2 + i
                add_term(u_idx, v_idx, w_idx, 1.0, 1.0, 1.0, col)
                col += 1
    
    # Verify col == 16
    assert col == 16, f"Expected 16 columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
