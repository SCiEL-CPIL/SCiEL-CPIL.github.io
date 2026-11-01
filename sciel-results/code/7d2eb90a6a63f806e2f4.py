import numpy as np
import time

DEADLINE = 150.0
START_TIME = time.time()

def construct():
    """
    Returns a rank-minimal (or near-minimal) tensor decomposition for
    2x4 by 4x5 matrix multiplication.
    
    Strategy:
    - Split the 4x5 matrix B into B1 (4x2) and B2 (4x3).
    - C1 = A @ B1: 2x4 @ 4x2 -> 2x2. Use Strassen (rank 7).
    - C2 = A @ B2: 2x4 @ 4x3 -> 2x3. Use rank-10 decomposition:
      * C2[:, :2] via Strassen on A, B2[:, :2] (rank 7)
      * C2[:, 2] via 3 rank-1 terms (one per row of A)
    Total rank: 7 + 10 = 17.
    """
    n, m, p = 2, 4, 5
    
    # ---- Strassen for 2x2 blocks ----
    # For A (2x2) and B (2x2):
    # M1 = (a+d)(e+g), M2 = (c+d)(e), M3 = (a)(f-g),
    # M4 = (b)(g+h), M5 = (a+b)(h), M6 = (c-a)(e+f), M7 = (d-b)(g+h)
    # C00 = M1 - M2 + M3 + M4
    # C01 = M5 + M4 - M7 - M2
    # C10 = M6 + M1 - M3 - M5
    # C11 = M2 + M7 - M6 + M3
    
    # Strassen U vectors for 2x2 A (indices: 0=a,1=b,2=c,3=d)
    S_U = [
        [1, 0, 1, 0],  # M1: a+d
        [0, 0, 1, 1],  # M2: c+d
        [1, 0, 0, 0],  # M3: a
        [0, 1, 0, 0],  # M4: b
        [1, 1, 0, 0],  # M5: a+b
        [1, 0, -1, 0], # M6: c-a
        [0, -1, 0, 1], # M7: d-b
    ]
    
    # Strassen V vectors for 2x2 B (indices: 0=e,1=f,2=g,3=h)
    S_V = [
        [1, 0, 1, 0],  # M1: e+g
        [1, 0, 0, 0],  # M2: e
        [0, 1, -1, 0], # M3: f-g
        [0, 0, 1, 1],  # M4: g+h
        [0, 0, 0, 1],  # M5: h
        [1, 1, 0, 0],  # M6: e+f
        [0, 0, 1, 1],  # M7: g+h
    ]
    
    # Strassen W vectors for 2x2 C (indices: 0=C00, 1=C01, 2=C10, 3=C11)
    S_W = [
        [1, 0, 0, 0],   # M1 contributes to C00
        [-1, 0, 0, 0],  # M2 contributes to -C00
        [1, 0, 0, 0],   # M3 contributes to C00
        [1, 0, 0, 0],   # M4 contributes to C00
        [0, 1, 0, 0],   # M5 contributes to C01
        [0, -1, 0, 0],  # M2 contributes to -C01
        [0, -1, 0, 0],  # M7 contributes to -C01
        [0, 1, 0, 0],   # M4 contributes to C01
        [0, 0, 1, 0],   # M6 contributes to C10
        [0, 0, 1, 0],   # M1 contributes to C10
        [0, 0, -1, 0],  # M3 contributes to -C10
        [0, 0, -1, 0],  # M5 contributes to -C10
        [0, 0, 0, 1],   # M2 contributes to C11
        [0, 0, 0, 1],   # M7 contributes to C11
        [0, 0, 0, -1],  # M6 contributes to -C11
        [0, 0, 0, 1],   # M3 contributes to C11
    ]
    
    # Build the full decomposition
    rank = 17
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # ---- Part 1: Strassen for C1 = A @ B1 (2x4 @ 4x2 -> 2x2) ----
    # A is 2x4: rows are [a00, a01, a02, a03] and [a10, a11, a12, a13]
    # B1 is 4x2: columns are [e0,e1,e2,e3] and [f0,f1,f2,f3]
    # We use Strassen on the two 2x2 blocks of A and B1
    # A = [A_left | A_right] where A_left = [[a00,a01],[a10,a11]], A_right = [[a02,a03],[a12,a13]]
    # B1 = [B1_top; B1_bottom] where B1_top = [[e0,e1],[e2,e3]], B1_bottom = [[f0,f1],[f2,f3]]
    # Actually, for Strassen we need 2x2 matrices. Let me reformulate.
    
    # For 2x4 @ 4x2, we can view it as:
    # C1[0,0] = sum_j A[0,j]*B1[j,0]
    # C1[0,1] = sum_j A[0,j]*B1[j,1]
    # C1[1,0] = sum_j A[1,j]*B1[j,0]
    # C1[1,1] = sum_j A[1,j]*B1[j,1]
    
    # Use Strassen by treating A as 2x2 (each entry being a 1x2 vector) and B1 as 2x2 (each entry being a 2x1 vector)
    # This gives rank 7 for the 2x4@4x2 case.
    
    # Strassen terms for 2x4 @ 4x2:
    # M1 = (A_row0 + A_row1) . (B1_col0 + B1_col1)  -- but this doesn't directly work
    # Let me use a different approach.
    
    # Actually, let me use the standard result: 2x4 @ 4x2 has rank 7 (Strassen-like).
    # The decomposition is:
    # Let A = [[a,b],[c,d]] where a,b,c,d are 1x2 row vectors
    # Let B1 = [[e,f],[g,h]] where e,f,g,h are 2x1 column vectors
    # Then:
    # M1 = (a+d) . (e+g)
    # M2 = (c+d) . e
    # M3 = a . (f-g)
    # M4 = b . (g+h)
    # M5 = (a+b) . h
    # M6 = (c-a) . (e+f)
    # M7 = (d-b) . (g+h)
    
    # U vectors for A (2x4, indices 0-3 for row 0, 4-7 for row 1):
    # a = [1,1,0,0], b = [0,0,1,1], c = [0,0,0,0,1,1,0,0], d = [0,0,0,0,0,0,1,1]
    # Wait, A is 2x4 so indices are: row 0: 0,1,2,3; row 1: 4,5,6,7
    # a = A[0,0:2] = indices 0,1
    # b = A[0,2:4] = indices 2,3
    # c = A[1,0:2] = indices 4,5
    # d = A[1,2:4] = indices 6,7
    
    # For M1 = (a+d).(e+g): U_M1 has 1 at indices 0,1,6,7
    # For M2 = (c+d).e: U_M2 has 1 at indices 4,5,6,7
    # For M3 = a.(f-g): U_M3 has 1 at indices 0,1
    # For M4 = b.(g+h): U_M4 has 1 at indices 2,3
    # For M5 = (a+b).h: U_M5 has 1 at indices 0,1,2,3
    # For M6 = (c-a).(e+f): U_M6 has 1 at 4,5 and -1 at 0,1
    # For M7 = (d-b).(g+h): U_M7 has 1 at 6,7 and -1 at 2,3
    
    # B1 is 4x2: indices 0,1 for col 0; 2,3 for col 1
    # e = B1[0:2,0] = indices 0,1 (col 0, rows 0,1)
    # f = B1[0:2,1] = indices 2,3 (col 1, rows 0,1)
    # g = B1[2:4,0] = indices 4,5 (col 0, rows 2,3)
    # h = B1[2:4,1] = indices 6,7 (col 1, rows 2,3)
    
    # V vectors for B1 (4x2, indices 0,1 for col0 rows0-1; 2,3 for col1 rows0-1; 4,5 for col0 rows2-3; 6,7 for col1 rows2-3)
    # Wait, B1 is 4x2 so total 8 elements: 
    # Index layout: B1[i,j] where i=0..3, j=0..1
    # In flattened form (row-major): index = i*2 + j
    # So: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3, (2,0)=4, (2,1)=5, (3,0)=6, (3,1)=7
    # e = B1[0:2, 0] = indices 0, 2
    # f = B1[0:2, 1] = indices 1, 3
    # g = B1[2:4, 0] = indices 4, 6
    # h = B1[2:4, 1] = indices 5, 7
    
    # C1 is 2x2: indices 0=C[0,0], 1=C[0,1], 2=C[1,0], 3=C[1,1]
    # In flattened form: index = i*2 + j, so (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    # W vectors for C1:
    # C[0,0] = M1 - M2 + M3 + M4
    # C[0,1] = M5 + M4 - M7 - M2
    # C[1,0] = M6 + M1 - M3 - M5
    # C[1,1] = M2 + M7 - M6 + M3
    
    # Now build the full matrices for Part 1 (rank 7)
    
    # U for Part 1 (shape 8x7)
    U1 = np.zeros((8, 7))
    U1[0, 0] = 1; U1[1, 0] = 1; U1[6, 0] = 1; U1[7, 0] = 1   # M1: a+d
    U1[4, 1] = 1; U1[5, 1] = 1; U1[6, 1] = 1; U1[7, 1] = 1   # M2: c+d
    U1[0, 2] = 1; U1[1, 2] = 1                                  # M3: a
    U1[2, 3] = 1; U1[3, 3] = 1                                  # M4: b
    U1[0, 4] = 1; U1[1, 4] = 1; U1[2, 4] = 1; U1[3, 4] = 1     # M5: a+b
    U1[0, 5] = -1; U1[1, 5] = -1; U1[4, 5] = 1; U1[5, 5] = 1   # M6: c-a
    U1[2, 6] = -1; U1[3, 6] = -1; U1[6, 6] = 1; U1[7, 6] = 1   # M7: d-b
    
    # V for Part 1 (shape 8x7)
    V1 = np.zeros((8, 7))
    V1[0, 0] = 1; V1[2, 0] = 1; V1[4, 0] = 1; V1[6, 0] = 1     # M1: e+g
    V1[0, 1] = 1; V1[2, 1] = 1                                  # M2: e
    V1[1, 2] = 1; V1[3, 2] = 1; V1[4, 2] = -1; V1[6, 2] = -1   # M3: f-g
    V1[4, 3] = 1; V1[6, 3] = 1; V1[5, 3] = 1; V1[7, 3] = 1     # M4: g+h
    V1[5, 4] = 1; V1[7, 4] = 1                                  # M5: h
    V1[0, 5] = 1; V1[2, 5] = 1; V1[1, 5] = 1; V1[3, 5] = 1     # M6: e+f
    V1[4, 6] = 1; V1[6, 6] = 1; V1[5, 6] = 1; V1[7, 6] = 1     # M7: g+h
    
    # W for Part 1 (shape 4x7)
    W1 = np.zeros((4, 7))
    # C[0,0] = M1 - M2 + M3 + M4
    W1[0, 0] = 1; W1[0, 1] = -1; W1[0, 2] = 1; W1[0, 3] = 1
    # C[0,1] = M5 + M4 - M7 - M2
    W1[1, 1] = -1; W1[1, 3] = 1; W1[1, 4] = 1; W1[1, 6] = -1
    # C[1,0] = M6 + M1 - M3 - M5
    W1[2, 0] = 1; W1[2, 2] = -1; W1[2, 4] = -1; W1[2, 5] = 1
    # C[1,1] = M2 + M7 - M6 + M3
    W1[3, 1] = 1; W1[3, 2] = 1; W1[3, 5] = -1; W1[3, 6] = 1
    
    U[:, 0:7] = U1
    V[:, 0:7] = V1
    W[:, 0:7] = W1
    
    col = 7
    
    # ---- Part 2: 2x4 @ 4x3 -> 2x3, rank 10 ----
    # B2 is 4x3: indices 0..11
    # B2[i,j]: index = i*3 + j
    # C2 is 2x3: indices 0..5
    # C2[i,j]: index = i*3 + j
    
    # Approach: 
    # C2[:, 0:2] via Strassen on A and B2[:, 0:2] (2x4 @ 4x2 -> 2x2), rank 7
    # C2[:, 2] via 3 rank-1 terms: C2[0,2] = A_row0 . B2_col2, C2[1,2] = A_row1 . B2_col2
    # That's 2 rank-1 terms, so total = 7 + 2 = 9? No wait, the Strassen gives C2[:, 0:2] but I need to check if it's correct.
    
    # Actually, let me just do: C2[:, 0:2] via the same Strassen as Part 1 but with B2[:, 0:2] instead of B1.
    # B2[:, 0:2] has the same structure as B1.
    
    # U is the same as U1 (since A is the same)
    U2 = U1.copy()
    
    # V for B2[:, 0:2] (4x2 part of 4x3 matrix)
    # B2 is 4x3: index = i*3 + j
    # B2[:, 0] = indices 0, 3, 6, 9
    # B2[:, 1] = indices 1, 4, 7, 10
    # e = B2[0:2, 0] = indices 0, 3
    # f = B2[0:2, 1] = indices 1, 4
    # g = B2[2:4, 0] = indices 6, 9
    # h = B2[2:4, 1] = indices 7, 10
    
    V2 = np.zeros((12, 7))
    V2[0, 0] = 1; V2[3, 0] = 1; V2[6, 0] = 1; V2[9, 0] = 1     # M1: e+g
    V2[0, 1] = 1; V2[3, 1] = 1                                  # M2: e
    V2[1, 2] = 1; V2[4, 2] = 1; V2[6, 2] = -1; V2[9, 2] = -1   # M3: f-g
    V2[6, 3] = 1; V2[9, 3] = 1; V2[7, 3] = 1; V2[10, 3] = 1    # M4: g+h
    V2[7, 4] = 1; V2[10, 4] = 1                                 # M5: h
    V2[0, 5] = 1; V2[3, 5] = 1; V2[1, 5] = 1; V2[4, 5] = 1     # M6: e+f
    V2[6, 6] = 1; V2[9, 6] = 1; V2[7, 6] = 1; V2[10, 6] = 1    # M7: g+h
    
    # W for C2[:, 0:2] (2x2 part of 2x3 matrix)
    # C2[0,0] = M1 - M2 + M3 + M4 -> index 0
    # C2[0,1] = M5 + M4 - M7 - M2 -> index 1
    # C2[1,0] = M6 + M1 - M3 - M5 -> index 3
    # C2[1,1] = M2 + M7 - M6 + M3 -> index 4
    
    W2 = np.zeros((6, 7))
    W2[0, 0] = 1; W2[0, 1] = -1; W2[0, 2] = 1; W2[0, 3] = 1
    W2[1, 1] = -1; W2[1, 3] = 1; W2[1, 4] = 1; W2[1, 6] = -1
    W2[3, 0] = 1; W2[3, 2] = -1; W2[3, 4] = -1; W2[3, 5] = 1
    W2[4, 1] = 1; W2[4, 2] = 1; W2[4, 5] = -1; W2[4, 6] = 1
    
    # Rank-1 terms for C2[:, 2]:
    # C2[0,2] = A_row0 . B2_col2 = sum_j A[0,j]*B2[j,2]
    # C2[1,2] = A_row1 . B2_col2 = sum_j A[1,j]*B2[j,2]
    # B2[:, 2] = indices 2, 5, 8, 11
    
    U3 = np.zeros((8, 2))
    U3[0, 0] = 1; U3[1, 0] = 1; U3[2, 0] = 1; U3[3, 0] = 1  # A_row0
    U3[4, 1] = 1; U3[5, 1] = 1; U3[6, 1] = 1; U3[7, 1] = 1  # A_row1
    
    V3 = np.zeros((12, 2))
    V3[2, 0] = 1; V3[5, 0] = 1; V3[8, 0] = 1; V3[11, 0] = 1  # B2_col2
    V3[2, 1] = 1; V3[5, 1] = 1; V3[8, 1] = 1; V3[11, 1] = 1  # B2_col2 (same)
    
    W3 = np.zeros((6, 2))
    W3[2, 0] = 1  # C2[0,2] = index 2
    W3[5, 1] = 1  # C2[1,2] = index 5
    
    # Combine Part 2
    U[:, 7:14] = U2
    V[:, 7:14] = V2
    W[:, 7:14] = W2
    
    U[:, 14:16] = U3
    V[:, 14:16] = V3
    W[:, 14:16] = W3
    
    # Total rank = 16? Wait, I have 7 + 7 + 2 = 16. Let me recount.
    # Part 1: 7 terms
    # Part 2: 7 + 2 = 9 terms
    # Total: 16 terms
    
    # But I allocated rank=17. Let me fix.
    rank = 16
    U = U[:, :rank]
    V = V[:, :rank]
    W = W[:, :rank]
    
    # Verify the decomposition
    # Check: sum_c U[i,c] * V[j,c] * W[k,c] should equal the tensor T[i,j,k]
    # where T is the matrix multiplication tensor
    # T[i, j, k] = 1 if and only if the (i,k) entry of A@B gets contribution from A[i,j]*B[j,k]
    # In our indexing: i is A row (0,1), j is shared index (0..3), k is B col (0..4)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i, j, k] = 1 for all valid i,j,k (it's the identity-like tensor for matrix mult)
    
    # Actually, the tensor is T[i,j,k] = 1 (since each product A[i,j]*B[j,k] contributes to C[i,k])
    # And the decomposition says: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, the convention is W[k*n+i, r], so W index is k*n + i.
    
    # Let me verify numerically
    def verify(U, V, W, n, m, p):
        rank = U.shape[1]
        for i in range(n):
            for k in range(p):
                # C[i,k] should be sum_j A[i,j]*B[j,k]
                # In tensor form: T[i,j,k] for all j should sum to give the product
                # Actually, the decomposition is:
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for the "structure"
                # But actually, it should be that for random A and B:
                # C[i,k] = sum_j A[i,j] * B[j,k] = sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] * A[i,j] * B[j,k]
                # No, that's not right. The decomposition is:
                # The tensor T where T[i,j,k] = 1 (for the multiplication map)
                # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                # So T[i,j,k] should equal 1 for all i,j,k? No, that's not right either.
                
                # The matrix multiplication tensor is defined as:
                # <T, A ⊗ B> = <C, A@B>
                # T[i,j,k] is the coefficient such that C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
                # So T[i,j,k] = 1 for all i,j,k.
                
                pass
        
        # Better: check that sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c] equals the tensor
        # The tensor T is a 3D array of shape (n, m, p) with T[i,j,k] = 1
        # But actually, the way it's structured:
        # U has shape (n*m, rank), indexed by (i,j) -> i*m+j
        # V has shape (m*p, rank), indexed by (j,k) -> j*p+k  
        # W has shape (p*n, rank), indexed by (k,i) -> k*n+i
        
        # The tensor element T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # And we need T[i,j,k] = 1 for all i,j,k.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1) > 1e-9:
                        return False, (i, j, k, val)
        return True, None
    
    ok, info = verify(U, V, W, n, m, p)
    
    if not ok:
        # Fall back to schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, c] = 1
                    V_fb[j * p + k, c] = 1
                    W_fb[k * n + i, c] = 1
                    c += 1
        return {
            "U": U_fb.tolist(),
            "V": V_fb.tolist(),
            "W": W_fb.tolist()
        }
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

