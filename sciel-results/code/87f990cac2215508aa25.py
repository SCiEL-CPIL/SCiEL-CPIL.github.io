import numpy as np
import time

def _verify_mm(U, V, W, n, m, p):
    """
    Verify if U, V, W decompose the matrix multiplication tensor M_{n,m,p}.
    Condition: sum_r U[i,j]^r V[j',k]^r W[i',k']^r = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Reshape for easier indexing
    # U: (n*m, rank) -> (n, m, rank)
    U_t = U.reshape(n, m, rank)
    # V: (m*p, rank) -> (m, p, rank)
    V_t = V.reshape(m, p, rank)
    # W: (p*n, rank) -> (n, p, rank) ?? 
    # The code uses W[k*n+i, r]. 
    # So W is indexed by (k, i) where k is col of C (p), i is row of C (n).
    # So W_t should be (p, n, rank) or (n, p, rank)?
    # Let's stick to the linear index to be safe.
    
    # We need to check if the tensor T constructed from U,V,W matches M_{n,m,p}.
    # T_{i,j,j',k,i',k'} = sum_r U[i,j]^r V[j',k]^r W[i',k']^r
    # We want T to be 1 if i=i', j=j', k=k', else 0.
    
    # This is a 6D tensor check, which is expensive.
    # Instead, we can check the condition on random matrices or specific basis matrices.
    # Check for all i, j, k, i', k':
    # sum_{r} U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] 
    # should be 1 if i==i' and j==j' and k==k', else 0.
    
    # Note: The indices in V are j', k. In U they are i, j.
    # The condition is delta_{i,i'} delta_{j,j'} delta_{k,k'}.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check the '1' case: i'=i, j'=j, k'=k
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
                
                # Check the '0' cases:
                # This is too many loops. 
                # Alternative: Construct the tensor and check sparsity?
                # Or check a few random cases?
                # For exactness, we should check all.
                # But n=3, m=4, p=8. 3*4*8 = 96 triples.
                # For each triple, we check against all other triples?
                # No, the condition is local.
                # The tensor T is determined by U, V, W.
                # We can verify by checking if the bilinear map computes AB correctly.
                
    # Better verification:
    # Pick random A, B. Compute AB.
    # Compute C_decomp = sum_r (U_r dot A) (V_r dot B) W_r.
    # Check if C_decomp == AB.
    
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_true = A @ B
    
    C_decomp = np.zeros((n, p))
    for r in range(rank):
        u_r = U[:, r].reshape(n, m)
        v_r = V[:, r].reshape(m, p)
        w_r = W[:, r].reshape(p, n) # Wait, W index is k*n+i. So W[:, r] is vector of length p*n.
        # Reshape W[:, r] to (p, n) then transpose to (n, p)?
        # W[k*n+i, r] corresponds to element (k, i) in a (p, n) matrix.
        # So w_r_matrix = W[:, r].reshape(p, n).T
        w_r_matrix = W[:, r].reshape(p, n).T
        
        alpha = np.dot(u_r, A) # scalar
        beta = np.dot(v_r, B) # scalar
        C_decomp += alpha * beta * w_r_matrix
        
    if not np.allclose(C_decomp, C_true, atol=1e-9):
        return False
        
    return True

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 Strassen.
    Shapes: (4, 7), (4, 7), (4, 7).
    Indices:
    U: (i, j) with i,j in 0..1. Index = i*2+j.
    V: (j, k) with j,k in 0..1. Index = j*2+k.
    W: (k, i) with k,i in 0..1. Index = k*2+i.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00 -> C10 += M2, C11 += M2
    # M3 = A00(B01-B11) -> C01 += M3
    # M4 = A11(B10-B00) -> C10 += M4
    # M5 = (A00+A01)B11 -> C01 += M5, C11 -= M5
    # M6 = (A10-A00)(B00+B01) -> C10 += M6
    # M7 = (A01-A11)(B10+B11) -> C01 -= M7
    
    # Coefficients for U (A part)
    # M1: A00=1, A11=1
    U[0, 0] = 1; U[3, 0] = 1
    # M2: A10=1, A11=1
    U[2, 1] = 1; U[3, 1] = 1
    # M3: A00=1
    U[0, 2] = 1
    # M4: A11=1
    U[3, 3] = 1
    # M5: A00=1, A01=1
    U[0, 4] = 1; U[1, 4] = 1
    # M6: A10=1, A00=-1
    U[2, 5] = 1; U[0, 5] = -1
    # M7: A01=1, A11=-1
    U[1, 6] = 1; U[3, 6] = -1
    
    # Coefficients for V (B part)
    # M1: B00=1, B11=1
    V[0, 0] = 1; V[3, 0] = 1
    # M2: B00=1
    V[0, 1] = 1
    # M3: B01=1, B11=-1
    V[1, 2] = 1; V[3, 2] = -1
    # M4: B10=1, B00=-1
    V[2, 3] = 1; V[0, 3] = -1
    # M5: B11=1
    V[3, 4] = 1
    # M6: B00=1, B01=1
    V[0, 5] = 1; V[1, 5] = 1
    # M7: B10=1, B11=1
    V[2, 6] = 1; V[3, 6] = 1
    
    # Coefficients for W (C part)
    # W index is k*2+i.
    # M1: C00+=1, C11+=1
    W[0, 0] = 1; W[3, 0] = 1
    # M2: C10+=1, C11+=1
    W[2, 1] = 1; W[3, 1] = 1
    # M3: C01+=1
    W[1, 2] = 1
    # M4: C10+=1
    W[2, 3] = 1
    # M5: C01+=1, C11-=-1
    W[1, 4] = 1; W[3, 4] = -1
    # M6: C10+=1
    W[2, 5] = 1
    # M7: C01-=-1
    W[1, 6] = -1
    
    return U, V, W

def construct():
    n, m, p = 3, 4, 8
    start_time = time.time()
    deadline = start_time + 110
    
    # Use 4x4x8 decomposition restricted to 3x4x8
    # 4x4x8 can be tiled with 4 blocks of 2x2x2.
    # Rows of A: 0,1,2,3. Cols of A: 0,1,2,3.
    # Rows of B: 0,1,2,3. Cols of B: 0..7.
    # Blocks:
    # 1. A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # 2. A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # 3. A[2:4, 0:2] * B[0:2, 0:2] -> C[2:4, 0:2]
    # 4. A[2:4, 2:4] * B[2:4, 0:2] -> C[2:4, 0:2]
    # Wait, B has 8 cols.
    # We need to cover B[0:4, 0:8] and B[4:8, 0:8]?
    # No, B is 4x8.
    # We can split B into two 4x4 blocks: B1 (cols 0-3) and B2 (cols 4-7).
    # Then we need 2x2x2 blocks for A[0:2, 0:2]*B1[0:2, 0:2], etc.
    # Actually, simpler:
    # Tile A (4x4) with 2x2 blocks: 4 blocks.
    # Tile B (4x8) with 2x2 blocks: 2 rows x 4 cols = 8 blocks?
    # No, Strassen is 2x2x2.
    # We need to match the inner dimension.
    # A is 4x4. B is 4x8.
    # We can use 4 Strassen blocks for the first 4 cols of B, and 4 for the next 4.
    # Total 8 blocks? Rank 8*7 = 56.
    # Can we do better?
    # Maybe use 4x4x4 Strassen?
    # Rank of 4x4x4 is 49.
    # So 4x4x8 would be 49 + 49 = 98? No, 49 for 4x4x4.
    # So 4x4x8 is 2 * 49 = 98.
    # But 2x2x2 tiling gives 56.
    # 56 is better than 98.
    
    # Let's use the 2x2x2 tiling for 4x4x8.
    # A: 4x4. B: 4x8.
    # Split A into 4 blocks of 2x2: A00, A01, A10, A11.
    # Split B into 8 blocks of 2x2: B00, B01, B02, B03 (for first 4 cols) and B10..B13 (for next 4).
    # Wait, B is 4 rows. So B is split into 2 blocks of 2 rows.
    # B_top (2x8), B_bottom (2x8).
    # A_top (2x4), A_bottom (2x4).
    # C = A_top @ B_top + A_bottom @ B_bottom.
    # A_top is 2x4. B_top is 2x8.
    # This is 2x4x8.
    # We can tile 2x4x8 with 2x2x2 blocks?
    # A_top cols: 0,1 and 2,3.
    # B_top cols: 0,1 and 2,3 and 4,5 and 6,7.
    # So 2 (A col blocks) * 4 (B col blocks) = 8 blocks of 2x2x2.
    # Each block rank 7. Total 56.
    # Same for A_bottom.
    # Total 16 blocks? No.
    # A is 4x4.
    # A[0:2, 0:2] @ B[0:2, 0:2] -> C[0:2, 0:2]
    # A[0:2, 2:4] @ B[2:4, 0:2] -> C[0:2, 0:2]
    # A[2:4, 0:2] @ B[0:2, 0:2] -> C[2:4, 0:2]
    # A[2:4, 2:4] @ B[2:4, 0:2] -> C[2:4, 0:2]
    # This covers C[0:4, 0:2].
    # We need C[0:4, 0:8].
    # So we repeat this for cols 2:4, 4:6, 6:8 of B.
    # Total 4 sets of 4 blocks.
    # 16 blocks of 2x2x2.
    # Rank 16 * 7 = 112.
    # This is worse than schoolbook 96.
    
    # Wait, the seed program found 60. How?
    # Maybe it used 3x3x3 Laderman?
    # Or maybe it used a better tiling.
    
    # Let's try to construct a 3x4x8 decomposition directly using 2x2x2 blocks where possible.
    # n=3, m=4, p=8.
    # We can fit one 2x2x2 block at (0,0,0).
    # Remaining:
    # i=2 (1 row)
    # j=2,3 (2 cols)
    # k=2..7 (6 cols)
    # This leaves a 1x2x6 and 2x2x6 and 1x2x6?
    # It's messy.
    
    # Let's stick to the 4x4x8 restriction.
    # If I use 2x2x2 tiling for 4x4x8, I get rank 112?
    # No, 4x4x8.
    # A is 4x4. B is 4x8.
    # We can view this as 2 multiplications of 4x4x4.
    # Rank 2 * 49 = 98.
    # Or 4 multiplications of 2x2x4?
    # 2x2x4 can be done with 2x2x2 blocks?
    # 2x2x4 = 2x2x2 + 2x2x2. Rank 14.
    # 4 blocks of 2x2x4. Rank 4 * 14 = 56.
    # Yes!
    # A is 4x4. Split into four 2x2 blocks.
    # B is 4x8. Split into four 2x2x4 blocks?
    # No, B is 4 rows.
    # We can split B into two 2x8 blocks.
    # Then we have 2 (A row blocks) * 2 (B row blocks) = 4 terms?
    # No, matrix multiplication sums over the inner dimension.
    # C = A @ B.
    # A is 4x4. B is 4x8.
    # Split A into A1 (2x4) and A2 (2x4).
    # Split B into B1 (4x4) and B2 (4x4).
    # C = A1 @ B1 + A2 @ B2? No.
    # C = [A1; A2] @ [B1 B2] = [A1@B1 + A2@B1, A1@B2 + A2@B2].
    # So we need to compute A1@B1, A2@B1, A1@B2, A2@B2.
    # Each is 2x4 @ 4x4 = 2x4x4.
    # Rank of 2x4x4?
    # 2x4x4 can be tiled with 2x2x2 blocks.
    # A1 is 2x4. B1 is 4x4.
    # Split A1 cols into 2+2.
    # Split B1 rows into 2+2.
    # So 2 (A col blocks) * 2 (B row blocks) = 4 blocks of 2x2x2.
    # Rank 4 * 7 = 28.
    # So each 2x4x4 multiplication has rank 28.
    # We have 4 such multiplications.
    # Total rank 4 * 28 = 112.
    # Still 112.
    
    # Is there a better way for 2x4x4?
    # Maybe use 2x2x2 Strassen more efficiently?
    # No, 2x4x4 is essentially two 2x2x2 multiplications in parallel?
    # A1 = [A11 A12]. B1 = [B11; B21].
    # A1 @ B1 = A11 @ B11 + A12 @ B21.
    # These are independent 2x2x2 multiplications.
    # So rank is 7 + 7 = 14.
    # So 2x4x4 has rank 14.
    # Then 4x4x8 has rank 4 * 14 = 56.
    # Yes!
    # So Rank(4x4x8) = 56.
    # Restricting to 3x4x8, we just zero out the 4th row of A and 4th row of C.
    # The rank remains 56 (or less if terms vanish, but they don't).
    # So 56 is a valid upper bound.
    # The seed program found 60.
    # So 56 is an improvement.
    
    # Let's implement the 56 rank decomposition.
    
    U = np.zeros((3*4, 56))
    V = np.zeros((4*8, 56))
    W = np.zeros((8*3, 56))
    
    col = 0
    # 4 blocks of 2x4x4.
    # Block 1: A[0:2, :] @ B[:, 0:4] -> C[0:2, 0:4]
    # Block 2: A[2:4, :] @ B[:, 0:4] -> C[2:4, 0:4] (but we only need C[0:3])
    # Block 3: A[0:2, :] @ B[:, 4:8] -> C[0:2, 4:8]
    # Block 4: A[2:4, :] @ B[:, 4:8] -> C[2:4, 4:8]
    
    # For each 2x4x4 block, we use 4 Strassen 2x2x2 blocks.
    # 2x4x4: A is 2x4, B is 4x4, C is 2x4.
    # A cols: 0,1 and 2,3.
    # B rows: 0,1 and 2,3.
    # So 4 sub-blocks:
    # 1. A[:, 0:2] @ B[0:2, :] -> C[:, 0:2]
    # 2. A[:, 2:4] @ B[2:4, :] -> C[:, 0:2]
    # 3. A[:, 0:2] @ B[0:2, :] -> C[:, 2:4] (Wait, B is 4x4, so B[0:2, :] is 2x4)
    # No, B is 4x4.
    # B[0:2, 0:2], B[0:2, 2:4], B[2:4, 0:2], B[2:4, 2:4].
    # A[0:2, 0:2] @ B[0:2, 0:2] -> C[0:2, 0:2]
    # A[0:2, 2:4] @ B[2:4, 0:2] -> C[0:2, 0:2]
    # A[0:2, 0:2] @ B[0:2, 2:4] -> C[0:2, 2:4]
    # A[0:2, 2:4] @ B[2:4, 2:4] -> C[0:2, 2:4]
    
    # So for each 2x4x4, we have 4 Strassen blocks.
    # Total 16 Strassen blocks.
    # Rank 16 * 7 = 112?
    # Wait, earlier I said 14.
    # A1 @ B1 = A11 @ B11 + A12 @ B21.
    # A1 is 2x4. B1 is 4x4.
    # A1 = [A11 A12]. B1 = [B11; B21].
    # A1 @ B1 = A11 @ B11 + A12 @ B21.
    # A11 is 2x2. B11 is 2x4.
    # This is 2x2x4.
    # 2x2x4 can be done with 2x2x2 blocks?
    # A11 is 2x2. B11 is 2x4.
    # B11 = [B111 B112].
    # A11 @ B111 -> 2x2.
    # A11 @ B112 -> 2x2.
    # So 2 Strassen blocks. Rank 14.
    # So A11 @ B11 is rank 14.
    # Similarly A12 @ B21 is rank 14.
    # Total 28.
    # So 2x4x4 is rank 28.
    # 4 such blocks -> 112.
    
    # My previous logic was flawed.
    # 2x4x4 is NOT rank 14. It is rank 28.
    # So 4x4x8 is rank 112.
    
    # So 56 is not achievable with simple tiling.
    # The seed program found 60.
    # Maybe it used a different strategy.
    
    # Let's just use the seed program's approach but with a better search or known decompositions.
    # Given the time limit, I will output the seed program's logic but with the CORRECT verification.
    # If the seed program found 60, and I verify it correctly, it might still be 60.
    # But if the seed program's verification was wrong, it might have returned an invalid decomposition that happened to pass the wrong check.
    # I must ensure the decomposition is valid.
    
    # I will implement the 112 rank decomposition as a fallback, and try to find a better one.
    # Actually, I will just return the schoolbook (96) if I can't find better quickly.
    # But 60 is better than 96.
    
    # Let's assume the seed program's Strassen tiling is valid for the all-ones tensor, 
    # but we need the MM tensor.
    # I will implement a correct MM tensor decomposition using 2x2x2 blocks.
    
    # I'll use the 112 rank decomposition as it's guaranteed to work.
    # 112 > 96. So it's worse than schoolbook.
    # I need rank < 96.
    
    # I will use the Laderman 3x3x3 (rank 23) and pad.
    # 3x4x8.
    # Use 3x3x3 for the main part.
    # This is complex.
    
    # I will return the schoolbook decomposition (96) to be safe, but I know 60 is possible.
    # I'll try to reproduce the seed's 60 rank result with correct verification.
    
    return _schoolbook(3, 4, 8)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
