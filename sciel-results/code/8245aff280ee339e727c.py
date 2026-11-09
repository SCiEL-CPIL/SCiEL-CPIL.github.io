"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 130s to leave margin before 150s kill

    # Schoolbook rank
    rank_sb = n * m * p  # 128

    # We'll try to find a lower rank via random initialization + gradient descent
    # on the matrix multiplication tensor.
    
    # The tensor T[i,j,k] = delta for the multiplication:
    # (A * B)[i,k] = sum_j A[i,j] * B[j,k]
    # T is a 3-tensor of shape (n, m, p) where T[i,j,k] = 1 for all i,j,k
    # Wait, that's not right. The multiplication tensor has entries:
    # T_{i,j,k} = 1 if we're computing (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    # Actually the tensor is: T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # No wait. Let me think more carefully.
    
    # The matrix multiplication tensor T of size (n*m) x (m*p) x (p*n) is defined as:
    # T[a, b, c] = 1 where a = i*m + j (row of A, col of A), 
    #             b = j*p + k (row of B, col of B),
    #             c = k*n + i (row of result, col of result)
    # for all i in [0,n), j in [0,m), k in [0,p)
    # And T[a,b,c] = 0 otherwise.
    
    # A rank-r decomposition means:
    # T[a,b,c] = sum_{t=0}^{r-1} U[a,t] * V[b,t] * W[c,t]
    
    # For the schoolbook: r = n*m*p, with U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    
    # To find lower rank, we can use random tensor decomposition / CP decomposition.
    # But exact decomposition with small integers is hard.
    
    # Known results: For 4x4 * 4x8, the border rank / rank is at most:
    # Using Strassen-like methods. The rank of 4x4x4 is at most 49 (Strassen) or 48.
    # For 4x4x8, we can split the 8 into two 4s and use two 4x4x4 multiplications: rank <= 2*49 = 98
    # Or better: use the fact that 4x4x8 can be done with rank at most... 
    # Actually, let's just try to find a good decomposition numerically.
    
    # Strategy: Try CP decomposition with decreasing rank, using random init and
    # gradient descent to minimize ||T - sum U*V*W||_F^2, then check if we get exact.
    
    # Build the tensor T as a 3D array
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
            # T[i*m+j, j*p+k, k*n+i] = 1
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Reshape for computation: T_flat[a, b, c]
    # We want to minimize sum_{a,b,c} (T[a,b,c] - sum_t U[a,t]*V[b,t]*W[c,t])^2
    
    # Try various ranks
    best_result = None
    best_rank = rank_sb
    
    # Start with a reasonable upper bound. Known: 4x4x4 rank <= 49 (Strassen 49, or 48 with improvements)
    # For 4x4x8: split into two 4x4x4: rank <= 98. 
    # But there might be better. Let's try ranks from some upper bound down.
    
    # Actually, let's use a smarter approach: initialize with the schoolbook and try to
    # merge/combine terms, or use random CP decomposition.
    
    # Let's try random CP decomposition with gradient descent for various ranks.
    # Use a time budget to try multiple ranks.
    
    np.random.seed(42)
    
    # We'll try ranks in a specific order
    # Upper bound: 98 (two Strassen 4x4x4)
    # Try: 98, 96, 94, 92, 90, 88, 86, 84, 82, 80, 78, 76, 74, 72, 70, 68, 66, 64, 62, 60
    # But that's a lot. Let's be more strategic.
    
    # Actually, for exact integer decompositions, random continuous methods may not find them.
    # Let's try a different approach: use the known Strassen decomposition structure.
    
    # Strassen's 7-multiplication algorithm for 2x2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For 4x4x4, we can use block Strassen: split 4x4 into 2x2 blocks, each 2x2x2 uses 7 mults, total 7*4=28? No.
    # Block Strassen for 4x4: treat as 2x2 blocks of 2x2 matrices. Each "multiplication" of 2x2 matrices
    # can be done with 7 scalar multiplications (Strassen). So 7 block multiplications, each of rank 7*2*2=28? 
    # No, each block multiplication is a 2x2 * 2x2 -> 2x2, which has rank 8 (schoolbook) or 7 (Strassen).
    # So total rank = 7 * 7 = 49. Yes, that's the 49.
    
    # For 4x4x8: We can view this as 4x4 * 4x8. Split the 4x8 into two 4x4 blocks.
    # Then we need two 4x4x4 multiplications. Using Strassen for each: 49 + 49 = 98.
    # But we can do better by sharing some intermediate computations? Maybe not directly.
    
    # Alternatively, split 4x4 A into 2x2 blocks and 4x8 B into 2x4 blocks.
    # A = [[A11, A12], [A21, A22]] each 2x2
    # B = [[B11, B12, B13, B14], [B21, B22, B23, B24]] each 2x4
    # C = A*B is 4x8:
    # C11 = A11*B11 + A12*B21 (2x2 * 2x4 -> 2x4)
    # C12 = A11*B12 + A12*B22
    # C13 = A11*B13 + A12*B23
    # C14 = A11*B14 + A12*B24
    # C21 = A21*B11 + A22*B21
    # etc.
    
    # Each 2x2 * 2x4 multiplication: rank? 
    # 2x2 * 2x4: n=2, m=2, p=4. Schoolbook rank = 16.
    # Can we do better? Using Strassen-like: treat 2x4 as two 2x2 blocks.
    # 2x2 * 2x4 = 2x2 * [2x2 | 2x2] -> two 2x2*2x2 results.
    # Each 2x2*2x2 has rank 7 (Strassen). So 2x2*2x4 has rank 14? 
    # But the two 2x2*2x2 multiplications are independent, so rank = 7+7 = 14.
    
    # So for 4x4*4x8 with 2x2 blocks:
    # We have 4 block rows of A (each 2x2) times 2 block columns of B... wait.
    # A is 4x4 = 2x2 blocks of 2x2: 4 blocks total (2x2 grid)
    # B is 4x8 = 2x4 blocks of 2x4: 2 blocks total (1x2 grid, each 2x4)
    # C is 4x8 = 2x4 blocks of 2x4: 4 blocks total (2x2 grid)
    
    # C[i,j] = sum_k A[i,k] * B[k,j] where i,k are 2x2 block indices, j is 2x4 block index.
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]  (2x2 * 2x4 + 2x2 * 2x4)
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    # Each term A[i,k]*B[k,j] is a 2x2 * 2x4 multiplication with rank 14.
    # Total: 8 terms * 14 = 112. That's worse than 98.
    
    # Better: use Strassen at the 4x4 level for the 4x4*4x4 parts.
    # Split 4x8 into [4x4 | 4x4]. 
    # C_left = A * B_left (4x4 * 4x4), rank <= 49
    # C_right = A * B_right (4x4 * 4x4), rank <= 49
    # Total rank = 98.
    
    # Can we do better? Perhaps by not fully separating them.
    # The tensor 4x4x8 might have rank less than 98.
    
    # Let me try a numerical approach: use random CP decomposition with gradient descent.
    # If we can find an exact (or near-exact) decomposition with rank < 98, great.
    
    # For exactness, we need the reconstruction error to be essentially 0.
    
    def try_cp_decomposition(rank, max_iters=5000, lr=0.01, seed=0):
        """Try CP decomposition of T with given rank using gradient descent."""
        np.random.seed(seed)
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        # Gradient descent on F = sum_{a,b,c} (T[a,b,c] - sum_t U[a,t]*V[b,t]*W[c,t])^2
        # This is expensive for large tensors. Let's use a smarter approach.
        
        # Actually, let's use the fact that we can reshape and use matrix operations.
        # T_mat1[a, b, c] -> for fixed c, we have a matrix of size (n*m) x (m*p)
        # T_c[a,b] = T[a,b,c]
        # Reconstruction: sum_t U[:,t] * V[:,t]^T * W[c,t]
        
        # Gradient wrt U[a,t]: -2 * sum_{b,c} (T[a,b,c] - recon[a,b,c]) * V[b,t] * W[c,t]
        # This is still expensive.
        
        # Alternative: use ALS (Alternating Least Squares) for CP decomposition.
        # Fix V and W, solve for U: minimize ||T - U * (V ⊙ W)||_F^2
        # where (V ⊙ W) is the Khatri-Rao product.
        
        # For each t, U[:,t] = argmin ||T_unfolded - U[:,t] * (V[:,t] ⊙ W[:,t])^T||^2
        # Wait, let me think about this more carefully.
        
        # T[a,b,c] ≈ sum_t U[a,t] V[b,t] W[c,t]
        # For fixed V, W: 
        # sum_{b,c} (T[a,b,c] - sum_t U[a,t] V[b,t] W[c,t])^2 V[b,t'] W[c,t'] = sum_{b,c} T[a,b,c] V[b,t'] W[c,t']
        # This gives a linear system for U[:,t] for each t'.
        
        # Actually, it's simpler: for each t, the contribution is U[:,t] * (V[:,t] ⊙ W[:,t])^T
        # where (V[:,t] ⊙ W[:,t])[b*n*m + ... ] hmm, let me reshape.
        
        # Let me define: for each t, define M_t[a, (b,c)] = U[a,t] * V[b,t] * W[c,t]
        # Then T[a, (b,c)] = sum_t M_t[a, (b,c)]
        
        # Reshape T to (n*m) x (m*p * p*n)
        T_2d = T.reshape(n * m, m * p * p * n)
        
        # For ALS on U:
        # Fix V, W. For each t, we want to find U[:,t] to minimize
        # ||T_2d - sum_t U[:,t] * (V[:,t] ⊙ W[:,t]_reshaped)^T||_F^2
        
        # (V[:,t] ⊙ W[:,t]) has length m*p * p*n = m*p^2*n
        # Element (b, c) of the Khatri-Rao is V[b,t] * W[c,t]
        
        # So the gradient equation for U[:,t] is:
        # sum_t' U[:,t'] * (V[:,t'] ⊙ W[:,t'])^T * (V[:,t] ⊙ W[:,t]) = T_2d * (V[:,t] ⊙ W[:,t])
        
        # Let G = sum_t' U[:,t'] * (V[:,t'] ⊙ W[:,t'])^T  (this is the current reconstruction)
        # Then: G * (V[:,t] ⊙ W[:,t]) = T_2d * (V[:,t] ⊙ W[:,t])
        # But G depends on all U, so this is coupled.
        
        # Actually, for ALS, we fix all other U[:,t'] for t' != t and solve for U[:,t].
        # T_2d - sum_{t' != t} U[:,t'] * (V[:,t'] ⊙ W[:,t'])^T = U[:,t] * (V[:,t] ⊙ W[:,t])^T
        # Let R = T_2d - sum_{t' != t} U[:,t'] * (V[:,t'] ⊙ W[:,t'])^T
        # Then R = U[:,t] * (V[:,t] ⊙ W[:,t])^T
        # This is a rank-1 approximation problem. The optimal U[:,t] is:
        # U[:,t] = R * (V[:,t] ⊙ W[:,t]) / ||V[:,t] ⊙ W[:,t]||^2
        
        # This is getting complex. Let me just use a simple gradient-based approach with 
        # a time limit and see if we can find something.
        
        # Actually, let me try a much simpler approach: since we know the structure,
        # let's just return the schoolbook or a Strassen-based construction.
        
        # For now, let's construct the Strassen-based 4x4x4 decomposition and use it
        # twice for 4x4x8.
        
        return None  # Placeholder
    
    # Let's construct the Strassen 4x4x4 decomposition explicitly.
    # 4x4 matrix A split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12], [B21, B22]] where each is 2x2
    # C = A*B = [[C11, C12], [C21, C22]]
    
    # Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mi is a 2x2 * 2x2 multiplication, which has rank 8 (schoolbook) or 7 (Strassen).
    # If we use Strassen recursively, each 2x2*2x2 has rank 7.
    # Total rank = 7 * 7 = 49.
    
    # To build the explicit factor matrices for 4x4x4 with rank 49:
    # We need to express each of the 7 Strassen multiplications as 7 scalar multiplications.
    
    # This is complex but doable. Let me define the 2x2x2 Strassen decomposition first.
    
    # 2x2x2 Strassen:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For the 4x4x4 case, each "block" is a 2x2 matrix, so each Mi involves multiplying
    # two 2x2 matrices, which we do with 7 scalar multiplications.
    
    # Let me index the 4x4 matrices as:
    # A[i,j] for i,j in 0..3
    # B[j,k] for j,k in 0..3
    # C[i,k] for i,k in 0..3
    
    # Block indices: A[0:2, 0:2] = A11, A[0:2, 2:4] = A12, A[2:4, 0:2] = A21, A[2:4, 2:4] = A22
    # Similarly for B.
    
    # This is getting very involved. Let me just use the schoolbook for now and try to
    # improve it with a simple optimization.
    
    # Actually, let me try a different strategy: use the fact that we can find a 
    # decomposition by solving a system of equations. But that's also hard.
    
    # Given the complexity, let me just return the schoolbook decomposition (rank 128)
    # and hope that the evolution finds something better. But the task asks for rank-minimal.
    
    # Let me at least try to construct the rank-98 decomposition (two Strassen 4x4x4).
    
    # For 4x4x4 Strassen decomposition, I'll build it explicitly.
    
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
        n2, m2, p2 = 2, 2, 2
        U = np.zeros((n2 * m2, 7))
        V = np.zeros((m2 * p2, 7))
        W = np.zeros((p2 * n2, 7))
        
        # Indexing: A[i,j] -> U[i*m2+j, t], B[j,k] -> V[j*p2+k, t], C[i,k] -> W[k*n2+i, t]
        
        # M1 = (a+d)(e+h): A terms: a=A[0,0], d=A[1,1]; B terms: e=B[0,0], h=B[1,1]
        # U[0,0]=1 (a), U[3,0]=1 (d); V[0,0]=1 (e), V[3,0]=1 (h)
        # C11 += M1: W[0,0]=1; C22 += M1: W[3,0]=1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (c+d)e: A terms: c=A[1,0], d=A[1,1]; B term: e=B[0,0]
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        # C21 += M2: W[1, 1] = 1
        W[1, 1] = 1
        
        # M3 = a(f-h): A term: a=A[0,0]; B terms: f=B[0,1], h=B[1,1]
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        # C12 += M3: W[1, 2] = 1
        W[1, 2] = 1
        
        # M4 = d(g-e): A term: d=A[1,1]; B terms: g=B[1,0], e=B[0,0]
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        # C11 += M4: W[0, 3] = 1; C21 += M4: W[1, 3] = 1
        W[0, 3] = 1; W[1, 3] = 1
        
        # M5 = (a+b)h: A terms: a=A[0,0], b=A[0,1]; B term: h=B[1,1]
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        # C11 -= M5: W[0, 4] = -1; C12 += M5: W[1, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (c-a)(e+f): A terms: c=A[1,0], a=A[0,0]; B terms: e=B[0,0], f=B[0,1]
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        # C22 += M6: W[3, 5] = 1
        W[3, 5] = 1
        
        # M7 = (b-d)(g+h): A terms: b=A[0,1], d=A[1,1]; B terms: g=B[1,0], h=B[1,1]
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        # C11 += M7: W[0, 6] = 1
        W[0, 6] = 1
        
        return U, V, W
    
    # Verify Strassen 2x2x2
    U2, V2, W2 = strassen_2x2x2()
    T2 = np.einsum('at,bt,ct->abc', U2, V2, W2)
    T2_expected = np.zeros((4, 4, 4))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                T2_expected[i*2+j, j*2+k, k*2+i] = 1
    if not np.allclose(T2, T2_expected):
        print("Strassen 2x2x2 verification failed!")
    
    # Now build 4x4x4 Strassen decomposition (rank 49)
    def strassen_4x4x4():
        """Return U, V, W for 4x4x4 Strassen decomposition (rank 49)."""
        n4, m4, p4 = 4, 4, 4
        U4 = np.zeros((n4 * m4, 49))
        V4 = np.zeros((m4 * p4, 49))
        W4 = np.zeros((p4 * n4, 49))
        
        # 7 Strassen multiplications, each is a 2x2*2x2 with rank 7
        # For each of the 7 multiplications, we need to specify which 2x2 blocks of A, B are involved
        # and how the 2x2 result maps to the 4x4 C.
        
        # Strassen formulas for 4x4 (with 2x2 blocks):
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Block indices in 4x4:
        # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4], A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
        # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4], B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
        # C11 = C[0:2, 0:2], C12 = C[0:2, 2:4], C21 = C[2:4, 0:2], C22 = C[2:4, 2:4]
        
        # For each Strassen multiplication Mi, we have:
        # Mi = Ai * Bi where Ai and Bi are linear combinations of 2x2 blocks.
        # The result Mi is a 2x2 matrix, and it contributes to certain blocks of C.
        
        # Let's define for each of the 7 multiplications:
        # - The A-side: a 2x2 matrix that is a linear combination of A blocks
        # - The B-side: a 2x2 matrix that is a linear combination of B blocks
        # - The C-side: which blocks of C get contributions and with what signs
        
        # Multiplication 1: (A11+A22)(B11+B22)
        # A side: A11 + A22 -> 2x2 matrix where element (i,j) = A[0:2,0:2][i,j] + A[2:4,2:4][i,j]
        # B side: B11 + B22
        # C side: C11 += M1, C22 += M1
        
        # Multiplication 2: (A21+A22)B11
        # A side: A21 + A22
        # B side: B11
        # C side: C21 += M2
        
        # Multiplication 3: A11(B12-B22)
        # A side: A11
        # B side: B12 - B22
        # C side: C12 += M3
        
        # Multiplication 4: A22(B21-B11)
        # A side: A22
        # B side: B21 - B11
        # C side: C11 += M4, C21 += M4
        
        # Multiplication 5: (A11+A12)B22
        # A side: A11 + A12
        # B side: B22
        # C side: C11 -= M5, C12 += M5
        
        # Multiplication 6: (A21-A11)(B11+B12)
        # A side: A21 - A11
        # B side: B11 + B12
        # C side: C22 += M6
        
        # Multiplication 7: (A12-A22)(B21+B22)
        # A side: A12 - A22
        # B side: B21 + B22
        # C side: C11 += M7
        
        # For each multiplication, we use the 2x2x2 Strassen decomposition.
        # The 2x2x2 Strassen has U2, V2, W2 of shape (4,7), (4,7), (4,7).
        # We need to map the 2x2 indices to the 4x4 indices.
        
        # For a 2x2 block at position (row_block, col_block) in the 4x4 matrix:
        # Element (i,j) of the block maps to 4x4 index (2*row_block + i, 2*col_block + j)
        # In the U matrix (size 16 x rank), index = (2*row_block + i) * 4 + (2*col_block + j)
        # In the V matrix (size 16 x rank), index = (2*row_block + i) * 4 + (2*col_block + j)
        # In the W matrix (size 16 x rank), index = (2*col_block + j) * 4 + (2*row_block + i)
        
        # Let me define the 7 multiplications more precisely.
        
        # For each multiplication s (0..6):
        # A side: a 2x2 matrix A_s where A_s[i,j] = sum of coefficients * A_block[i,j]
        # B side: a 2x2 matrix B_s where B_s[i,j] = sum of coefficients * B_block[i,j]
        # C side: contributions to C blocks
        
        # Let me define:
        # mul_defs[s] = (A_blocks, B_blocks, C_contributions)
        # A_blocks: list of (block_row, block_col, coeff) for the A side
        # B_blocks: list of (block_row, block_col, coeff) for the B side
        # C_contributions: list of (block_row, block_col, coeff) for the C side
        
        # Block (br, bc) refers to the 2x2 block at row br, col bc in the 4x4 matrix.
        
        mul_defs = [
            # M1: (A11+A22)(B11+B22)
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
            # M2: (A21+A22)B11
            ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1)]),
            # M3: A11(B12-B22)
            ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1)]),
            # M4: A22(B21-B11)
            ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
            # M5: (A11+A12)B22
            ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
            # M6: (A21-A11)(B11+B12)
            ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
            # M7: (A12-A22)(B21+B22)
            ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]),
        ]
        
        for s, (A_blocks, B_blocks, C_contribs) in enumerate(mul_defs):
            # For this multiplication, use the 2x2x2 Strassen decomposition
            for t in range(7):
                col = s * 7 + t
                
                # A side: for each (i,j) in 2x2, U4[4x4_index, col] += coeff * U2[i*2+j, t]
                for i in range(2):
                    for j in range(2):
                        u2_val = U2[i * 2 + j, t]
                        if u2_val == 0:
                            continue
                        for (br, bc, coeff) in A_blocks:
                            # 4x4 index: (2*br + i) * 4 + (2*bc + j)
                            idx = (2 * br + i) * 4 + (2 * bc + j)
                            U4[idx, col] += coeff * u2_val
                
                # B side: for each (j,k) in 2x2, V4[4x4_index, col] += coeff * V2[j*2+k, t]
                for j in range(2):
                    for k in range(2):
                        v2_val = V2[j * 2 + k, t]
                        if v2_val == 0:
                            continue
                        for (br, bc, coeff) in B_blocks:
                            # 4x4 index: (2*br + j) * 4 + (2*bc + k)
                            idx = (2 * br + j) * 4 + (2 * bc + k)
                            V4[idx, col] += coeff * v2_val
                
                # C side: for each (i,k) in 2x2, W4[4x4_index, col] += coeff * W2[k*2+i, t]
                for i in range(2):
                    for k in range(2):
                        w2_val = W2[k * 2 + i, t]
                        if w2_val == 0:
                            continue
                        for (br, bc, coeff) in C_contribs:
                            # 4x4 index: (2*bc + k) * 4 + (2*br + i)
                            idx = (2 * bc + k) * 4 + (2 * br + i)
                            W4[idx, col] += coeff * w2_val
        
        return U4, V4, W4
    
    # Verify 4x4x4 Strassen
    U4, V4, W4 = strassen_4x4x4()
    T4 = np.einsum('at,bt,ct->abc', U4, V4, W4)
    T4_expected = np.zeros((16, 16, 16))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                T4_expected[i*4+j, j*4+k, k*4+i] = 1
    if not np.allclose(T4, T4_expected):
        print("Strassen 4x4x4 verification failed!")
        print("Max error:", np.max(np.abs(T4 - T4_expected)))
    
    # Now build 4x4x8 decomposition using two 4x4x4 Strassen decompositions.
    # Split B (4x8) into B_left (4x4) and B_right (4x4).
    # C_left = A * B_left, C_right = A * B_right.
    
    # For the 4x4x8 tensor:
    # U has shape (16, rank), V has shape (32, rank), W has shape (32, rank)
    # V[0:16, :] corresponds to B_left, V[16:32, :] corresponds to B_right
    # W[0:16, :] corresponds to C_left, W[16:32, :] corresponds to C_right
    
    # Wait, let me check the indexing:
    # V index = j*p + k where j in [0,4), k in [0,8). So V has 32 rows.
    # For B_left (k in [0,4)): V index = j*8 + k for k in [0,4)
    # For B_right (k in [4,8)): V index = j*8 + k for k in [4,8)
    
    # W index = k*n + i where k in [0,8), i in [0,4). So W has 32 rows.
    # For C_left (k in [0,4)): W index = k*4 + i for k in [0,4)
    # For C_right (k in [4,8)): W index = k*4 + i for k in [4,8)
    
    # So:
    # U4_left = U4 (same A)
    # V4_left[j*4+k, t] -> V[j*8+k, t] for k in [0,4)
    # W4_left[k*4+i, t] -> W[k*4+i, t] for k in [0,4)
    
    # U4_right = U4 (same A)
    # V4_right[j*4+k, t] -> V[j*8+4+k, t] for k in [0,4)
    # W4_right[k*4+i, t] -> W[(4+k)*4+i, t] for k in [0,4)
    
    U8 = np.zeros((16, 98))
    V8 = np.zeros((32, 98))
    W8 = np.zeros((32, 98))
    
    # Left half (columns 0-47)
    U8[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V8[j*8+k, 0:49] = V4[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W8[k*4+i, 0:49] = W4[k*4+i, 0:49]
    
    # Right half (columns 49-97)
    U8[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V8[j*8+4+k, 49:98] = V4[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W8[(4+k)*4+i, 49:98] = W4[k*4+i, 0:49]
    
    # Verify 4x4x8
    T8 = np.einsum('at,bt,ct->abc', U8, V8, W8)
    T8_expected = np.zeros((16, 32, 32))
    for i in range(4):
        for j in range(4):
            for k in range(8):
                T8_expected[i*4+j, j*8+k, k*4+i] = 1
    if not np.allclose(T8, T8_expected):
        print("4x4x8 Strassen verification failed!")
        print("Max error:", np.max(np.abs(T8 - T8_expected)))
    else:
        print("4x4x8 Strassen verification passed! Rank 98.")
    
    # Convert to lists
    result = {
        "U": U8.tolist(),
        "V": V8.tolist(),
        "W": W8.tolist()
    }
    
    return result
# EVOLVE-BLOCK-END
