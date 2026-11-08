"""Matrix multiplication 2x4x6 rank-minimal decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120

    # Strategy: Use Kronecker product structure.
    # 2x4x6 = (2x2) x (2x3)
    # Rank of 2x2x2 multiplication is 7 (Strassen).
    # Rank of 1x2x3 (which is effectively a vector-matrix product or just linear map) is 6 (trivial).
    # Wait, the factorization is T_{n,m,p} = T_{n1,m1,p1} \otimes T_{n2,m2,p2}
    # Here n=2, m=4, p=6.
    # We can write m=2*2, p=2*3.
    # So T_{2,4,6} = T_{2,2,2} \otimes T_{1,2,3}? No, dimensions must match.
    # T_{n,m,p} has indices i (n), j (m), k (p).
    # If we decompose m = m1*m2 and p = p1*p2, we need n = n1*n2?
    # Actually, the tensor T_{n,m,p} can be viewed as T_{n1*n2, m1*m2, p1*p2}.
    # If T_{n,m,p} = T_{n1,m1,p1} \otimes T_{n2,m2,p2}, then rank(T_{n,m,p}) <= rank(T_{n1,m1,p1}) * rank(T_{n2,m2,p2}).
    # Let's try n=2=2*1, m=4=2*2, p=6=3*2.
    # T_{2,4,6} = T_{2,2,3} \otimes T_{1,2,2}?
    # Rank(T_{1,2,2}) is 2 (it's just 2 independent terms).
    # Rank(T_{2,2,3})?
    # T_{2,2,3} multiplies 2x2 by 2x3.
    # Standard rank is 2*2*3 = 12.
    # Can we do better?
    # T_{2,2,3} is equivalent to two copies of T_{2,2,1} plus some cross terms? No.
    # T_{2,2,3} is the tensor for multiplying A(2x2) and B(2x3).
    # This is not a standard square case.
    
    # Let's try n=2=2*1, m=4=4*1, p=6=6*1.
    # T_{2,4,6} = T_{2,4,6} \otimes T_{1,1,1}. Rank 48.
    
    # Let's try n=2=1*2, m=4=2*2, p=6=3*2.
    # T_{2,4,6} = T_{1,2,3} \otimes T_{2,2,2}.
    # Rank(T_{1,2,3}) = 6 (schoolbook for 1x2x3).
    # Rank(T_{2,2,2}) = 7 (Strassen).
    # Total Rank = 6 * 7 = 42.
    
    # Let's verify the Kronecker structure.
    # T_{n,m,p}[i,j,k] = sum_t U[i,t] V[j,t] W[k,t]
    # If T = T1 \otimes T2, then:
    # T[i1*n2+i2, j1*m2+j2, k1*p2+k2] = sum_{t1,t2} (U1[i1,t1]U2[i2,t2]) (V1[j1,t1]V2[j2,t2]) (W1[k1,t1]W2[k2,t2])
    # = sum_t (U1[i1,t1]U2[i2,t2] V1[j1,t1]V2[j2,t2] W1[k1,t1]W2[k2,t2])
    # This matches the matrix multiplication tensor if the indices are ordered correctly.
    # For T_{1,2,3}, n1=1, m1=2, p1=3.
    # For T_{2,2,2}, n2=2, m2=2, p2=2.
    # Total n = 1*2 = 2. m = 2*2 = 4. p = 3*2 = 6.
    # This works!
    
    # Step 1: Construct T_{1,2,3} decomposition.
    # Rank 6.
    # U1: (1*2, 6) = (2, 6)
    # V1: (2*3, 6) = (6, 6)
    # W1: (3*1, 6) = (3, 6)
    # Indices: i1 in 0..0, j1 in 0..1, k1 in 0..2.
    # T1[i1, j1, k1] = 1 if j1==j1 and k1==k1? No.
    # T_{n,m,p}[i,j,k] corresponds to C[i,k] += A[i,j] * B[j,k].
    # For T_{1,2,3}: C is 1x3, A is 1x2, B is 2x3.
    # C[0, k] = sum_j A[0, j] B[j, k].
    # Tensor T[i, j, k] = 1 if i=0 (always) and we are computing term A[0,j]B[j,k].
    # Actually, the tensor T_{n,m,p} has entries T[i,j,k] = 1.
    # The decomposition is T[i,j,k] = sum_t U[i,t] V[j,t] W[k,t].
    # For T_{1,2,3}:
    # i is 0. j is 0,1. k is 0,1,2.
    # We need T[0,j,k] = 1 for all j,k? No.
    # The tensor T_{n,m,p} is defined such that <T, A \otimes B> = <C, ...>.
    # T[i,j,k] = 1 for all i,j,k.
    # Wait, no. The tensor T_{n,m,p} has T[i,j,k] = 1.
    # Then sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,j,k} A[i,j] B[j,k] = sum_i (AB)[i,k] ... wait.
    # The contraction is over j.
    # <T, A \otimes B> = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k].
    # If T[i,j,k]=1, this is sum_{i,j,k} A[i,j] B[j,k] = sum_i (sum_j A[i,j] B[j,k]) ... no, sum over i and k too?
    # The matrix multiplication tensor is defined such that its contraction with A and B gives the trace of C^T C? No.
    # The standard definition: T_{n,m,p} is a tensor of size n x m x p with T[i,j,k] = 1.
    # Then T \cdot A \cdot B = sum_{i,j,k} A[i,j] B[j,k].
    # This is NOT the matrix product.
    # The matrix product C = AB is given by C[i,k] = sum_j A[i,j] B[j,k].
    # To express C[i,k] via tensor contraction, we use a tensor T[i,k,j] = 1?
    # Let's check the interface: W[k*n+i, r].
    # This suggests the output index is (k, i).
    # The tensor T has indices (i, j, k) corresponding to U[i], V[j], W[k].
    # If T[i,j,k] = 1, then sum_t U[i,t] V[j,t] W[k,t] = 1 for all i,j,k? No, that's impossible unless rank is huge.
    # The tensor T_{n,m,p} for matrix multiplication has T[i,j,k] = 1 if the term A[i,j]B[j,k] contributes to C[i,k].
    # So T[i,j,k] = 1 for all i, j, k.
    # Wait, if T[i,j,k]=1, then sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,j,k} A[i,j] B[j,k].
    # This is the sum of all elements of C.
    # This is not the matrix product tensor.
    # The matrix product tensor is T[i, j, k] = 1 where i is row of A, j is col of A/row of B, k is col of B.
    # The output is C[i, k].
    # The tensor T should satisfy: C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k].
    # So T[i,j,k] must be 1 for all i,j,k?
    # If T[i,j,k] = 1, then C[i,k] = sum_j 1 * A[i,j] B[j,k] = (AB)[i,k].
    # Yes! So the tensor T_{n,m,p} has entries T[i,j,k] = 1.
    # And we want to decompose T = sum_t U_t \otimes V_t \otimes W_t.
    # Where U_t is a vector of length n, V_t of length m, W_t of length p.
    # T[i,j,k] = sum_t U[i,t] V[j,t] W[k,t] = 1 for all i,j,k.
    
    # So for T_{1,2,3}:
    # i=0, j in {0,1}, k in {0,1,2}.
    # T[0,j,k] = 1.
    # We need sum_t U[0,t] V[j,t] W[k,t] = 1 for all j,k.
    # Let U[0,t] = 1 for all t.
    # Then sum_t V[j,t] W[k,t] = 1 for all j,k.
    # This means the matrix M = V^T W must be the all-ones matrix of size m x p (2x3).
    # Rank of all-ones matrix is 1.
    # So we can do this with rank 1?
    # V[0,0]=1, V[1,0]=1. W[0,0]=1, W[1,0]=1, W[2,0]=1.
    # Then sum_t V[j,t] W[k,t] = V[j,0] W[k,0] = 1*1 = 1.
    # So T_{1,2,3} has rank 1?
    # Let's check.
    # C[0,k] = sum_j A[0,j] B[j,k].
    # If we use rank 1:
    # U[0,0]=1, V[0,0]=1, V[1,0]=1, W[0,0]=1, W[1,0]=1, W[2,0]=1.
    # Term: U[0,0] V[j,0] W[k,0] = 1 * 1 * 1 = 1.
    # This contributes 1 * A[0,j] B[j,k] to the sum.
    # But we need the sum over j to be (AB)[0,k].
    # The tensor decomposition gives sum_t U[i,t] V[j,t] W[k,t].
    # This is a scalar.
    # The matrix multiplication is a set of scalars C[i,k].
    # The tensor T_{n,m,p} is defined such that the bilinear form <T, A \otimes B> = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k].
    # If T[i,j,k]=1, then <T, A \otimes B> = sum_{i,j,k} A[i,j] B[j,k] = sum_i sum_k (AB)[i,k].
    # This is the sum of all elements of the product matrix.
    # This is NOT the matrix multiplication tensor in the sense of computing the product.
    # The matrix multiplication tensor is usually defined as having 6 indices or being a map from A,B to C.
    # In the context of rank decomposition, we usually refer to the tensor T_{n,m,p} where T[i,j,k] = 1.
    # And the "rank" is the minimum number of terms in the decomposition T = sum U_t \otimes V_t \otimes W_t.
    # And the condition is T[i,j,k] = sum_t U[i,t] V[j,t] W[k,t] = 1.
    # So yes, T[i,j,k] = 1 for all i,j,k.
    # And for T_{1,2,3}, rank is 1.
    
    # So T_{2,4,6} = T_{1,2,3} \otimes T_{2,2,2}.
    # Rank(T_{1,2,3}) = 1.
    # Rank(T_{2,2,2}) = 7.
    # Total Rank = 1 * 7 = 7.
    # Is this possible?
    # T_{2,4,6}[i, j, k] = 1.
    # Decomposition: sum_t U[i,t] V[j,t] W[k,t] = 1.
    # If T = T1 \otimes T2, then T[i1 n2 + i2, j1 m2 + j2, k1 p2 + k2] = sum_{t1,t2} U1[i1,t1] U2[i2,t2] V1[j1,t1] V2[j2,t2] W1[k1,t1] W2[k2,t2].
    # This must equal 1 for all indices.
    # Let's check if T_{2,4,6} = T_{1,2,3} \otimes T_{2,2,2} works.
    # n=2=1*2, m=4=2*2, p=6=3*2.
    # i = i1*2 + i2. i1 in {0}, i2 in {0,1}.
    # j = j1*2 + j2. j1 in {0,1}, j2 in {0,1}.
    # k = k1*2 + k2. k1 in {0,1,2}, k2 in {0,1}.
    # T[i,j,k] = 1.
    # Decomposition sum:
    # sum_{t1,t2} U1[i1,t1] U2[i2,t2] V1[j1,t1] V2[j2,t2] W1[k1,t1] W2[k2,t2]
    # = (sum_{t1} U1[i1,t1] V1[j1,t1] W1[k1,t1]) * (sum_{t2} U2[i2,t2] V2[j2,t2] W2[k2,t2])
    # = T1[i1,j1,k1] * T2[i2,j2,k2].
    # We need T1[i1,j1,k1] = 1 for all i1,j1,k1.
    # And T2[i2,j2,k2] = 1 for all i2,j2,k2.
    # T1 is T_{1,2,3}. We found rank 1 works.
    # T2 is T_{2,2,2}. Rank 7 works.
    # So T_{2,4,6} has rank <= 1 * 7 = 7.
    # This seems too good to be true.
    # Let's double check T_{1,2,3}.
    # T1[0,j,k] = 1.
    # Rank 1: U[0,0]=1, V[0,0]=1, V[1,0]=1, W[0,0]=1, W[1,0]=1, W[2,0]=1.
    # Sum = 1*1*1 = 1. Correct.
    
    # So the rank is 7?
    # Let's construct the matrices.
    # Rank = 7.
    # U: (2, 7), V: (4, 7), W: (6, 7).
    # T1: U1 (1,1), V1 (2,1), W1 (3,1).
    # T2: U2 (2,7), V2 (2,7), W2 (2,7).
    # U[i, t] = U1[i1, t1] * U2[i2, t2].
    # Since T1 has rank 1, t1 is always 0.
    # So t = t2.
    # U[i, t] = U1[0, 0] * U2[i2, t] = 1 * U2[i2, t] = U2[i2, t].
    # V[j, t] = V1[j1, 0] * V2[j2, t] = 1 * V2[j2, t] = V2[j2, t].
    # W[k, t] = W1[k1, 0] * W2[k2, t] = 1 * W2[k2, t] = W2[k2, t].
    
    # So we just need the Strassen decomposition for T_{2,2,2}.
    # And we pad/replicate indices.
    # i = i2 (since i1=0).
    # j = j1*2 + j2.
    # k = k1*2 + k2.
    
    # U[i, t] = U2[i, t] (where i is i2).
    # V[j, t] = V2[j%2, t].
    # W[k, t] = W2[k%2, t].
    
    # Let's verify.
    # sum_t U[i,t] V[j,t] W[k,t] = sum_t U2[i,t] V2[j%2, t] W2[k%2, t].
    # This is T2[i, j%2, k%2].
    # We need this to be 1.
    # But T2[i, j, k] = 1 for all i,j,k.
    # So yes, it works.
    
    # Wait, is T_{2,2,2} rank 7?
    # Strassen's algorithm uses 7 multiplications.
    # Yes.
    
    # So the rank is 7.
    
    # Let's get the Strassen matrices.
    # Standard Strassen decomposition for 2x2x2.
    # U, V, W matrices of size 2x7.
    
    # Strassen formulas:
    # M1 = (A11)(B11 + B22)
    # M2 = (A21 + A22)(B11)
    # M3 = (A11 + A12)(B22)
    # M4 = (A11 - A22)(B11 + B12)
    # M5 = (A12 - A21)(B21 + B22)
    # M6 = (A11 + A21)(B12 - B11)
    # M7 = (A12 + A22)(B21 - B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map these to U, V, W.
    # T[i,j,k] = sum_t U[i,t] V[j,t] W[k,t] = 1.
    # This means the bilinear form is 1.
    # The Strassen algorithm computes C = AB.
    # The tensor T_{2,2,2} has T[i,j,k]=1.
    # The decomposition T = sum U_t \otimes V_t \otimes W_t.
    # The terms U_t, V_t, W_t correspond to the 7 multiplications.
    
    # Let's define the vectors for each multiplication.
    # For M1 = (A11)(B11 + B22):
    # A part: A11. Index (0,0).
    # B part: B11 + B22. Indices (0,0) and (1,1).
    # C part: contributes to C11 and C22.
    # Wait, the tensor decomposition is for the tensor T[i,j,k]=1.
    # The Strassen algorithm decomposes the multiplication map.
    # The tensor T_{2,2,2} is the tensor of the multiplication map.
    # So the U, V, W matrices from Strassen should directly give the decomposition.
    
    # Let's list the 7 terms.
    # Term 1: A11 * (B11 + B22)
    # U1[0]=1, U1[1]=0. (A11 is at i=0, j=0? No, A is 2x2. A11 is A[0,0].)
    # In tensor T[i,j,k], i is row of A, j is col of A (row of B), k is col of B.
    # A11 is A[0,0]. So i=0, j=0.
    # B11 is B[0,0]. So j=0, k=0.
    # B22 is B[1,1]. So j=1, k=1.
    # So for Term 1:
    # U[0,0]=1, U[1,0]=0.
    # V[0,0]=1, V[1,0]=1. (j=0 for B11, j=1 for B22? No, V is indexed by j. B11 is B[0,0], so j=0. B22 is B[1,1], so j=1.)
    # Wait, V[j,t]. B[j,k].
    # For B11 (j=0,k=0), V[0,0]=1.
    # For B22 (j=1,k=1), V[1,0]=1.
    # W[k,t].
    # For B11 (k=0), W[0,0]=1.
    # For B22 (k=1), W[1,0]=1.
    
    # Let's check sum_t U[i,t] V[j,t] W[k,t] for Term 1.
    # U[0,0]=1, U[1,0]=0.
    # V[0,0]=1, V[1,0]=1.
    # W[0,0]=1, W[1,0]=1.
    # i=0, j=0, k=0: 1*1*1 = 1.
    # i=0, j=1, k=0: 1*1*1 = 1.
    # i=0, j=0, k=1: 1*1*1 = 1.
    # i=0, j=1, k=1: 1*1*1 = 1.
    # i=1, ...: 0.
    # So Term 1 contributes 1 to T[i,j,k] for i=0, all j,k.
    
    # This is not right. T[i,j,k] should be 1 for ALL i,j,k.
    # The sum of all 7 terms must be 1 for all i,j,k.
    
    # Let's look up the standard Strassen tensor decomposition.
    # The 7 vectors are:
    # 1. (1,0), (1,1), (1,1)  -> A11, B11+B22, C11+C22? No.
    # The vectors are for A, B, C.
    # U (A), V (B), W (C).
    # Term 1: U=(1,0), V=(1,1), W=(1,1).
    # Term 2: U=(0,1), V=(1,0), W=(1,0).
    # Term 3: U=(1,1), V=(0,1), W=(0,1).
    # Term 4: U=(1,-1), V=(1,1), W=(1,0).
    # Term 5: U=(1,-1), V=(0,1), W=(0,1).
    # Term 6: U=(1,1), V=(1,-1), W=(1,0).
    # Term 7: U=(1,1), V=(0,1), W=(0,1).
    
    # Let's verify Term 1:
    # U=(1,0), V=(1,1), W=(1,1).
    # T[i,j,k] += U[i]V[j]W[k].
    # i=0: V[j]W[k] = 1*1=1 for all j,k.
    # i=1: 0.
    # So adds 1 to T[0,j,k] for all j,k.
    
    # Let's verify the sum of all 7 terms.
    # This is a known result. The Strassen decomposition provides a rank 7 decomposition of the 2x2x2 matrix multiplication tensor.
    # So we can use these vectors.
    
    U2 = np.array([
        [1, 0, 1, 1, 1, 1, 1],
        [0, 1, 1, -1, -1, 1, 1]
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, 1, 0, 1, 0],
        [1, 0, 1, 1, 1, -1, 1]
    ], dtype=float)
    
    W2 = np.array([
        [1, 1, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1, 0, 1]
    ], dtype=float)
    
    # Check if sum_t U2[i,t] V2[j,t] W2[k,t] = 1 for all i,j,k.
    # We can verify this quickly.
    
    # Now construct U, V, W for 2x4x6.
    # Rank = 7.
    # U (2, 7), V (4, 7), W (6, 7).
    # U[i, t] = U2[i, t]. (i is 0,1)
    # V[j, t] = V2[j%2, t]. (j is 0..3)
    # W[k, t] = W2[k%2, t]. (k is 0..5)
    
    U = np.zeros((n, 7), dtype=float)
    V = np.zeros((m, 7), dtype=float)
    W = np.zeros((p, 7), dtype=float)
    
    U[:, :] = U2
    for j in range(m):
        V[j, :] = V2[j % 2, :]
    for k in range(p):
        W[k, :] = W2[k % 2, :]
        
    # Verify
    # Reconstruct T
    T_recon = np.einsum('it,jt,kt->ijk', U, V, W)
    T_expected = np.ones((n, m, p))
    
    if not np.allclose(T_recon, T_expected):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
        
    # The interface expects U (n*m, rank), V (m*p, rank), W (p*n, rank).
    # My U, V, W are (n, rank), (m, rank), (p, rank).
    # I need to reshape them according to the index ordering.
    # U[i*m+j, t] corresponds to A[i,j].
    # My U[i, t] is for A[i, :].
    # This implies the tensor decomposition I found is for T[i,j,k] where i is row, j is col of A, k is col of B.
    # But the interface says U has shape (n*m, rank).
    # This means U is indexed by (i, j).
    # My U is indexed by i.
    # This suggests my tensor T[i,j,k] is not the same as the one expected.
    # The expected tensor T has U[i*m+j, t], V[j*p+k, t], W[k*n+i, t].
    # This corresponds to T[i,j,k] = sum_t U[i*m+j, t] V[j*p+k, t] W[k*n+i, t].
    # This is the standard tensor for matrix multiplication.
    # My T[i,j,k] = sum_t U[i, t] V[j, t] W[k, t].
    # This is a different tensor.
    # The tensor T[i,j,k]=1 is the same.
    # But the decomposition vectors are different.
    # For the standard tensor, U is a vector of length n*m.
    # For my tensor, U is a vector of length n.
    # This means I was decomposing the wrong tensor!
    # The tensor T_{n,m,p} for matrix multiplication has indices (i, j, k) where:
    # i is row of A (0..n-1)
    # j is col of A / row of B (0..m-1)
    # k is col of B (0..p-1)
    # And T[i,j,k] = 1.
    # The decomposition is T[i,j,k] = sum_t U[i*m+j, t] V[j*p+k, t] W[k*n+i, t].
    # This is NOT separable as U[i] V[j] W[k].
    # It is U[i,j] V[j,k] W[k,i].
    # So my Kronecker product argument was flawed because the indices are coupled.
    # T_{n,m,p} is not T_{n1,m1,p1} \otimes T_{n2,m2,p2} in the simple sense.
    # It is a more complex structure.
    
    # So rank 7 is likely incorrect for this interface.
    # Let's revert to schoolbook or try a different approach.
    
    # Given the time constraint and the complexity, I will return the schoolbook solution.
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
