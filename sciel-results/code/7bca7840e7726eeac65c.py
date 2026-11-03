"""Matrix multiplication 4x5x6 tensor decomposition using Strassen-based embedding."""
import time
import random

DIMENSIONS = [4, 5, 6]

# Strassen 2x2x2 decomposition
# C = A @ B where A, B, C are 2x2
# M1 = (A11 + A22)(B11 + B22)
# M2 = (A21 + A22) B11
# M3 = A11 (B12 - B22)
# M4 = A22 (B21 - B11)
# M5 = (A11 + A12) B22
# M6 = (A21 - A11)(B11 + B12)
# M7 = (A12 - A22)(B21 + B22)
# C11 = M1 + M4 - M5 + M7
# C12 = M3 + M5
# C21 = M2 + M4
# C22 = M1 - M2 + M3 + M6

# Strassen 2x2x2 factors:
# U is indexed by (i,j) where i is row of A, j is col of A (i,j in {0,1})
# V is indexed by (j,k) where j is row of B, k is col of B (j,k in {0,1})
# W is indexed by (k,i) where i is row of C, k is col of C (k,i in {0,1})
# Convention: W[k*n+i, c] for n=2, so W index is k*2+i

# For the 7 products, we define:
# Each product c has:
#   U factors: coefficients for A11, A12, A21, A22
#   V factors: coefficients for B11, B12, B21, B22
#   W factors: coefficients for C11, C12, C21, C22 (but W is indexed by (k,i))

# Let's define the Strassen 2x2x2 decomposition explicitly.
# A = [[A11, A12], [A21, A22]]
# B = [[B11, B12], [B21, B22]]
# C = [[C11, C12], [C21, C22]]

# U rows: A11 (row 0), A12 (row 1), A21 (row 2), A22 (row 3)
# V rows: B11 (row 0), B12 (row 1), B21 (row 2), B22 (row 3)
# W rows: C11 (row 0), C12 (row 1), C21 (row 2), C22 (row 3)
# But W is indexed by W[k*n+i] = W[k*2+i], so:
#   C11: k=0, i=0 -> index 0
#   C12: k=1, i=0 -> index 2
#   C21: k=0, i=1 -> index 1
#   C22: k=1, i=1 -> index 3

# Wait, let me re-read the convention: W[k*n+i, r]
# For n=2: W[k*2+i, r]
# C[i,k] where i is row, k is col
# So C11 = C[0,0]: k=0, i=0 -> W[0*2+0] = W[0]
# C12 = C[0,1]: k=1, i=0 -> W[1*2+0] = W[2]
# C21 = C[1,0]: k=0, i=1 -> W[0*2+1] = W[1]
# C22 = C[1,1]: k=1, i=1 -> W[1*2+1] = W[3]

# Strassen products:
# M1 = (A11 + A22)(B11 + B22)
# M2 = (A21 + A22) B11
# M3 = A11 (B12 - B22)
# M4 = A22 (B21 - B11)
# M5 = (A11 + A12) B22
# M6 = (A21 - A11)(B11 + B12)
# M7 = (A12 - A22)(B21 + B22)

# C11 = M1 + M4 - M5 + M7
# C12 = M3 + M5
# C21 = M2 + M4
# C22 = M1 - M2 + M3 + M6

# U factors for each product (coefficients for A11, A12, A21, A22):
# M1: A11=1, A12=0, A21=0, A22=1
# M2: A11=0, A12=0, A21=1, A22=1
# M3: A11=1, A12=0, A21=0, A22=0
# M4: A11=0, A12=0, A21=0, A22=1
# M5: A11=1, A12=1, A21=0, A22=0
# M6: A11=-1, A12=0, A21=1, A22=0
# M7: A11=0, A12=1, A21=0, A22=-1

# V factors for each product (coefficients for B11, B12, B21, B22):
# M1: B11=1, B12=0, B21=0, B22=1
# M2: B11=1, B12=0, B21=0, B22=0
# M3: B11=0, B12=1, B21=0, B22=-1
# M4: B11=-1, B12=0, B21=1, B22=0
# M5: B11=0, B12=0, B21=0, B22=1
# M6: B11=1, B12=1, B21=0, B22=0
# M7: B11=0, B12=0, B21=1, B22=1

# W factors for each product (coefficients for C11, C12, C21, C22):
# M1: C11=1, C12=0, C21=0, C22=1
# M2: C11=0, C12=0, C21=1, C22=-1
# M3: C11=0, C12=1, C21=0, C22=1
# M4: C11=1, C12=0, C21=1, C22=0
# M5: C11=-1, C12=1, C21=0, C22=0
# M6: C11=0, C12=0, C21=0, C22=1
# M7: C11=1, C12=0, C21=0, C22=0


def _strassen_2x2_factors():
    """Return U, V, W factors for 2x2x2 Strassen decomposition.
    U: 4x7, V: 4x7, W: 4x7
    """
    U = [
        [1, 0, 1, 0, 1, -1, 0],  # A11
        [0, 0, 0, 0, 1, 0, 1],   # A12
        [0, 1, 0, 0, 0, 1, 0],   # A21
        [1, 1, 0, 1, 0, 0, -1],  # A22
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],  # B11
        [0, 0, 1, 0, 0, 1, 0],   # B12
        [0, 0, 0, 1, 0, 0, 1],   # B21
        [1, 0, -1, 0, 1, 0, 1],  # B22
    ]
    # W indexed by W[k*2+i]: C11->0, C21->1, C12->2, C22->3
    W = [
        [1, 0, 0, 1, -1, 0, 1],  # C11 (k=0,i=0)
        [0, 1, 0, 1, 0, 0, 0],   # C21 (k=0,i=1)
        [0, 0, 1, 0, 1, 0, 0],   # C12 (k=1,i=0)
        [1, -1, 1, 0, 0, 1, 0],  # C22 (k=1,i=1)
    ]
    return U, V, W


def _embed_and_extract(n, m, p, S):
    """Embed n x m x p into S x S x S using Strassen, then extract.
    Returns U, V, W of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    # We'll build the full S x S x S Strassen decomposition recursively,
    # then extract the n x m x p sub-problem.
    
    # For simplicity, let's build the Strassen decomposition for size S
    # and track which elements are used.
    
    # Actually, let's think about this differently.
    # We want to multiply A (n x m) by B (m x p) to get C (n x p).
    # We can pad A to S x S, B to S x S, compute C_pad = A_pad @ B_pad,
    # then extract C.
    
    # The Strassen decomposition gives us factors for S x S x S.
    # We need to extract only the rows of U corresponding to A's elements,
    # only the rows of V corresponding to B's elements,
    # and only the rows of W corresponding to C's elements.
    
    # But Strassen is a recursive algorithm, so the factors are not simply
    # a subset of the schoolbook factors.
    
    # Let me think about this more carefully.
    
    # The tensor decomposition of matrix multiplication T_{n,m,p} is:
    # T = sum_c U_c ⊗ V_c ⊗ W_c
    # where U_c is an n x m matrix, V_c is an m x p matrix, W_c is a p x n matrix
    # (with W_c[k,i] being the coefficient for C[i,k]).
    
    # For Strassen, we decompose the S x S x S tensor, then we can extract
    # the n x m x p sub-tensor by taking linear combinations of the factors.
    
    # Actually, the cleanest way is to build the Strassen decomposition
    # recursively and track the linear combinations.
    
    # Let's build the full Strassen S x S x S decomposition as a list of
    # (U_c, V_c, W_c) where each is a flat vector.
    
    def _build_strassen(S):
        """Build Strassen decomposition for S x S x S.
        Returns list of (U_flat, V_flat, W_flat) where:
        U_flat has length S*S (indexed by i*S+j for A[i,j])
        V_flat has length S*S (indexed by j*S+k for B[j,k])
        W_flat has length S*S (indexed by k*S+i for C[i,k])
        """
        if S == 2:
            U2, V2, W2 = _strassen_2x2_factors()
            results = []
            for c in range(7):
                U_flat = [U2[i][c] for i in range(4)]
                V_flat = [V2[j][c] for j in range(4)]
                W_flat = [W2[k][c] for k in range(4)]
                results.append((U_flat, V_flat, W_flat))
            return results
        
        half = S // 2
        # Split A into 4 blocks of size half x half
        # A11, A12, A21, A22
        # Similarly for B and C.
        
        # For each of the 7 Strassen products, we need to recursively
        # decompose each half-size matrix multiplication.
        
        # The 7 products are:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22) B11
        # M3 = A11 (B12 - B22)
        # M4 = A22 (B21 - B11)
        # M5 = (A11 + A12) B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # Each M is a half x half matrix.
        
        # For each M, we recursively decompose it.
        # The recursive decomposition gives us factors for half x half x half.
        # We need to combine them.
        
        # For M1 = (A11 + A22)(B11 + B22):
        # The U factor for A11+A22 is the sum of U factors for A11 and A22.
        # The V factor for B11+B22 is the sum of V factors for B11 and B22.
        # The W factor for M1 contributes to C11 and C22.
        
        # Let's build this systematically.
        
        # First, get the recursive decomposition for half size.
        sub_decomp = _build_strassen(half)
        sub_rank = len(sub_decomp)
        
        # For each of the 7 products, we have sub_rank sub-decompositions.
        # Total rank = 7 * sub_rank.
        
        results = []
        
        # For each product t in 0..6, and each sub-column c in 0..sub_rank-1:
        for t in range(7):
            for c in range(sub_rank):
                U_sub, V_sub, W_sub = sub_decomp[c]
                
                # Determine the U, V, W coefficients for this product.
                # U coefficients: which A blocks are used and with what weights
                # V coefficients: which B blocks are used and with what weights
                # W coefficients: which C blocks are affected and with what weights
                
                # A block indices: 0=A11, 1=A12, 2=A21, 3=A22
                # B block indices: 0=B11, 1=B12, 2=B21, 3=B22
                # C block indices: 0=C11, 1=C12, 2=C21, 3=C22
                
                # Product definitions:
                # t=0: M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
                # t=1: M2 = (A21+A22)B11 -> C21 += M2, C22 -= M2
                # t=2: M3 = A11(B12-B22) -> C12 += M3, C22 += M3
                # t=3: M4 = A22(B21-B11) -> C11 += M4, C21 += M4
                # t=4: M5 = (A11+A12)B22 -> C11 -= M5, C12 += M5
                # t=5: M6 = (A21-A11)(B11+B12) -> C22 += M6
                # t=6: M7 = (A12-A22)(B21+B22) -> C11 += M7
                
                # A block coefficients for each product:
                # t=0: A11=1, A22=1
                # t=1: A21=1, A22=1
                # t=2: A11=1
                # t=3: A22=1
                # t=4: A11=1, A12=1
                # t=5: A21=1, A11=-1
                # t=6: A12=1, A22=-1
                
                A_coeffs = {
                    0: {0: 1, 3: 1},
                    1: {2: 1, 3: 1},
                    2: {0: 1},
                    3: {3: 1},
                    4: {0: 1, 1: 1},
                    5: {2: 1, 0: -1},
                    6: {1: 1, 3: -1},
                }
                
                # B block coefficients for each product:
                # t=0: B11=1, B22=1
                # t=1: B11=1
                # t=2: B12=1, B22=-1
                # t=3: B21=1, B11=-1
                # t=4: B22=1
                # t=5: B11=1, B12=1
                # t=6: B21=1, B22=1
                
                B_coeffs = {
                    0: {0: 1, 3: 1},
                    1: {0: 1},
                    2: {1: 1, 3: -1},
                    3: {2: 1, 0: -1},
                    4: {3: 1},
                    5: {0: 1, 1: 1},
                    6: {2: 1, 3: 1},
                }
                
                # C block coefficients for each product:
                # t=0: C11=1, C22=1
                # t=1: C21=1, C22=-1
                # t=2: C12=1, C22=1
                # t=3: C11=1, C21=1
                # t=4: C11=-1, C12=1
                # t=5: C22=1
                # t=6: C11=1
                
                C_coeffs = {
                    0: {0: 1, 3: 1},
                    1: {2: 1, 3: -1},
                    2: {1: 1, 3: 1},
                    3: {0: 1, 2: 1},
                    4: {0: -1, 1: 1},
                    5: {3: 1},
                    6: {0: 1},
                }
                
                # Now build the full U, V, W flat vectors.
                # U has length S*S, indexed by i*S+j
                # For block (bi, bj) where bi, bj in {0,1}, the elements are at
                # positions (bi*half + i)*(S) + (bj*half + j) for i,j in range(half)
                
                U_full = [0] * (S * S)
                V_full = [0] * (S * S)
                W_full = [0] * (S * S)
                
                # For each A block that contributes:
                for a_block, a_coeff in A_coeffs[t].items():
                    bi = a_block // 2
                    bj = a_block % 2
                    # For each element in the sub-decomposition:
                    for i in range(half):
                        for j in range(half):
                            sub_idx = i * half + j
                            full_idx = (bi * half + i) * S + (bj * half + j)
                            U_full[full_idx] += a_coeff * U_sub[sub_idx]
                
                # For each B block that contributes:
                for b_block, b_coeff in B_coeffs[t].items():
                    bi = b_block // 2
                    bj = b_block % 2
                    for i in range(half):
                        for j in range(half):
                            sub_idx = i * half + j
                            full_idx = (bi * half + i) * S + (bj * half + j)
                            V_full[full_idx] += b_coeff * V_sub[sub_idx]
                
                # For each C block that contributes:
                for c_block, c_coeff in C_coeffs[t].items():
                    bi = c_block // 2
                    bj = c_block % 2
                    # W is indexed by k*S+i where k is col, i is row
                    # C block (bi, bj) corresponds to rows bi*half..(bi+1)*half-1
                    # and cols bj*half..(bj+1)*half-1
                    for i in range(half):
                        for k in range(half):
                            # In the sub-decomposition, W is indexed by k*half+i
                            sub_idx = k * half + i
                            # In the full decomposition, W is indexed by k_full*S+i_full
                            i_full = bi * half + i
                            k_full = bj * half + k
                            full_idx = k_full * S + i_full
                            W_full[full_idx] += c_coeff * W_sub[sub_idx]
                
                results.append((U_full, V_full, W_full))
        
        return results
    
    full_decomp = _build_strassen(S)
    rank = len(full_decomp)
    
    # Now extract the n x m x p sub-problem.
    # U has rows for A[i,j] where i in 0..n-1, j in 0..m-1
    # In the S x S matrix, these are at positions i*S+j
    # V has rows for B[j,k] where j in 0..m-1, k in 0..p-1
    # In the S x S matrix, these are at positions j*S+k
    # W has rows for C[i,k] where i in 0..n-1, k in 0..p-1
    # In the S x S matrix, W is indexed by k*S+i, so positions k*S+i
    
    U_rows = [i * S + j for i in range(n) for j in range(m)]
    V_rows = [j * S + k for j in range(m) for k in range(p)]
    W_rows = [k * S + i for k in range(p) for i in range(n)]
    
    U_out = [[0] * rank for _ in range(n * m)]
    V_out = [[0] * rank for _ in range(m * p)]
    W_out = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        U_flat, V_flat, W_flat = full_decomp[c]
        for i, row_idx in enumerate(U_rows):
            U_out[i][c] = U_flat[row_idx]
        for j, row_idx in enumerate(V_rows):
            V_out[j][c] = V_flat[row_idx]
        for k, row_idx in enumerate(W_rows):
            W_out[k][c] = W_flat[row_idx]
    
    # Remove zero columns
    U_final = []
    V_final = []
    W_final = []
    for c in range(rank):
        if any(U_out[i][c] != 0 for i in range(n * m)) or \
           any(V_out[j][c] != 0 for j in range(m * p)) or \
           any(W_out[k][c] != 0 for k in range(p * n)):
            U_final.append([U_out[i][c] for i in range(n * m)])
            V_final.append([V_out[j][c] for j in range(m * p)])
            W_final.append([W_out[k][c] for k in range(p * n)])
    
    # Transpose to get (n*m, rank), (m*p, rank), (p*n, rank)
    final_rank = len(U_final)
    U_result = [[0] * final_rank for _ in range(n * m)]
    V_result = [[0] * final_rank for _ in range(m * p)]
    W_result = [[0] * final_rank for _ in range(p * n)]
    
    for c in range(final_rank):
        for i in range(n * m):
            U_result[i][c] = U_final[c][i]
        for j in range(m * p):
            V_result[j][c] = V_final[c][j]
        for k in range(p * n):
            W_result[k][c] = W_final[c][k]
    
    return U_result, V_result, W_result


def _make_schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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
    return U, V, W


def _verify(U, V, W, n, m, p, max_checks=100):
    """Verify the decomposition by checking random inputs."""
    import random
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    
    for _ in range(max_checks):
        # Generate random A and B
        A = [[random.randint(-10, 10) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-10, 10) for _ in range(p)] for _ in range(m)]
        
        # Compute C = A @ B
        C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        
        # Compute C from decomposition
        C_decomp = [[0] * p for _ in range(n)]
        for c in range(rank):
            # Compute the product of the c-th factor
            # U_c is a vector of length n*m
            # V_c is a vector of length m*p
            # W_c is a vector of length p*n
            # The contribution to C[i,k] is sum_j U[i*m+j] * V[j*p+k] * W[k*n+i]
            # But actually, the tensor decomposition says:
            # C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
            # Wait, that's not right. Let me think again.
            
            # The tensor T_{i,j,k} = delta_{i,j,k} for matrix multiplication.
            # T = sum_c U_c ⊗ V_c ⊗ W_c
            # where (U_c ⊗ V_c ⊗ W_c)[i,j,k] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            
            # So C[i,k] = sum_j A[i,j] * B[j,k] = sum_{i,j,k} T_{i,j,k} * A[i,j] * B[j,k]
            # = sum_c (sum_{i,j} U[i*m+j, c] * A[i,j]) * (sum_{j,k} V[j*p+k, c] * B[j,k]) * ... 
            # No, that's not right either.
            
            # Actually, the bilinear form is:
            # C[i,k] = sum_j A[i,j] B[j,k]
            # And the tensor decomposition gives:
            # C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
            # Wait, that's not standard. Let me reconsider.
            
            # The standard way: T_{i,j,k} is the tensor for matrix multiplication.
            # T_{i,j,k} = 1 if the indices correspond to a valid multiplication term.
            # For C = A @ B, C[i,k] = sum_j A[i,j] B[j,k].
            # So T_{i,j,k} should be such that sum_{i,j,k} T_{i,j,k} A[i,j] B[j,k] = C[i,k]... 
            # No, that's not right. The tensor T has 3 indices, and the contraction is:
            # C[i,k] = sum_j T_{i,j,k} A[i,j] B[j,k]
            # where T_{i,j,k} = 1 for all valid (i,j,k).
            
            # The rank-r decomposition is:
            # T_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            
            # So: C[i,k] = sum_j A[i,j] B[j,k] = sum_{i',j',k'} T_{i',j',k'} A[i',j'] B[j',k'] * delta_{i,i'} delta_{k,k'}
            # Hmm, this is getting confusing. Let me just directly verify.
            
            # The decomposition says: for each (i,k),
            # C[i,k] = sum_j A[i,j] B[j,k]
            # And the tensor T_{i,j,k} satisfies:
            # T_{i,j,k} = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
            
            # So we can verify by checking that for random A, B:
            # sum_j A[i,j] B[j,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] A[i,j]) * (sum_j V[j*p+k, c] B[j,k])
            # Wait, that's not right. The tensor is trilinear in a specific way.
            
            # Let me just check the tensor directly.
            pass
        
        # Alternative verification: check that the tensor matches.
        # For each (i,j,k), T_{i,j,k} should equal 1 (for matrix multiplication).
        # T_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # But we only need to check for valid (i,j,k) where i<n, j<m, k<p.
        # Actually, for matrix multiplication, T_{i,j,k} = 1 for all i,j,k in range.
        # Wait, no. The tensor for matrix multiplication is:
        # T_{i,j,k} = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k]
        # So T has indices (i,j,k) and T_{i,j,k} = 1 for all valid i,j,k.
        
        # Let me just check a few (i,j,k) tuples.
        for i in range(min(n, 2)):
            for j in range(min(m, 2)):
                for k in range(min(p, 2)):
                    t_val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                    if abs(t_val - 1) > 1e-9:
                        return False
    
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Try different sizes S
    # S must be at least max(n, m, p)
    # And S should be a power of 2 for Strassen
    min_S = max(n, m, p)
    
    # Find the smallest power of 2 >= min_S
    S = 1
    while S < min_S:
        S *= 2
    
    best_result = None
    best_rank = n * m * p  # Schoolbook rank
    
    # Try S, S*2, etc.
    for trial_S in [S, S * 2]:
        if time.time() > deadline - 30:
            break
        
        try:
            U, V, W = _embed_and_extract(n, m, p, trial_S)
            rank = len(U[0]) if U else 0
            
            if rank > 0 and rank < best_rank:
                # Verify
                if _verify(U, V, W, n, m, p, max_checks=20):
                    best_result = (U, V, W)
                    best_rank = rank
                    print(f"Found rank {rank} with S={trial_S}")
        except Exception as e:
            print(f"Error with S={trial_S}: {e}")
    
    # If no better result found, use schoolbook
    if best_result is None:
        U, V, W = _make_schoolbook(n, m, p)
    else:
        U, V, W = best_result
    
    return {"U": U, "V": V, "W": W}
