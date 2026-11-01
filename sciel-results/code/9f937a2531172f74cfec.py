"""Program for matrix multiplication tensor decomposition: 2x2 by 2x16."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 16
    
    # We use a tensor product decomposition:
    # The multiplication tensor M_{n,m,p} can be viewed as a product structure
    # when p is even: group the p output columns into pairs.
    # 
    # For each pair of output columns (2t, 2t+1), we have a sub-tensor that
    # multiplies A (n x m) by a 2-column block B_t (m x 2).
    # This sub-tensor has the structure of a 2x2x2 matrix multiplication
    # (with n=2, m=2, and the "2" being the pair of columns).
    #
    # We use Strassen's 7-term decomposition for each 2x2x2 block.
    # Total rank = 7 * (p/2) = 7 * 8 = 56.
    
    num_blocks = p // 2  # 8 blocks
    rank = 7 * num_blocks  # 56
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 32 x 56
    
    # Strassen's 7 products for 2x2 matrix multiplication:
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = A*B where C_{ik} = sum_j A_{ij} B_{jk}
    #
    # Strassen:
    # M1 = (a11 + a22)(b11 + b22)
    # M2 = (a21 + a22)b11
    # M3 = a11(b12 - b22)
    # M4 = a22(b21 - b11)
    # M5 = (a11 + a12)b22
    # M6 = (a21 - a11)(b11 + b12)
    # M7 = (a12 - a22)(b21 + b22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # The 7 Strassen factors (each factor r=0..6):
    # For factor r, we need U_r, V_r, W_r such that:
    # (sum_r U_r V_r^T) * (something) ... 
    # Actually, the tensor decomposition means:
    # T_{i,j,k} = sum_r U_{i,r} * V_{j,r} * W_{k,r}
    # where T is the matrix multiplication tensor.
    
    # For the 2x2x2 block, indices are:
    # i in {0,1} (row of A, row of C)
    # j in {0,1} (col of A, row of B)
    # k in {0,1} (col of B, col of C)
    
    # Strassen factors for the 2x2x2 tensor:
    # Each factor r has components (u_i, v_j, w_k)
    # The product u_i * v_j * w_k summed over r should equal T_{i,j,k} = delta_{j,k} (for matrix mult, T_{i,j,k} = delta_{j,k}... wait)
    
    # Actually for matrix multiplication A(n x m) * B(m x p) -> C(n x p):
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor is T_{i,j,k} = delta_{j,j'} ... no.
    # The tensor for matrix mult is: T[i,j,k] = 1 if the computation of C[i,k] involves A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all valid i,j,k (it's just the structure).
    # Wait, no. The tensor is defined such that:
    # C_{i,k} = sum_{j} A_{i,j} B_{j,k}
    # This is a linear map. The tensor decomposition gives:
    # For each term r: C += U_r * V_r^T * W_r (outer products)
    
    # Let me think more carefully. The matrix multiplication tensor M has entries:
    # M[i,j,k] = 1 if i is row of A/C, j is col of A / row of B, k is col of B/C
    # The bilinear map is: (A, B) -> C where C_{i,k} = sum_j A_{i,j} B_{j,k}
    # 
    # Tensor decomposition of rank R means:
    # For each r in 0..R-1:
    #   U[:, r] is a vector in R^n (rows of A)
    #   V[:, r] is a vector in R^m (cols of A = rows of B)  
    #   W[:, r] is a vector in R^p (cols of B = cols of C)
    # And: C_{i,k} = sum_r U[i,r] * (sum_j A[i',j] ... 
    
    # OK let me just use the standard formulation. The tensor M_{n,m,p} for matrix mult:
    # M = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k (where the sum is over all i,j,k with appropriate ranges)
    # No wait, that's not right either.
    
    # The correct tensor: for matrix multiplication, the structure tensor is:
    # T = sum_{j} (sum_i e_i ⊗ e_j ⊗ e_j) ... 
    # 
    # Let me use a different approach. I'll directly construct the factors.
    # 
    # For the full problem, C = A * B where A is 2x2, B is 2x16, C is 2x16.
    # C_{i,k} = sum_{j=0}^{1} A_{i,j} * B_{j,k}
    # 
    # We want to write this as: C = sum_r u_r * v_r^T * w_r (outer product structure)
    # where u_r in R^2, v_r in R^2, w_r in R^16
    # 
    # Actually the factor matrices are:
    # U: (n*m) x rank = 4 x 56  -> U[i*m+j, r] = u_r[i] if we split
    # V: (m*p) x rank = 32 x 56 -> V[j*p+k, r] = v_r[j]*w_r[k]... 
    
    # Hmm, I'm getting confused. Let me look at the schoolbook again.
    # In the schoolbook:
    # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # So for each (i,j,k): U has a 1 at row i*m+j, V has a 1 at row j*p+k, W has a 1 at row k*n+i
    # And the contribution to C is: for each r, C += (U[:,r] reshaped to n x m) * (V[:,r] reshaped to m x p)
    # Wait no, it's: C_{i,k} += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] ... 
    
    # Actually I think the convention is:
    # C_{i,k} = sum_r (sum_j U[i*m+j, r] * V[j*p+k, r]) * ... 
    # No, I think it's a 3-tensor contraction.
    
    # Let me just trust the schoolbook structure and adapt it for Strassen.
    
    # For Strassen on 2x2x2 blocks, the 7 factors are:
    # Factor 0 (M1): 
    #   U components: for A, u = [1, 0, 0, 1] (a11 + a22) -> rows (i*m+j): (0*2+0, 1*2+1) = (0, 3)
    #   V components: for B, v = [1, 1, 1, 1] (b11+b22) -> rows (j*p+k): but here p=2 for the block
    #   W components: for C, w = [1, 0, 0, 1] (c11+c22) -> rows (k*n+i): (0*2+0, 1*2+1) = (0, 3)
    # 
    # This is getting complicated with the indexing. Let me just implement it directly.
    
    # For the full 2x2x16 problem, let's group B into 8 pairs of columns.
    # For block t (columns 2t and 2t+1 of B), we have a local 2x2x2 problem.
    
    # The 7 Strassen factors for a 2x2x2 block:
    # Each factor r has:
    #   u_r: 2-vector (for A rows)
    #   v_r: 2-vector (for B rows / A cols)  
    #   w_r: 2-vector (for C cols)
    # 
    # But in our global indexing:
    #   U row index: i*m + j = i*2 + j, where i in {0,1} (A row), j in {0,1} (A col)
    #   V row index: j*p + k = j*16 + k, where j in {0,1} (B row), k in {0,...,15} (B col)
    #   W row index: k*n + i = k*2 + i, where k in {0,...,15} (C col), i in {0,1} (C row)
    
    # For block t, the local B columns are 2t and 2t+1.
    # Local C columns are also 2t and 2t+1.
    
    # Strassen factors (u, v, w) where each is a 2-vector:
    # M1: u=[1,1], v=[1,1], w=[1,1]
    # M2: u=[0,1], v=[1,0], w=[0,1]  -- wait, let me get these right
    
    # Standard Strassen for C = AB:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # As tensor factors (each factor is a rank-1 tensor u ⊗ v ⊗ w):
    # M1: u = [1, 1], v = [1, 1], w = [1, 1]
    #   Contribution to C11: u[0]*v[0]*w[0] = 1, to C12: u[0]*v[0]*w[1] = 1, etc.
    #   Wait, this isn't quite right. The factor contributes u[i]*v[j]*w[k] to T[i,j,k]
    #   and we need sum_r T_r[i,j,k] = delta_{j,k} (the matrix mult tensor)
    
    # Actually for matrix multiplication, the tensor is:
    # T[i,j,k] = delta_{j,k} is NOT right.
    # 
    # The matrix multiplication map is: (A, B) -> C where C = AB
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # 
    # As a trilinear form: for fixed A and B, C is determined.
    # The "tensor" here is the structure of the multiplication.
    # 
    # A rank-1 term in the decomposition is: for each r, we have vectors
    # u_r (length n), v_r (length m), w_r (length p) such that:
    # C_{i,k} = sum_r u_r[i] * (sum_j A_{i',j} ...) 
    # 
    # I think the correct interpretation is:
    # The bilinear map (A, B) -> C can be written as:
    # C = sum_r (u_r A) (B v_r) w_r^T ... no this is getting too complex.
    
    # Let me just use the schoolbook pattern and verify it works, then try to find Strassen-like savings.
    
    # Schoolbook: for each (i,j,k), one factor with U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
    # This gives C_{i,k} = sum_j A_{i,j} B_{j,k} correctly.
    
    # Now, can we combine some of these? The 2x2x2 blocks might allow Strassen.
    
    # For block t, the relevant indices are:
    # A: all of A (2x2)
    # B: columns 2t, 2t+1
    # C: columns 2t, 2t+1
    # 
    # The 8 schoolbook factors for this block are (i,j,k_local) where k_local in {0,1}:
    # (0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)
    
    # Strassen says we can do it in 7 instead of 8.
    
    # The 7 Strassen factors for the local 2x2x2:
    # In local indices (i, j, k) each in {0,1}:
    
    # M1: u=[1,1], v=[1,1], w=[1,1]
    # M2: u=[0,1], v=[1,0], w=[0,1]
    # M3: u=[1,0], v=[0,1], w=[1,0]  -- wait, need to be careful
    
    # Let me look up the exact Strassen tensor decomposition.
    # The 7 factors (u, v, w) for 2x2x2 matrix mult:
    
    factors_222 = [
        ([1, 1], [1, 1], [1, 1]),  # M1
        ([0, 1], [1, 0], [0, 1]),  # M2 -- hmm, not sure
        ([1, 0], [0, 1], [1, 0]),  # M3
        ([0, 1], [1, -1], [1, 0]), # M4
        ([1, 1], [0, 1], [0, 1]),  # M5
        ([1, -1], [1, 1], [0, 1]), # M6
        ([1, -1], [0, 1], [1, 0]), # M7
    ]
    
    # I'm not confident about the exact factors. Let me just use the schoolbook approach
    # since I'm not certain about the Strassen tensor factors for this specific indexing convention.
    
    # Given the uncertainty, I'll implement the schoolbook decomposition which is guaranteed correct.
    rank = n * m * p  # 64
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
# EVOLVE-BLOCK-END
