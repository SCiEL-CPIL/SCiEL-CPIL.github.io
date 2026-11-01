"""
Rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x11 matrix.

Uses a hybrid decomposition:
- Split the 11 columns into 5 blocks of 2 and 1 block of 1.
- For each 2-column block, use Strassen's rank-7 decomposition for 2x2x2.
- For the 1-column block, use schoolbook rank-4 decomposition for 2x2x1.

Total rank = 5*7 + 4 = 39.
"""

DIMENSIONS = [2, 2, 11]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=11
    
    # Number of 2-column blocks and 1-column blocks
    # 11 = 5*2 + 1
    num_2blocks = 5
    num_1blocks = 1
    rank = num_2blocks * 7 + num_1blocks * 4  # 35 + 4 = 39
    
    # Build U, V, W matrices
    # U: shape (n*m, rank) = (4, rank)
    # V: shape (m*p, rank) = (22, rank)
    # W: shape (p*n, rank) = (22, rank)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2x2:
    # For A (2x2), B (2x2), C = A*B (2x2):
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B10)
    # M7 = (A01 - A11)(B01 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # In the tensor decomposition, for each Strassen term r (0..6), we have:
    # U[r] corresponds to the A coefficients
    # V[r] corresponds to the B coefficients
    # W[r] corresponds to the C coefficients
    
    # Strassen U rows (for A entries):
    # A is 2x2, indexed as A[i,j] where i=0,1 (row), j=0,1 (col)
    # U index = i*m + j = i*2 + j
    # 
    # M1: (A00 + A11) -> U = [1, 0, 0, 1]
    # M2: (A10 + A11) -> U = [0, 0, 1, 1]
    # M3: A00 -> U = [1, 0, 0, 0]
    # M4: A11 -> U = [0, 0, 0, 1]
    # M5: (A00 + A01) -> U = [1, 1, 0, 0]
    # M6: (A10 - A00) -> U = [-1, 0, 1, 0]
    # M7: (A01 - A11) -> U = [0, 1, 0, -1]
    
    strassen_U = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1],  # M7
    ]
    
    # Strassen V rows (for B entries):
    # B is 2x2, indexed as B[j,k] where j=0,1 (row), k=0,1 (col)
    # V index = j*p + k, but for a 2x2 block starting at column offset c, 
    # the local B is B[j, c+k] for k=0,1
    # V local index = j*2 + k
    #
    # M1: (B00 + B11) -> V = [1, 0, 0, 1]
    # M2: B00 -> V = [1, 0, 0, 0]
    # M3: (B01 - B11) -> V = [0, 1, -1, 0]... wait
    # Let me re-derive. B00 = B[0,0], B01 = B[0,1], B10 = B[1,0], B11 = B[1,1]
    # Local V index: j*2 + k where j is row, k is col
    # So V[0] = B00, V[1] = B01, V[2] = B10, V[3] = B11
    #
    # M1: (B00 + B11) -> V = [1, 0, 0, 1]
    # M2: B00 -> V = [1, 0, 0, 0]
    # M3: (B01 - B11) -> V = [0, 1, 0, -1]
    # M4: (B10 - B00) -> V = [-1, 0, 1, 0]
    # M5: B11 -> V = [0, 0, 0, 1]
    # M6: (B00 + B10) -> V = [1, 0, 1, 0]
    # M7: (B01 + B11) -> V = [0, 1, 0, 1]
    
    strassen_V = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [-1, 0, 1, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 0, 1, 0],   # M6
        [0, 1, 0, 1],   # M7
    ]
    
    # Strassen W rows (for C entries):
    # C is 2x2, indexed as C[i,k] where i=0,1 (row), k=0,1 (col)
    # W index = k*n + i = k*2 + i (trace convention: W[k*n + i])
    # So W local index: k*2 + i
    # W[0] = C[0,0] (k=0, i=0), W[1] = C[1,0] (k=0, i=1), W[2] = C[0,1] (k=1, i=0), W[3] = C[1,1] (k=1, i=1)
    #
    # C00 = M1 + M4 - M5 + M7 -> W = [1, 0, 0, 0] with coeffs [1, 0, 0, 1, -1, 0, 1]
    # C10 = M2 + M4 -> W = [0, 1, 0, 0] with coeffs [0, 1, 0, 1, 0, 0, 0]
    # C01 = M3 + M5 -> W = [0, 0, 1, 0] with coeffs [0, 0, 1, 0, 1, 0, 0]
    # C11 = M1 - M2 + M3 + M6 -> W = [0, 0, 0, 1] with coeffs [1, -1, 1, 0, 0, 1, 0]
    
    # For each Strassen term r, W[r] has a single 1 at the appropriate position? No.
    # Actually, in the tensor decomposition, each term r contributes to all C entries.
    # The W matrix for term r should have the C coefficients.
    # But wait - in a rank-1 term, W is a single vector. The sum over r gives the full C.
    # So for each r, W[r] is a vector over all C entries, and the coefficient is embedded.
    #
    # Actually no. In the tensor decomposition:
    # C[i,k] = sum_r U[i,m,j... wait, let me think again.
    
    # The tensor is T[(i,j), (j,k), (k,i)] = delta_{i,k?} no.
    # For matrix mult: C[i,k] = sum_j A[i,j] B[j,k]
    # In trace form: sum_{i,j,k} A[i,j] B[j,k] C[k,i] = sum_{i,k} C[i,k] C[k,i]? No.
    # 
    # Actually the standard tensor for matmult M_{n,m,p} has entries T[(i,j), (j,k), (k,i)]
    # which is 1 if... hmm. Let me just use the schoolbook as reference.
    #
    # Schoolbook: for each (i,j,k), U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1.
    # This gives: sum_r (sum_{i,j} A[i,j] U[i*m+j,r]) (sum_{j,k} B[j,k] V[j*p+k,r]) (sum_{k,i} C[k,i] W[k*n+i,r])
    # = sum_{i,j,k} A[i,j] B[j,k] C[k,i]
    # = sum_{i,k} C[i,k] C[k,i]? No, = sum_{i,j,k} A[i,j] B[j,k] C[k,i]
    # For this to equal sum_{i,k} C[i,k] (the product), we need the trace form.
    # 
    # Actually, I think the convention is that the tensor decomposition satisfies:
    # C[i,k] = sum_r (sum_j A[i,j] U[i*m+j,r]) * (sum_j B[j,k] V[j*p+k,r]) * W[k*n+i,r]
    # Wait, that doesn't work dimensionally unless W is just a scalar per term.
    
    # Let me look at the schoolbook more carefully.
    # U[i*m+j, r] where r=(i',j',k'). U[i*m+j, r] = delta_{i,i'} delta_{j,j'}
    # V[j*p+k, r] = delta_{j,j'} delta_{k,k'}
    # W[k*n+i, r] = delta_{k,k'} delta_{i,i'}
    # 
    # Product: sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] 
    # = sum_{i',j',k'} delta_{i,i'} delta_{j,j'} delta_{j,j'} delta_{k,k'} delta_{k,k'} delta_{i,i'}
    # = delta_{j,j'} delta_{k,k'}... this is getting confusing.
    
    # Let me just directly verify: the product (A*U_col)(B*V_col)(C*W_col) summed over columns.
    # For schoolbook column (i0,j0,k0):
    # A part: sum_{i,j} A[i,j] * delta_{i,i0} delta_{j,j0} = A[i0,j0]
    # B part: sum_{j,k} B[j,k] * delta_{j,j0} delta_{k,k0} = B[j0,k0]
    # C part: sum_{k,i} C[k,i] * delta_{k,k0} delta_{i,i0} = C[k0,i0]
    # Product: A[i0,j0] * B[j0,k0] * C[k0,i0]
    # Sum over (i0,j0,k0): sum A[i,j] B[j,k] C[k,i] = trace(ABC)
    
    # So the tensor being decomposed is the trace(ABC) tensor, not the matmult tensor directly.
    # But for rank purposes, these are related. The task says "exactly decompose the matrix multiplication tensor".
    # 
    # I think the correct interpretation is that we need:
    # C[i,k] = sum_r U_r[i,j] V_r[j,k] W_r[k,i] summed appropriately.
    # 
    # Actually, I think the standard formulation is:
    # The matmult tensor M has M[(i,j), (j,k), (k,i)] = delta_{i,i}... no.
    # M[(i,j), (j,k), (k,i)] is the tensor where contracting with A, B, C gives trace(ABC).
    # But we want C = AB, which is different.
    
    # Let me just go with the schoolbook convention and adapt Strassen to it.
    # In the schoolbook, each term (i,j,k) has U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1.
    # 
    # For Strassen on a 2x2x2 block (columns c, c+1):
    # We need 7 terms. Each term r has U_r, V_r, W_r vectors.
    # The product (A·U_r)(B·V_r)(C·W_r) should sum to the correct contribution.
    
    # For the 2x2x2 block starting at column offset c (local columns 0,1):
    # Local B matrix is B[:, c:c+2], local C is C[:, c:c+2].
    # 
    # Term r contributes: (sum_{i,j} A[i,j] U_r[i*m+j]) * (sum_{j,l} B[j, c+l] V_r[j*2+l]) * (sum_{l,i} C[i, c+l] W_r[l*2+i])
    # Wait, W index is k*n+i where k is the column index. So W[l*2+i] for local column l.
    # But W has global index k*n+i where k is 0..10. For local column l, global column = c+l.
    # So W_r[(c+l)*2 + i].
    
    # This is getting complex. Let me just build it directly.
    
    # Strassen U (same for all blocks, operates on A):
    # U_r[i*m+j] for i=0,1; j=0,1
    sU = [
        [1, 0, 0, 1],   # r=0: M1
        [0, 0, 1, 1],   # r=1: M2
        [1, 0, 0, 0],   # r=2: M3
        [0, 0, 0, 1],   # r=3: M4
        [1, 1, 0, 0],   # r=4: M5
        [-1, 0, 1, 0],  # r=5: M6
        [0, 1, 0, -1],  # r=6: M7
    ]
    
    # Strassen V (operates on 2-column block of B):
    # V_r[j*p + (c+l)] where j=0,1; l=0,1 is local column
    sV = [
        [1, 0, 0, 1],   # r=0: (B00+B11) -> B[0,c]+B[1,c+1] -> V[0*p+c]=1, V[1*p+c+1]=1
        [1, 0, 0, 0],   # r=1: B00 -> V[0*p+c]=1
        [0, 1, 0, -1],  # r=2: (B01-B11) -> V[0*p+c+1]=1, V[1*p+c+1]=-1
        [-1, 0, 1, 0],  # r=3: (B10-B00) -> V[0*p+c]=-1, V[1*p+c]=1
        [0, 0, 0, 1],   # r=4: B11 -> V[1*p+c+1]=1
        [1, 0, 1, 0],   # r=5: (B00+B10) -> V[0*p+c]=1, V[1*p+c]=1
        [0, 1, 0, 1],   # r=6: (B01+B11) -> V[0*p+c+1]=1, V[1*p+c+1]=1
    ]
    # sV[r][j*2+l] gives coefficient for B[j, c+l]
    
    # Strassen W (operates on 2-column block of C):
    # W_r[k*n + i] where k is global column, i=0,1
    # For local column l, global column = c+l, so index = (c+l)*2 + i
    # 
    # C00 = M1 + M4 - M5 + M7
    # C10 = M2 + M4
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6
    #
    # W_r[(c+l)*2 + i] for each r:
    # r=0 (M1): C00 += 1, C11 += 1 -> W[0*2+0]=1, W[1*2+1]=1 (local)
    # r=1 (M2): C10 += 1, C11 += -1 -> W[0*2+1]=1, W[1*2+1]=-1 (local)
    # r=2 (M3): C01 += 1, C11 += 1 -> W[1*2+0]=1, W[1*2+1]=1 (local)
    # r=3 (M4): C00 += 1, C10 += 1 -> W[0*2+0]=1, W[0*2+1]=1 (local)
    # r=4 (M5): C00 += -1, C01 += 1 -> W[0*2+0]=-1, W[1*2+0]=1 (local)
    # r=5 (M6): C11 += 1 -> W[1*2+1]=1 (local)
    # r=6 (M7): C00 += 1 -> W[0*2+0]=1 (local)
    
    sW = [
        [1, 0, 0, 1],   # r=0: C00=1, C11=1
        [0, 1, 0, -1],  # r=1: C10=1, C11=-1
        [0, 0, 1, 1],   # r=2: C01=1, C11=1
        [1, 1, 0, 0],   # r=3: C00=1, C10=1
        [-1, 0, 1, 0],  # r=4: C00=-1, C01=1
        [0, 0, 0, 1],   # r=5: C11=1
        [1, 0, 0, 0],   # r=6: C00=1
    ]
    # sW[r][l*2+i] gives coefficient for C[i, c+l]
    
    # Build the decomposition
    for b in range(num_2blocks):
        c = b * 2  # start column of this block
        for r in range(7):
            # U
            for ij in range(4):
                U[ij][col] = sU[r][ij]
            # V
            for j in range(2):
                for l in range(2):
                    V[j * p + (c + l)][col] = sV[r][j * 2 + l]
            # W
            for l in range(2):
                for i in range(2):
                    W[(c + l) * n + i][col] = sW[r][l * 2 + i]
            col += 1
    
    # Schoolbook for the remaining 1 column (column 10)
    c = 10
    for i in range(2):
        for j in range(2):
            for k in range(1):  # only 1 column
                U[i * m + j][col] = 1
                V[j * p + (c + k)][col] = 1
                W[(c + k) * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
