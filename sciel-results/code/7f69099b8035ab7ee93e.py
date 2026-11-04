"""Matrix multiplication 2x4x4 rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 4]

# Deadline guard
START_TIME = time.time()
DEADLINE = 150.0  # seconds
SAFETY_MARGIN = 5.0  # return before this margin


def _time_left():
    return DEADLINE - SAFETY_MARGIN - (time.time() - START_TIME)


def _validate(U, V, W, n=2, m=4, p=4, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Build the multiplication tensor T[i,j,k] = delta
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition requires: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # Actually, the standard form is: (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor T_{i,j,k} where the multiplication map is linear in A and B.
    
    # The tensor for matrix multiplication: T[i,j,k] = 1 if we're computing (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    # In the trace convention W[k*n+i, r], the output index is (k, i) -> row k, col i? 
    # Wait, the task says "output factor follows the official trace convention W[k*n+i,r]"
    # So W index is k*n + i, meaning row = k, col = i? Or is it the standard (AB)_{i,k}?
    
    # Let me re-read: "multiplying a 2 by 4 matrix by a 4 by 4 matrix"
    # A is 2x4, B is 4x4, C = AB is 2x4
    # So i in [0,1] (rows of A/C), j in [0,3] (cols of A / rows of B), k in [0,3] (cols of B/C)
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    
    # U[i*m + j, c] - index by (i, j) where i is row of A, j is col of A
    # V[j*p + k, c] - index by (j, k) where j is row of B, k is col of B  
    # W[k*n + i, c] - index by (k, i) where k is col of C, i is row of C
    
    # The decomposition: C_{i,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] ... no
    # Actually for matrix multiplication tensor decomposition:
    # T_{i,j,k} = sum_c U_{i,j}^{(c)} * V_{j,k}^{(c)} * W_{k,i}^{(c)}? 
    # No, the standard is: the multiplication tensor has entries T_{i,j,k} = 1 for the "middle" index
    # and the decomposition is: T = sum_c u_c ⊗ v_c ⊗ w_c
    # where (u_c)_{i,j} = U[i*m+j, c], (v_c)_{j,k} = V[j*p+k, c], (w_c)_{k,i} = W[k*n+i, c]
    
    # For matrix multiplication, T_{i,j,k} = delta_{i,k}? No.
    # T_{i,j,k} = 1 always? No, that's not right either.
    
    # The matrix multiplication tensor: for computing C = AB where A is n×m, B is m×p
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor T has entries T_{i,j,k} = 1 for all i,j,k? No.
    
    # Actually, the bilinear map (A, B) -> C is given by:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # This is a trilinear form if we think of it as <C, A ⊗ B> but actually it's bilinear.
    
    # The rank of the matrix multiplication tensor: we want to write
    # C_{i,k} = sum_c (sum_j U_{i,j}^{(c)} A_{i,j}) * (sum_j V_{j,k}^{(c)} B_{j,k}) * W_{k,i}^{(c)}?
    
    # No. The standard tensor decomposition for matrix multiplication:
    # The tensor T ∈ F^{n×m} ⊗ F^{m×p} ⊗ F^{p×n} (with the last index being (k,i) in trace convention)
    # T_{i,j,k,i'} = delta_{i,i'} * delta_{j,k}? No...
    
    # Let me think more carefully. The multiplication map is:
    # mu: F^{n×m} × F^{m×p} -> F^{n×p}
    # mu(A,B)_{i,k} = sum_j A_{i,j} B_{j,k}
    
    # As a tensor in (F^{n×m})^* ⊗ (F^{m×p})^* ⊗ F^{n×p}, the entries are:
    # T_{(i,j),(j',k),(i',k')} = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    
    # But with the trace convention for W being W[k*n+i, r], the output index is (k, i) not (i, k).
    # So T_{(i,j),(j',k),(k'',i'')} = delta_{i,i''} * delta_{j,j'} * delta_{k,k''}
    
    # The rank decomposition: T = sum_c u_c ⊗ v_c ⊗ w_c
    # where u_c ∈ F^{n×m}, v_c ∈ F^{m×p}, w_c ∈ F^{p×n} (indexed by (k,i))
    
    # So: sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k''*n+i'', c] = delta_{i,i''} * delta_{j,j'} * delta_{k,k''}
    
    # For the schoolbook: for each (i,j,k), we have one term with U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
    # Check: sum over (i',j',k') terms... no, each term c corresponds to one (i,j,k) triple.
    # Term c=(i,j,k): U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1
    # Then sum_c U[i*m+j,c] V[j'*p+k',c] W[k''*n+i'',c] = 1 iff i=i', j=j', k=k'' and also i''=i, k''=k
    # Wait that gives delta_{i,i'} * delta_{j,j'} * delta_{k,k'} * delta_{i'',i} * delta_{k'',k}
    # Hmm, that's not quite right. Let me re-examine.
    
    # Actually I think the convention might be simpler. Let me just verify with the schoolbook.
    # In the schoolbook, for each (i,j,k): U[i*m+j, col]=1, V[j*p+k, col]=1, W[k*n+i, col]=1
    # The product sum_c U[i*m+j,c]*V[j'*p+k',c]*W[k''*n+i'',c] should equal the tensor entry.
    
    # For a fixed c corresponding to (i0,j0,k0):
    # U[i*m+j,c] = 1 if i=i0 and j=j0, else 0
    # V[j'*p+k',c] = 1 if j'=j0 and k'=k0, else 0
    # W[k''*n+i'',c] = 1 if k''=k0 and i''=i0, else 0
    
    # So the sum over c gives: 1 if there exists (i0,j0,k0) such that i=i0, j=j0, j'=j0, k'=k0, k''=k0, i''=i0
    # i.e., i=j'... no: i=i0, j=j0, j'=j0, k'=k0, k''=k0, i''=i0
    # So we need j=j' and k=k'' and i=i'' and i=i0, j=j0, k=k0 for some (i0,j0,k0)
    # This means: the sum is 1 iff j=j' and k=k'' and i=i''
    
    # So the tensor entry T[(i,j),(j',k),(k'',i'')] = 1 if j=j' and k=k'' and i=i'', else 0.
    # This matches: delta_{j,j'} * delta_{k,k''} * delta_{i,i''}
    
    # Great! So the validation is:
    # For all (i,j), (j',k), (k'',i''): sum_c U[i*m+j,c]*V[j'*p+k,c]*W[k''*n+i'',c] == delta_{j,j'}*delta_{k,k''}*delta_{i,i''}
    
    n, m, p = n, m, p
    # Build target tensor
    # Target[(i,j),(j',k),(k'',i'')] = 1 if j==j' and k==k'' and i==i''
    
    # Efficient check: for each c, the contribution is outer product
    # Let's use numpy for speed
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape: U_arr is (n*m, rank) -> (n, m, rank)
    U_r = U_arr.reshape(n, m, rank)
    V_r = V_arr.reshape(m, p, rank)
    W_r = W_arr.reshape(p, n, rank)
    
    # Compute: result[i, j, jp, k, kp, ip] = sum_c U_r[i,j,c] * V_r[jp,k,c] * W_r[kp,ip,c]
    # We want result[i, j, jp, k, kp, ip] = 1 if j==jp and k==kp and i==ip
    
    # Use einsum: sum over c
    # U_r[i,j,c] * V_r[jp,k,c] * W_r[kp,ip,c] summed over c
    # = np.einsum('ijc,jkc,kpc->ijjkp p', ...) hmm indices are getting confusing
    
    # Let me use explicit indices:
    # A[i,j,c] = U_r[i,j,c]
    # B[jp,k,c] = V_r[jp,k,c]  
    # C[kp,ip,c] = W_r[kp,ip,c]
    # result[i,j,jp,k,kp,ip] = sum_c A[i,j,c]*B[jp,k,c]*C[kp,ip,c]
    
    # This is a 6D array of size 2*4*4*4*4*2 = 512, which is small.
    result = np.zeros((n, m, m, p, p, n))
    for c in range(rank):
        # A[:, :, c] is (n,m), B[:, :, c] is (m,p), C[:, :, c] is (p,n)
        # outer product: A[i,j] * B[jp,k] * C[kp,ip]
        result += np.einsum('ij,jk,pi->ijjkpi', U_r[:, :, c], V_r[:, :, c], W_r[:, :, c])
    
    # Target: T[i,j,jp,k,kp,ip] = 1 if j==jp and k==kp and i==ip
    target = np.zeros((n, m, m, p, p, n))
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            if j == jp and k == kp and i == ip:
                                target[i, j, jp, k, kp, ip] = 1.0
    
    if not np.allclose(result, target, atol=tol):
        return False
    return True


def _try_rank(rank, n=2, m=4, p=4, max_iters=500, seed=42):
    """Try to find a rank-`rank` decomposition using random initialization and gradient-like refinement."""
    if rank < 1:
        return None
    
    rng = np.random.RandomState(seed)
    
    # Try multiple random initializations
    best_error = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    for init in range(20):
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = rng.randn(p * n, rank)
        
        # Normalize to avoid scale issues
        for c in range(rank):
            norm = np.sqrt(np.sum(U[:, c]**2) * np.sum(V[:, c]**2) * np.sum(W[:, c]**2))
            if norm > 1e-10:
                scale = norm ** (1/3)
                U[:, c] /= scale
                V[:, c] /= scale
                W[:, c] /= scale
        
        # Iterative refinement: alternating least squares style
        # For each c, given U, V, W, we can try to optimize.
        # Actually, let's use a simpler approach: since the tensor is small, 
        # use a direct algebraic approach.
        
        # For matrix multiplication, there are known lower bounds.
        # For 2x4x4, the Strassen-like approach gives rank 28 as mentioned.
        # Let's try to construct it directly.
        
        break  # We'll use constructive approaches instead
    
    return None


def _strassen_2x2(A_blocks, B_blocks):
    """
    Strassen's 7-multiplication algorithm for 2x2 block matrices.
    A_blocks: list of 4 blocks [[A11, A12], [A21, A22]]
    B_blocks: list of 4 blocks [[B11, B12], [B21, B22]]
    Returns 7 products M1..M7 and the reconstruction formulas.
    """
    A11, A12, A21, A22 = A_blocks
    B11, B12, B21, B22 = B_blocks
    
    # The 7 Strassen products:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    return [
        (A11 + A22, B11 + B22),  # M1
        (A21 + A22, B11),        # M2
        (A11, B12 - B22),        # M3
        (A22, B21 - B11),        # M4
        (A11 + A12, B22),        # M5
        (A21 - A11, B11 + B12),  # M6
        (A12 - A22, B21 + B22),  # M7
    ]


def construct():
    """
    Construct a low-rank exact tensor decomposition for 2x4x4 matrix multiplication.
    
    Strategy: Use Strassen's algorithm on 2x2 blocks.
    A is 2x4, B is 4x4.
    
    View A as 2 rows of 4 elements. View B as 4x4.
    
    One approach: split the 4 columns of A into two groups of 2, and the 4x4 B 
    into four 2x2 blocks. Then we have 4 block multiplications:
    A[:, 0:2] * B[0:2, 0:2], A[:, 0:2] * B[0:2, 2:4], 
    A[:, 2:4] * B[2:4, 0:2], A[:, 2:4] * B[2:4, 2:4]
    
    Each is a 2x2 * 2x2 multiplication, which can be done with Strassen's 7 mults.
    Total: 4 * 7 = 28 multiplications.
    
    Let me construct this explicitly.
    
    A is 2x4: A[i, j] for i in {0,1}, j in {0,1,2,3}
    B is 4x4: B[j, k] for j in {0,1,2,3}, k in {0,1,2,3}
    C = A*B is 2x4: C[i, k] for i in {0,1}, k in {0,1,2,3}
    
    C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k]
    
    Split j into {0,1} and {2,3}:
    C[i,k] = sum_{j=0}^{1} A[i,j] B[j,k] + sum_{j=2}^{3} A[i,j] B[j,k]
    
    Let A1 = A[:, 0:2] (2x2), A2 = A[:, 2:4] (2x2)
    Let B11 = B[0:2, 0:2], B12 = B[0:2, 2:4], B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    
    Then C[:, 0:2] = A1 * B11 + A2 * B21
    And C[:, 2:4] = A1 * B12 + A2 * B22
    
    Now apply Strassen to each of the 4 products A1*B11, A1*B12, A2*B21, A2*B22.
    
    But wait, Strassen computes C = A*B using 7 mults where A and B are 2x2.
    The "multiplications" in Strassen are scalar multiplications of the combined terms.
    
    For A1 * B11:
    M1 = (A1[0,0]+A1[1,1]) * (B11[0,0]+B11[1,1])
    M2 = (A1[1,0]+A1[1,1]) * B11[0,0]
    M3 = A1[0,0] * (B11[0,1]-B11[1,1])
    M4 = A1[1,1] * (B11[1,0]-B11[0,0])
    M5 = (A1[0,0]+A1[0,1]) * B11[1,1]
    M6 = (A1[1,0]-A1[0,0]) * (B11[0,0]+B11[0,1])
    M7 = (A1[0,1]-A1[1,1]) * (B11[1,0]+B11[1,1])
    
    Result:
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Each M is a product of a linear combination of A entries and a linear combination of B entries.
    In the tensor decomposition, each such "multiplication" corresponds to one rank-1 term.
    
    So for each of the 4 block products, we get 7 rank-1 terms, total 28.
    
    Let me construct U, V, W for all 28 terms.
    
    For a term corresponding to block product (A_block, B_block) with Strassen index s:
    U[i*m+j, c] = coefficient of A[i,j] in the A-part of the s-th Strassen product
    V[j'*p+k, c] = coefficient of B[j',k] in the B-part of the s-th Strassen product
    W[k''*n+i'', c] = coefficient in the reconstruction for C[i'',k'']
    
    Let me define the 4 block products:
    Block 0: A1 * B11 -> contributes to C[:, 0:2]
    Block 1: A1 * B12 -> contributes to C[:, 2:4]
    Block 2: A2 * B21 -> contributes to C[:, 0:2]
    Block 3: A2 * B22 -> contributes to C[:, 2:4]
    
    For block b, the A-block is either A1 (b in {0,1}) or A2 (b in {2,3}).
    A1 uses A columns 0,1; A2 uses A columns 2,3.
    
    For block b, the B-block is:
    b=0: B11 = B[0:2, 0:2]
    b=1: B12 = B[0:2, 2:4]
    b=2: B21 = B[2:4, 0:2]
    b=3: B22 = B[2:4, 2:4]
    
    Strassen indices 0..6 for each block.
    
    Let me define the Strassen coefficients:
    For a 2x2 matrix A with entries A[0,0], A[0,1], A[1,0], A[1,1]:
    A-part coefficients for M1..M7:
    M1: A[0,0]+A[1,1] -> [1, 0, 0, 1]
    M2: A[1,0]+A[1,1] -> [0, 0, 1, 1]
    M3: A[0,0] -> [1, 0, 0, 0]
    M4: A[1,1] -> [0, 0, 0, 1]
    M5: A[0,0]+A[0,1] -> [1, 1, 0, 0]
    M6: A[1,0]-A[0,0] -> [-1, 0, 1, 0]
    M7: A[0,1]-A[1,1] -> [0, 1, 0, -1]
    
    For a 2x2 matrix B with entries B[0,0], B[0,1], B[1,0], B[1,1]:
    B-part coefficients for M1..M7:
    M1: B[0,0]+B[1,1] -> [1, 0, 0, 1]
    M2: B[0,0] -> [1, 0, 0, 0]
    M3: B[0,1]-B[1,1] -> [0, 1, 0, -1]
    M4: B[1,0]-B[0,0] -> [-1, 0, 1, 0]
    M5: B[1,1] -> [0, 0, 0, 1]
    M6: B[0,0]+B[0,1] -> [1, 1, 0, 0]
    M7: B[1,0]+B[1,1] -> [0, 0, 1, 1]
    
    Reconstruction coefficients for C[0,0], C[0,1], C[1,0], C[1,1]:
    C[0,0] = M1 + M4 - M5 + M7 -> [1, 0, 0, 1, -1, 0, 1]
    C[0,1] = M3 + M5 -> [0, 0, 1, 0, 1, 0, 0]
    C[1,0] = M2 + M4 -> [0, 1, 0, 1, 0, 0, 0]
    C[1,1] = M1 - M2 + M3 + M6 -> [1, -1, 1, 0, 0, 1, 0]
    
    Now let me build the full decomposition.
    """
    n, m, p = 2, 4, 4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Strassen A-part coefficients (for 2x2, indexed as [A00, A01, A10, A11])
    strassen_A = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1],  # M7
    ]
    
    # Strassen B-part coefficients (for 2x2, indexed as [B00, B01, B10, B11])
    strassen_B = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [-1, 0, 1, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1],   # M7
    ]
    
    # Reconstruction: C[ci, ck] gets coefficient from M1..M7
    # ci, ck in {0,1}
    recon = {
        (0, 0): [1, 0, 0, 1, -1, 0, 1],   # C00 = M1 + M4 - M5 + M7
        (0, 1): [0, 0, 1, 0, 1, 0, 0],    # C01 = M3 + M5
        (1, 0): [0, 1, 0, 1, 0, 0, 0],    # C10 = M2 + M4
        (1, 1): [1, -1, 1, 0, 0, 1, 0],   # C11 = M1 - M2 + M3 + M6
    }
    
    # Block definitions:
    # Block 0: A1 (cols 0,1) * B11 (rows 0,1; cols 0,1) -> C cols 0,1
    # Block 1: A1 (cols 0,1) * B12 (rows 0,1; cols 2,3) -> C cols 2,3
    # Block 2: A2 (cols 2,3) * B21 (rows 2,3; cols 0,1) -> C cols 0,1
    # Block 3: A2 (cols 2,3) * B22 (rows 2,3; cols 2,3) -> C cols 2,3
    
    blocks = [
        {'a_cols': [0, 1], 'b_rows': [0, 1], 'c_cols': [0, 1]},  # Block 0
        {'a_cols': [0, 1], 'b_rows': [0, 1], 'c_cols': [2, 3]},  # Block 1
        {'a_cols': [2, 3], 'b_rows': [2, 3], 'c_cols': [0, 1]},  # Block 2
        {'a_cols': [2, 3], 'b_rows': [2, 3], 'c_cols': [2, 3]},  # Block 3
    ]
    
    col = 0
    for block in blocks:
        a_cols = block['a_cols']  # [c0, c1] - the two columns of A used
        b_rows = block['b_rows']  # [r0, r1] - the two rows of B used
        c_cols = block['c_cols']  # [k0, k1] - the two columns of C affected
        
        for s in range(7):
            # A-part: coefficients for A[0, a_cols[0]], A[0, a_cols[1]], A[1, a_cols[0]], A[1, a_cols[1]]
            # In U, index is i*m + j where i is row (0 or 1) and j is column (0..3)
            a_coeffs = strassen_A[s]  # [coeff for A[0,0], A[0,1], A[1,0], A[1,1]] in local 2x2
            # Local A[0,0] = A[0, a_cols[0]], A[0,1] = A[0, a_cols[1]]
            # Local A[1,0] = A[1, a_cols[0]], A[1,1] = A[1, a_cols[1]]
            U[0 * m + a_cols[0]][col] += a_coeffs[0]
            U[0 * m + a_cols[1]][col] += a_coeffs[1]
            U[1 * m + a_cols[0]][col] += a_coeffs[2]
            U[1 * m + a_cols[1]][col] += a_coeffs[3]
            
            # B-part: coefficients for B[b_rows[0], 0], B[b_rows[0], 1], B[b_rows[1], 0], B[b_rows[1], 1]
            # In local 2x2: B[0,0]=B[b_rows[0], c_cols[0]], B[0,1]=B[b_rows[0], c_cols[1]]
            # Wait, no. The B-block is B[b_rows[0]:b_rows[1]+1, c_cols[0]:c_cols[1]+1]
            # But actually B is 4x4, and we're taking a 2x2 sub-block.
            # Local B[0,0] = B[b_rows[0], c_cols[0]], B[0,1] = B[b_rows[0], c_cols[1]]
            # Local B[1,0] = B[b_rows[1], c_cols[0]], B[1,1] = B[b_rows[1], c_cols[1]]
            b_coeffs = strassen_B[s]
            # In V, index is j*p + k where j is row (0..3) and k is column (0..3)
            V[b_rows[0] * p + c_cols[0]][col] += b_coeffs[0]
            V[b_rows[0] * p + c_cols[1]][col] += b_coeffs[1]
            V[b_rows[1] * p + c_cols[0]][col] += b_coeffs[2]
            V[b_rows[1] * p + c_cols[1]][col] += b_coeffs[3]
            
            # W-part: reconstruction coefficients
            # W index is k*n + i where k is column of C (0..3) and i is row of C (0,1)
            for ci in range(2):
                for ck in range(2):
                    coeff = recon[(ci, ck)][s]
                    if coeff != 0:
                        # This term contributes to C[ci, c_cols[ck]]
                        c_col_actual = c_cols[ck]
                        W[c_col_actual * n + ci][col] += coeff
            
            col += 1
    
    # Verify
    if _validate(U, V, W):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][column] = 1
                    V_sb[j * p + k][column] = 1
                    W_sb[k * n + i][column] = 1
                    column += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
