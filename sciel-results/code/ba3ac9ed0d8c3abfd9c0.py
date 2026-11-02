"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    # Schoolbook baseline
    rank_school = n * m * p  # 80
    U_sb = np.zeros((n * m, rank_school))
    V_sb = np.zeros((m * p, rank_school))
    W_sb = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    # Try to find better decompositions using randomized approaches
    # For matrix multiplication tensor T_{i,j,k} = delta_{i,k} (in appropriate indexing)
    # We need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta condition
    
    # The tensor for multiplying A (n x m) by B (m x p) to get C (n x p):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # Actually the standard formulation: T[a, b, c] where a indexes (i,j), b indexes (j,k), c indexes (k,i)
    # T[(i,j), (j',k), (k',i')] = delta_{j,j'} * delta_{k,k'} * delta_{i,i'}
    
    # Rank-1 decomposition: sum_c u_c[a] * v_c[b] * w_c[c]
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if this is a valid (i,j,k) triple, 0 otherwise
    # Actually for matrix mult: the tensor is T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # with the constraint that the same j appears in both A and B indices, and same k,i in output.
    
    # Let me re-read the convention: W[k*n+i, r]
    # So the tensor entry is T[i*m+j, j*p+k, k*n+i] which should equal 1 for all valid (i,j,k)
    # and 0 otherwise. But since i<m... wait, i ranges over n, j over m, k over p.
    # The tensor is defined on indices (a, b, c) where a in [0, n*m), b in [0, m*p), c in [0, p*n)
    # T[a, b, c] = 1 iff there exist i,j,k such that a=i*m+j, b=j'*p+k, c=k'*n+i' 
    # with j=j', k=k', i=i'. Otherwise 0.
    
    # So T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, and 0 elsewhere.
    
    # This is the standard matrix multiplication tensor.
    
    # Known results: 
    # 2x2x2 has rank 7 (Strassen)
    # 4x4x4 has rank 49 (Landsberg) or 48? Actually best known is 49 for 4x4x4.
    # 4x4x5: unknown, but at most 80 (schoolbook).
    
    # Let's try to use Strassen-like constructions or known results.
    
    # For 2x2x2, Strassen's algorithm gives rank 7:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # Let's try to find a rank-7 decomposition for 2x2x2 first, then extend.
    
    # Actually, let me try a different approach: use numerical optimization to find
    # low-rank decompositions.
    
    # The tensor T has size (n*m) x (m*p) x (p*n) = 16 x 20 x 20
    # We want to find rank r such that T = sum_{c=1}^{r} u_c v_c^T w_c (outer products)
    
    # This is a hard problem. Let's try randomized initialization with alternating least squares.
    
    def verify(U, V, W, tol=1e-6):
        """Check if the decomposition is exact."""
        n_, m_, p_ = DIMENSIONS
        # Build the target tensor implicitly
        # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k, else 0
        # Reconstruct: sum_c U[a,c]*V[b,c]*W[c_,c] where c_ is the third index
        # For each (a,b), check that the vector over c_ matches target
        for i in range(n_):
            for j in range(m_):
                a = i * m_ + j
                for k in range(p_):
                    b = j * p_ + k
                    c_idx = k * n_ + i
                    # Check T[a,b,c_idx] = 1
                    val = np.sum(U[a, :] * V[b, :] * W[c_idx, :])
                    if abs(val - 1.0) > tol:
                        return False
                    # Check T[a,b,c'] = 0 for c' != c_idx
                    for i2 in range(n_):
                        for k2 in range(p_):
                            c2 = k2 * n_ + i2
                            if c2 != c_idx:
                                val0 = np.sum(U[a, :] * V[b, :] * W[c2, :])
                                if abs(val0) > tol:
                                    return False
        return True
    
    # Try rank 7 for 2x2x2 as a building block
    # Actually, let's just try to improve on schoolbook with some heuristics.
    
    # Known: 3x3x3 has rank at most 23 (Bini et al.), 4x4x4 at most 49
    # 4x4x5: we can pad 4x4x4 to 4x4x5 by adding zeros, but that doesn't help.
    # Alternatively, 4x4x5 can be seen as 4x4x4 + 4x4x1, and 4x4x1 has rank 4 (each row of A times each column... no)
    # Actually 4x4x1: C[i,0] = sum_j A[i,j] * B[j,0], tensor T[i*m+j, j*p+0, 0*n+i] = T[i*4+j, j*1+0, i] 
    # This is just the standard matrix-vector product tensor, which has rank 4*1=4? No, it's rank 4*4=16? 
    # Wait, for m=4, p=1: T[i*4+j, j, i] = 1. This is a permutation-like tensor.
    # Actually this has rank 16 (schoolbook) but can it be lower?
    # T[i*4+j, j, i] = 1 iff the middle index j matches. This is like a "copy" tensor.
    # It's the tensor for multiplying a 4x4 matrix by a 4x1 vector.
    # Rank is at most 16 (schoolbook), and I believe it's exactly 16 since it's a "slice" structure.
    
    # Let me try a different approach: use the fact that 4x4x5 = 4x4x4 + 4x4x1
    # and see if we can share some rank-1 terms.
    
    # Actually, let's just try to use known good decompositions.
    
    # For now, let's try a simple improvement: use the 2x2x2 Strassen decomposition
    # and tile it.
    
    # 4x4x4 can be decomposed using 2x2x2 Strassen blocks:
    # Split A, B, C into 2x2 blocks of 2x2 matrices.
    # Each 2x2x2 block multiplication uses rank 7.
    # But the blocks interact, so we need cross terms.
    # Total: 4 blocks * 7 + cross terms... this gets complicated.
    
    # Let's try a completely different approach: use numerical tensor decomposition.
    
    # Initialize with random rank-r decomposition and use ALS to fit the tensor.
    
    def try_rank(r, n_iter=500, seed=42):
        if time.time() > deadline - 5:
            return None, None, None, False
        
        rng = np.random.RandomState(seed)
        n_, m_, p_ = DIMENSIONS
        
        U = rng.randn(n_ * m_, r) * 0.1
        V = rng.randn(m_ * p_, r) * 0.1
        W = rng.randn(p_ * n_, r) * 0.1
        
        for it in range(n_iter):
            if time.time() > deadline - 5:
                break
            
            # Update U: for each a, U[a,:] = (T[a,:,:] * (V ⊙ W)) / (V ⊙ W)
            # T[a,b,c] is 1 if b=j*p+k and c=k*n+i where a=i*m+j, else 0
            # This is complex. Let's use a simpler approach.
            
            # Actually, let's just verify the schoolbook and return it.
            # Finding a better decomposition in 130s is very challenging.
            
            # Let's try a structured approach: use the 2x2x2 Strassen algorithm
            # and extend to 4x4x5.
            break
        
        return U, V, W, verify(U, V, W)
    
    # Let's construct the Strassen-based 4x4x4 decomposition and extend to 4x4x5.
    
    # For 2x2x2, Strassen's 7 multiplications:
    # We need U, V, W of shape (4,7), (4,7), (4,7) for 2x2x2.
    # A is 2x2, B is 2x2, C is 2x2.
    # a = i*2+j, b = j*2+k, c = k*2+i
    
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For the tensor decomposition, we need:
    # sum_c U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c] = 1 if valid, 0 otherwise
    
    # Let me define the 7 rank-1 components for 2x2x2:
    # Component c corresponds to one of the 7 products.
    # For component 1: U[i*2+j, 0] = 1 if (i,j) in {(0,0),(1,1)}, else 0
    #                V[j*2+k, 0] = 1 if (j,k) in {(0,0),(1,1)}, else 0
    #                W[k*2+i, 0] = 1 if (k,i) in {(0,0),(1,1)}, else 0
    # But this doesn't quite work because the same component needs to contribute to multiple C entries.
    
    # Actually, the tensor decomposition is:
    # T[i*2+j, j*2+k, k*2+i] = sum_c U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c]
    
    # For Strassen, we have 7 intermediate products M1..M7.
    # Each M_c is a product of a linear combination of A entries and a linear combination of B entries.
    # The result C entries are linear combinations of the M_c.
    
    # So: C[i,k] = sum_c (sum_j alpha_{c,i,j} * A[i,j]) * (sum_j beta_{c,k,j} * B[j,k])
    # = sum_c sum_j alpha_{c,i,j} * beta_{c,k,j} * A[i,j] * B[j,k]
    
    # And we need C[i,k] = sum_j A[i,j] * B[j,k], so:
    # sum_c sum_j alpha_{c,i,j} * beta_{c,k,j} = 1 for all i,k,j... wait, that's not right.
    
    # Actually: C[i,k] = sum_j A[i,j] * B[j,k]
    # And we compute C[i,k] = sum_c gamma_{c,i,k} * M_c where M_c = (sum_j alpha_{c,i,j} A[i,j]) * (sum_j beta_{c,j,k} B[j,k])
    
    # So: sum_c gamma_{c,i,k} * alpha_{c,i,j} * beta_{c,j,k} = 1 for all i,j,k (the j that appears in A[i,j] and B[j,k])
    
    # In tensor form: T[i*2+j, j*2+k, k*2+i] = sum_c U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c]
    # where U[i*2+j, c] = alpha_{c,i,j}, V[j*2+k, c] = beta_{c,j,k}, W[k*2+i, c] = gamma_{c,i,k}
    
    # For Strassen 2x2x2:
    # M1 = (A00+A11)(B00+B11): alpha_1 = {A00:1, A11:1}, beta_1 = {B00:1, B11:1}
    #   gamma_1: C00 += 1, C11 += 1, so W[0*2+0,0]=1, W[1*2+1,0]=1
    # M2 = (A10+A11)B00: alpha_2 = {A10:1, A11:1}, beta_2 = {B00:1}
    #   gamma_2: C10 += 1, C11 -= 1, so W[0*2+1,1]=1, W[1*2+1,1]=-1
    # M3 = A00(B10-B11): alpha_3 = {A00:1}, beta_3 = {B10:1, B11:-1}
    #   gamma_3: C01 += 1, C11 += 1, so W[1*2+0,2]=1, W[1*2+1,2]=1
    # M4 = A11(B01-B00): alpha_4 = {A11:1}, beta_4 = {B01:1, B00:-1}
    #   gamma_4: C00 -= 1, C10 += 1, so W[0*2+0,3]=-1, W[0*2+1,3]=1
    # M5 = (A00+A01)B11: alpha_5 = {A00:1, A01:1}, beta_5 = {B11:1}
    #   gamma_5: C00 += 1, C01 += 1, so W[0*2+0,4]=1, W[1*2+0,4]=1
    # M6 = (A10-A00)(B00+B01): alpha_6 = {A10:1, A00:-1}, beta_6 = {B00:1, B01:1}
    #   gamma_6: C11 += 1, so W[1*2+1,5]=1
    # M7 = (A01-A11)(B10+B11): alpha_7 = {A01:1, A11:-1}, beta_7 = {B10:1, B11:1}
    #   gamma_7: C00 -= 1, so W[0*2+0,6]=-1
    
    # Let me verify this gives the correct tensor.
    
    # Build 2x2x2 Strassen decomposition:
    n2, m2, p2 = 2, 2, 2
    r2 = 7
    U2 = np.zeros((n2 * m2, r2))
    V2 = np.zeros((m2 * p2, r2))
    W2 = np.zeros((p2 * n2, r2))
    
    # M1 = (A00+A11)(B00+B11), contributes to C00 (+1), C11 (+1)
    U2[0*2+0, 0] = 1  # A00
    U2[1*2+1, 0] = 1  # A11
    V2[0*2+0, 0] = 1  # B00
    V2[1*2+1, 0] = 1  # B11
    W2[0*2+0, 0] = 1  # C00
    W2[1*2+1, 0] = 1  # C11
    
    # M2 = (A10+A11)B00, contributes to C10 (+1), C11 (-1)
    U2[1*2+0, 1] = 1  # A10
    U2[1*2+1, 1] = 1  # A11
    V2[0*2+0, 1] = 1  # B00
    W2[0*2+1, 1] = 1  # C10
    W2[1*2+1, 1] = -1  # C11
    
    # M3 = A00(B10-B11), contributes to C01 (+1), C11 (+1)
    U2[0*2+0, 2] = 1  # A00
    V2[1*2+0, 2] = 1  # B10
    V2[1*2+1, 2] = -1  # B11
    W2[1*2+0, 2] = 1  # C01
    W2[1*2+1, 2] = 1  # C11
    
    # M4 = A11(B01-B00), contributes to C00 (-1), C10 (+1)
    U2[1*2+1, 3] = 1  # A11
    V2[0*2+1, 3] = 1  # B01
    V2[0*2+0, 3] = -1  # B00
    W2[0*2+0, 3] = -1  # C00
    W2[0*2+1, 3] = 1  # C10
    
    # M5 = (A00+A01)B11, contributes to C00 (+1), C01 (+1)
    U2[0*2+0, 4] = 1  # A00
    U2[0*2+1, 4] = 1  # A01
    V2[1*2+1, 4] = 1  # B11
    W2[0*2+0, 4] = 1  # C00
    W2[1*2+0, 4] = 1  # C01
    
    # M6 = (A10-A00)(B00+B01), contributes to C11 (+1)
    U2[1*2+0, 5] = 1  # A10
    U2[0*2+0, 5] = -1  # A00
    V2[0*2+0, 5] = 1  # B00
    V2[0*2+1, 5] = 1  # B01
    W2[1*2+1, 5] = 1  # C11
    
    # M7 = (A01-A11)(B10+B11), contributes to C00 (-1)
    U2[0*2+1, 6] = 1  # A01
    U2[1*2+1, 6] = -1  # A11
    V2[1*2+0, 6] = 1  # B10
    V2[1*2+1, 6] = 1  # B11
    W2[0*2+0, 6] = -1  # C00
    
    # Verify 2x2x2 decomposition
    def verify_2x2x2(U2, V2, W2, tol=1e-9):
        n_, m_, p_ = 2, 2, 2
        for i in range(n_):
            for j in range(m_):
                for k in range(p_):
                    a = i * m_ + j
                    b = j * p_ + k
                    c = k * n_ + i
                    val = np.sum(U2[a, :] * V2[b, :] * W2[c, :])
                    if abs(val - 1.0) > tol:
                        return False
                    for i2 in range(n_):
                        for k2 in range(p_):
                            c2 = k2 * n_ + i2
                            if c2 != c:
                                val0 = np.sum(U2[a, :] * V2[b, :] * W2[c2, :])
                                if abs(val0) > tol:
                                    return False
        return True
    
    # Check if our Strassen decomposition is correct
    strassen_valid = verify_2x2x2(U2, V2, W2)
    
    if not strassen_valid:
        # Fall back to schoolbook
        U_out = U_sb.tolist()
        V_out = V_sb.tolist()
        W_out = W_sb.tolist()
        return {"U": U_out, "V": V_out, "W": W_out}
    
    # Now extend to 4x4x4 using the 2x2x2 Strassen block.
    # Split 4x4 into 2x2 blocks of 2x2 matrices.
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # Similarly for B and C.
    
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Each product Aij*Bkl is a 2x2x2 tensor, which we can decompose with rank 7.
    # Total rank for 4x4x4 using this approach: 8 * 7 = 56.
    # But we can do better by sharing some terms.
    
    # Actually, the standard way to use Strassen for 4x4x4 is to recursively apply Strassen,
    # giving rank 7^2 = 49 for 4x4x4.
    
    # Let's construct the 4x4x4 decomposition using recursive Strassen.
    # This is complex, so let's just use the 8-block approach with rank 56 for now.
    
    # For 4x4x4:
    # A is 4x4, split into 2x2 blocks of 2x2: A00, A01, A10, A11
    # B is 4x4, split into 2x2 blocks of 2x2: B00, B01, B10, B11
    # C is 4x4, split into 2x2 blocks of 2x2: C00, C01, C10, C11
    
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Each term Aij*Bkl is a 2x2x2 multiplication.
    # For Aij*Bkl, the tensor is T2[i*2+j, j*2+k, k*2+i] for the local indices.
    # In the global 4x4x4 tensor, this corresponds to:
    # A index: (2*block_i + i, 2*block_j + j) where block_i, block_j are the 2x2 block indices
    # B index: (2*block_j + j, 2*block_k + k)
    # C index: (2*block_k + k, 2*block_i + i)
    
    # Let's build the 4x4x4 decomposition:
    n4, m4, p4 = 4, 4, 4
    r4 = 8 * 7  # 8 blocks * 7 rank per block
    U4 = np.zeros((n4 * m4, r4))
    V4 = np.zeros((m4 * p4, r4))
    W4 = np.zeros((p4 * n4, r4))
    
    # Block structure:
    # (block_i, block_j) for A blocks: (0,0)=A00, (0,1)=A01, (1,0)=A10, (1,1)=A11
    # (block_j, block_k) for B blocks
    # (block_k, block_i) for C blocks
    
    # For each pair of A-block (bi, bj) and B-block (bj, bk) where they share the middle index bj:
    # We need A[bi,bj] * B[bj,bk] to contribute to C[bk,bi]
    
    # The 8 products are:
    # A00*B00 -> C00 (bi=0,bj=0,bk=0)
    # A01*B10 -> C00 (bi=0,bj=1,bk=0)
    # A00*B01 -> C01 (bi=0,bj=0,bk=1)
    # A01*B11 -> C01 (bi=0,bj=1,bk=1)
    # A10*B00 -> C10 (bi=1,bj=0,bk=0)
    # A11*B10 -> C10 (bi=1,bj=1,bk=0)
    # A10*B01 -> C11 (bi=1,bj=0,bk=1)
    # A11*B11 -> C11 (bi=1,bj=1,bk=1)
    
    products = [
        (0, 0, 0),  # A00*B00 -> C00
        (0, 1, 0),  # A01*B10 -> C00
        (0, 0, 1),  # A00*B01 -> C01
        (0, 1, 1),  # A01*B11 -> C01
        (1, 0, 0),  # A10*B00 -> C10
        (1, 1, 0),  # A11*B10 -> C10
        (1, 0, 1),  # A10*B01 -> C11
        (1, 1, 1),  # A11*B11 -> C11
    ]
    
    for prod_idx, (bi, bj, bk) in enumerate(products):
        base_col = prod_idx * 7
        for c in range(7):
            # U: A index = (2*bi + i, 2*bj + j)
            for i in range(2):
                for j in range(2):
                    a_idx = (2 * bi + i) * 4 + (2 * bj + j)
                    U4[a_idx, base_col + c] = U2[i * 2 + j, c]
            
            # V: B index = (2*bj + j, 2*bk + k)
            for j in range(2):
                for k in range(2):
                    b_idx = (2 * bj + j) * 4 + (2 * bk + k)
                    V4[b_idx, base_col + c] = V2[j * 2 + k, c]
            
            # W: C index = (2*bk + k, 2*bi + i)
            for k in range(2):
                for i in range(2):
                    c_idx = (2 * bk + k) * 4 + (2 * bi + i)
                    W4[c_idx, base_col + c] = W2[k * 2 + i, c]
    
    # Verify 4x4x4 decomposition
    def verify_4x4x4(U4, V4, W4, tol=1e-9):
        n_, m_, p_ = 4, 4, 4
        for i in range(n_):
            for j in range(m_):
                for k in range(p_):
                    a = i * m_ + j
                    b = j * p_ + k
                    c = k * n_ + i
                    val = np.sum(U4[a, :] * V4[b, :] * W4[c, :])
                    if abs(val - 1.0) > tol:
                        return False
                    for i2 in range(n_):
                        for k2 in range(p_):
                            c2 = k2 * n_ + i2
                            if c2 != c:
                                val0 = np.sum(U4[a, :] * V4[b, :] * W4[c2, :])
                                if abs(val0) > tol:
                                    return False
        return True
    
    four_valid = verify_4x4x4(U4, V4, W4)
    
    if not four_valid:
        # Fall back to schoolbook
        U_out = U_sb.tolist()
        V_out = V_sb.tolist()
        W_out = W_sb.tolist()
        return {"U": U_out, "V": V_out, "W": W_out}
    
    # Now extend to 4x4x5.
    # C is 4x5, so C = [[C00, C01, C02, C03, C04], ...] where each Cij is 2x2? No.
    # Actually, for 4x4x5: A is 4x4, B is 4x5, C is 4x5.
    # Split A into 2x2 blocks of 2x2: A00, A01, A10, A11 (each 2x2)
    # Split B into 2x5 blocks: B0 (2x5), B1 (2x5) where B0 = B[0:2,:], B1 = B[2:4,:]
    # C is 4x5: C0 = C[0:2,:], C1 = C[2:4,:]
    
    # C0 = A00 * B0 + A01 * B1
    # C1 = A10 * B0 + A11 * B1
    
    # Each Aij is 2x2, Bk is 2x5.
    # Aij * Bk is a 2x2x5 multiplication, which can be seen as 2x2x5.
    # We can split Bk into two 2x2 and one 2x1: Bk = [Bk0, Bk1, Bk2] where Bk0, Bk1 are 2x2, Bk2 is 2x1.
    # Aij * Bk0 is 2x2x2 (rank 7 with Strassen)
    # Aij * Bk1 is 2x2x2 (rank 7 with Strassen)
    # Aij * Bk2 is 2x2x1 (rank ?)
    
    # For 2x2x1: A is 2x2, B is 2x1, C is 2x1.
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # Tensor: T[i*2+j, j*1+k, k*2+i] = T[i*2+j, j, 2*i] for k=0
    # T[0,0,0]=1, T[1,1,0]=1, T[2,0,2]=1, T[3,1,2]=1, all others 0.
    # This is a rank-4 tensor (each term is a separate rank-1 component).
    # Actually, can we do better? T[i*2+j, j, 2*i] = 1.
    # This is like a permutation tensor. Rank is 4 (I believe).
    
    # So for 2x2x5: 2x2x2 + 2x2x2 + 2x2x1 = 7 + 7 + 4 = 18 rank.
    # For 4x4x5: 4 * 18 = 72 rank.
    
    # Let's construct this.
    
    # First, 2x2x1 decomposition:
    n21, m21, p21 = 2, 2, 1
    r21 = 4
    U21 = np.zeros((n21 * m21, r21))
    V21 = np.zeros((m21 * p21, r21))
    W21 = np.zeros((p21 * n21, r21))
    
    # T[i*2+j, j, 2*i] = 1
    # (i,j) = (0,0): a=0, b=0, c=0
    # (i,j) = (0,1): a=1, b=1, c=0
    # (i,j) = (1,0): a=2, b=0, c=2
    # (i,j) = (1,1): a=3, b=1, c=2
    
    U21[0, 0] = 1; V21[0, 0] = 1; W21[0, 0] = 1
    U21[1, 1] = 1; V21[1, 1] = 1; W21[0, 1] = 1
    U21[2, 2] = 1; V21[0, 2] = 1; W21[2, 2] = 1
    U21[3, 3] = 1; V21[1, 3] = 1; W21[2, 3] = 1
    
    # Verify 2x2x1
    def verify_2x2x1(U21, V21, W21, tol=1e-9):
        n_, m_, p_ = 2, 2, 1
        for i in range(n_):
            for j in range(m_):
                for k in range(p_):
                    a = i * m_ + j
                    b = j * p_ + k
                    c = k * n_ + i
                    val = np.sum(U21[a, :] * V21[b, :] * W21[c, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    if not verify_2x2x1(U21, V21, W21):
        # Fall back to schoolbook
        U_out = U_sb.tolist()
        V_out = V_sb.tolist()
        W_out = W_sb.tolist()
        return {"U": U_out, "V": V_out, "W": W_out}
    
    # Now build 4x4x5:
    n5, m5, p5 = 4, 4, 5
    # A: 4x4, split into 2x2 blocks: A00, A01, A10, A11
    # B: 4x5, split into rows: B0 = B[0:2, :], B1 = B[2:4, :]
    # C: 4x5, split into rows: C0 = C[0:2, :], C1 = C[2:4, :]
    
    # C0 = A00 * B0 + A01 * B1
    # C1 = A10 * B0 + A11 * B1
    
    # Each Aij is 2x2, Bk is 2x5.
    # Bk = [Bk_col0, Bk_col1, Bk_col2, Bk_col3, Bk_col4] where each col is 2x1.
    # Or split Bk into 2x2 + 2x2 + 2x1:
    # Bk = [Bk_00, Bk_01, Bk_02] where Bk_00 = Bk[:,0:2], Bk_01 = Bk[:,2:4], Bk_02 = Bk[:,4:5]
    
    # Aij * Bk_00 is 2x2x2 (rank 7)
    # Aij * Bk_01 is 2x2x2 (rank 7)
    # Aij * Bk_02 is 2x2x1 (rank 4)
    
    # Total per Aij*Bk: 7 + 7 + 4 = 18
    # Total for 4x4x5: 4 * 18 = 72
    
    r5 = 72
    U5 = np.zeros((n5 * m5, r5))
    V5 = np.zeros((m5 * p5, r5))
    W5 = np.zeros((p5 * n5, r5))
    
    # Products:
    # (bi, bj, bk, col_range) where bi, bj are A block indices, bk is B row block, col_range is B column range
    # A[bi,bj] * B[bk, cols] -> C[bk, bi, cols]
    
    # Let's enumerate:
    # A00 * B0_00 -> C0_00 (bi=0, bj=0, bk=0, cols=0:2)
    # A00 * B0_01 -> C0_01 (bi=0, bj=0, bk=0, cols=2:4)
    # A00 * B0_02 -> C0_02 (bi=0, bj=0, bk=0, cols=4:5)
    # A01 * B1_00 -> C0_10 (bi=0, bj=1, bk=1, cols=0:2)
    # A01 * B1_01 -> C0_11 (bi=0, bj=1, bk=1, cols=2:4)
    # A01 * B1_02 -> C0_12 (bi=0, bj=1, bk=1, cols=4:5)
    # A10 * B0_00 -> C1_00 (bi=1, bj=0, bk=0, cols=0:2)
    # A10 * B0_01 -> C1_01 (bi=1, bj=0, bk=0, cols=2:4)
    # A10 * B0_02 -> C1_02 (bi=1, bj=0, bk=0, cols=4:5)
    # A11 * B1_00 -> C1_10 (bi=1, bj=1, bk=1, cols=0:2)
    # A11 * B1_01 -> C1_11 (bi=1, bj=1, bk=1, cols=2:4)
    # A11 * B1_02 -> C1_12 (bi=1, bj=1, bk=1, cols=4:5)
    
    # For each product, we add 18 rank-1 terms (7 for 2x2x2 part 1, 7 for 2x2x2 part 2, 4 for 2x2x1 part)
    
    products_5 = [
        # (bi, bj, bk, col_start, col_end)
        (0, 0, 0, 0, 2),   # A00 * B0[:,0:2]
        (0, 0, 0, 2, 4),   # A00 * B0[:,2:4]
        (0, 0, 0, 4, 5),   # A00 * B0[:,4:5]
        (0, 1, 1, 0, 2),   # A01 * B1[:,0:2]
        (0, 1, 1, 2, 4),   # A01 * B1[:,2:4]
        (0, 1, 1, 4, 5),   # A01 * B1[:,4:5]
        (1, 0, 0, 0, 2),   # A10 * B0[:,0:2]
        (1, 0, 0, 2, 4),   # A10 * B0[:,2:4]
        (1, 0, 0, 4, 5),   # A10 * B0[:,4:5]
        (1, 1, 1, 0, 2),   # A11 * B1[:,0:2]
        (1, 1, 1, 2, 4),   # A11 * B1[:,2:4]
        (1, 1, 1, 4, 5),   # A11 * B1[:,4:5]
    ]
    
    col_offset = 0
    for (bi, bj, bk, col_start, col_end) in products_5:
        col_width = col_end - col_start
        if col_width == 2:
            # 2x2x2 product, use Strassen rank 7
            for c in range(7):
                # U: A index = (2*bi + i, 2*bj + j)
                for i in range(2):
                    for j in range(2):
                        a_idx = (2 * bi + i) * 4 + (2 * bj + j)
                        U5[a_idx, col_offset + c] = U2[i * 2 + j, c]
                
                # V: B index = (2*bk + j, col_start + k)
                for j in range(2):
                    for k in range(2):
                        b_idx = (2 * bk + j) * 5 + (col_start + k)
                        V5[b_idx, col_offset + c] = V2[j * 2 + k, c]
                
                # W: C index = (col_start + k, 2*bi + i)
                for k in range(2):
                    for i in range(2):
                        c_idx = (col_start + k) * 4 + (2 * bi + i)
                        W5[c_idx, col_offset + c] = W2[k * 2 + i, c]
            col_offset += 7
        elif col_width == 1:
            # 2x2x1 product, use rank 4
            for c in range(4):
                # U: A index = (2*bi + i, 2*bj + j)
                for i in range(2):
                    for j in range(2):
                        a_idx = (2 * bi + i) * 4 + (2 * bj + j)
                        U5[a_idx, col_offset + c] = U21[i * 2 + j, c]
                
                # V: B index = (2*bk + j, col_start)
                for j in range(2):
                    b_idx = (2 * bk + j) * 5 + col_start
                    V5[b_idx, col_offset + c] = V21[j, c]
                
                # W: C index = (col_start, 2*bi + i)
                for i in range(2):
                    c_idx = col_start * 4 + (2 * bi + i)
                    W5[c_idx, col_offset + c] = W21[i, c]
            col_offset += 4
    
    # Verify 4x4x5 decomposition
    def verify_4x4x5(U5, V5, W5, tol=1e-9):
        n_, m_, p_ = 4, 4, 5
        for i in range(n_):
            for j in range(m_):
                for k in range(p_):
                    a = i * m_ + j
                    b = j * p_ + k
                    c = k * n_ + i
                    val = np.sum(U5[a, :] * V5[b, :] * W5[c, :])
                    if abs(val - 1.0) > tol:
                        return False
                    for i2 in range(n_):
                        for k2 in range(p_):
                            c2 = k2 * n_ + i2
                            if c2 != c:
                                val0 = np.sum(U5[a, :] * V5[b, :] * W5[c2, :])
                                if abs(val0) > tol:
                                    return False
        return True
    
    five_valid = verify_4x4x5(U5, V5, W5)
    
    if five_valid:
        # Convert to lists with integer/half-integer entries
        U_out = U5.tolist()
        V_out = V5.tolist()
        W_out = W5.tolist()
        return {"U": U_out, "V": V_out, "W": W_out}
    else:
        # Fall back to schoolbook
        U_out = U_sb.tolist()
        V_out = V_sb.tolist()
        W_out = W_sb.tolist()
        return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
