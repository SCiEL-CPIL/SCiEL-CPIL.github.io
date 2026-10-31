"""Matrix multiplication 4,4,4 - rank 49 Strassen decomposition via Kronecker product."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Rank-49 exact tensor decomposition for 4x4 matrix multiplication.
    
    Uses the tensor (Kronecker) product of the rank-7 Strassen decomposition
    for 2x2 matrix multiplication with itself to get a rank-49 decomposition
    for 4x4 matrix multiplication.
    
    The 4x4 matrices are viewed as 2x2 block matrices with 2x2 blocks.
    Strassen's algorithm for 2x2 block multiplication uses 7 scalar multiplications,
    each of which is a 2x2 matrix product. Each 2x2 product uses 7 multiplications,
    giving 7*7 = 49 total multiplications.
    """
    n, m, p = DIMENSIONS
    assert (n, m, p) == (4, 4, 4)
    
    _t0 = time.time()
    _DEADLINE = 150.0
    _MARGIN = 15.0
    
    rank = 49
    
    # Define the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    # For C = A*B where A, B, C are 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # U factors (for A): each row is a linear combination of A entries
    # V factors (for B): each row is a linear combination of B entries
    # W factors (for C): each column is a linear combination of C entries
    
    # U_strassen: 4x7 matrix, rows correspond to A00,A01,A10,A11
    U_strassen = np.array([
        [1, 0, 1, 0, 1, -1, 0],   # A00
        [0, 0, 0, 0, 1, 0, 1],    # A01
        [0, 1, 0, 0, 0, 1, 0],    # A10
        [1, 1, 0, 1, 0, 0, -1],   # A11
    ], dtype=object)
    
    # V_strassen: 4x7 matrix, rows correspond to B00,B01,B10,B11
    V_strassen = np.array([
        [1, 1, 0, -1, 0, 1, 0],   # B00
        [1, 0, 1, 0, 0, 1, 0],    # B01
        [0, 0, -1, 1, 0, 0, 1],   # B10
        [1, 0, -1, 0, 1, 0, 1],   # B11
    ], dtype=object)
    
    # W_strassen: 4x7 matrix, rows correspond to C00,C01,C10,C11
    W_strassen = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [0, 1, 0, 1, 0, 0, 0],    # C10
        [1, -1, 1, 0, 0, 1, 0],   # C11
    ], dtype=object)
    
    # Verify the 2x2 Strassen decomposition
    # C[i,k] = sum_r U[i,r] * V[k,r] * W[i,k,r]... wait, let me think about this more carefully.
    
    # The tensor decomposition of matrix multiplication for 2x2 matrices:
    # C[i,k] = sum_r U_A[i,r] * U_B[k,r] * W[i,k,r]
    # where U_A has shape (2,7), U_B has shape (2,7), W has shape (2,2,7)
    
    # Actually, let me reconsider the tensor structure.
    # For matrix multiplication C = A*B:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # A tensor decomposition writes this as:
    # C[i,k] = sum_r (sum_j U[i,j,r] * A[i,j]) * (sum_j V[j,k,r] * B[j,k])
    # But that's not quite right either. The standard form is:
    # 
    # C[i,k] = sum_r U_A[i,r] * V_B[k,r] * W_C[i,k,r]
    # where U_A[i,r] = sum_j U[i,j,r] A[i,j], etc.
    
    # Let me use the explicit form from the task description:
    # U: (n*m, rank) = (4, 7) for 2x2
    # V: (m*p, rank) = (4, 7) for 2x2  
    # W: (p*n, rank) = (4, 7) for 2x2
    #
    # The decomposition is:
    # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * ... 
    # No, that's not right.
    
    # Let me re-read the task: "output factor follows the official trace convention W[k*n+i,r]"
    # So W has shape (p*n, rank) = (4, 7) for 2x2, indexed as W[k*n+i, r]
    
    # The standard tensor decomposition for matrix mult:
    # C[i,k] = sum_r U[i,r] * V[k,r] * W[i,k,r]
    # But here U has shape (n*m, rank), so U[i*m+j, r] for A[i,j]
    # V has shape (m*p, rank), so V[j*p+k, r] for B[j,k]
    # W has shape (p*n, rank), so W[k*n+i, r] for C[i,k]
    
    # The identity is:
    # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    # Wait, that gives a product of two sums, which would be degree 2 in each.
    
    # Actually, I think the correct form is:
    # C[i,k] = sum_r U_A[i, r] * U_B[k, r] * W[i, k, r]
    # where U_A[i, r] = sum_j U[i*m+j, r] * A[i,j]
    # and U_B[k, r] = sum_j V[j*p+k, r] * B[j,k]
    
    # Let me just verify with the Strassen factors directly.
    
    # For 2x2:
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = A*B = [[C00, C01], [C10, C11]]
    
    # U_A[i, r] for i=0 (A00, A01) and i=1 (A10, A11)
    # Actually U has shape (n*m, rank) = (4, 7)
    # U[0, r] = coeff for A00, U[1, r] = coeff for A01, U[2, r] = coeff for A10, U[3, r] = coeff for A11
    
    # So U_A for A00 (i=0, j=0): U[0, r]
    # U_A for A01 (i=0, j=1): U[1, r]
    # U_A for A10 (i=1, j=0): U[2, r]
    # U_A for A11 (i=1, j=1): U[3, r]
    
    # Similarly for V:
    # V[0, r] = coeff for B00, V[1, r] = coeff for B01, V[2, r] = coeff for B10, V[3, r] = coeff for B11
    
    # And W:
    # W[0, r] = coeff for C00 (k=0, i=0), W[1, r] = coeff for C01 (k=1, i=0), 
    # W[2, r] = coeff for C10 (k=0, i=1), W[3, r] = coeff for C11 (k=1, i=1)
    
    # The decomposition should satisfy:
    # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
    
    # Wait, that would make it cubic. Let me reconsider.
    
    # I think the correct form is that the tensor T[i,k,j] for matrix mult satisfies:
    # T[i,k,j] = delta(i,j,k) meaning T[i,k,j] = 1 if the "index" works out, else 0
    # For matrix mult: C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor has T[i,j,k] = 1 (it's the permutation tensor for the middle index)
    
    # A rank-r decomposition: T[i,j,k] = sum_{r=1}^r U[i,r] V[j,r] W[k,r]
    # But that's for a 3-index tensor where all indices are "free".
    
    # For matrix mult, the tensor is T[i,j,k] where C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] should satisfy: sum_j T[i,j,k] * A[i,j] * B[j,k] = C[i,k]
    # This means T[i,j,k] = delta_{j,j} = 1 for all i,j,k? No, that's not right.
    
    # Actually, for matrix mult C = A*B:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The "tensor" here is just the identity in the middle index.
    # A tensor decomposition writes:
    # C[i,k] = sum_r (sum_j U[i,j,r] A[i,j]) * (sum_j V[j,k,r] B[j,k])
    # But this is still not quite standard.
    
    # Let me look at it differently. The standard bilinear decomposition of matrix mult:
    # C = sum_{r=1}^r lambda_r (a_r * b_r)
    # where a_r = sum_{i,j} U_r[i,j] A[i,j], b_r = sum_{j,k} V_r[j,k] B[j,k]
    # and C = sum_r lambda_r * outer(a_r, b_r)
    # i.e., C[i,k] = sum_r lambda_r * a_r[i] * b_r[k]
    
    # In the task's notation:
    # U has shape (n*m, rank), so U[i*m+j, r] = U_r[i,j]
    # V has shape (m*p, rank), so V[j*p+k, r] = V_r[j,k]
    # W has shape (p*n, rank), so W[k*n+i, r] = lambda_r for the (i,k) output
    
    # So: C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    
    # Let me verify this with Strassen.
    
    # For 2x2, let's check if my U_strassen, V_strassen, W_strassen work.
    
    # Actually, I realize I had the indices slightly off. Let me redefine.
    
    # U[i*m+j, r] for 2x2: U[0,r]=A00_coeff, U[1,r]=A01_coeff, U[2,r]=A10_coeff, U[3,r]=A11_coeff
    # V[j*p+k, r] for 2x2: V[0,r]=B00_coeff, V[1,r]=B01_coeff, V[2,r]=B10_coeff, V[3,r]=B11_coeff
    # W[k*n+i, r] for 2x2: W[0,r]=C00_coeff, W[1,r]=C01_coeff, W[2,r]=C10_coeff, W[3,r]=C11_coeff
    
    # So my U_strassen above has rows for A00,A01,A10,A11 which matches.
    # V_strassen has rows for B00,B01,B10,B11 which matches.
    # W_strassen has rows for C00,C01,C10,C11 which matches.
    
    # Let me verify the Strassen decomposition numerically.
    
    def verify_strassen_2x2():
        """Verify the 2x2 Strassen decomposition."""
        rng = np.random.default_rng(42)
        for trial in range(20):
            A = rng.integers(-3, 4, size=(2, 2)).astype(object)
            B = rng.integers(-3, 4, size=(2, 2)).astype(object)
            C_expected = A @ B
            
            C_calc = np.zeros((2, 2), dtype=object)
            for r in range(7):
                a_comb = [U_strassen[i, r] * A[i//2, i%2] for i in range(4)]
                # a_comb[0] = U[0,r]*A00, a_comb[1] = U[1,r]*A01, etc.
                # But we need sum_j U[i*m+j, r] * A[i,j] for each i
                # For i=0: U[0,r]*A[0,0] + U[1,r]*A[0,1]
                # For i=1: U[2,r]*A[1,0] + U[3,r]*A[1,1]
                a0 = U_strassen[0, r] * A[0, 0] + U_strassen[1, r] * A[0, 1]
                a1 = U_strassen[2, r] * A[1, 0] + U_strassen[3, r] * A[1, 1]
                
                b0 = V_strassen[0, r] * B[0, 0] + V_strassen[1, r] * B[0, 1]
                b1 = V_strassen[2, r] * B[1, 0] + V_strassen[3, r] * B[1, 1]
                
                # C[i,k] += W[k*2+i, r] * a_i * b_k
                C_calc[0, 0] += W_strassen[0, r] * a0 * b0
                C_calc[0, 1] += W_strassen[1, r] * a0 * b1
                C_calc[1, 0] += W_strassen[2, r] * a1 * b0
                C_calc[1, 1] += W_strassen[3, r] * a1 * b1
            
            if not all(C_calc[i, k] == C_expected[i, k] for i in range(2) for k in range(2)):
                print(f"Trial {trial} failed")
                print(f"A = {A.tolist()}")
                print(f"B = {B.tolist()}")
                print(f"C_expected = {C_expected.tolist()}")
                print(f"C_calc = {C_calc.tolist()}")
                return False
        return True
    
    # Let me check if the Strassen factors are correct.
    # The issue might be in my factor definitions.
    
    # Standard Strassen for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For the bilinear form:
    # a_r = linear combo of A entries
    # b_r = linear combo of B entries
    # C[i,k] = sum_r lambda_r(i,k) * a_r * b_r
    
    # So for each r, we need:
    # a_r = sum_{i,j} U[i,j,r] A[i,j]
    # b_r = sum_{j,k} V[j,k,r] B[j,k]
    # lambda_r(i,k) = W[k*n+i, r]
    
    # Let me redefine more carefully.
    
    # r=0 (M1): a0 = A00+A11, b0 = B00+B11
    # U[:,0] = [1, 0, 0, 1] (A00=1, A01=0, A10=0, A11=1)
    # V[:,0] = [1, 0, 0, 1] (B00=1, B01=0, B10=0, B11=1)
    # W[:,0] = [1, 0, 0, 1] (C00=1, C01=0, C10=0, C11=1)
    
    # r=1 (M2): a1 = A10+A11, b1 = B00
    # U[:,1] = [0, 0, 1, 1]
    # V[:,1] = [1, 0, 0, 0]
    # W[:,1] = [0, 0, 1, -1] (C10=1, C11=-1)
    
    # r=2 (M3): a2 = A00, b2 = B01-B11
    # U[:,2] = [1, 0, 0, 0]
    # V[:,2] = [0, 1, 0, -1]
    # W[:,2] = [0, 1, 0, 1] (C01=1, C11=1)
    
    # r=3 (M4): a3 = A11, b3 = B10-B00
    # U[:,3] = [0, 0, 0, 1]
    # V[:,3] = [-1, 0, 1, 0]
    # W[:,3] = [1, 0, 1, 0] (C00=1, C10=1)
    
    # r=4 (M5): a4 = A00+A01, b4 = B11
    # U[:,4] = [1, 1, 0, 0]
    # V[:,4] = [0, 0, 0, 1]
    # W[:,4] = [-1, 1, 0, 0] (C00=-1, C01=1)
    
    # r=5 (M6): a5 = A10-A00, b5 = B00+B01
    # U[:,5] = [-1, 0, 1, 0]
    # V[:,5] = [1, 1, 0, 0]
    # W[:,5] = [0, 0, 0, 1] (C11=1)
    
    # r=6 (M7): a6 = A01-A11, b6 = B10+B11
    # U[:,6] = [0, 1, 0, -1]
    # V[:,6] = [0, 0, 1, 1]
    # W[:,6] = [1, 0, 0, 0] (C00=1)
    
    # Let me create the correct matrices.
    
    U_s = np.array([
        [1, 0, 1, 0, 1, -1, 0],   # A00
        [0, 0, 0, 0, 1, 0, 1],    # A01
        [0, 1, 0, 0, 0, 1, 0],    # A10
        [1, 1, 0, 1, 0, 0, -1],   # A11
    ], dtype=object)
    
    V_s = np.array([
        [1, 1, 0, -1, 0, 1, 0],   # B00
        [0, 0, 1, 0, 0, 1, 0],    # B01
        [0, 0, 0, 1, 0, 0, 1],    # B10
        [1, 0, -1, 0, 1, 0, 1],   # B11
    ], dtype=object)
    
    W_s = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [0, 1, 0, 1, 0, 0, 0],    # C10
        [1, -1, 1, 0, 0, 1, 0],   # C11
    ], dtype=object)
    
    # Verify
    def verify_2x2(U_s, V_s, W_s):
        rng = np.random.default_rng(42)
        for trial in range(50):
            A = rng.integers(-3, 4, size=(2, 2)).astype(object)
            B = rng.integers(-3, 4, size=(2, 2)).astype(object)
            C_expected = A @ B
            
            C_calc = np.zeros((2, 2), dtype=object)
            for r in range(7):
                a0 = U_s[0, r] * A[0, 0] + U_s[1, r] * A[0, 1]
                a1 = U_s[2, r] * A[1, 0] + U_s[3, r] * A[1, 1]
                b0 = V_s[0, r] * B[0, 0] + V_s[1, r] * B[0, 1]
                b1 = V_s[2, r] * B[1, 0] + V_s[3, r] * B[1, 1]
                
                C_calc[0, 0] += W_s[0, r] * a0 * b0
                C_calc[0, 1] += W_s[1, r] * a0 * b1
                C_calc[1, 0] += W_s[2, r] * a1 * b0
                C_calc[1, 1] += W_s[3, r] * a1 * b1
            
            if not all(C_calc[i, k] == C_expected[i, k] for i in range(2) for k in range(2)):
                return False
        return True
    
    # Check
    if not verify_2x2(U_s, V_s, W_s):
        raise AssertionError("2x2 Strassen verification failed")
    
    # Now build the 4x4 decomposition using the Kronecker product.
    # The 4x4 matrix multiplication tensor can be decomposed using the
    # tensor product of the 2x2 tensor with itself.
    
    # For 4x4 matrices viewed as 2x2 block matrices with 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # Similarly for B and C.
    
    # Strassen gives: C_block[i,j] = sum_r lambda_r(i,j) * a_block_r * b_block_r
    # where a_block_r is a 2x2 matrix (linear combo of A blocks)
    # and b_block_r is a 2x2 matrix (linear combo of B blocks)
    
    # Each block product a_block_r * b_block_r is a 2x2 matrix product,
    # which can be decomposed using the 2x2 Strassen decomposition.
    
    # So the total number of scalar multiplications is 7 * 7 = 49.
    
    # The decomposition:
    # C[i,k] = sum_{r1=0}^6 sum_{r2=0}^6 lambda(r1)(i_block, k_block) * 
    #           (sum_{j1} U_s[i_block*2 + j1_block, r1] * A[i_block*2+j1_block, ...]) *
    #           ...
    
    # This is getting complex. Let me think of it as a Kronecker product of tensors.
    
    # The 2x2 matrix mult tensor T2[i,j,k] has rank 7.
    # The 4x4 matrix mult tensor T4 can be seen as T2 ⊗ T2 in some sense.
    
    # Actually, the 4x4 matrix mult tensor is NOT simply the Kronecker product of two 2x2 tensors.
    # It's more subtle because the block structure involves Strassen's algebraic identities.
    
    # Let me construct it explicitly.
    
    # U for 4x4 has shape (16, 49)
    # V for 4x4 has shape (16, 49)
    # W for 4x4 has shape (16, 49)
    
    # Index for U: i*m + j = i*4 + j, where i,j in 0..3
    # Index for V: j*p + k = j*4 + k, where j,k in 0..3
    # Index for W: k*n + i = k*4 + i, where k,i in 0..3
    
    U = np.zeros((16, 49), dtype=object)
    V = np.zeros((16, 49), dtype=object)
    W = np.zeros((16, 49), dtype=object)
    
    col = 0
    for r1 in range(7):
        for r2 in range(7):
            # This corresponds to the outer Strassen term r1 and inner Strassen term r2.
            
            # The outer term r1 gives:
            # a_block_r1 = linear combo of A blocks using U_s[:, r1]
            # b_block_r1 = linear combo of B blocks using V_s[:, r1]
            # lambda_r1(i_block, k_block) = W_s[k_block*2 + i_block, r1]
            
            # a_block_r1 is a 2x2 matrix:
            # a_block_r1[i_block, j_block] = sum_{j_block'} U_s[i_block*2 + j_block', r1] * A[i_block*2+j_block', ...]
            # Wait, I need to be more careful.
            
            # U_s has shape (4, 7) where U_s[idx, r] is the coefficient for A[idx//2, idx%2] in the r-th term.
            # For the outer Strassen, idx corresponds to block indices.
            
            # Let me define:
            # For outer term r1, the "a" factor is a 2x2 matrix of linear combinations of A entries.
            # a_r1[i_block, j_block] = U_s[i_block*2 + j_block, r1] * A[i_block*2 + j_block, ...]
            # But A[i_block*2 + j_block] is a 2x2 block, so we need to expand.
            
            # Actually, I think the cleanest way is:
            # The 4x4 matrix A can be indexed as A[bi*2 + bi2, bj*2 + bj2] where bi,bj are block indices (0,1)
            # and bi2,bj2 are local indices within the block (0,1).
            
            # For the outer Strassen term r1, we form:
            # a_outer_r1[bi, bj] = sum over A entries in block (bi, bj) with coefficients from U_s
            
            # But U_s only has 4 entries (for 2x2), corresponding to block-level entries.
            # So U_s[bi*2+bj, r1] is the coefficient for the entire block A[bi, bj].
            
            # Wait, that's not right either. U_s has shape (4,7) where the 4 rows correspond to
            # A00, A01, A10, A11 (the 4 entries of a 2x2 matrix).
            
            # For the 4x4 case, when we apply Strassen at the block level:
            # We have 4 blocks A00, A01, A10, A11, each of which is 2x2.
            # The outer Strassen uses U_s to form linear combinations of these blocks.
            # So a_outer_r1 is a 2x2 matrix of 2x2 blocks.
            # a_outer_r1[bi, bj] = U_s[bi*2+bj, r1] * A_block[bi, bj]
            # where A_block[bi, bj] is the 2x2 block at position (bi, bj).
            
            # Similarly, b_outer_r1[bi, bj] = V_s[bi*2+bj, r1] * B_block[bi, bj]
            
            # Then the product a_outer_r1 * b_outer_r1 is a 2x2 matrix of 2x2 blocks,
            # and each block product is a 2x2 matrix multiplication.
            
            # For each output block position (ci, ck), the contribution is:
            # C_block[ci, ck] += W_s[ck*2+ci, r1] * (a_outer_r1 * b_outer_r1)[ci, ck]
            
            # And (a_outer_r1 * b_outer_r1)[ci, ck] = sum_{bj} a_outer_r1[ci, bj] * b_outer_r1[bj, ck]
            # = sum_{bj} U_s[ci*2+bj, r1] * A_block[ci, bj] * V_s[bj*2+ck, r1] * B_block[bj, ck]
            
            # Each A_block[ci, bj] * B_block[bj, ck] is a 2x2 matrix product,
            # which we decompose using the inner Strassen.
            
            # So for the inner Strassen term r2:
            # The contribution to C_block[ci, ck] from this term is:
            # W_s[ck*2+ci, r1] * sum_{bj} U_s[ci*2+bj, r1] * V_s[bj*2+ck, r1] *
            #   (inner_strassen_contribution for A_block[ci,bj] * B_block[bj,ck] using r2)
            
            # The inner Strassen contribution for a 2x2 product P = X*Y is:
            # P[i,k] = sum_{r2} W_s[k*2+i, r2] * a_inner_r2 * b_inner_r2
            # where a_inner_r2 = sum_{j} U_s[i*2+j, r2] * X[i,j]
            # and b_inner_r2 = sum_{j} V_s[j*2+k, r2] * Y[j,k]
            
            # So for the 4x4 case, for fixed (r1, r2, ci, ck):
            # The contribution to C at position (ci*2+i2, ck*2+k2) is:
            # W_s[ck*2+ci, r1] * W_s[k2*2+i2, r2] *
            #   (sum_{bj, j2} U_s[ci*2+bj, r1] * U_s[i2*2+j2, r2] * A[ci*2+i2, bj*2+j2]) *
            #   (sum_{j2} V_s[bj*2+ck, r1] * V_s[j2*2+k2, r2] * B[bj*2+j2, ck*2+k2])
            
            # Wait, I'm getting confused. Let me be very explicit.
            
            # A has entries A[ai, aj] for ai, aj in 0..3.
            # Block indices: ai = bi*2 + ai2, aj = bj*2 + aj2
            # Similarly for B: B[bj, bk] where bj = bj_block*2 + bj2, bk = ck_block*2 + ck2
            # C: C[ai, ak] where ai = ci*2 + ai2, ak = ck*2 + ak2
            
            # For fixed (r1, r2):
            # U[ai*4+aj, col] should be the coefficient such that:
            # sum_{aj} U[ai*4+aj, col] * A[ai, aj] gives the "a" factor for this term.
            
            # From the derivation:
            # The a factor for output position (ci, ai2, ck, ak2) with terms (r1, r2) is:
            # sum_{bj, bj2} U_s[ci*2+bj, r1] * U_s[ai2*2+bj2, r2] * A[ci*2+ai2, bj*2+bj2]
            
            # So U[ci*2+ai2, bj*2+bj2, col] should contribute U_s[ci*2+bj, r1] * U_s[ai2*2+bj2, r2]
            # But U is indexed by (ai*4+aj, col), so:
            # U[ai*4+aj, col] += U_s[ai//2 * 2 + aj//2, r1] * U_s[ai%2 * 2 + aj%2, r2]
            # where ai//2 = ci, aj//2 = bj, ai%2 = ai2, aj%2 = aj2
            
            # Similarly for V:
            # V[bj*4+bk, col] += V_s[bj//2 * 2 + bk//2, r1] * V_s[bj%2 * 2 + bk%2, r2]
            # where bj//2 = bj_block, bk//2 = ck_block, bj%2 = bj2, bk%2 = ck2
            
            # And W:
            # W[ak*4+ai, col] += W_s[ak//2 * 2 + ai//2, r1] * W_s[ak%2 * 2 + ai%2, r2]
            # where ak//2 = ck, ai//2 = ci, ak%2 = ak2, ai%2 = ai2
            
            # Let me implement this.
            
            for ai in range(4):
                for aj in range(4):
                    ci = ai // 2
                    ai2 = ai % 2
                    bj = aj // 2
                    aj2 = aj % 2
                    U[ai*4+aj, col] += U_s[ci*2+bj, r1] * U_s[ai2*2+aj2, r2]
            
            for bj in range(4):
                for bk in range(4):
                    bj_block = bj // 2
                    bj2 = bj % 2
                    ck_block = bk // 2
                    ck2 = bk % 2
                    V[bj*4+bk, col] += V_s[bj_block*2+ck_block, r1] * V_s[bj2*2+ck2, r2]
            
            for ai in range(4):
                for ak in range(4):
                    ci = ai // 2
                    ai2 = ai % 2
                    ck = ak // 2
                    ak2 = ak % 2
                    W[ak*4+ai, col] += W_s[ck*2+ci, r1] * W_s[ak2*2+ai2, r2]
            
            col += 1
    
    assert col == 49, f"Expected 49, got {col}"
    
    # Verify the 4x4 decomposition
    rng = np.random.default_rng(0)
    for trial in range(20):
        A = rng.integers(-3, 4, size=(4, 4)).astype(object)
        B = rng.integers(-3, 4, size=(4, 4)).astype(object)
        C_expected = A @ B
        
        C_calc = np.zeros((4, 4), dtype=object)
        for r in range(49):
            # a_i = sum_j U[i*4+j, r] * A[i,j]
            # b_k = sum_j V[j*4+k, r] * B[j,k]
            # C[i,k] += W[k*4+i, r] * a_i * b_k
            
            a = np.zeros(4, dtype=object)
            for i in range(4):
                for j in range(4):
                    a[i] += U[i*4+j, r] * A[i, j]
            
            b = np.zeros(4, dtype=object)
            for k in range(4):
                for j in range(4):
                    b[k] += V[j*4+k, r] * B[j, k]
            
            for i in range(4):
                for k in range(4):
                    C_calc[i, k] += W[k*4+i, r] * a[i] * b[k]
        
        if not all(C_calc[i, k] == C_expected[i, k] for i in range(4) for k in range(4)):
            raise AssertionError(f"4x4 verification failed at trial {trial}")
    
    if time.time() > _DEADLINE - _MARGIN:
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
