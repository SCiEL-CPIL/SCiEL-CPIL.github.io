import numpy as np
import time

def construct():
    n, m, p = 2, 4, 8
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    def verify(U, V, W):
        """Verify tensor decomposition for 2x4x8 matrix multiplication."""
        rank = U.shape[1]
        np.random.seed(123)
        A_test = np.random.rand(n, m)
        B_test = np.random.rand(m, p)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            u_r = U[:, r].reshape(n, m)   # u_r[i,a]
            v_r = V[:, r].reshape(m, p)   # v_r[a,j]
            w_r = W[:, r].reshape(p, n)   # w_r[j,i]
            for i in range(n):
                for j in range(p):
                    ua = np.dot(u_r[i, :], A_test[i, :])
                    vb = np.dot(v_r[:, j], B_test[:, j])
                    C_computed[i, j] += w_r[j, i] * ua * vb
        
        return np.allclose(C_computed, C_expected, atol=1e-10)
    
    # Build the rank-56 Strassen decomposition
    rank_56 = 56
    U = np.zeros((8, rank_56))
    V = np.zeros((32, rank_56))
    W = np.zeros((16, rank_56))
    
    col = 0
    blocks = [
        (0, 0, 0, 0),
        (0, 0, 2, 2),
        (0, 0, 4, 4),
        (0, 0, 6, 6),
        (2, 2, 0, 0),
        (2, 2, 2, 2),
        (2, 2, 4, 4),
        (2, 2, 6, 6),
    ]
    
    for (a_col_off, b_row_off, b_col_off, c_col_off) in blocks:
        a00 = 0*4 + a_col_off
        a01 = 0*4 + a_col_off + 1
        a10 = 1*4 + a_col_off
        a11 = 1*4 + a_col_off + 1
        
        b00 = b_row_off*8 + b_col_off
        b01 = b_row_off*8 + b_col_off + 1
        b10 = (b_row_off+1)*8 + b_col_off
        b11 = (b_row_off+1)*8 + b_col_off + 1
        
        c00 = c_col_off*2 + 0
        c01 = (c_col_off+1)*2 + 0
        c10 = c_col_off*2 + 1
        c11 = (c_col_off+1)*2 + 1
        
        U[a00, col] += 1
        U[a11, col] += 1
        V[b00, col] += 1
        V[b11, col] += 1
        W[c00, col] += 1
        W[c11, col] += 1
        col += 1
        
        U[a10, col] += 1
        U[a11, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1
        W[c11, col] += -1
        col += 1
        
        U[a00, col] += 1
        V[b01, col] += 1
        V[b11, col] += -1
        W[c01, col] += 1
        W[c11, col] += 1
        col += 1
        
        U[a11, col] += 1
        V[b10, col] += 1
        V[b00, col] += -1
        W[c00, col] += 1
        W[c10, col] += 1
        col += 1
        
        U[a00, col] += 1
        U[a01, col] += 1
        V[b11, col] += 1
        W[c00, col] += -1
        W[c01, col] += 1
        col += 1
        
        U[a10, col] += 1
        U[a00, col] += -1
        V[b00, col] += 1
        V[b01, col] += 1
        W[c11, col] += 1
        col += 1
        
        U[a01, col] += 1
        U[a11, col] += -1
        V[b10, col] += 1
        V[b11, col] += 1
        W[c00, col] += 1
        col += 1
    
    assert col == 56
    
    # Verify the 56-rank solution
    if not verify(U, V, W):
        # Try alternative W indexing
        def verify_alt(U, V, W):
            rank = U.shape[1]
            np.random.seed(123)
            A_test = np.random.rand(n, m)
            B_test = np.random.rand(m, p)
            C_expected = A_test @ B_test
            C_computed = np.zeros((n, p))
            for r in range(rank):
                u_r = U[:, r].reshape(n, m)
                v_r = V[:, r].reshape(m, p)
                w_r = W[:, r].reshape(n, p)  # w_r[i,j]
                for i in range(n):
                    for j in range(p):
                        ua = np.dot(u_r[i, :], A_test[i, :])
                        vb = np.dot(v_r[:, j], B_test[:, j])
                        C_computed[i, j] += w_r[i, j] * ua * vb
            return np.allclose(C_computed, C_expected, atol=1e-10)
        
        if verify_alt(U, V, W):
            # W is indexed as W[i*p+j, r], but shape is (p*n, rank) = (16, rank)
            # Since n*p = p*n = 16, we need to figure out the correct indexing
            # The task says W[k*n+i, r] where k is... let me just return and see
            pass
    
    # Try to find a Winograd/FAT-based decomposition
    # For 2x4x8: treat as 2 independent 1x4x8 convolutions
    # Each 1x4x8: C[j] = sum_a A[a] * B[a,j] for j=0..7
    # This is 4x8 = 32 multiplications naively per row
    # Winograd can reduce this
    
    # Winograd minimum convolution for m=4, n=8:
    # We want to compute c[k] = sum_{a=0}^{3} x[a] * y[a+k] for k=0..7
    # where y has length 11 (padded), but we only use y[0..7]
    
    # Actually for our problem: C[i,j] = sum_{a=0}^{3} A[i,a] * B[a,j]
    # For fixed i: 8 dot products of length 4
    # Winograd F(4,4) uses 4 multiplications per 4-output block
    # But we have 8 outputs... 
    # Use F(4,8): evaluate at points, multiply, interpolate
    
    # Let me try a structured approach using polynomial evaluation
    # A[i,a] as coefficients of polynomial P(x) = sum_a A[i,a] * x^a
    # B[a,j] as coefficients... this doesn't directly work for matrix mult
    
    # Alternative: Use the fact that we can split the 4-length dimension
    # A = [A1 | A2] where A1, A2 are 2x2
    # B = [B1; B2] where B1, B2 are 2x8
    # C = A1*B1 + A2*B2
    # Each 2x2x8 has some rank, and we add them
    
    # 2x2x8: C[i,j] = sum_{a=0}^{1} A[i,a]*B[a,j]
    # This is 2*2*8 = 32 mults naively
    # Strassen for 2x2x2 gives rank 7, so 2x2x8 = 4*7 = 28?
    # But we can do better by not splitting B into 2x2 blocks
    
    # Let me try: 2x2x8 can be done with rank = 2*8 = 16 naively (each output needs 2 mults)
    # Or use Winograd: for each row i, C[i,:] = A[i,0]*B[0,:] + A[i,1]*B[1,:]
    # That's 2*8 = 16 multiplications per 2x2x8 block
    # Two such blocks: 2*16 = 32 for the full 2x4x8? No, that's worse than 56.
    
    # Wait, I'm confusing myself. Let me be more careful.
    # 2x4x8: C[i,j] = sum_{a=0}^{3} A[i,a]*B[a,j]
    # Split a into a1 (0,1) and a2 (2,3):
    # C[i,j] = sum_{a1=0}^{1} A[i,a1]*B[a1,j] + sum_{a2=0}^{1} A[i,2+a2]*B[2+a2,j]
    # First term: 2x2x8 (A is 2x2, B is 2x8)
    # Second term: 2x2x8 (A is 2x2, B is 2x8)
    # Total: 2 * rank(2x2x8)
    
    # rank(2x2x8): C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j]
    # Naive: 2*2*8 = 32
    # But we can write: for each j, C[:,j] = A @ B[:,j] where A is 2x2, B[:,j] is 2x1
    # 2x2x1 has rank... it's just 2 mults per j, so 2*8 = 16 total
    # Can we do better? The tensor is sum_{i,a,j} A[i,a]*B[a,j]*C[i,j]
    # For 2x2x1: rank is 2 (just two multiplications)
    # For 2x2x8: we have 8 independent 2x2x1 problems, so rank <= 16
    # Lower bound: n*p = 2*8 = 16, so rank(2x2x8) = 16
    
    # So 2x4x8 via splitting: 2*16 = 32? That can't be right because lower bound is 16.
    # Wait, the lower bound for 2x4x8 is n*p = 2*8 = 16.
    # But the naive count is 2*4*8 = 64.
    # Strassen gives 56.
    # Can we get below 56?
    
    # The splitting gives 2 * rank(2x2x8) = 2*16 = 32. But this assumes the two 2x2x8
    # decompositions are independent, which they are. So rank <= 32?
    # That seems too good. Let me double-check.
    
    # For 2x2x8: U shape (2*2, r) = (4, r), V shape (2*8, r) = (16, r), W shape (8*2, r) = (16, r)
    # C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j]
    # Rank-1 term: U[i,0]=1, U[i,1]=0 for row 0; U[2+i,0]=0, U[2+i,1]=1 for row 1
    # Actually: for the term A[i,0]*B[0,j]: U[i,0]=1, V[0*8+j]=1, W[j*2+i]=1
    # For the term A[i,1]*B[1,j]: U[i,1]=1, V[1*8+j]=1, W[j*2+i]=1
    # So for each j, we need 2 terms. Total: 8*2 = 16 terms.
    # U: for term (j, a): U[a*2+i, r] = 1 for all i. So U has 4 entries per term? No...
    # U shape is (n*m, r) = (4, r). U[i*m+a, r] = U[i*2+a, r]
    # For term computing A[i,0]*B[0,j]: U[i*2+0, r] = 1 for i=0,1. So U[0,r]=1, U[2,r]=1.
    # V[a*p+j, r] = V[0*8+j, r] = 1. So V[j,r] = 1.
    # W[j*n+i, r] = W[j*2+i, r] = 1 for i=0,1. So W[2j,r]=1, W[2j+1,r]=1.
    # 
    # For term computing A[i,1]*B[1,j]: U[i*2+1, r] = 1. So U[1,r]=1, U[3,r]=1.
    # V[1*8+j, r] = 1. So V[8+j,r] = 1.
    # W[j*2+i, r] = 1. So W[2j,r]=1, W[2j+1,r]=1.
    #
    # Total terms: 8 (for a=0) + 8 (for a=1) = 16 terms.
    # This is a valid rank-16 decomposition of 2x2x8.
    
    # Now for 2x4x8, split into two 2x2x8:
    # C[i,j] = sum_{a=0}^{1} A[i,a]*B[a,j] + sum_{a=0}^{1} A[i,2+a]*B[2+a,j]
    # First part: 2x2x8 with A1 = A[:, 0:2], B1 = B[0:2, :]
    # Second part: 2x2x8 with A2 = A[:, 2:4], B2 = B[2:4, :]
    # 
    # For the first part (16 terms):
    #   U1[i*2+a, r] = 1 means U[i*4 + a, r] = 1 for a=0,1 (since m=4 now, U[i*4+a, r])
    #   Wait, U shape for 2x4x8 is (8, r). U[i*4+a, r].
    #   For A[i,a] with a in {0,1}: U[i*4+a, r] = 1
    #   V[a*8+j, r] = 1 for a in {0,1}: V[0*8+j, r] and V[1*8+j, r]
    #   W[j*2+i, r] = 1
    #
    # For the second part (16 terms):
    #   U[i*4+2+a, r] = 1 for a in {0,1}: U[i*4+2, r] and U[i*4+3, r]
    #   V[(2+a)*8+j, r] = 1: V[2*8+j, r] and V[3*8+j, r]
    #   W[j*2+i, r] = 1
    #
    # Total: 32 terms. But wait, can we combine terms from both parts?
    # If for the same (j, i), we have terms from both parts, the W entries are the same.
    # But U and V are different, so they're distinct rank-1 terms.
    # 
    # So rank <= 32? Let me verify this is correct.
    
    # Build the rank-32 decomposition
    U32 = np.zeros((8, 32))
    V32 = np.zeros((32, 32))
    W32 = np.zeros((16, 32))
    
    col = 0
    for j in range(8):
        # Part 1: a in {0, 1}
        for a in range(2):
            for i in range(2):
                U32[i*4 + a, col] = 1
            V32[a*8 + j, col] = 1
            for i in range(2):
                W32[j*2 + i, col] = 1
            col += 1
        # Part 2: a in {2, 3}
        for a in range(2):
            for i in range(2):
                U32[i*4 + 2 + a, col] = 1
            V32[(2+a)*8 + j, col] = 1
            for i in range(2):
                W32[j*2 + i, col] = 1
            col += 1
    
    assert col == 32
    
    if verify(U32, V32, W32):
        # Great! We have a rank-32 solution
        # Try to reduce further using Winograd on the 4-length dimension
        # For each j, C[:,j] = A @ B[:,j] where A is 2x4, B[:,j] is 4x1
        # 2x4x1: C[i] = sum_a A[i,a]*B[a]
        # Rank of 2x4x1: lower bound is max(2,4,1) = 4? Actually it's at most 4.
        # Naive: 2*4 = 8 mults. Can we do 4?
        # C[0] = A[0,0]*B[0] + A[0,1]*B[1] + A[0,2]*B[2] + A[0,3]*B[3]
        # C[1] = A[1,0]*B[0] + A[1,1]*B[1] + A[1,2]*B[2] + A[1,3]*B[3]
        # This is 2x4x1 tensor. Rank <= 4 (use B[0],B[1],B[2],B[3] as the 4 factors)
        # Actually rank = 4: for each a, term is A[:,a] * B[a] * C[:,a]... no
        # Term a: U[i*4+a, r] = A coefficient, V[a*1+0, r] = 1, W[0*2+i, r] = 1
        # That gives 4 terms, each contributing to both C[0] and C[1].
        # But we need different weights for C[0] and C[1]... 
        # Actually: C[i] = sum_a A[i,a]*B[a]. 
        # Term a: U[i*4+a, r] = 1 for all i (but A[i,a] is different for i=0,1)
        # Hmm, U[i*4+a, r] should extract A[i,a]. So U[i*4+a, r] = 1.
        # V[a, r] = 1 (to extract B[a]).
        # W[0*2+i, r] = 1 (to add to C[i]).
        # So term a contributes A[i,a]*B[a] to C[i] for both i=0,1.
        # Total: 4 terms. Rank(2x4x1) = 4.
        # 
        # So for 2x4x8: 8 * rank(2x4x1) = 8*4 = 32. Confirmed.
        # 
        # Can we do better? The lower bound is n*p = 2*8 = 16.
        # rank(2x4x8) is known to be... let me think. 
        # It's at most 32 (as shown). The Strassen-based gives 56 which is worse!
        # So our rank-32 is already better.
        # 
        # Can we get below 32? We'd need to share factors across different j values.
        # The tensor is sum_{i,a,j} A[i,a]*B[a,j]*C[i,j]
        # = sum_j (sum_{i,a} A[i,a]*B[a,j]*C[i,j])
        # Each j gives a 2x4x1 tensor. If we can combine across j, we might reduce rank.
        # But B[a,j] has different j indices, so the V factors are different.
        # Unless we use a transform on j...
        
        # Winograd on the j dimension: use a transform to diagonalize
        # For each (i,a), we have B[a,:] which is a length-8 vector
        # C[i,:] = sum_a A[i,a] * B[a,:]
        # This is a linear combination of rows of B with coefficients from A
        # 
        # Use a 8-point transform T on the j dimension:
        # B' = T @ B (8x8 matrix)
        # C' = T @ C
        # Then C'[i,k] = sum_j T[k,j] * C[i,j] = sum_j T[k,j] * sum_a A[i,a]*B[a,j]
        # = sum_a A[i,a] * sum_j T[k,j]*B[a,j] = sum_a A[i,a] * B'[a,k]
        # So C' = A @ B'
        # This doesn't help directly since it's still the same multiplication.
        # 
        # The Winograd idea is to use a transform that makes the convolution separable.
        # For our problem, it's not a convolution, it's a general matrix product.
        # 
        # I think rank-32 might be close to optimal for this structure.
        # The theoretical lower bound is 16, but achieving that would require
        # each rank-1 term to contribute to all 16 output entries, which seems hard.
        
        return {
            "U": U32.tolist(),
            "V": V32.tolist(),
            "W": W32.tolist()
        }
    
    # If rank-32 doesn't verify, try to fix the indexing
    # Let me try the alternative W convention
    def verify_alt(U, V, W):
        rank = U.shape[1]
        np.random.seed(123)
        A_test = np.random.rand(n, m)
        B_test = np.random.rand(m, p)
        C_expected = A_test @ B_test
        C_computed = np.zeros((n, p))
        for r in range(rank):
            u_r = U[:, r].reshape(n, m)
            v_r = V[:, r].reshape(m, p)
            w_r = W[:, r].reshape(n, p)  # w_r[i,j]
            for i in range(n):
                for j in range(p):
                    ua = np.dot(u_r[i, :], A_test[i, :])
                    vb = np.dot(v_r[:, j], B_test[:, j])
                    C_computed[i, j] += w_r[i, j] * ua * vb
        return np.allclose(C_computed, C_expected, atol=1e-10)
    
    if verify_alt(U32, V32, W32):
        # W is indexed as W[i*p+j, r] but shape is (16, r) = (p*n, r)
        # We need to permute W to match W[k*n+i, r] convention
        # W_alt[i*p+j, r] should become W_new[j*n+i, r]
        W_permuted = np.zeros((16, 32))
        for i in range(2):
            for j in range(8):
                W_permuted[j*2 + i, :] = W32[i*8 + j, :]
        return {
            "U": U32.tolist(),
            "V": V32.tolist(),
            "W": W_permuted.tolist()
        }
    
    # Fallback: return the rank-56 solution
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
