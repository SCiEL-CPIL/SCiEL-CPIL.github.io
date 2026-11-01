"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def _strassen_2x2_decomp():
    """
    Strassen's 7-rank decomposition for 2x2 matrix multiplication.
    Returns (U, V, W) each of shape (4, 7) where:
      U indexes input A (row, col) -> index r*2+c
      V indexes input B (row, col) -> index r*2+c  
      W indexes output C (row, col) -> index r*2+c
    Convention: C[r,c] = sum_k U[r*2+c, k] * V[k, ...] ... 
    Actually for tensor: T[i,j,k] = sum_m U[i*2+j, m] * V[m, k*2+?] ...
    
    Let me be precise. The matrix multiplication tensor T of size 2x2x2:
    T[a, b, c] = 1 if a+c = b (0-indexed: T[i,j,k] where output[i,k] += A[i,j]*B[j,k])
    
    Standard indexing: T[i, j, k] = delta_{i+k, j} for 2x2x2? No.
    For multiplying A (2x2) by B (2x2) to get C (2x2):
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Tensor T[i,j,k] where i indexes A rows, j indexes A cols/B rows, k indexes B cols/C cols.
    T[i,j,k] = 1 always (it's a trilinear form).
    
    Factorization: T[i,j,k] = sum_m U[i,j,m] * V[j,k,m] * W[i,k,m]
    
    Strassen's 7 products for 2x2:
    P1 = (A00+A11)*(B00+B11)
    P2 = (A10+A11)*B00
    P3 = A00*(B01-B11)
    P4 = A11*(B10-B00)
    P5 = (A00+A01)*B11
    P6 = (A10-A00)*(B00+B01)
    P7 = (A01-A11)*(B10+B11)
    
    C00 = P1 + P4 - P5 + P7
    C01 = P3 + P5
    C10 = P2 + P4
    C11 = P1 - P2 + P3 + P6
    """
    # U[i,j,m]: coefficient of A[i,j] in product m
    # V[j,k,m]: coefficient of B[j,k] in product m
    # W[i,k,m]: coefficient of product m in C[i,k]
    
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # Index: A[i,j] -> i*2+j, B[j,k] -> j*2+k, C[i,k] -> i*2+k
    
    # P1 = (A00+A11)*(B00+B11)
    U[0*2+0, 0] = 1  # A[0,0]
    U[1*2+1, 0] = 1  # A[1,1]
    V[0*2+0, 0] = 1  # B[0,0]
    V[1*2+1, 0] = 1  # B[1,1]
    W[0*2+0, 0] = 1  # C[0,0]
    W[1*2+1, 0] = 1  # C[1,1]
    
    # P2 = (A10+A11)*B00
    U[1*2+0, 1] = 1  # A[1,0]
    U[1*2+1, 1] = 1  # A[1,1]
    V[0*2+0, 1] = 1  # B[0,0]
    W[1*2+0, 1] = 1  # C[1,0]
    W[1*2+1, 1] = -1 # C[1,1]
    
    # P3 = A00*(B01-B11)
    U[0*2+0, 2] = 1  # A[0,0]
    V[0*2+1, 2] = 1  # B[0,1]
    V[1*2+1, 2] = -1 # B[1,1]
    W[0*2+1, 2] = 1  # C[0,1]
    W[1*2+1, 2] = 1  # C[1,1]
    
    # P4 = A11*(B10-B00)
    U[1*2+1, 3] = 1  # A[1,1]
    V[1*2+0, 3] = 1  # B[1,0]
    V[0*2+0, 3] = -1 # B[0,0]
    W[0*2+0, 3] = 1  # C[0,0]
    W[1*2+0, 3] = 1  # C[1,0]
    
    # P5 = (A00+A01)*B11
    U[0*2+0, 4] = 1  # A[0,0]
    U[0*2+1, 4] = 1  # A[0,1]
    V[1*2+1, 4] = 1  # B[1,1]
    W[0*2+0, 4] = -1 # C[0,0]
    W[0*2+1, 4] = 1  # C[0,1]
    
    # P6 = (A10-A00)*(B00+B01)
    U[1*2+0, 5] = 1  # A[1,0]
    U[0*2+0, 5] = -1 # A[0,0]
    V[0*2+0, 5] = 1  # B[0,0]
    V[0*2+1, 5] = 1  # B[0,1]
    W[1*2+1, 5] = 1  # C[1,1]
    
    # P7 = (A01-A11)*(B10+B11)
    U[0*2+1, 6] = 1  # A[0,1]
    U[1*2+1, 6] = -1 # A[1,1]
    V[1*2+0, 6] = 1  # B[1,0]
    V[1*2+1, 6] = 1  # B[1,1]
    W[0*2+0, 6] = 1  # C[0,0]
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    n, m, p = n, m, p
    rank = U.shape[1]
    
    # Build the full multiplication tensor
    # T[a, b, c] where a in [0, n*m), b in [0, m*p), c in [0, p*n)
    # T[a, b, c] = 1 iff a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
    
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)  
    # W shape: (p*n, rank)
    
    # The decomposition says: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # Check all entries
    for a in range(n * m):
        i, j = divmod(a, m)
        for b in range(m * p):
            j2, k = divmod(b, p)
            expected = 1.0 if j == j2 else 0.0
            for c in range(p * n):
                k2, i2 = divmod(c, n)
                if j == j2 and k == k2 and i == i2:
                    expected = 1.0
                else:
                    expected = 0.0
                
                actual = 0.0
                for r in range(rank):
                    actual += U[a, r] * V[b, r] * W[c, r]
                
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Strategy: Decompose 2x4 * 4x8 into 4 independent 2x2 * 2x2 multiplications
    # A is 2x4, B is 4x8, C is 2x8
    # Split B into 4 blocks of 4x2: B[:, 0:2], B[:, 2:4], B[:, 4:6], B[:, 6:8]
    # Each block multiplication: A (2x4) * B_block (4x2) = C_block (2x2)
    # 
    # For each block s (s=0,1,2,3):
    #   A is 2x4, B_s is 4x2, C_s is 2x2
    #   We can view this as: reshape A into two 2x2 blocks? No, A is 2x4.
    #   Actually: C_s[i,k] = sum_j A[i,j] * B_s[j,k] for j in 0..3
    #   This is a 2x4x2 tensor.
    #
    # Alternative view: Split A's columns into two groups of 2:
    #   A = [A_left | A_right] where A_left is 2x2, A_right is 2x2
    #   B_s = [B_s_top; B_s_bottom] where B_s_top is 2x2, B_s_bottom is 2x2
    #   C_s = A_left * B_s_top + A_right * B_s_bottom
    #
    # So each block multiplication is a sum of two 2x2x2 multiplications.
    # Each 2x2x2 multiplication can be done with rank 7 (Strassen).
    # So each block needs rank 14, and 4 blocks need rank 56.
    #
    # But we can do better! Notice that the structure is:
    #   C_s = A_left * B_s_top + A_right * B_s_bottom
    #
    # Let's think of this differently. The full problem is 2x4x8.
    # 
    # Better approach: Use the fact that 2x4x8 can be decomposed by
    # viewing it as a Kronecker product structure.
    #
    # Actually, let me try a direct block approach:
    # 
    # A (2x4) * B (4x8) = C (2x8)
    # 
    # Split A into 2 blocks of 2x2 horizontally: A = [A0 | A1]
    # Split B into 4 blocks of 2x2 vertically: B = [B00; B01; B10; B11] where each is 2x2
    #   Actually B is 4x8, so split rows into 2 groups of 2, cols into 4 groups of 2
    #   B = [[B00, B01, B02, B03], [B10, B11, B12, B13]] where each Bxy is 2x2
    #
    # C = [A0 | A1] * [[B00, B01, B02, B03], [B10, B11, B12, B13]]
    #   = [A0*B00 + A1*B10, A0*B01 + A1*B11, A0*B02 + A1*B12, A0*B03 + A1*B13]
    #
    # Each output block Cx (2x2) = A0*Bx0 + A1*Bx1
    #
    # For each x (0,1,2,3): Cx = A0*Bx0 + A1*Bx1
    # This is a sum of two 2x2x2 matrix multiplications.
    #
    # Using Strassen for each: rank 7 each, so 14 per x, total 56.
    #
    # Can we do better? Let's try to share Strassen computations.
    #
    # Alternative: View the whole thing as a 2x2x2 tensor in a different way.
    #
    # Let me try: The tensor is T[i, j, k] where i in {0,1}, j in {0,1,2,3}, k in {0,...,7}
    # T[i,j,k] = 1 (it's always 1 for the multiplication tensor)
    #
    # Wait, no. The tensor T[i,j,k] for matrix mult satisfies:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor is T[i,j,k] = 1 for all valid i,j,k (it's the identity in the sense
    # that the trilinear form is sum_{i,j,k} A[i,j]*B[j,k]*C[i,k] with T[i,j,k]=1).
    #
    # The rank of the 2x4x8 multiplication tensor... 
    #
    # Known results: The rank of matrix multiplication M_{n,m,p} satisfies
    # R(M_{n,m,p}) >= n*m*p / something... 
    #
    # For 2x4x8: Lower bound is at least max(2*4, 4*8, 2*8) = 32? No, that's not right.
    # The trivial lower bound is max(n*m, m*p, n*p) but that's not tight.
    # Actually the rank is at least n*m*p / (something) but the standard lower bound
    # from slice rank or other methods gives various bounds.
    #
    # Let me just implement the block Strassen approach with rank 56 and see if I can
    # find a better decomposition.
    #
    # Actually, wait. Let me reconsider the block structure.
    #
    # A is 2x4, B is 4x8. 
    # 
    # Think of A as a 2x2 block matrix with 2x2 blocks? No, A is 2x4, so it's one row
    # of two 2x2 blocks: A = [A00 A01] where A00, A01 are 2x2.
    # 
    # B is 4x8, so it's a 2x4 block matrix of 2x2 blocks:
    # B = [[B00, B01, B02, B03],
    #      [B10, B11, B12, B13]]
    #
    # C = A * B = [A00*B00 + A01*B10, A00*B01 + A01*B11, A00*B02 + A01*B12, A00*B03 + A01*B13]
    #
    # Each column of C is a 2x2 matrix = A00*Bx0 + A01*Bx1
    #
    # So we have 4 independent "sums of two 2x2 multiplications".
    #
    # For each such sum, we need to compute A00*Bx0 + A01*Bx1.
    # This can be viewed as a 2x2x2 tensor product summed twice.
    #
    # Using Strassen: Each 2x2x2 mult is rank 7. Sum of two independent ones is rank 14.
    # Total: 4 * 14 = 56.
    #
    # But maybe we can do better by exploiting structure. Let's try a different decomposition.
    #
    # Alternative: Use the fact that 2x4x8 = 2x2x2 ⊗ something?
    # 2x4x8 = (2x2x2) ⊗ (1x2x4)? That doesn't quite work dimensionally.
    #
    # Let me just go with rank 56 initially and then try to improve.
    #
    # Actually, I realize I should think about this more carefully.
    #
    # The matrix multiplication tensor M_{n,m,p} has entries T[i,j,k] = 1 for all
    # i in [n], j in [m], k in [p]. The rank is the minimum number of rank-1 tensors
    # that sum to T.
    #
    # For M_{2,4,8}:
    # - Schoolbook: rank = 2*4*8 = 64
    # - Can we do better?
    #
    # One approach: M_{2,4,8} can be seen as 4 copies of M_{2,2,2} (by splitting
    # the 8 columns into 4 groups of 2). Each M_{2,2,2} has rank 7 (Strassen).
    # But they're not independent - they share the A matrix.
    #
    # More precisely: M_{2,4,8}[i,j,k] = 1 for all i,j,k.
    # Split k into k = 2*x + y where x in {0,1,2,3}, y in {0,1}.
    # Split j into j = 2*z + w where z in {0,1}, w in {0,1}.
    #
    # Then M_{2,4,8}[i, 2z+w, 2x+y] = 1.
    #
    # This is equivalent to: for each (i, z, x), the 2x2 sub-tensor in (w, y) is all ones.
    # The 2x2x2 all-ones tensor... wait, that's not M_{2,2,2}.
    #
    # M_{2,2,2}[i', j', k'] = 1 for all i', j', k' (it's the multiplication tensor,
    # which is indeed all ones in the standard basis).
    #
    # So M_{2,4,8} decomposes as: for each combination of the "block" indices, we have
    # a M_{2,2,2} tensor. But the block indices are: i (2 choices), z (2 choices), x (4 choices).
    # That's 2*2*4 = 16 blocks, each being M_{2,2,2}.
    #
    # But wait, that's not right either. Let me re-examine.
    #
    # M_{n,m,p}[i,j,k] = 1 for all i in [n], j in [m], k in [p].
    # This is just the all-ones tensor of shape (n,m,p)!
    #
    # The rank of the all-ones tensor of shape (n,m,p) is min(n,m,p) * ... no.
    # The rank of the all-ones tensor J_{n,m,p} is min(n,m,p) if we're talking about
    # the standard tensor rank. Actually no, the rank of J_{n,m,p} is min(n,m,p).
    #
    # Wait, that can't be right for matrix multiplication. Let me reconsider.
    #
    # OH WAIT. I've been confusing myself. The matrix multiplication tensor is NOT
    # the all-ones tensor. Let me reclarify.
    #
    # For matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # The tensor T is defined such that the trilinear form is:
    # sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] * C[i,k]
    #
    # For this to equal the matrix multiplication, we need:
    # sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] * C[i,k] = sum_{i,k} C[i,k] * sum_j A[i,j]*B[j,k]
    #
    # So T[i,j,k] = 1 for all i,j,k. Yes, it IS the all-ones tensor!
    #
    # But then the rank of matrix multiplication would be the rank of the all-ones
    # tensor, which is min(n,m,p) = min(2,4,8) = 2. That's clearly wrong because
    # we know Strassen gives rank 7 for 2x2x2, not rank 2.
    #
    # I'm making an error. Let me look this up properly.
    #
    # The matrix multiplication tensor M_{n,m,p} is an element of (R^n ⊗ R^m ⊗ R^p)^*.
    # It's defined by M_{n,m,p}(A, B, C) = sum_{i,j,k} A[i,j] * B[j,k] * C[k,i]
    # (note the indices on C are swapped).
    #
    # So M_{n,m,p}[i,j,k] = delta_{j, k+i}? No...
    #
    # Let me use the standard definition. The tensor M_{n,m,p} satisfies:
    # M_{n,m,p}(A, B, C) = trace(C^T * A * B) = sum_{i,k} C[k,i] * sum_j A[i,j] * B[j,k]
    #
    # So M_{n,m,p}[i,j,k] = 1 if we're computing sum A[i,j]*B[j,k]*C[k,i],
    # which means the tensor has 1s where the indices are "compatible" in the
    # multiplication sense.
    #
    # In the standard basis: M_{n,m,p} = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k
    # where i in [n], j in [m], k in [p].
    #
    # So it IS the all-ones tensor! And its rank is... 
    #
    # The rank of the all-ones tensor J_{n,m,p} (where every entry is 1) is:
    # - If we're in the standard tensor rank (minimum number of rank-1 tensors),
    #   the rank of J_{n,m,p} is min(n,m,p).
    #
    # But that contradicts Strassen! For 2x2x2, min(2,2,2) = 2, but Strassen says 7.
    #
    # I must be misunderstanding the tensor definition. Let me think again.
    #
    # AH, I see my error. The matrix multiplication tensor is NOT the all-ones tensor
    # in the way I was thinking. Let me use the correct definition.
    #
    # The matrix multiplication tensor M_{n,m,p} is in (R^{n×m} ⊗ R^{m×p} ⊗ R^{n×p}).
    # It's defined by: M_{n,m,p}(A, B, C) = sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i}
    #
    # In terms of basis vectors: if we flatten A to index a = i*m+j, B to b = j*p+k,
    # C to c = k*n+i, then:
    # M_{n,m,p}[a,b,c] = 1 iff a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
    #                    = 0 otherwise
    #
    # So it's NOT all ones! It's 1 only when the indices are "compatible" with
    # matrix multiplication. For n=m=p=2:
    # a = i*2+j, b = j*2+k, c = k*2+i
    # The number of (a,b,c) tuples where this holds is n*m*p = 8.
    # The total number of entries is 4*4*4 = 64.
    # So 8 entries are 1, 56 are 0.
    #
    # THAT's the correct tensor. And its rank for 2x2x2 is 7 (Strassen).
    #
    # OK so now back to 2x4x8:
    # a = i*4+j for i in {0,1}, j in {0,1,2,3}
    # b = j*8+k for j in {0,1,2,3}, k in {0,...,7}
    # c = k*2+i for k in {0,...,7}, i in {0,1}
    #
    # The tensor has 8*2*4*8 = 512 non-zero entries (out of 8*32*16 = 4096 total).
    #
    # Now, how to decompose this efficiently?
    #
    # Key insight: Split the problem into 4 independent sub-problems by
    # grouping the k index.
    #
    # Let k = 2*x + y where x in {0,1,2,3}, y in {0,1}.
    # Then c = (2x+y)*2 + i = 4x + 2y + i.
    #
    # For fixed x, we have a sub-tensor involving:
    #   a = i*4+j (i in {0,1}, j in {0,1,2,3})
    #   b = j*8 + (2x+y) = 8j + 2x + y
    #   c = 4x + 2y + i
    #
    # This sub-tensor for fixed x has the structure of a 2x4x2 multiplication tensor!
    # Because it's multiplying a 2x4 matrix by a 4x2 matrix to get a 2x2 matrix.
    #
    # So M_{2,4,8} = sum_{x=0}^{3} M_{2,4,2}^{(x)} where each M_{2,4,2}^{(x)}
    # is supported on disjoint slices of the tensor.
    #
    # The rank of M_{2,4,2} is what we need. Let's call it r.
    # Then rank(M_{2,4,8}) <= 4*r.
    #
    # What is rank(M_{2,4,2})?
    # M_{2,4,2}: a = i*4+j, b = j*2+k, c = k*2+i
    # i in {0,1}, j in {0,1,2,3}, k in {0,1}
    #
    # This is the tensor for multiplying a 2x4 matrix by a 4x2 matrix.
    #
    # We can further decompose: split j into j = 2*z + w where z in {0,1}, w in {0,1}.
    # Then b = (2z+w)*2 + k = 4z + 2w + k.
    #
    # For fixed z, we get a 2x2x2 sub-tensor:
    #   a = i*4 + (2z+w) = 4i + 2z + w
    #   b = 4z + 2w + k
    #   c = 2k + i
    #
    # So M_{2,4,2} = sum_{z=0}^{1} M_{2,2,2}^{(z)}
    #
    # Each M_{2,2,2} has rank 7 (Strassen).
    # So rank(M_{2,4,2}) <= 2*7 = 14.
    # And rank(M_{2,4,8}) <= 4*14 = 56.
    #
    # Can we do better than 56? The sub-tensors for different (x,z) pairs have
    # disjoint supports in the (a,b,c) space, so their ranks add. We can't
    # share rank-1 components between them unless they have overlapping supports.
    #
    # Actually, they DO share the A matrix entries (the a index). So there might
    # be room for optimization.
    #
    # For now, let me implement the rank-56 decomposition and verify it.
    # Then I'll try to find improvements.
    
    def build_rank_56():
        """Build a rank-56 decomposition using 4 blocks of 2x4x2, each using 2 Strassen 2x2x2."""
        n, m, p = 2, 4, 8
        U = np.zeros((n*m, 56), dtype=int)  # (8, 56)
        V = np.zeros((m*p, 56), dtype=int)  # (32, 56)
        W = np.zeros((p*n, 56), dtype=int)  # (16, 56)
        
        # Strassen decomposition for 2x2x2
        # U_s[i*2+j, r], V_s[j*2+k, r], W_s[k*2+i, r] for r in 0..6
        U_s, V_s, W_s = _strassen_2x2_decomp()
        
        # For each x in 0..3 (group of 2 k-values), and z in 0..1 (group of 2 j-values):
        #   Sub-tensor: 2x2x2 multiplication
        #   i in {0,1}, w in {0,1}, y in {0,1}
        #   a = i*4 + (2*z + w)
        #   b = (2*z + w)*8 + (2*x + y)
        #   c = (2*x + y)*2 + i
        #   
        #   In terms of the 2x2x2 Strassen indices:
        #   local_i = i, local_j = w, local_k = y
        #   U_s[local_i*2 + local_j, r] -> U[a, base + r]
        #   V_s[local_j*2 + local_k, r] -> V[b, base + r]
        #   W_s[local_k*2 + local_i, r] -> W[c, base + r]
        
        col = 0
        for x in range(4):
            for z in range(2):
                for r in range(7):
                    for i in range(2):
                        for w in range(2):
                            a = i * 4 + (2 * z + w)
                            U[a, col] += U_s[i * 2 + w, r]
                        for y in range(2):
                            for w in range(2):
                                b = (2 * z + w) * 8 + (2 * x + y)
                                V[b, col] += V_s[w * 2 + y, r]
                            for i in range(2):
                                c = (2 * x + y) * 2 + i
                                W[c, col] += W_s[y * 2 + i, r]
                    col += 1
        
        return U.tolist(), V.tolist(), W.tolist()
    
    U_list, V_list, W_list = build_rank_56()
    
    # Verify
    U_arr = np.array(U_list)
    V_arr = np.array(V_list)
    W_arr = np.array(W_list)
    
    if not _verify_decomposition(U_arr, V_arr, W_arr, n, m, p):
        # Fallback to schoolbook if verification fails
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
        return {"U": U, "V": V, "W": W}
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Try to find a better decomposition using optimization
    # For now, return the rank-56 solution
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
