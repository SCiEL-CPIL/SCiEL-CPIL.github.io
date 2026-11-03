"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

import time
import sys

# EVOLVE-BLOCK-START
def _to_list(mat):
    """Convert a list of lists / numpy array to a clean list of lists of ints (or floats)."""
    if hasattr(mat, 'tolist'):
        mat = mat.tolist()
    # If entries are numpy scalars, convert to Python
    out = []
    for row in mat:
        out_row = []
        for v in row:
            if hasattr(v, 'item'):
                v = v.item()
            out_row.append(v)
        out.append(out_row)
    return out

def _verify(U, V, W, n, m, p):
    """Check tensor decomposition. Returns True if valid."""
    import numpy as np
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Tensor: T[i,j,k,r] = U[i*m+j, r] * V[j*p+k, r] should equal W[k*n+i, r] for each r
    # Actually the decomposition says: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, the standard formulation: (AB)[i,k] = sum_j A[i,j] B[j,k]
    # Tensor T[i,j,k] with A at position i (row), B at position j (col of A, row of B), k (col of B)
    # Actually let me re-read the interface.
    # U shape (n*m, rank): indexed by (i,j) where i is row of A (n), j is col of A / row of B (m)
    # V shape (m*p, rank): indexed by (j,k) where j is row of B (m), k is col of B (p)
    # W shape (p*n, rank): indexed by (k,i) where k is col of B (p), i is row of A (n)
    # The decomposition: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = (A*B)[i,k]
    # So for the tensor to be correct: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should give
    # 1 if this corresponds to a valid multiplication term, 0 otherwise.
    # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k) since
    # (AB)[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor is: T[i,j,k] = 1 for all i in range(n), j in range(m), k in range(p)
    # And the decomposition requires: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1 for all i,j,k
    # Wait, that's not right either. The tensor entries are:
    # T[i,j,k] = 1 (since A[i,j]*B[j,k] contributes to C[i,k])
    # And the factorization is: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Hmm, but that would require the sum to equal 1 for every (i,j,k).
    
    # Let me check with schoolbook:
    # For each (i,j,k), there's exactly one r where U[i,j,r]=1, V[j,k,r]=1, W[k,i,r]=1
    # So sum_r ... = 1. Yes.
    
    # So we need: for all i in range(n), j in range(m), k in range(p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    # Build the check
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want T[i,j,k] == 1 for all i,j,k
    
    # Efficient check using einsum
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    # Check T == ones
    return np.allclose(T, np.ones((n, m, p)), atol=1e-9)

def _strassen_2x2(A, B):
    """Strassen multiply for 2x2 matrices. Returns 2x2 result.
    A, B are lists of lists or numpy arrays of shape (2,2)."""
    import numpy as np
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    a11, a12, a21, a22 = A[0,0], A[0,1], A[1,0], A[1,1]
    b11, b12, b21, b22 = B[0,0], B[0,1], B[1,0], B[1,1]
    
    m1 = (a11 + a22) * (b11 + b22)
    m2 = (a21 + a22) * b11
    m3 = a11 * (b12 - b22)
    m4 = a22 * (b21 - b11)
    m5 = (a11 + a12) * b22
    m6 = (a21 - a11) * (b11 + b12)
    m7 = (a12 - a22) * (b21 + b22)
    
    c11 = m1 + m4 - m5 + m7
    c12 = m3 + m5
    c21 = m2 + m4
    c22 = m1 - m2 + m3 + m6
    
    return np.array([[c11, c12], [c21, c22]])

def _build_strassen_factors_2x2_2x2_2x2():
    """Build U, V, W for 2x2 * 2x2 = 2x2 using Strassen.
    n=2, m=2, p=2. Rank 7.
    U shape (4,7), V shape (4,7), W shape (4,7)
    """
    import numpy as np
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen for 2x2:
    # C = A*B where A, B are 2x2
    # The 7 products are:
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b12-b22)
    # m4 = a22(b21-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    
    # For the tensor decomposition, we need:
    # For each product r, U[i,j,r] * V[j,k,r] * W[k,i,r] should combine to give the contribution
    # to C[i,k] from that product.
    
    # Actually, let me think about this differently.
    # The standard way: for each of the 7 Strassen products, we have a rank-1 contribution.
    # Product r involves:
    #   U entries: which (i,j) pairs get which coefficients
    #   V entries: which (j,k) pairs get which coefficients  
    #   W entries: which (k,i) pairs get which coefficients
    #   such that sum over r of U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all (i,j,k)
    #   (since the full tensor is all-ones for 2x2*2x2)
    
    # Wait no. The tensor T[i,j,k] = 1 for all i,j,k means the multiplication is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # And the decomposition is: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where T[i,j,k] = 1 for all valid (i,j,k).
    
    # For Strassen, we're grouping the 4 terms of the sum in a clever way.
    # Let me think of it as: the schoolbook has 4 rank-1 terms (one for each j).
    # Strassen combines them into 7 rank-1 terms with different structure.
    
    # Actually, I think the cleanest way is to just construct the factors directly.
    # For 2x2*2x2, the tensor is 2x2x2 = 8 entries, all 1.
    # Schoolbook: rank 4 (one for each j).
    # Strassen: rank 7.
    
    # Let me use a known construction. For Strassen 2x2:
    # The 7 "multiplication" operations each correspond to a rank-1 tensor.
    
    # Product 1: (a11+a22)(b11+b22)
    #   U: (1,1) gets 1, (2,2) gets 1  [using 1-indexed: (i=1,j=1) and (i=2,j=2)]
    #   V: (1,1) gets 1, (2,2) gets 1  [j=1,k=1 and j=2,k=2]
    #   W: (1,1) gets 1, (2,2) gets 1  [k=1,i=1 and k=2,i=2]
    #   Contribution: U[i,j,1]*V[j,k,1]*W[k,i,1]
    #   For (i=1,j=1,k=1): 1*1*1 = 1
    #   For (i=1,j=1,k=2): 1*0*0 = 0 (V[1,2,1]=0)
    #   Hmm, this doesn't work cleanly.
    
    # Let me try a different approach. I'll use the fact that we can verify the result.
    # Let me just try to build it and verify.
    
    # Actually, I recall that the Strassen decomposition for the 2x2 matrix multiplication tensor
    # can be written with the following factors (0-indexed):
    
    # Let me use a known table. For Strassen 2x2*2x2:
    # r=0: U[0,0]=1,U[1,1]=1; V[0,0]=1,V[1,1]=1; W[0,0]=1,W[1,1]=1
    # r=1: U[1,0]=1,U[1,1]=1; V[0,0]=1; W[0,0]=1
    # r=2: U[0,0]=1; V[0,1]=1,V[1,1]=-1; W[1,0]=1
    # r=3: U[1,1]=1; V[1,0]=1,V[0,0]=-1; W[0,1]=1
    # r=4: U[0,0]=1,U[0,1]=1; V[1,1]=1; W[1,1]=1
    # r=5: U[1,0]=1,U[0,0]=-1; V[0,0]=1,V[0,1]=1; W[0,0]=1
    # r=6: U[0,1]=1,U[1,1]=-1; V[1,0]=1,V[1,1]=1; W[1,0]=1
    
    # Let me verify this:
    # T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # Should equal 1 for all (i,j,k) in {0,1}^3.
    
    U[0,0] = 1; U[1,1] = 1; V[0,0] = 1; V[1,1] = 1; W[0,0] = 1; W[1,1] = 1
    U[1,0] = 1; U[1,1] += 1; V[0,0] += 1; W[0,0] += 1
    U[0,0] += 1; V[0,1] = 1; V[1,1] = -1; W[1,0] = 1
    U[1,1] += 1; V[1,0] = 1; V[0,0] = -1; W[0,1] = 1
    U[0,0] += 1; U[0,1] = 1; V[1,1] += 1; W[1,1] += 1
    U[1,0] += 1; U[0,0] = -1; V[0,0] += 1; V[0,1] += 1; W[0,0] += 1
    U[0,1] += 1; U[1,1] = -1; V[1,0] += 1; V[1,1] += 1; W[1,0] += 1
    
    # This is getting messy. Let me just write it out explicitly.
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # r=0: (a00+a11)(b00+b11)
    U[0,0] = 1; U[1,0] = 1
    V[0,0] = 1; V[1,0] = 1
    W[0,0] = 1; W[1,0] = 1
    
    # r=1: (a10+a11)b00
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[0,1] = 1
    
    # r=2: a00(b01-b11)
    U[0,2] = 1
    V[0,2] = 1; V[1,2] = -1
    W[1,2] = 1
    
    # r=3: a11(b10-b00)
    U[3,3] = 1
    V[1,3] = 1; V[0,3] = -1
    W[0,3] = 1
    
    # r=4: (a00+a01)b11
    U[0,4] = 1; U[1,4] = 1
    V[1,4] = 1
    W[1,4] = 1
    
    # r=5: (a10-a00)(b00+b01)
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[0,5] = 1
    
    # r=6: (a01-a11)(b10+b11)
    U[1,6] = 1; U[3,6] = -1
    V[1,6] = 1; V[2,6] = 1
    W[1,6] = 1
    
    # Wait, I'm confusing myself with the indexing.
    # U[i*m+j, r] where i is row of A (0,1), j is col of A (0,1)
    # So U[0] = U[0*2+0] = U[0,0] corresponds to A[0,0]
    # U[1] = U[0*2+1] = U[0,1] corresponds to A[0,1]
    # U[2] = U[1*2+0] = U[1,0] corresponds to A[1,0]
    # U[3] = U[1*2+1] = U[1,1] corresponds to A[1,1]
    
    # V[j*p+k, r] where j is row of B (0,1), k is col of B (0,1)
    # V[0] = V[0*2+0] = V[0,0] corresponds to B[0,0]
    # V[1] = V[0*2+1] = V[0,1] corresponds to B[0,1]
    # V[2] = V[1*2+0] = V[1,0] corresponds to B[1,0]
    # V[3] = V[1*2+1] = V[1,1] corresponds to B[1,1]
    
    # W[k*n+i, r] where k is col of B (0,1), i is row of A (0,1)
    # W[0] = W[0*2+0] = W[0,0] corresponds to (k=0,i=0)
    # W[1] = W[0*2+1] = W[0,1] corresponds to (k=0,i=1)
    # W[2] = W[1*2+0] = W[1,0] corresponds to (k=1,i=0)
    # W[3] = W[1*2+1] = W[1,1] corresponds to (k=1,i=1)
    
    # Strassen products:
    # m1 = (a00+a11)(b00+b11)
    #   U: A[0,0]=1, A[1,1]=1  => U[0]=1, U[3]=1
    #   V: B[0,0]=1, B[1,1]=1  => V[0]=1, V[3]=1
    #   W: C[0,0]=1, C[1,1]=1  => W[0]=1, W[3]=1
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    
    # m2 = (a10+a11)b00
    #   U: A[1,0]=1, A[1,1]=1  => U[2]=1, U[3]=1
    #   V: B[0,0]=1  => V[0]=1
    #   W: C[0,1]=1  => W[1]=1
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[1,1] = 1
    
    # m3 = a00(b01-b11)
    #   U: A[0,0]=1  => U[0]=1
    #   V: B[0,1]=1, B[1,1]=-1  => V[1]=1, V[3]=-1
    #   W: C[1,0]=1  => W[2]=1
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[2,2] = 1
    
    # m4 = a11(b10-b00)
    #   U: A[1,1]=1  => U[3]=1
    #   V: B[1,0]=1, B[0,0]=-1  => V[2]=1, V[0]=-1
    #   W: C[0,0]=1  => W[0]=1
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[0,3] = 1
    
    # m5 = (a00+a01)b11
    #   U: A[0,0]=1, A[0,1]=1  => U[0]=1, U[1]=1
    #   V: B[1,1]=1  => V[3]=1
    #   W: C[1,1]=1  => W[3]=1
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[3,4] = 1
    
    # m6 = (a10-a00)(b00+b01)
    #   U: A[1,0]=1, A[0,0]=-1  => U[2]=1, U[0]=-1
    #   V: B[0,0]=1, B[0,1]=1  => V[0]=1, V[1]=1
    #   W: C[0,0]=1  => W[0]=1
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[0,5] = 1
    
    # m7 = (a01-a11)(b10+b11)
    #   U: A[0,1]=1, A[1,1]=-1  => U[1]=1, U[3]=-1
    #   V: B[1,0]=1, B[1,1]=1  => V[2]=1, V[3]=1
    #   W: C[1,0]=1  => W[2]=1
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[2,6] = 1
    
    return _to_list(U), _to_list(V), _to_list(W)

def _verify_strassen_2x2():
    """Verify the 2x2 Strassen factors."""
    U, V, W = _build_strassen_factors_2x2_2x2_2x2()
    return _verify(U, V, W, 2, 2, 2)

def construct():
    """Construct a rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    # Schoolbook rank = n*m*p = 4*4*6 = 96
    # Try to do better with Strassen-based block decomposition.
    
    # Strategy: Split the 4x4 * 4x6 multiplication into blocks.
    # Option 1: 2x2 blocks for A (2x2 blocks of 2x2), and 2x3 blocks for B.
    #   A is 4x4: 2x2 blocks, each 2x2.
    #   B is 4x6: 2x3 blocks, each 2x2.
    #   C is 4x6: 2x3 blocks, each 2x2.
    #   C[i_block, j_block] = sum_{k_block} A[i_block, k_block] * B[k_block, j_block]
    #   i_block in {0,1}, j_block in {0,1,2}, k_block in {0,1}
    #   For each (i_block, j_block): 2 terms (k_block=0,1), each a 2x2 * 2x2 multiply.
    #   Total block multiplications: 2 * 3 * 2 = 12 block 2x2*2x2 multiplications.
    #   Each 2x2*2x2 can be done with Strassen (rank 7) or schoolbook (rank 4).
    #   With Strassen: 12 * 7 = 84.
    #   With schoolbook: 12 * 4 = 48. Wait, 48 < 96, so even schoolbook blocks would help?
    #   No, because the block structure doesn't reduce the total scalar operations in a simple way.
    #   Actually, for the tensor decomposition, the rank is additive across independent block multiplications
    #   if the blocks are independent.
    
    # Actually, I need to think about this more carefully. The tensor decomposition rank is not simply
    # the sum of the ranks of the block multiplications unless the blocks are truly independent
    # (i.e., no shared indices).
    
    # Let me reconsider. For a block decomposition:
    # C = A * B where A is (nb_a x nb_b) blocks of size (ha x hb), B is (nb_b x nb_c) blocks of size (hb x hc).
    # C is (nb_a x nb_c) blocks of size (ha x hc).
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # Each block product A[i,k] * B[k,j] is a (ha x hb) * (hb x hc) = (ha x hc) multiplication.
    # The tensor for the whole problem is:
    # T[i_a, i_b, j_b, j_c] where i_a is row block of A, i_b is col block of A (row block of B),
    # j_b is row block of B, j_c is col block of B.
    # Actually, let me use the standard tensor formulation.
    
    # The matrix multiplication tensor for (n x m) * (m x p) has:
    # T[i, j, k] for i in range(n), j in range(m), k in range(p)
    # T[i,j,k] = 1 (since (AB)[i,k] = sum_j A[i,j] B[j,k])
    # Factorization: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # For block decomposition with block sizes:
    # n = nb_a * ha, m = nb_b * hb, p = nb_c * hc
    # where ha=hb (inner dimension of A blocks), hb=hc (inner dimension of B blocks)
    # For our case: n=4, m=4, p=6
    # Option: ha=2, hb=2, hc=2, nb_a=2, nb_b=2, nb_c=3
    # A blocks: (2x2) blocks, nb_a=2 rows, nb_b=2 cols => A is 2x2 blocks of 2x2 = 4x4 ✓
    # B blocks: (2x2) blocks, nb_b=2 rows, nb_c=3 cols => B is 2x3 blocks of 2x2 = 4x6 ✓
    # C blocks: (2x2) blocks, nb_a=2 rows, nb_c=3 cols => C is 2x3 blocks of 2x2 = 4x6 ✓
    
    # The tensor T[i,j,k] where i = ia*ha + ii, j = ib*hb + jj, k = ic*hc + kk
    # T[ia*ha+ii, ib*hb+jj, ic*hc+kk] = 1 if ia,ib,ic are valid and the block product is defined
    # Actually, T[i,j,k] = 1 for all valid (i,j,k) regardless of block structure.
    
    # The factorization for the block case:
    # For each block product (ia, ib, ic) (i.e., C[ia,ic] += A[ia,ib] * B[ib,ic]):
    #   The contribution to T[ia*ha+ii, ib*hb+jj, ic*hc+kk] is:
    #   U_block[ia,ib,ii,jj, r_block] * V_block[ib,ic,jj,kk, r_block] * W_block[ic,ia,kk,ii, r_block]
    #   where r_block ranges over the rank of the block multiplication.
    
    # If we use Strassen for each 2x2*2x2 block multiplication (rank 7), and there are
    # nb_a * nb_b * nb_c = 2 * 2 * 3 = 12 block multiplications, the total rank is 12 * 7 = 84.
    
    # But wait, I need to make sure the factors don't overlap in a way that creates conflicts.
    # Since each block multiplication contributes to a disjoint set of (i,j,k) indices
    # (specifically, for fixed (ia, ib, ic), it contributes to (ii, jj, kk) in the block),
    # and different (ia, ib, ic) tuples give disjoint (i,j,k) ranges, the factors are independent.
    # So the total rank is indeed the sum of the ranks of the individual block multiplications.
    
    # Let me construct this.
    ha = hb = hc = 2
    nb_a = n // ha  # 2
    nb_b = m // hb  # 2
    nb_c = p // hc  # 3
    
    # For each block triple (ia, ib, ic), we have a 2x2 * 2x2 = 2x2 multiplication.
    # We use Strassen (rank 7) for each.
    
    # Get the Strassen factors for 2x2*2x2
    U_block, V_block, W_block = _build_strassen_factors_2x2_2x2_2x2()
    # U_block: (4, 7), V_block: (4, 7), W_block: (4, 7)
    
    total_rank = nb_a * nb_b * nb_c * 7
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for ia in range(nb_a):
        for ib in range(nb_b):
            for ic in range(nb_c):
                for r in range(7):
                    c = col_offset + r
                    # U: index is (i*m + j) where i = ia*ha + ii, j = ib*hb + jj
                    for ii in range(ha):
                        for jj in range(hb):
                            i = ia * ha + ii
                            j = ib * hb + jj
                            U[i * m + j][c] = U_block[ii * hb + jj][r]
                    # V: index is (j*p + k) where j = ib*hb + jj, k = ic*hc + kk
                    for jj in range(hb):
                        for kk in range(hc):
                            j = ib * hb + jj
                            k = ic * hc + kk
                            V[j * p + k][c] = V_block[jj * hc + kk][r]
                    # W: index is (k*n + i) where k = ic*hc + kk, i = ia*ha + ii
                    for kk in range(hc):
                        for ii in range(ha):
                            k = ic * hc + kk
                            i = ia * ha + ii
                            W[k * n + i][c] = W_block[kk * ha + ii][r]
                col_offset += 7
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fall back to schoolbook
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
# EVOLVE-BLOCK-END
