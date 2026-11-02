"""Matrix multiplication 2x3x8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Use Strassen decomposition for the 2x2x2 core, and handle the extra
    # row/column with standard multiplication.
    #
    # The 2x3x8 multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # i in {0,1}, j in {0,1,2}, k in {0,...,7}
    #
    # Strategy: Split j into two parts: j=0,1 (2x2 core) and j=2 (scalar column)
    #
    # For the 2x2 core (i in {0,1}, j in {0,1}, k in {0,...,7}):
    #   C_core[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    #   This is a 2x2x8 matrix product.
    #
    # For the extra column j=2:
    #   C_extra[i,k] = A[i,2]*B[2,k]
    #   This is 2x1x8, which has rank 2*1*8 = 16 (but actually rank 2 since
    #   each output depends on one product per i... wait no, it's rank 2*8=16? No.)
    #
    # Actually, let me think more carefully.
    #
    # The full tensor T = sum_{i,j,k} A_{ij} B_{jk} C_{ik}
    # With n=2, m=3, p=8.
    #
    # Let's use a direct approach:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    #
    # For the 2x2 part (j=0,1), we can use Strassen-like decomposition.
    # For the 2x1 part (j=2), it's just 2*8 = 16 scalar products.
    #
    # But actually, let's think about this differently.
    #
    # The 2x2x8 matrix multiplication can be decomposed using Strassen for the
    # 2x2 part applied to each pair of columns. But Strassen gives rank 7 for
    # 2x2x2. For 2x2x8, we can apply Strassen to each group of 2 columns,
    # giving 7 * 4 = 28 terms for the 2x2x8 part.
    #
    # The 2x1x8 part (j=2) requires 2*8 = 16 terms.
    #
    # Total: 28 + 16 = 44 terms.
    #
    # But wait, can we do better? Let me think about the 2x2x8 case more carefully.
    #
    # Actually, for 2x2xp matrix multiplication, the rank is known to be 7*ceil(p/2) 
    # for even p? No, that's not quite right either.
    #
    # Let me just use a simple but effective decomposition:
    #
    # For C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    #
    # Term for (i,j,k): U has 1 at position (i*m+j, r), V has 1 at (j*p+k, r), W has 1 at (k*n+i, r)
    #
    # Schoolbook: rank = 2*3*8 = 48
    #
    # Let's try to reduce this. The key insight is that for fixed i and k,
    # C[i,k] is a bilinear form in A and B. 
    #
    # Let's use the following approach:
    # For the j=0,1 part (2x2x8), use Strassen decomposition applied to each 
    # pair of output columns.
    #
    # Strassen for 2x2x2:
    #   M1 = (A11+A22)(B11+B22)
    #   M2 = (A21+A22)B11
    #   M3 = A11(B21-B22)
    #   M4 = A22(B12-B11)
    #   M5 = (A11+A12)B22
    #   M6 = (A21-A11)(B11+B12)
    #   M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # For 2x2x8, we process columns in pairs: (0,1), (2,3), (4,5), (6,7)
    # Each pair uses 7 Strassen terms, total 28 terms.
    #
    # For j=2 (2x1x8), we need 2*8 = 16 terms (one for each i,k pair).
    #
    # Total rank: 28 + 16 = 44.
    
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Strassen decomposition for 2x2x8 (j=0,1 part)
    # Process output columns in pairs
    for pair_idx in range(4):
        k1 = pair_idx * 2
        k2 = pair_idx * 2 + 1
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,k1]+B[0,k2]+B[1,k1]+B[1,k2])... 
        # Wait, Strassen works on 2x2 matrices. Let me be more careful.
        #
        # For columns k1 and k2, we have a 2x2 "block" of B:
        #   B_block = [[B[0,k1], B[0,k2]],
        #              [B[1,k1], B[1,k2]]]
        #
        # And A is 2x2:
        #   A = [[A[0,0], A[0,1]],
        #        [A[1,0], A[1,1]]]
        #
        # C_block = A * B_block
        #
        # Strassen:
        # M1 = (A00+A11)(B00+B11) where B00=B[0,k1], B11=B[1,k2]... 
        # No wait, the indices are confusing. Let me use standard Strassen notation.
        #
        # A = [[a11, a12], [a21, a22]] = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
        # B = [[b11, b12], [b21, b22]] = [[B[0,k1], B[0,k2]], [B[1,k1], B[1,k2]]]
        # C = [[c11, c12], [c21, c22]] = [[C[0,k1], C[0,k2]], [C[1,k1], C[1,k2]]]
        #
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b21-b22)
        # M4 = a22(b12-b11)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        #
        # c11 = M1 + M4 - M5 + M7
        # c12 = M3 + M5
        # c21 = M2 + M4
        # c22 = M1 - M2 + M3 + M6
        
        # Each Mi is a product of a linear combo of A entries and a linear combo of B entries.
        # In the tensor decomposition, each Mi corresponds to one rank-1 term.
        
        # M1: (A[0,0]+A[1,1]) * (B[0,k1]+B[1,k2])
        U[0*m+0, col] = 1   # A[0,0]
        U[1*m+1, col] = 1   # A[1,1]
        V[0*p+k1, col] = 1  # B[0,k1]
        V[1*p+k2, col] = 1  # B[1,k2]
        W[k1*n+0, col] = 1  # C[0,k1]
        W[k2*n+1, col] = 1  # C[1,k2]
        col += 1
        
        # M2: (A[1,0]+A[1,1]) * B[0,k1]
        U[1*m+0, col] = 1   # A[1,0]
        U[1*m+1, col] = 1   # A[1,1]
        V[0*p+k1, col] = 1  # B[0,k1]
        W[k1*n+1, col] = 1  # C[1,k1]
        col += 1
        
        # M3: A[0,0] * (B[1,k1]-B[0,k2])
        U[0*m+0, col] = 1   # A[0,0]
        V[1*p+k1, col] = 1  # B[1,k1]
        V[0*p+k2, col] = -1 # -B[0,k2]
        W[k2*n+0, col] = 1  # C[0,k2]
        col += 1
        
        # M4: A[1,1] * (B[0,k2]-B[0,k1])
        U[1*m+1, col] = 1   # A[1,1]
        V[0*p+k2, col] = 1  # B[0,k2]
        V[0*p+k1, col] = -1 # -B[0,k1]
        W[k1*n+0, col] = 1  # C[0,k1]
        W[k1*n+1, col] = 1  # C[1,k1]
        col += 1
        
        # M5: (A[0,0]+A[0,1]) * B[1,k2]
        U[0*m+0, col] = 1   # A[0,0]
        U[0*m+1, col] = 1   # A[0,1]
        V[1*p+k2, col] = 1  # B[1,k2]
        W[k1*n+0, col] = -1 # -C[0,k1]
        W[k2*n+0, col] = 1  # C[0,k2]
        col += 1
        
        # M6: (A[1,0]-A[0,0]) * (B[0,k1]+B[0,k2])
        U[1*m+0, col] = 1   # A[1,0]
        U[0*m+0, col] = -1  # -A[0,0]
        V[0*p+k1, col] = 1  # B[0,k1]
        V[0*p+k2, col] = 1  # B[0,k2]
        W[k2*n+1, col] = 1  # C[1,k2]
        col += 1
        
        # M7: (A[0,1]-A[1,1]) * (B[1,k1]+B[1,k2])
        U[0*m+1, col] = 1   # A[0,1]
        U[1*m+1, col] = -1  # -A[1,1]
        V[1*p+k1, col] = 1  # B[1,k1]
        V[1*p+k2, col] = 1  # B[1,k2]
        W[k1*n+0, col] = 1  # C[0,k1]
        col += 1
    
    # Now handle j=2 part: C[i,k] += A[i,2] * B[2,k] for i in {0,1}, k in {0,...,7}
    # This is 2*8 = 16 terms
    for i in range(2):
        for k in range(8):
            U[i*m+2, col] = 1   # A[i,2]
            V[2*p+k, col] = 1   # B[2,k]
            W[k*n+i, col] = 1   # C[i,k]
            col += 1
    
    assert col == rank, f"Column count {col} != rank {rank}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
