"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x10 matrix.
    
    Uses Strassen's algorithm for each 2x2x2 sub-block. Since the right matrix is 2x10,
    we can view it as 5 independent 2x2 blocks. Each 2x2x2 multiplication can be done
    with rank 7 (Strassen) instead of rank 8 (schoolbook).
    
    Total rank = 5 * 7 = 35.
    
    Interface: U (n*m, rank) = (4, 35), V (m*p, rank) = (20, 35), W (p*n, rank) = (20, 35)
    with entries that are integers or half-integers.
    """
    n, m, p = DIMENSIONS  # 2, 2, 10
    
    # Strassen's 7 products for 2x2 * 2x2:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B10)
    # M7 = (A01 - A11) * (B01 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # The tensor for 2x2*2x2 has entries T[i,j,k] = delta_{i,k} for the appropriate slice
    # Actually, the matrix multiplication tensor T_{ijk} where C_ik = sum_j A_ij B_jk
    # For 2x2*2x2: T is a 2x2x2 tensor with T[i,j,k] = 1 if the output C[i,k] gets A[i,j]*B[j,k]
    
    # Let's think about this more carefully.
    # We want to decompose the tensor T such that T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # where the multiplication is C[i,k] = sum_j A[i,j] B[j,k] = sum_r (sum_j U[i,r] A[i,j]) ... 
    # 
    # Actually, the standard tensor decomposition for matrix multiplication:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # This can be written as: C[i,k] = sum_r (sum_j U[i,j,r] A[i,j]) * (sum_k W[k,j,r] B[j,k]) ... no
    #
    # Let me re-derive. The tensor T for multiplying (n x m) by (m x p) to get (n x p) is:
    # T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 if this term A[i,j]*B[j,k] contributes to C[i,k]
    # Actually T is defined such that the multiplication tensor satisfies:
    # C[i,k] = sum_{j} A[i,j] B[j,k]
    # In tensor form: T_{i,j,k} = delta_{i,k} for the right indices... 
    #
    # The correct formulation: We have a 3-tensor T with indices (i, j, k) where:
    # - i indexes rows of A and C (size n)
    # - j indexes columns of A / rows of B (size m)  
    # - k indexes columns of B and C (size p)
    # 
    # T[i, j, k] = 1 if and only if the product A[i,j] * B[j,k] appears in C[i,k]
    # This is always true, so T[i,j,k] = 1 for all valid i,j,k? No, that's not right either.
    #
    # Let me think again. The matrix multiplication tensor is a 3-way tensor of order (n, m, p).
    # The entry T[i,j,k] is 1 if the (i,j) entry of A and the (j,k) entry of B multiply to contribute
    # to the (i,k) entry of C. Since every A[i,j]*B[j,k] contributes to C[i,k], we have T[i,j,k] = 1 
    # for all i, j, k. But that's just the all-ones tensor, which doesn't seem right.
    #
    # I think the correct way to view it: The multiplication map is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # This is a trilinear form. The tensor T has entries T[i,k,j] = delta_{i,k}... no.
    #
    # Standard convention: T is a tensor of shape (n, p, m) or (n, m, p).
    # Using the convention from the problem: W has shape (p*n, rank) and is indexed W[k*n+i, r].
    # So the output C[i,k] is indexed as k*n + i in W.
    # 
    # The tensor T has shape (n*m, m*p, p*n) in the flattened sense, but really it's a 3-tensor
    # with modes (n, m, p) where:
    # - Mode 1 (size n): row index of A and C
    # - Mode 2 (size m): column index of A / row index of B
    # - Mode 3 (size p): column index of B and C
    #
    # T[i, j, k] = 1 for all i, j, k? That can't be right because then the rank would be 1.
    #
    # I think I'm confusing myself. Let me look at this differently.
    #
    # The matrix multiplication tensor T_{M_{n,m,p}} is defined such that:
    # <T, A ⊗ B ⊗ C^*> = sum_{i,j,k} A[i,j] B[j,k] C[k,i]
    # 
    # So T[i, j, k] = 1 if and only if A[i,j] * B[j,k] * C[k,i] appears in the sum.
    # This means T[i,j,k] = 1 for all i,j,k. But again, that gives rank 1.
    #
    # Wait, I think the issue is that the tensor is not all ones. Let me reconsider.
    #
    # Actually, I believe the tensor is:
    # T[i, j, k] = 1 for the "right" combination, but the key is that the three indices are NOT
    # all free. The constraint is that the middle index j must match between A and B.
    #
    # I think the correct statement is: T is a tensor of shape (n, m, p) where
    # T[i, j, k] = 1 if the term A[i,j] * B[j,k] contributes to C[i,k].
    # Since it always does, T[i,j,k] = 1 for all i,j,k.
    #
    # But that contradicts the fact that schoolbook has rank n*m*p.
    #
    # Let me look at this from the decomposition perspective.
    # Schoolbook: for each (i,j,k), we have a term A[i,j] * B[j,k] -> C[i,k].
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1 for r corresponding to (i,j,k).
    #
    # So the tensor T has entries:
    # T_flat[i*m+j, j*p+k, k*n+i] = 1
    #
    # In 3D form, T[i, j, k] where the first index is from U (size n*m), second from V (size m*p),
    # third from W (size p*n). But these are flattened indices.
    #
    # The 3D tensor T has shape (n*m, m*p, p*n) and T[a, b, c] = 1 iff:
    # a = i*m + j, b = j*p + k, c = k*n + i for some valid i, j, k.
    #
    # So T is a sparse tensor with n*m*p non-zero entries.
    #
    # Now, for Strassen on 2x2*2x2:
    # n=m=p=2, so T has shape (4, 4, 4) with 8 non-zero entries.
    #
    # The 8 non-zero entries are at:
    # (i,j,k) -> (i*2+j, j*2+k, k*2+i)
    # (0,0,0) -> (0, 0, 0)
    # (0,0,1) -> (0, 1, 2)
    # (0,1,0) -> (1, 0, 1)
    # (0,1,1) -> (1, 1, 3)
    # (1,0,0) -> (2, 0, 0)
    # (1,0,1) -> (2, 1, 2)
    # (1,1,0) -> (3, 0, 1)
    # (1,1,1) -> (3, 1, 3)
    #
    # Strassen decomposes this into 7 terms.
    #
    # For the 2x2*2x10 case, we have 5 independent 2x2*2x2 blocks (columns 0-1, 2-3, 4-5, 6-7, 8-9).
    # Each block can use Strassen with rank 7, giving total rank 35.
    
    # Strassen coefficients for 2x2*2x2:
    # U rows: 0=A00, 1=A10, 2=A11, 3=A01 (wait, let me check the indexing)
    # U index: i*m + j = i*2 + j
    # i=0,j=0 -> 0 (A00)
    # i=0,j=1 -> 1 (A01)
    # i=1,j=0 -> 2 (A10)
    # i=1,j=1 -> 3 (A11)
    #
    # V index: j*p + k = j*2 + k
    # j=0,k=0 -> 0 (B00)
    # j=0,k=1 -> 1 (B01)
    # j=1,k=0 -> 2 (B10)
    # j=1,k=1 -> 3 (B11)
    #
    # W index: k*n + i = k*2 + i
    # k=0,i=0 -> 0 (C00)
    # k=0,i=1 -> 1 (C10)
    # k=1,i=0 -> 2 (C01)
    # k=1,i=1 -> 3 (C11)
    
    # Strassen's 7 products:
    # M1 = (A00+A11)(B00+B11) -> contributes to C00, C11
    # M2 = (A10+A11)B00 -> contributes to C10, C11
    # M3 = A00(B10-B11) -> contributes to C01, C11
    # M4 = A11(B01-B00) -> contributes to C00, C10
    # M5 = (A00+A01)B11 -> contributes to C00, C01
    # M6 = (A10-A00)(B00+B10) -> contributes to C11
    # M7 = (A01-A11)(B01+B11) -> contributes to C00
    
    # Let me verify:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Check C00:
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # M4 = A11(B01-B00) = A11B01 - A11B00
    # -M5 = -(A00+A01)B11 = -A00B11 - A01B11
    # M7 = (A01-A11)(B01+B11) = A01B01 + A01B11 - A11B01 - A11B11
    # Sum: A00B00 + A00B11 + A11B00 + A11B11 + A11B01 - A11B00 - A00B11 - A01B11 + A01B01 + A01B11 - A11B01 - A11B11
    # = A00B00 + A01B01
    # Correct! C00 = A00B00 + A01B01
    
    # Check C01:
    # M3 = A00(B10-B11) = A00B10 - A00B11
    # M5 = (A00+A01)B11 = A00B11 + A01B11
    # Sum: A00B10 + A01B11
    # Correct! C01 = A00B10 + A01B11
    
    # Check C10:
    # M2 = (A10+A11)B00 = A10B00 + A11B00
    # M4 = A11(B01-B00) = A11B01 - A11B00
    # Sum: A10B00 + A11B01
    # Correct! C10 = A10B00 + A11B01
    
    # Check C11:
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # -M2 = -(A10+A11)B00 = -A10B00 - A11B00
    # M3 = A00(B10-B11) = A00B10 - A00B11
    # M6 = (A10-A00)(B00+B10) = A10B00 + A10B10 - A00B00 - A00B10
    # Sum: A00B00 + A00B11 + A11B00 + A11B11 - A10B00 - A11B00 + A00B10 - A00B11 + A10B00 + A10B10 - A00B00 - A00B10
    # = A11B11 + A10B10
    # Correct! C11 = A10B10 + A11B11
    
    # Great, Strassen works. Now let's build the decomposition.
    
    rank = 35
    U = [[0] * rank for _ in range(n * m)]  # 4 x 35
    V = [[0] * rank for _ in range(m * p)]  # 20 x 35
    W = [[0] * rank for _ in range(p * n)]  # 20 x 35
    
    # For each of the 5 blocks (block b handles columns 2b and 2b+1 of B, and columns 2b and 2b+1 of C)
    for b in range(5):
        base_r = b * 7  # starting rank index for this block
        
        # U indices for this block: A00=0, A01=1, A10=2, A11=3 (local)
        # Global U index: same since n*m=4 and we're using all 4 rows
        
        # V indices for this block: B00=2b, B01=2b+1, B10=2b+2, B11=2b+3 (local j*p+k)
        # j=0: k=0 -> 0*p+0=0, k=1 -> 1; j=1: k=0 -> 10, k=1 -> 11
        # So for block b: B00=2b, B01=2b+1, B10=10+2b, B11=11+2b
        
        # W indices for this block: C00=2b, C10=2b+1, C01=10+2b, C11=11+2b (local k*n+i)
        # k=0: i=0 -> 0, i=1 -> 1; k=1: i=0 -> 10, i=1 -> 11
        # So for block b: C00=2b, C10=2b+1, C01=10+2b, C11=11+2b
        
        # Strassen products:
        # M1 = (A00+A11)(B00+B11): U: A00=+1, A11=+1; V: B00=+1, B11=+1; W: C00=+1, C11=+1
        r = base_r
        U[0][r] += 1  # A00
        U[3][r] += 1  # A11
        V[2*b][r] += 1  # B00
        V[11+2*b][r] += 1  # B11
        W[2*b][r] += 1  # C00
        W[11+2*b][r] += 1  # C11
        
        # M2 = (A10+A11)B00: U: A10=+1, A11=+1; V: B00=+1; W: C10=+1, C11=-1
        r = base_r + 1
        U[2][r] += 1  # A10
        U[3][r] += 1  # A11
        V[2*b][r] += 1  # B00
        W[2*b+1][r] += 1  # C10
        W[11+2*b][r] += -1  # C11
        
        # M3 = A00(B10-B11): U: A00=+1; V: B10=+1, B11=-1; W: C01=+1, C11=+1
        r = base_r + 2
        U[0][r] += 1  # A00
        V[10+2*b][r] += 1  # B10
        V[11+2*b][r] += -1  # B11
        W[10+2*b][r] += 1  # C01
        W[11+2*b][r] += 1  # C11
        
        # M4 = A11(B01-B00): U: A11=+1; V: B01=+1, B00=-1; W: C00=+1, C10=+1
        r = base_r + 3
        U[3][r] += 1  # A11
        V[2*b+1][r] += 1  # B01
        V[2*b][r] += -1  # B00
        W[2*b][r] += 1  # C00
        W[2*b+1][r] += 1  # C10
        
        # M5 = (A00+A01)B11: U: A00=+1, A01=+1; V: B11=+1; W: C00=-1, C01=+1
        r = base_r + 4
        U[0][r] += 1  # A00
        U[1][r] += 1  # A01
        V[11+2*b][r] += 1  # B11
        W[2*b][r] += -1  # C00
        W[10+2*b][r] += 1  # C01
        
        # M6 = (A10-A00)(B00+B10): U: A10=+1, A00=-1; V: B00=+1, B10=+1; W: C11=+1
        r = base_r + 5
        U[2][r] += 1  # A10
        U[0][r] += -1  # A00
        V[2*b][r] += 1  # B00
        V[10+2*b][r] += 1  # B10
        W[11+2*b][r] += 1  # C11
        
        # M7 = (A01-A11)(B01+B11): U: A01=+1, A11=-1; V: B01=+1, B11=+1; W: C00=+1
        r = base_r + 6
        U[1][r] += 1  # A01
        U[3][r] += -1  # A11
        V[2*b+1][r] += 1  # B01
        V[11+2*b][r] += 1  # B11
        W[2*b][r] += 1  # C00
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
