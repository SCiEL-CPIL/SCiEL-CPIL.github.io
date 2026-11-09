"""Optimized matrix multiplication tensor decomposition for 2x5 by 5x6."""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 5, 6]

# Precompute Strassen rank-7 tensors for 2x2 matrix multiplication
# Strassen's algorithm: 7 multiplications for 2x2 matrices
# The 7 rank-1 tensors (u, v, w) where u is 4x1, v is 4x1, w is 4x1
# Indexing: for 2x2, n=m=p=2, so U is 4xrank, V is 4xrank, W is 4xrank
# U[i*2+j], V[j*2+k], W[k*2+i]

def _strassen_tensors():
    """Return the 7 Strassen rank-1 tensors as (u, v, w) tuples.
    Each u, v, w is a list of 4 entries (for 2x2 case).
    """
    # Strassen's 7 products:
    # m1 = (a00 + a11)(b00 + b11)
    # m2 = (a10 + a11)b00
    # m3 = a00(b10 - b11)
    # m4 = a11(b01 - b00)
    # m5 = (a00 + a01)b11
    # m6 = (a10 - a00)(b00 + b01)
    # m7 = (a01 - a11)(b10 + b11)
    #
    # Output:
    # c00 = m1 - m4 + m5 + m7
    # c01 = m3 + m4
    # c10 = m2 + m5
    # c11 = m1 - m2 + m3 + m6
    #
    # For each Strassen product mi, we have:
    # u[i*2+j] = coefficient of a_ij in the left factor
    # v[j*2+k] = coefficient of b_jk in the right factor
    # w[k*2+i] = coefficient in the output c_ik
    
    tensors = []
    
    # m1 = (a00 + a11)(b00 + b11)
    # u: a00=1, a11=1 -> [1, 0, 0, 1]
    # v: b00=1, b11=1 -> [1, 0, 0, 1]
    # w: c00 += 1, c11 += 1 -> [1, 0, 0, 1]
    tensors.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    
    # m2 = (a10 + a11)b00
    # u: a10=1, a11=1 -> [0,0,1,1]
    # v: b00=1 -> [1,0,0,0]
    # w: c10 += 1, c11 -= 1 -> [0,0,1,-1]
    tensors.append(([0,0,1,1], [1,0,0,0], [0,0,1,-1]))
    
    # m3 = a00(b10 - b11)
    # u: a00=1 -> [1,0,0,0]
    # v: b10=1, b11=-1 -> [0,0,1,-1]
    # w: c01 += 1, c11 += 1 -> [0,1,0,1]
    tensors.append(([1,0,0,0], [0,0,1,-1], [0,1,0,1]))
    
    # m4 = a11(b01 - b00)
    # u: a11=1 -> [0,0,0,1]
    # v: b01=1, b00=-1 -> [-1,1,0,0]
    # w: c00 -= 1, c01 += 1 -> [-1,1,0,0]
    tensors.append(([0,0,0,1], [-1,1,0,0], [-1,1,0,0]))
    
    # m5 = (a00 + a01)b11
    # u: a00=1, a01=1 -> [1,1,0,0]
    # v: b11=1 -> [0,0,0,1]
    # w: c00 += 1, c10 += 1 -> [1,0,1,0]
    tensors.append(([1,1,0,0], [0,0,0,1], [1,0,1,0]))
    
    # m6 = (a10 - a00)(b00 + b01)
    # u: a10=1, a00=-1 -> [-1,0,1,0]
    # v: b00=1, b01=1 -> [1,1,0,0]
    # w: c11 += 1 -> [0,0,0,1]
    tensors.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
    
    # m7 = (a01 - a11)(b10 + b11)
    # u: a01=1, a11=-1 -> [0,1,0,-1]
    # v: b10=1, b11=1 -> [0,0,1,1]
    # w: c00 += 1 -> [1,0,0,0]
    tensors.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))
    
    return tensors

def _verify_decomposition(n, m, p, U, V, W):
    """Verify that the decomposition is exact.
    Check that for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _verify_zero_conditions(n, m, p, U, V, W):
    """Verify that all other tensor entries are zero.
    The tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    should be 1 if this corresponds to a valid (i,j,k) triple for matmul, 0 otherwise.
    Actually, the matrix multiplication tensor has T[i,j,k]=1 for all i,j,k in range.
    Wait - the matrix multiplication tensor is defined such that:
    (AB)[i,k] = sum_j A[i,j] B[j,k]
    So the tensor T[i,j,k] = 1 for all valid i,j,k.
    But the decomposition must satisfy:
    T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    And T[i,j,k] should be 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    Actually, re-reading the problem: the tensor for matrix multiplication of (n x m) by (m x p)
    has entries T[i,j,k] = 1 for all i,j,k. The rank is the minimum number of rank-1 tensors
    that sum to this tensor.
    
    So we just need to verify that for all i,j,k, the sum equals 1.
    """
    return _verify_decomposition(n, m, p, U, V, W)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Start with schoolbook
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    # Try Strassen-based construction
    # Split the 5x6 matrix into blocks.
    # A is 2x5, B is 5x6.
    # We can think of A as having columns grouped, and B as having rows grouped.
    # 
    # One approach: partition the 5 columns of A and 5 rows of B into groups,
    # and the 6 columns of B into groups.
    #
    # For Strassen on 2x2: we need to express the multiplication as 7 products.
    # For 2x5 x 5x6, we can try to use block structure.
    #
    # Let's try: A = [A1 | A2 | A3] where A1 is 2x2, A2 is 2x2, A3 is 2x1
    # B = [B1; B2; B3] where B1 is 2x6, B2 is 2x6, B3 is 1x6
    #
    # AB = A1*B1 + A2*B2 + A3*B3
    #
    # Each A_i * B_i is a 2x6 product.
    # For A1*B1: 2x2 * 2x6. We can use Strassen-like tricks.
    #
    # Actually, let me think about this differently.
    # The known minimal rank for 2x2 matrix multiplication is 7 (Strassen).
    # For 2x5 x 5x6, we can try various block decompositions.
    
    # Approach: Use the fact that we can decompose 5 = 2+2+1 and 6 = 2+2+2
    # A (2x5) = [A1 | A2 | A3], A1:2x2, A2:2x2, A3:2x1
    # B (5x6) = [B1; B2; B3], B1:2x6, B2:2x6, B3:1x6
    # C = A1*B1 + A2*B2 + A3*B3
    
    # For A1*B1 (2x2 * 2x6):
    #   B1 = [B1a | B1b | B1c], each 2x2
    #   A1*B1 = A1*B1a + A1*B1b + A1*B1c
    #   Each A1*B1x is 2x2 * 2x2, which can be done with rank 7 (Strassen)
    #   So A1*B1 has rank at most 21
    #
    # Similarly for A2*B2: rank at most 21
    # A3*B3: 2x1 * 1x6. This is just outer product of column and row.
    #   A3 = [a0; a1] (2x1), B3 = [b0 b1 b2 b3 b4 b5] (1x6)
    #   A3*B3 = [a0*b0, a0*b1, ..., a1*b5]
    #   This can be done with rank 6 (one for each output entry)
    #   Or we can be smarter: it's a rank-1 matrix (outer product), so rank 1?
    #   No, the tensor for this specific multiplication...
    #   Actually A3*B3 gives a 2x6 matrix where C[i,k] = A3[i,0]*B3[0,k]
    #   The tensor here has T[i,0,k] = 1 for all i,k. That's a rank-6 tensor (or can be done with 6 rank-1 terms).
    #   But wait, can we do better? The tensor is sum over i,k of e_i ⊗ e_0 ⊗ e_k.
    #   This has rank 6 (since it's a 2x1x6 tensor with all ones, which has rank min(2,1,6)*... no.
    #   Actually, the tensor T[i,j,k] where j is fixed at 0, and T[i,0,k]=1 for all i,k.
    #   This is the outer product of a 2-vector of ones and a 6-vector of ones, tensored with e_0.
    #   So it's rank 1! T = (1,1) ⊗ e_0 ⊗ (1,1,1,1,1,1)
    #   So rank 1.
    
    # Total: 21 + 21 + 1 = 43. Better than 60.
    
    # But we can do even better by sharing Strassen structures.
    # Let me construct this explicitly.
    
    strassen = _strassen_tensors()
    
    # Build the decomposition
    # Columns:
    # 0-20: A1*B1 (3 blocks of 2x2*2x2, each Strassen rank 7)
    # 21-41: A2*B2 (3 blocks of 2x2*2x2, each Strassen rank 7)
    # 42: A3*B3 (rank 1)
    
    total_rank = 43
    U = [[0] * total_rank for _ in range(n * m)]  # 10 x 43
    V = [[0] * total_rank for _ in range(m * p)]  # 30 x 43
    W = [[0] * total_rank for _ in range(p * n)]  # 12 x 43
    
    col_offset = 0
    
    # Process A1*B1: A1 is columns 0,1 of A (2x2), B1 is rows 0,1 of B (2x6)
    # B1 = [B1a | B1b | B1c], columns 0-1, 2-3, 4-5
    for block in range(3):
        # B1 block columns: block*2, block*2+1
        for si in range(7):
            u, v, w = strassen[si]
            c = col_offset + si
            # u is for A1 (2x2): indices 0,1,2,3 correspond to a00,a01,a10,a11
            # In our U matrix, A1[i,j] is at U[i*5+j]
            # A1: i in 0,1; j in 0,1 -> U[i*5+0], U[i*5+1]
            for idx in range(4):
                ui, uj = divmod(idx, 2)  # u is indexed as [i*2+j]
                U[ui * m + uj][c] += u[idx]
            
            # v is for B1 block (2x2): v[j*2+k] where j is local row (0,1), k is local col
            # B1 is rows 0,1 of B, columns block*2, block*2+1
            # In V matrix, B[j,k] is at V[j*6+k]
            for idx in range(4):
                vj, vk = divmod(idx, 2)
                V[(block * 2 + vj) * p + (block * 2 + vk)][c] += v[idx]
            
            # w is for output C block (2x2): w[k*2+i] where k is local col, i is local row
            # Output C[i, block*2+k] is at W[(block*2+k)*2+i]
            for idx in range(4):
                wk, wi = divmod(idx, 2)
                W[(block * 2 + wk) * n + wi][c] += w[idx]
        
        col_offset += 7
    
    # Process A2*B2: A2 is columns 2,3 of A (2x2), B2 is rows 2,3 of B (2x6)
    for block in range(3):
        for si in range(7):
            u, v, w = strassen[si]
            c = col_offset + si
            # A2: i in 0,1; j in 2,3 -> U[i*5+2], U[i*5+3]
            for idx in range(4):
                ui, uj = divmod(idx, 2)
                U[ui * m + (2 + uj)][c] += u[idx]
            
            # B2: rows 2,3 of B, columns block*2, block*2+1
            for idx in range(4):
                vj, vk = divmod(idx, 2)
                V[(2 + vj) * p + (block * 2 + vk)][c] += v[idx]
            
            # Output: C[i, block*2+k]
            for idx in range(4):
                wk, wi = divmod(idx, 2)
                W[(block * 2 + wk) * n + wi][c] += w[idx]
        
        col_offset += 7
    
    # Process A3*B3: A3 is column 4 of A (2x1), B3 is row 4 of B (1x6)
    # Rank 1: U[0*5+4]=1, U[1*5+4]=1; V[4*6+k]=1 for all k; W[k*2+i]=1 for all k,i
    c = col_offset
    U[0 * m + 4][c] = 1
    U[1 * m + 4][c] = 1
    for k in range(p):
        V[4 * p + k][c] = 1
        for i in range(n):
            W[k * n + i][c] = 1
    col_offset += 1
    
    # Verify
    if _verify_decomposition(n, m, p, U, V, W):
        best_U, best_V, best_W = U, V, W
        best_rank = total_rank
    else:
        # Fall back to schoolbook
        best_U, best_V, best_W = U_sb, V_sb, W_sb
        best_rank = rank_sb
    
    # Try alternative: 5 = 2+1+2, 6 = 2+2+2
    # Or 5 = 3+2, 6 = 3+3
    # Let's also try a different partition
    
    # Try: A (2x5) = [A1 | A2], A1:2x3, A2:2x2
    # B (5x6) = [B1; B2], B1:3x6, B2:2x6
    # C = A1*B1 + A2*B2
    # A1*B1: 2x3 * 3x6. Split B1 into [B1a | B1b], each 3x3? No, 3x6 = 3x3 + 3x3
    # A1*B1a: 2x3 * 3x3. Rank? Schoolbook is 18. Can we do better?
    # 2x3 * 3x3: known minimal rank is not well-studied. Schoolbook = 18.
    # This doesn't seem obviously better.
    
    # Try yet another: use the fact that 2x5 can be seen as 2x(2+2+1)
    # and try to optimize the 2x2*2x6 part more carefully.
    
    # Actually, for 2x2 * 2x6, we can do better than 3*7=21.
    # 2x2 * 2x6: A is 2x2, B is 2x6.
    # C = A*B, C is 2x6.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # 
    # We can write this as:
    # C = [A[:,0] * B[0,:] + A[:,1] * B[1,:]]
    # 
    # One approach: use 4 rank-1 terms:
    # For each (i,k): C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This is the sum of two outer products: A[:,0]*B[0,:] + A[:,1]*B[1,:]
    # Each outer product is rank 1 in the tensor sense? No.
    # A[:,0] is 2x1, B[0,:] is 1x6. Their product is 2x6.
    # The tensor for A[:,0]*B[0,:] is: T[i,0,k] = 1 (for the part where j=0)
    # Wait, I need to be more careful.
    #
    # The full tensor for 2x2 * 2x6 has T[i,j,k] = 1 for i in 0,1; j in 0,1; k in 0,1,2,3,4,5.
    # That's 2*2*6 = 24 entries, all 1.
    #
    # We can decompose this as:
    # T = sum over j of (e_i) ⊗ (e_j) ⊗ (e_k) for all i,j,k
    # = (sum_i e_i) ⊗ (sum_j e_j) ⊗ (sum_k e_k)  -- NO, that gives all ones but with different structure.
    #
    # Actually, T[i,j,k] = 1 for all i,j,k means T = (1,1) ⊗ (1,1) ⊗ (1,1,1,1,1,1)
    # This is rank 1!
    #
    # Wait, that can't be right. Let me re-examine.
    #
    # The matrix multiplication tensor for (n x m) * (m x p) -> (n x p) is defined as:
    # T[i,j,k] = 1 if we're computing (AB)[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all valid i,j,k.
    #
    # The rank of this tensor is the matrix multiplication rank.
    # For 2x2*2x2, it's 7 (Strassen).
    #
    # But T = (1,1) ⊗ (1,1) ⊗ (1,1,1,1,1,1) would give T[i,j,k] = 1*1*1 = 1 for all i,j,k.
    # So that IS rank 1?!
    #
    # No wait, I'm confusing myself. Let me re-read the problem statement.
    #
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 5 matrix by a 5 by 6 matrix."
    #
    # The tensor is T where T[i,j,k] = 1 for all i in [0,2), j in [0,5), k in [0,6).
    # And the decomposition is T = sum_c U_c ⊗ V_c ⊗ W_c where
    # U_c[i*m+j] = U[i*m+j, c], etc.
    #
    # So T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # And we need T[i,j,k] = 1 for all i,j,k.
    #
    # If we set U[i*m+j, 0] = 1 for all i,j, V[j*p+k, 0] = 1 for all j,k, W[k*n+i, 0] = 1 for all k,i,
    # then T[i,j,k] = 1*1*1 = 1. Rank 1!
    #
    # But that's clearly wrong because the schoolbook is rank n*m*p.
    #
    # OH WAIT. I think I misunderstand the tensor. Let me re-read.
    #
    # "the output factor follows the official trace convention W[k*n+i,r]"
    #
    # Hmm, maybe the tensor is NOT all ones. Let me think again.
    #
    # Actually, I think the issue is that the tensor for matrix multiplication is:
    # M_{(i,k),j} = 1  (this is the standard matrix multiplication tensor)
    # where the first index is (i,k) for the output, the second is j for the summation index.
    #
    # But in the 3-tensor form T[i,j,k], we have T[i,j,k] = 1 for all i,j,k.
    #
    # And indeed, the rank of the all-ones tensor of size n*m*p is... 
    #
    # Wait, no. The all-ones tensor 1_{n} ⊗ 1_{m} ⊗ 1_{p} has rank 1.
    # But the matrix multiplication tensor is NOT the all-ones tensor in general.
    #
    # Let me reconsider. The matrix multiplication tensor T ∈ F^{n×m} ⊗ F^{m×p} ⊗ F^{n×p}
    # is defined by T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{i,k}
    #
    # So T[i,j] ⊗ T[j,k] ⊗ T[i,k]... 
    #
    # In coordinates: T_{(i,j),(j,k),(i,k)} = 1, and 0 otherwise.
    #
    # So if we flatten: T[i*m+j, j*p+k, i*p+k]... no wait.
    #
    # The three modes are:
    # Mode 1: (i,j) with i in [0,n), j in [0,m) -> size n*m
    # Mode 2: (j,k) with j in [0,m), k in [0,p) -> size m*p  
    # Mode 3: (i,k) with i in [0,n), k in [0,p) -> size n*p
    #
    # T[a, b, c] = 1 if a=(i,j), b=(j',k), c=(i',k') where j=j' and i=i' and k=k'
    # i.e., T[i*m+j, j*p+k, i*p+k] = 1... 
    #
    # Wait, mode 3 is (i,k) so the index is i*p+k, not k*n+i.
    #
    # But the problem says "W[k*n+i,r]" which suggests mode 3 index is k*n+i.
    #
    # So the convention is:
    # Mode 1: i*m + j (A[i,j])
    # Mode 2: j*p + k (B[j,k])
    # Mode 3: k*n + i (C[i,k] but indexed as k*n+i)
    #
    # And T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k.
    #
    # So T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some valid i,j,k.
    # Otherwise T[a,b,c] = 0.
    #
    # This is NOT the all-ones tensor. It's a sparse tensor.
    #
    # For the all-ones case, we'd need T[a,b,c]=1 for ALL a,b,c, which is not what we have.
    #
    # OK so my Strassen construction should be correct. Let me verify it.
    #
    # The verification function checks T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # for all i,j,k. That's the correct check.
    
    # Let me also try to see if we can do better with a different block structure.
    # 
    # Try: 5 = 2+3, 6 = 3+3
    # A (2x5) = [A1 | A2], A1:2x2, A2:2x3
    # B (5x6) = [B1; B2], B1:2x6, B2:3x6
    # C = A1*B1 + A2*B2
    # A1*B1: 2x2 * 2x6, rank at most 7*3=21 (using Strassen on each 2x2*2x2 block)
    #   But actually 2x2*2x6 can be done with rank 12? Let me think.
    #   2x2 * 2x6: T[i,j,k]=1 for i in 0,1; j in 0,1; k in 0..5
    #   We can group k into 3 pairs: (0,1),(2,3),(4,5)
    #   For each pair, 2x2*2x2 has rank 7. Total 21.
    #   Can we do better? Maybe 2x2*2x6 has rank 12 or 14?
    #   Known: rank(2x2*2x2) = 7. rank(2x2*2x6) is not standard.
    #   Lower bound: at least 2*2*6/... not helpful.
    #   Let's stick with 21 for now.
    #
    # A2*B2: 2x3 * 3x6. Schoolbook rank = 36.
    #   Can we do better? 2x3*3x6:
    #   Split B2 into [B2a | B2b], each 3x3.
    #   A2*B2a: 2x3*3x3, schoolbook = 18.
    #   A2*B2b: 2x3*3x3, schoolbook = 18.
    #   Total 36. Not better than 21+1=22 for the other part.
    #
    # So the 2+2+1 split seems better: 21+21+1 = 43.
    
    # Can we improve the 2x2*2x6 part?
    # 2x2 * 2x6: A = [[a00, a01], [a10, a11]], B = 2x6 matrix.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    #
    # We can write C = A * B = [A[:,0] * B[0,:] + A[:,1] * B[1,:]]
    #
    # Consider: for each column k of B, C[:,k] = A * B[:,k]
    # This is a 2x2 * 2x1 multiplication, which has rank... 
    # 2x2*2x1: T[i,j,k] where k=0. T[i,j,0]=1 for i in 0,1, j in 0,1.
    # This is a 2x2x1 tensor. Rank = 2 (it's the identity-like tensor).
    # Actually 2x2*2x1: C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0]
    # T[i,j,0] = 1. This can be done with rank 2:
    #   Term 1: U[i*2+0]=1, V[0*2+0]=1, W[0*2+i]=1 -> gives C[i,0] += A[i,0]*B[0,0]
    #   Term 2: U[i*2+1]=1, V[1*2+0]=1, W[0*2+i]=1 -> gives C[i,0] += A[i,1]*B[1,0]
    # So rank 2 for each column, total rank 12 for 6 columns.
    #
    # But Strassen gives 21 for three 2x2*2x2 blocks. 12 < 21!
    #
    # So let me use the column-wise approach for 2x2*2x6: rank 12.
    # Then total: 12 + 12 + 1 = 25!
    
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # New construction: 25 rank
    # A (2x5) = [A1 | A2 | A3], A1:2x2 (cols 0,1), A2:2x2 (cols 2,3), A3:2x1 (col 4)
    # B (5x6) = [B1; B2; B3], B1:2x6 (rows 0,1), B2:2x6 (rows 2,3), B3:1x6 (row 4)
    #
    # A1*B1: 2x2*2x6, rank 12 (column-wise)
    # A2*B2: 2x2*2x6, rank 12 (column-wise)
    # A3*B3: 2x1*1x6, rank 1
    # Total: 25
    
    total_rank2 = 25
    U2 = [[0] * total_rank2 for _ in range(n * m)]
    V2 = [[0] * total_rank2 for _ in range(m * p)]
    W2 = [[0] * total_rank2 for _ in range(p * n)]
    
    col_offset2 = 0
    
    # A1*B1: 2x2 * 2x6
    # For each column k (0..5) of B1, and each row j (0,1) of A1:
    # Term: U[ui*m+uj] = 1, V[j*p+k] = 1, W[k*n+ui] = 1
    # This gives C[ui, k] += A1[ui, j] * B1[j, k]
    for k in range(6):
        for j in range(2):
            c = col_offset2
            for ui in range(2):
                U2[ui * m + j][c] = 1
            V2[j * p + k][c] = 1
            for ui in range(2):
                W2[k * n + ui][c] = 1
            col_offset2 += 1
    
    # A2*B2: 2x2 * 2x6
    for k in range(6):
        for j in range(2):
            c = col_offset2
            for ui in range(2):
                U2[ui * m + (2 + j)][c] = 1
            V2[(2 + j) * p + k][c] = 1
            for ui in range(2):
                W2[k * n + ui][c] = 1
            col_offset2 += 1
    
    # A3*B3: 2x1 * 1x6, rank 1
    c = col_offset2
    for ui in range(2):
        U2[ui * m + 4][c] = 1
    for k in range(6):
        V2[4 * p + k][c] = 1
        for ui in range(2):
            W2[k * n + ui][c] = 1
    col_offset2 += 1
    
    if _verify_decomposition(n, m, p, U2, V2, W2):
        best_U, best_V, best_W = U2, V2, W2
        best_rank = total_rank2
    else:
        pass  # Keep previous best
    
    # Can we do even better?
    # The column-wise approach for 2x2*2x6 uses rank 12.
    # Can we share structure between the two 2x2*2x6 blocks?
    # A1*B1 and A2*B2 are independent, so probably not easily.
    #
    # What about a different split?
    # 5 = 3+2, 6 = 6
    # A (2x5) = [A1 | A2], A1:2x3, A2:2x2
    # B (5x6) = [B1; B2], B1:3x6, B2:2x6
    # A1*B1: 2x3*3x6. Column-wise: for each k, 2x3*3x1.
    #   2x3*3x1: C[i,0] = sum_j A[i,j]*B[j,0]. Rank = 3 (one for each j).
    #   Total for 6 columns: 18.
    # A2*B2: 2x2*2x6, rank 12.
    # Total: 30. Worse than 25.
    
    # 5 = 2+2+1, 6 = 2+2+2 (Strassen blocks)
    # Already did this: 43.
    
    # What if we use Strassen for the 2x2*2x2 sub-blocks but combine?
    # 2x2*2x6 with Strassen on three 2x2*2x2: rank 21.
    # Column-wise: rank 12. Column-wise is better.
    
    # Can we do 2x2*2x6 with rank < 12?
    # Lower bound: the tensor 2x2*2x6 has T[i,j,k]=1 for i in 0,1, j in 0,1, k in 0..5.
    # This tensor can be seen as: for each k, a 2x2 all-ones tensor.
    # The 2x2 all-ones tensor has rank 1 (it's (1,1)⊗(1,1)).
    # But we can't just sum them because the k indices are different.
    #
    # Actually, T = sum_k ( (1,1) ⊗ (1,1) ⊗ e_k )
    # = (1,1) ⊗ (1,1) ⊗ (1,1,1,1,1,1)
    # This is rank 1!
    #
    # Wait, so 2x2*2x6 has rank 1??
    # Let me check: T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k.
    # If we set U[i*2+j, 0] = 1, V[j*2+k, 0] = 1, W[k*2+i, 0] = 1,
    # then sum = 1*1*1 = 1. Yes!
    #
    # But this contradicts the fact that 2x2*2x2 has rank 7!
    #
    # OH. I see my error. The tensor T[i*m+j, j*p+k, k*n+i] = 1 is NOT the matrix multiplication tensor!
    #
    # The matrix multiplication tensor has T[i*m+j, j*p+k, k*n+i] = 1 ONLY when the indices are consistent,
    # but the "value" of the tensor entry is 1, and the decomposition must reproduce this.
    #
    # Wait, I'm going in circles. Let me carefully re-derive.
    #
    # Matrix multiplication: C = A*B where A is n×m, B is m×p.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # The tensor T ∈ F^{n×m} ⊗ F^{m×p} ⊗ F^{n×p} is defined as:
    # T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_{i,j} ⊗ e_{j,k} ⊗ e_{i,k}
    #
    # So T_{(i,j), (j',k), (i',k')} = 1 if j=j' and i=i' and k=k', else 0.
    #
    # In flattened form with the given conventions:
    # T[i*m+j, j'*p+k, k'*n+i'] = 1 if j=j', i=i', k=k', else 0.
    #
    # So T[a, b, c] is nonzero only when a=i*m+j, b=j*p+k, c=k*n+i for the SAME i,j,k.
    #
    # This means T is NOT the all-ones tensor. It's a specific sparse tensor.
    #
    # For 2x2*2x2: T has 8 nonzero entries (one for each (i,j,k) triple).
    # T[0,0,0]=1 (i=0,j=0,k=0), T[1,0,2]=1 (i=0,j=1,k=0 -> c=0*2+0=0... wait)
    #
    # Let me recompute. n=m=p=2.
    # i=0,j=0,k=0: a=0*2+0=0, b=0*2+0=0, c=0*2+0=0. T[0,0,0]=1.
    # i=0,j=0,k=1: a=0, b=1, c=1*2+0=2. T[0,1,2]=1.
    # i=0,j=1,k=0: a=1, b=0, c=0. T[1,0,0]=1.
    # i=0,j=1,k=1: a=1, b=1, c=2. T[1,1,2]=1.
    # i=1,j=0,k=0: a=2, b=0, c=1. T[2,0,1]=1.
    # i=1,j=0,k=1: a=2, b=1, c=3. T[2,1,3]=1.
    # i=1,j=1,k=0: a=3, b=0, c=1. T[3,0,1]=1.
    # i=1,j=1,k=1: a=3, b=1, c=3. T[3,1,3]=1.
    #
    # So T has 8 nonzero entries. The rank of this tensor is 7 (Strassen).
    #
    # If I set U[a,0]=1 for all a, V[b,0]=1 for all b, W[c,0]=1 for all c,
    # then sum_c U[a,c]*V[b,c]*W[c,c] = 1 for ALL a,b,c, not just the 8 valid ones.
    # So this would give T[a,b,c]=1 for all a,b,c, which is WRONG.
    #
    # The decomposition must give T[a,b,c]=1 for the 8 valid (a,b,c) and 0 for the rest.
    #
    # So my verification function is correct: it checks that for each valid (i,j,k),
    # the sum equals 1. But it does NOT check that invalid (a,b,c) give 0.
    #
    # Wait, but the problem says "exactly decompose the matrix multiplication tensor".
    # So we need T[a,b,c] to be correct for ALL a,b,c, not just the valid ones.
    #
    # Let me re-read the problem: "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    # with integer/half-integer entries that exactly decompose the matrix multiplication tensor"
    #
    # So yes, the full tensor must be reproduced. My verification only checks the valid entries.
    # I need to also check that invalid entries are 0.
    
    # Let me fix the verification.
    def _full_verify(n, m, p, U, V, W):
        rank = len(U[0]) if U else 0
        # Check all valid entries are 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                    if s != 1:
                        return False
        # Check all invalid entries are 0
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    # Determine if (a,b,c) is a valid (i,j,k) triple
                    i, j = divmod(a, m)
                    j2, k = divmod(b, p)
                    k2, i2 = divmod(c, n)
                    if not (j == j2 and i == i2 and k == k2):
                        s = sum(U[a][cc] * V[b][cc] * W[c][cc] for cc in range(rank))
                        if s != 0:
                            return False
        return True
    
    # Re-verify with full check
    if _full_verify(n, m, p, U2, V2, W2):
        best_U, best_V, best_W = U2, V2, W2
        best_rank = total_rank2
    else:
        # The column-wise construction might have nonzero invalid entries.
        # Let's check.
        pass
    
    # The column-wise construction: for A1*B1 part,
    # U[ui*m+j][c]=1, V[j*p+k][c]=1, W[k*n+ui][c]=1
    # This contributes 1 to T[i*m+j, j*p+k, k*n+i] when i=ui, which is correct.
    # But does it contribute to invalid entries?
    # T[a,b,c] gets contribution from term c if U[a,c]*V[b,c]*W[c,c] != 0.
    # For the column-wise term for (j,k): U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1.
    # So T[a,b,c] gets 1 if a=i*m+j, b=j*p+k, c=k*n+i for the specific i,j,k.
    # That's exactly the valid entry. So no invalid contributions.
    #
    # Wait, but multiple terms might contribute to the same (a,b,c).
    # For a given (a,b,c) = (i*m+j, j*p+k, k*n+i), only the term with that specific (j,k)
    # and that specific i will have all three factors nonzero.
    # Actually, for a fixed (a,b,c) = (i*m+j, j*p+k, k*n+i),
    # the term for (j',k') contributes if U[i*m+j, c']=1 and V[j*p+k, c']=1 and W[k*n+i, c']=1.
    # U[i*m+j, c']=1 only if c' is the column for (j, k') where k' is the column index in the loop.
    # V[j*p+k, c']=1 only if c' is the column for (j, k).
    # W[k*n+i, c']=1 only if c' is the column for (k, i) where i is the row.
    #
    # In my construction, for the A1*B1 part:
    # Column c corresponds to (j, k) where j in 0,1 and k in 0..5.
    # U[ui*m+j][c] = 1 for ui in 0,1. So U[i*m+j][c] = 1 for any i.
    # V[j*p+k][c] = 1.
    # W[k*n+ui][c] = 1 for ui in 0,1. So W[k*n+i][c] = 1 for any i.
    #
    # So for a valid (i,j,k): sum over all columns c' of U[i*m+j][c']*V[j*p+k][c']*W[k*n+i][c']
    # The only nonzero term is c' = the column for (j,k), which gives 1*1*1 = 1.
    # Other columns: either U[i*m+j][c']=0 (if c' is for different j) or V[j*p+k][c']=0 (if c' is for different k).
    # So the sum is exactly 1. Good.
    #
    # For an invalid (a,b,c): say a=i*m+j, b=j2*p+k, c=k2*n+i2 where j!=j2 or i!=i2 or k!=k2.
    # The term for column c'=(j',k') contributes if U[i*m+j][c']=1, V[j2*p+k][c']=1, W[k2*n+i2][c']=1.
    # U[i*m+j][c']=1 requires j'=j. V[j2*p+k][c']=1 requires j'=j2 and k'=k.
    # So we need j=j2 and k'=k. And W[k2*n+i2][c']=1 requires k'=k2 and i2 is any.
    # So we need j=j2, k=k2, and i2 can be anything.
    # But if j=j2 and k=k2, then (a,b,c) = (i*m+j, j*p+k, k*n+i2).
    # For this to be invalid, we need i != i2.
    # But W[k*n+i2][c']=1 for any i2, so the term contributes 1.
    # And the sum over all columns: only the column for (j,k) contributes, giving 1.
    # So T[i*m+j, j*p+k, k*n+i2] = 1 for i2 != i, which should be 0!
    #
    # So the column-wise construction is WRONG. It gives nonzero values for invalid entries.
    
    # I need to use the Strassen construction or another valid one.
    # Let me go back to the Strassen-based construction with rank 43.
    
    # Actually, let me re-examine the Strassen construction.
    # In the Strassen construction, for each Strassen term, we have specific u, v, w vectors.
    # The key is that Strassen's identity ensures that the sum of the 7 rank-1 tensors
    # gives exactly the 2x2*2x2 multiplication tensor, with zeros elsewhere.
    
    # Let me verify the Strassen construction more carefully.
    if _full_verify(n, m, p, U, V, W):
        best_U, best_V, best_W = U, V, W
        best_rank = total_rank
    else:
        # Fall back to schoolbook
        best_U, best_V, best_W = U_sb, V_sb, W_sb
        best_rank = rank_sb
    
    # The schoolbook is always valid.
    return {"U": best_U, "V": best_V, "W": best_W}
