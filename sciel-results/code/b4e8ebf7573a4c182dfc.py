"""Program for finding rank-minimal exact tensor decomposition for 2x3 x 3x4 matrix multiplication."""
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

def construct():
    """
    Find a low-rank exact tensor decomposition for multiplying a 2x3 matrix by a 3x4 matrix.
    
    The decomposition satisfies: for all (i,k,a,b,c,d):
        sum_r W[k*n+i, r] * U[a*m+b, r] * V[c*p+d, r] = 1 if a=i, b=c, d=k; else 0
    
    We use a constructive approach based on the structure of the tensor.
    """
    n, m, p = DIMENSIONS  # 2, 3, 4
    
    # We'll try to find a decomposition with rank as low as possible.
    # The schoolbook rank is n*m*p = 24.
    # Let's try to find a rank-20 or lower solution using a systematic construction.
    
    # Strategy: Use the fact that for each j (0..2), we can create columns that handle
    # multiple (i,k) pairs simultaneously, as long as we set W to zero at wrong positions.
    
    # Let's try a construction with rank 20.
    # For each j, we'll create columns that group the 8 (i,k) pairs into smaller groups.
    
    # Actually, let me think about this more carefully.
    # The key constraint is: for a given column r, if U has non-zeros at (i1,j) and (i2,j),
    # and V has non-zeros at (j,k1) and (j,k2), then:
    # - W[(k1,i1),r] and W[(k2,i2),r] can be non-zero (correct terms)
    # - W[(k2,i1),r] and W[(k1,i2),r] must be zero (cross-terms)
    
    # So for a fixed j, if we want to handle (i1,k1), (i1,k2), (i2,k1), (i2,k2) in one column,
    # we need W[(k2,i1),r] = 0 and W[(k1,i2),r] = 0, which is fine.
    
    # But we also need to ensure that across different columns, the sum works out.
    
    # Let me try a simpler approach: use the schoolbook decomposition as a starting point
    # and try to merge columns where possible.
    
    # Schoolbook: 24 columns, one for each (i,j,k).
    # Column (i,j,k): U[(i,j)]=1, V[(j,k)]=1, W[(k,i)]=1.
    
    # Can we merge columns? Two columns (i1,j1,k1) and (i2,j2,k2) can be merged if
    # the cross-terms vanish. This happens when the "wrong" W entries are zero.
    
    # Let me try a different approach: use numpy to solve for a rank-r decomposition
    # by setting up the linear system and using SVD or least squares.
    
    # The condition is: for all (i,k,a,b,c,d):
    # sum_r W[k*n+i, r] * U[a*m+b, r] * V[c*p+d, r] = delta(a=i, b=c, d=k)
    
    # This is a trilinear system, which is hard to solve directly.
    
    # Instead, let me try a constructive approach based on known results.
    # For 2x3x4, the minimum rank is believed to be around 19-20.
    
    # Let me try rank 20 with a specific construction.
    
    rank = 20
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Construction: For each j (0,1,2), create columns that handle the (i,k) pairs.
    # For j=0: 8 (i,k) pairs. We can use 4 columns, each handling 2 pairs.
    # For j=1: 8 (i,k) pairs. We can use 4 columns.
    # For j=2: 8 (i,k) pairs. We can use 4 columns.
    # Total: 12 columns. But we need to handle the cross-terms correctly.
    
    # Actually, let me try a more careful construction.
    
    # For each j, the (i,k) pairs are: (0,0), (0,1), (0,2), (0,3), (1,0), (1,1), (1,2), (1,3)
    
    # Let me try: for each j, use 3 columns.
    # Column 1 for j: handles (0,0), (0,1), (1,0), (1,1)
    #   U[(0,j)]=1, U[(1,j)]=1, V[(j,0)]=1, V[(j,1)]=1, W[(0,0)]=1, W[(1,0)]=1, W[(0,1)]=1, W[(1,1)]=1
    #   But this creates cross-terms: W[(1,0)] and W[(0,1)] are both non-zero, which is fine.
    #   Wait, the cross-terms are: U[(0,j)]*V[(j,1)]*W[(1,0)] and U[(1,j)]*V[(j,0)]*W[(0,1)].
    #   These contribute to C[0,1] and C[1,0] respectively, which are correct!
    #   Actually no: U[(0,j)]*V[(j,1)]*W[(1,0)] contributes A[0,j]*B[j,1]*W[(1,0)] to C[1,0]?
    #   No, W[(k,i)] multiplies the product to give C[i,k]. So W[(1,0)] means i=0, k=1.
    #   So U[(0,j)]*V[(j,1)]*W[(1,0)] contributes to C[0,1], which is correct!
    #   And U[(1,j)]*V[(j,0)]*W[(0,1)] contributes to C[1,0], which is also correct!
    
    # So actually, the "cross-terms" are not cross-terms at all - they are the correct terms!
    # The issue I had before was a misunderstanding of the index convention.
    
    # Let me re-examine: W is indexed by (k*n+i), so W[k*n+i] corresponds to output C[i,k].
    # So W[(k,i)] means the coefficient for C[i,k].
    
    # So for column r with U[(i,j),r]=u, V[(j,k),r]=v, W[(k,i),r]=w:
    # This contributes u*v*w to C[i,k] via the term A[i,j]*B[j,k].
    
    # If U has non-zeros at (i1,j) and (i2,j), and V has non-zeros at (j,k1) and (j,k2):
    # - U[(i1,j)]*V[(j,k1)]*W[(k1,i1)] contributes to C[i1,k1] ✓
    # - U[(i1,j)]*V[(j,k2)]*W[(k2,i1)] contributes to C[i1,k2] ✓
    # - U[(i2,j)]*V[(j,k1)]*W[(k1,i2)] contributes to C[i2,k1] ✓
    # - U[(i2,j)]*V[(j,k2)]*W[(k2,i2)] contributes to C[i2,k2] ✓
    
    # All four are correct terms! There are no cross-terms!
    
    # So for a fixed j, if we set U[(i,j),r]=1 for all i, V[(j,k),r]=1 for all k,
    # and W[(k,i),r]=1 for all (k,i), then this single column handles all 8 (i,k) pairs
    # for that j!
    
    # But wait, this would mean rank = m = 3, which is below the lower bound of 12.
    # So something is still wrong.
    
    # Let me check the lower bound again. The slice rank lower bound for matrix multiplication
    # of (n,m,p) is max(n*m, m*p, p*n) = max(6, 12, 8) = 12.
    
    # So rank 3 is impossible. Where is my error?
    
    # Ah, I see. The issue is that when we have U[(i,j),r]=1 for all i and V[(j,k),r]=1 for all k,
    # the product X_r = sum_{i,j} U[(i,j),r] A[i,j] = sum_i A[i,j] (for fixed j).
    # And Y_r = sum_{j,k} V[(j,k),r] B[j,k] = sum_k B[j,k] (for fixed j).
    # Then Z_r = X_r * Y_r = (sum_i A[i,j]) * (sum_k B[j,k]).
    # This is a product of sums, which expands to sum_{i,k} A[i,j] B[j,k].
    # So Z_r = sum_{i,k} A[i,j] B[j,k], which is a sum of m*n*p/m = n*p = 8 terms.
    # But Z_r is supposed to be a single scalar multiplication!
    
    # The issue is that in the bilinear algorithm, each Z_r = X_r * Y_r is a single scalar
    # multiplication. The number of scalar multiplications is r (the rank). But in my construction,
    # Z_r expands to 8 terms, which means we're doing 8 scalar multiplications per rank term,
    # for a total of 3*8 = 24 scalar multiplications, which is the schoolbook count.
    
    # So the rank in the tensor decomposition sense is NOT the same as the number of scalar
    # multiplications in the bilinear algorithm. The tensor rank is the minimum r such that
    # the tensor can be written as a sum of r rank-1 tensors, where each rank-1 tensor is
    # u ⊗ v ⊗ w with u, v, w being vectors.
    
    # In my construction, each "rank-1 tensor" u_r ⊗ v_r ⊗ w_r has:
    # u_r with 2 non-zeros, v_r with 4 non-zeros, w_r with 8 non-zeros.
    # The tensor u_r ⊗ v_r ⊗ w_r has 2*4*8 = 64 non-zero entries.
    # But the matrix multiplication tensor has only 24 non-zero entries (one for each (i,j,k)).
    # So the sum of 3 such tensors would have at most 3*64 = 192 non-zero entries,
    # but we need exactly 24 non-zero entries with value 1, and 0 elsewhere.
    
    # The problem is that the non-zero entries of u_r ⊗ v_r ⊗ w_r include entries that
    # should be 0 in the matrix multiplication tensor. For example, the entry at
    # ((i,j),(j',k),(k',i')) where j≠j' should be 0, but in my construction,
    # u_r[(i,j)] = 1 only if j matches, and v_r[(j',k)] = 1 only if j' matches,
    # so the product is 0 when j≠j'. So that's fine.
    
    # But what about ((i,j),(j,k),(k',i')) where k≠k'? In my construction,
    # w_r[(k',i')] = 1 for all k',i', so the product u_r[(i,j)]*v_r[(j,k)]*w_r[(k',i')]
    # = 1*1*1 = 1 for all k',i'. But we need this to be 1 only when k'=k and i'=i,
    # and 0 otherwise.
    
    # So my construction is wrong because w_r has non-zeros at positions where it should be 0.
    
    # The correct condition is: for the tensor to match the matrix multiplication tensor,
    # we need: sum_r u_r[(i,j)] v_r[(j,k)] w_r[(k,i)] = 1 for all (i,j,k), AND
    # sum_r u_r[(i,j)] v_r[(j',k)] w_r[(k',i')] = 0 for all other combinations.
    
    # In particular, for fixed (i,j,k), we need the sum to be 1.
    # For (i,j,k') with k'≠k, we need the sum to be 0.
    # For (i,j',k) with j'≠j, we need the sum to be 0.
    # Etc.
    
    # So my construction fails because for (i,j,k') with k'≠k, the sum is 1 (from the column
    # for j), but it should be 0.
    
    # So the construction is invalid. Let me go back to the schoolbook decomposition.
    
    # Schoolbook: rank = 24. This is guaranteed correct.
    
    # Let me just return the schoolbook decomposition.
    
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
