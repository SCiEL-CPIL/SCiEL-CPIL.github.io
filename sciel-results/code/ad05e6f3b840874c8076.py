"""Program for 3x4 * 4x7 matrix multiplication tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    n, m, p = 3, 4, 7  # Matrix sizes: (n x m) * (m x p)
    
    # The goal is to find a minimal rank tensor decomposition for C[i,k] = sum_j A[i,j]*B[j,k]
    # We use a heuristic approach: 
    # 1. Initialize with schoolbook (rank = n*m*p = 84)
    # 2. Try to reduce rank by checking linear dependencies in the U, V, W factors
    #    using numerical linear algebra (SVD/rank determination)
    
    # First, create the schoolbook decomposition
    rank_schoolbook = n * m * p
    U = np.zeros((n * m, rank_schoolbook), dtype=int)
    V = np.zeros((m * p, rank_schoolbook), dtype=int)
    W = np.zeros((p * n, rank_schoolbook), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Now try to find a smaller rank decomposition.
    # The tensor is T[i,k,j] = delta_{ik} * A_{ij} * B_{jk}... wait no.
    # The multiplication tensor has components T_{(i,k),j} where we want to represent:
    # C_{ik} = sum_j A_{ij} B_{jk}
    # In tensor notation: T_{ik,j} = 1 if we're computing C_{ik} from A_{i,j} and B_{j,k}
    # Actually, the standard decomposition uses:
    # C_{ik} = sum_r U_{i,j,r} V_{j,k,r} W_{r, i*k}  (with appropriate indexing)
    
    # Let's think of this differently. We have a 3-way tensor T of shape (n*m, m*p, p*n)
    # where T[(i,j), (j',k), (k',i')] = 1 if j=j' and k=k' and i=i' (i.e., the middle indices match)
    # This is a permutation-like tensor.
    
    # Actually, let's use the fact that this is a "bilinear map" and use known results.
    # For 3x4*4x7, the schoolbook rank is 84. 
    # Can we do better? 
    
    # Strategy: Use the fact that we can share computations across different (i,j,k) tuples.
    # For example, if we have two terms that involve the same A[i,j] or B[j,k], we might be able to combine them.
    
    # Let's try a different approach: represent the multiplication as a sum of rank-1 terms.
    # Each rank-1 term corresponds to choosing specific rows/columns to multiply.
    
    # For a 2x2x2 block, Strassen shows we can do it in 7 multiplications instead of 8.
    # Let's see if we can apply Strassen-like methods to parts of our 3x4x7 problem.
    
    # 3x4x7: 
    # i: 3 (rows of A, rows of C)
    # j: 4 (cols of A, rows of B)  
    # k: 7 (cols of B, cols of C)
    
    # Let's try to decompose this using smaller blocks.
    # Split k (7) into groups: {0,1}, {2,3}, {4,5}, {6}
    # Split j (4) into groups: {0,1}, {2,3}
    # Split i (3) into groups: {0}, {1,2}
    
    # For each combination of i-group, j-group, k-group, we have a sub-problem.
    # i=0, j={0,1}, k={0,1}: 1x2x2 -> schoolbook rank 4
    # i=0, j={0,1}, k={2,3}: 1x2x2 -> schoolbook rank 4
    # etc.
    
    # Total with this simple block decomposition:
    # i-group {0}: 1 row
    #   j-group {0,1}: 2x2 blocks for k-groups -> 4 blocks * 4 = 16
    #   j-group {2,3}: 2x2 blocks for k-groups -> 4 blocks * 4 = 16
    # i-group {1,2}: 2 rows
    #   j-group {0,1}: 2x2 blocks for k-groups -> 4 blocks * 4 = 16
    #   j-group {2,3}: 2x2 blocks for k-groups -> 4 blocks * 4 = 16
    # Total: 64 (still worse than 84? No, 64 < 84)
    
    # Wait, 1x2x2 has schoolbook rank 2*2 = 4 (one i, 2 j's, 2 k's).
    # 2x2x2 has schoolbook rank 2*2*2 = 8, but Strassen does it in 7.
    
    # Let's be more careful:
    # The sub-problem is: given A[i_group, j_group] and B[j_group, k_group], compute C[i_group, k_group]
    # The rank of this sub-problem (using schoolbook) is |i_group| * |j_group| * |k_group|
    
    # Let's try all possible splits and see if we can get below 84.
    
    # Actually, let's just implement a systematic search for rank-1 decompositions.
    # We'll build up the U, V, W matrices column by column, trying to minimize the rank.
    
    # The key insight: we need to represent the tensor T where
    # T[(i,j), (j',k), (k',i')] = 1 if j==j' and k==k' and i==i'
    # This is equivalent to: sum_j A[i,j]*B[j,k] = sum_r U[(i,j),r] * V[(j,k),r] * W[(k,i),r]  (wrong)
    
    # Let me restart with the correct tensor formulation.
    # We want: C[i,k] = sum_j A[i,j] * B[j,k]
    # In a rank-r decomposition: C[i,k] = sum_{r=1}^r U[i,j,r] * V[j',k',r] * W[k,i,r]
    # where j=j' and k=k' for the terms to contribute.
    # Actually, the standard form is:
    # C[i,k] = sum_r (sum_j U[i,j,r] * A[i,j]) * (sum_{j'} V[j',k,r] * B[j',k]) * W[r,k,i]
    # This is getting confused. Let's use the standard tensor decomposition.
    
    # The multiplication tensor is T in R^{(n*m) x (m*p) x (p*n)}
    # T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(k,i) for some i,j,k, else 0
    # This tensor has a Kronecker product structure.
    
    # For our case n=3, m=4, p=7, the tensor is of shape (12, 28, 21)
    # The number of non-zero entries is n*m*p = 84
    
    # The schoolbook decomposition has rank 84.
    # To get a lower rank, we need to find linear dependencies.
    
    # Let's try a numerical approach: 
    # 1. Start with schoolbook
    # 2. Check if any columns are linearly dependent
    # 3. If so, reduce rank
    
    # For the schoolbook decomposition:
    # U has 12 rows, 84 columns. Column (i,j,k) has a 1 at row (i,j).
    # V has 28 rows, 84 columns. Column (i,j,k) has a 1 at row (j,k).
    # W has 21 rows, 84 columns. Column (i,j,k) has a 1 at row (k,i).
    
    # Are there linear dependencies? 
    # U: Two columns (i,j,k1) and (i,j,k2) have the same row pattern (1 at position i*m+j).
    #    So if we sum them, we get 2 at that position. This means the columns are linearly dependent!
    #    Specifically, for fixed (i,j), the columns with different k are identical in U.
    #    Similarly, for fixed (j,k), columns with different i are identical in V.
    #    For fixed (k,i), columns with different j are identical in W.
    
    # This means we can combine terms! Let's work out the reduction.
    
    # For fixed (i,j), we have p=7 columns in U that are identical.
    # We can replace these 7 columns with fewer columns by adjusting W.
    
    # The idea: Instead of having separate rank-1 terms for each k, we can group them.
    # Let's define new basis vectors.
    
    # Start with the observation that U's columns depend only on (i,j), V's on (j,k), W's on (k,i).
    # The number of distinct U row patterns is n*m = 12
    # The number of distinct V row patterns is m*p = 28
    # The number of distinct W row patterns is p*n = 21
    
    # The minimum rank is at least max(12, 28, 21) = 28.
    # But we also need to satisfy the coupling constraint: the decomposition must reproduce the tensor.
    
    # Let's try rank = 28 and see if we can construct it.
    # For each (j,k) pair (there are 28), we have a column r=(j,k) in V with V[j,k,r]=1.
    # For U: U[i,j,r] should be non-zero only if... we need to define it.
    # For W: W[k,i,r] should be non-zero only if...
    
    # The constraint is: for each (i,j,k), we need sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    # and for invalid combinations (where the indices don't match in the right way), the sum should be 0.
    # Actually, the tensor only has 1's at positions where i,j,k are consistent.
    
    # This is a complex constraint satisfaction problem. Let's use a greedy/iterative approach.
    
    # Alternative: Use known results from literature on small matrix multiplication.
    # For 3x4*4x7, the best known rank might be achievable through Strassen-like methods.
    
    # Let's implement a Strassen-like decomposition for the 2x2x2 blocks and combine.
    
    def strassen_2x2x2(U, V, W, i_offset, j_offset, k_offset, start_col):
        """
        Use Strassen's 7-multiplication algorithm for a 2x2x2 block.
        A is 2x2, B is 2x2, C is 2x2.
        U corresponds to rows (i,j) for i in [i_offset, i_offset+2), j in [j_offset, j_offset+2)
        V corresponds to rows (j,k) for j in [j_offset, j_offset+2), k in [k_offset, k_offset+2)
        W corresponds to rows (k,i) for k in [k_offset, k_offset+2), i in [i_offset, i_offset+2)
        """
        col = start_col
        
        # Strassen's 7 products:
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # For each product Mr, we need to update U, V, W.
        # U[(i,j), r] = coefficient of A[i,j] in the "A-part" of Mr
        # V[(j,k), r] = coefficient of B[j,k] in the "B-part" of Mr
        # W[(k,i), r] = coefficient for C[k,i] (but C is 2x2, so k,i are local indices)
        
        # Let's define the 7 products.
        # Each product is of the form (linear combination of A) * (linear combination of B)
        # Then C entries are linear combinations of these products.
        
        products = [
            # (A_coeffs, B_coeffs, C_contribs)
            # A_coeffs: dict (i,j) -> coeff
            # B_coeffs: dict (j,k) -> coeff
            # C_contribs: dict (k,i) -> coeff
            (
                {(0,0):1, (1,1):1},  # A11 + A22
                {(0,0):1, (1,1):1},  # B11 + B22
                {(0,0):1, (1,1):1}   # contributes to C11 and C22
            ),
            (
                {(1,0):1, (1,1):1},  # A21 + A22
                {(0,0):1},           # B11
                {(1,0):1}            # contributes to C21
            ),
            (
                {(0,0):1},           # A11
                {(0,1):1, (1,1):-1}, # B12 - B22
                {(0,0):1, (1,1):1}   # contributes to C11 and C22
            ),
            (
                {(1,1):1},           # A22
                {(1,0):1, (0,0):-1}, # B21 - B11
                {(0,0):1}            # contributes to C11
            ),
            (
                {(0,0):1, (0,1):1},  # A11 + A12
                {(1,1):1},           # B22
                {(0,0):-1, (1,1):1}  # contributes to -C11 + C22
            ),
            (
                {(1,0):1, (0,0):-1}, # A21 - A11
                {(0,0):1, (0,1):1},  # B11 + B12
                {(1,1):1}            # contributes to C22
            ),
            (
                {(0,1):1, (1,1):-1}, # A12 - A22
                {(1,0):1, (1,1):1},  # B21 + B22
                {(0,0):1}            # contributes to C11
            ),
        ]
        
        for a_coeffs, b_coeffs, c_coeffs in products:
            for (li, lj), coeff_a in a_coeffs.items():
                row_a = (i_offset + li) * m + (j_offset + lj)
                U[row_a, col] += coeff_a
            for (lj, lk), coeff_b in b_coeffs.items():
                row_b = (j_offset + lj) * p + (k_offset + lk)
                V[row_b, col] += coeff_b
            for (lk, li), coeff_c in c_coeffs.items():
                row_c = (k_offset + lk) * n + (i_offset + li)
                W[row_c, col] += coeff_c
            col += 1
        
        return col
    
    def schoolbook_block(U, V, W, ni, nj, nk, i_offset, j_offset, k_offset, start_col):
        """
        Use schoolbook decomposition for a ni x nj x nk block.
        """
        col = start_col
        for ii in range(ni):
            for jj in range(nj):
                for kk in range(nk):
                    row_a = (i_offset + ii) * m + (j_offset + jj)
                    row_b = (j_offset + jj) * p + (k_offset + kk)
                    row_c = (k_offset + kk) * n + (i_offset + ii)
                    U[row_a, col] = 1
                    V[row_b, col] = 1
                    W[row_c, col] = 1
                    col += 1
        return col
    
    # Now let's construct the full decomposition using a mix of Strassen and schoolbook.
    # Strategy: 
    # i: 3 rows -> split into {0}, {1,2}
    # j: 4 cols -> split into {0,1}, {2,3}
    # k: 7 cols -> split into {0,1}, {2,3}, {4,5}, {6}
    
    # For each combination:
    # i_group, j_group, k_group -> block size (ni, nj, nk)
    # If nj==2 and nk==2, use Strassen (rank 7)
    # Otherwise, use schoolbook (rank ni*nj*nk)
    
    # Let's allocate columns.
    # First, count total rank needed.
    
    i_groups = [(0, 1), (1, 2)]  # (i_start, i_end)
    j_groups = [(0, 2), (2, 4)]
    k_groups = [(0, 2), (2, 4), (4, 6), (6, 7)]
    
    total_rank = 0
    block_info = []
    for i_start, i_end in i_groups:
        ni = i_end - i_start
        for j_start, j_end in j_groups:
            nj = j_end - j_start
            for k_start, k_end in k_groups:
                nk = k_end - k_start
                if nj == 2 and nk == 2:
                    block_rank = 7
                else:
                    block_rank = ni * nj * nk
                total_rank += block_rank
                block_info.append((ni, nj, nk, i_start, j_start, k_start, block_rank))
    
    # Initialize U, V, W with zeros
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    col = 0
    for ni, nj, nk, i_start, j_start, k_start, block_rank in block_info:
        if nj == 2 and nk == 2:
            col = strassen_2x2x2(U, V, W, i_start, j_start, k_start, col)
        else:
            col = schoolbook_block(U, V, W, ni, nj, nk, i_start, j_start, k_start, col)
    
    # Verify the construction (optional, for debugging)
    # Check that sum_r U[:,r] * V[:,r] * W[:,r] reproduces the tensor
    # This is a 3-way check, let's do it for a few entries
    
    # For now, return the construction.
    # Convert to lists of lists for the expected format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
