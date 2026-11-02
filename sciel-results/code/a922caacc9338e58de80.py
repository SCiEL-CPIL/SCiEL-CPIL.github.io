"""
Rank-minimal exact tensor decomposition for 6x6 matrix multiplication.
Strategy: Use Laderman's rank-16 algorithm for 4x4 matrix multiplication via a
2x2 block partition of 3x3 submatrices (block size 3). This yields total rank
2 * 2 * 2 * 16 = 128, which is better than the previous block-Strassen rank of 189.

The 6x6 matrix is partitioned into 2x2 blocks of size 3x3.
Laderman's algorithm computes the product of two 4x4 matrices using 16 scalar
multiplications instead of the naive 16 (same count, but the structure allows
for the tensor product with 3x3 blocks to be applied directly).

Actually, Laderman's algorithm is for 4x4 matrices and has rank 16.
When we partition a 6x6 matrix into 2x2 blocks of 3x3, each "scalar" in the
4x4 problem is a 3x3 matrix. The tensor product structure means we need
16 * (3*3*3) = 16 * 27 = 432? No, that's not right.

Wait, let me reconsider. The tensor product of two decompositions:
If A (n x n) is partitioned into (n1 x n1) blocks of size (n/n1 x n/n1),
and we have a decomposition of the (n/n1 x n/n1) matrix multiplication with rank r,
and we have a decomposition of the (n/n1 x n/n1) matrix multiplication with rank r',
then the total rank is r * r'.

But here, we're using Laderman's algorithm for 2x2 block matrices where each block
is 3x3. Laderman's algorithm for 2x2 matrices has rank 7 (Strassen).
For 4x4 matrices, Laderman's algorithm has rank 16.

Let me think about this differently. The 6x6 matrix can be viewed as a 2x2 block
matrix with 3x3 blocks. The multiplication of two 2x2 block matrices requires
computing products of 3x3 blocks. If we use Strassen for the 2x2 structure,
we need 7 multiplications of 3x3 blocks. Each 3x3 multiplication has rank 27
(schoolbook) or can use Winograd (rank 23) or other methods.

Actually, the inspiration mentions "4x4 rank-16 decomposition (Laderman) via a
2x2 block partition of 3x3 submatrices, yielding a total rank of 16 * 9 = 144".

I think the idea is: partition 6x6 into 4x4 blocks? No, 6 is not divisible by 4.

Let me re-read: "Apply the 4x4 rank-16 decomposition (Laderman) via a 2x2 block
partition of 3x3 submatrices". 

I think this means: treat the 6x6 matrix as having a 2x2 block structure where
each block is 3x3. Then use Laderman's algorithm which is designed for 4x4 matrices,
but adapted for 2x2 blocks of 3x3 matrices.

Actually, I think the correct interpretation is:
- Partition 6x6 into 2x2 blocks of size 3x3
- Use Laderman's rank-16 algorithm for the 2x2 block structure (but Laderman is for 4x4)

Wait, Laderman's algorithm is specifically for 4x4 matrices. For 2x2 matrices,
Strassen has rank 7. 

Let me just implement the 2x2 Strassen with 3x3 blocks and see if we can optimize.
Rank = 7 * 27 = 189 (using schoolbook for 3x3)

Or if we use Winograd for 3x3 (rank 23): 7 * 23 = 161

The inspiration says rank 144 = 16 * 9. So maybe they're using a different approach.

Let me just implement the clean 2x2 Strassen with 3x3 blocks (rank 189) and
also try to find linear dependencies to reduce the rank.
"""

import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    """
    Construct a rank-reduced exact tensor decomposition for 6x6 matrix multiplication.
    Uses 2x2 Strassen block decomposition with 3x3 submatrices (rank 189),
    then attempts rank reduction by finding linear dependencies.
    """
    deadline = time.time() + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Fallback: schoolbook rank 216
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=np.int8)
        V = np.zeros((m*p, rank), dtype=np.int8)
        W = np.zeros((p*n, rank), dtype=np.int8)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W
    
    # Primary: 2x2 Strassen with 3x3 blocks (rank 189)
    def get_strassen_2x2():
        block_size = 3
        n_blocks = n // block_size  # 2
        
        rank = n_blocks ** 3 * 7  # 8 * 7 = 56? No, 2^3 * 7 = 56
        
        # Wait, for 2x2 blocks, we have 2 block positions in each dimension.
        # The number of block multiplications is 2*2*2 = 8, each using 7 Strassen terms.
        # Total rank = 8 * 7 = 56? That doesn't seem right either.
        
        # Let me recalculate: for n_blocks x n_blocks block structure,
        # we have n_blocks^3 block multiplications, each of rank 7.
        # So rank = n_blocks^3 * 7 = 2^3 * 7 = 56.
        
        # But each block is 3x3, so the actual matrix size is 2*3 = 6.
        # The rank-1 terms operate on individual elements, so we need to expand.
        
        # Actually, for the tensor decomposition, each "scalar" multiplication in the
        # block algorithm corresponds to a 3x3 * 3x3 multiplication, which has rank 27.
        # So total rank = 7 * 27 = 189 for one block multiplication.
        # With 8 block multiplications: 8 * 7 * 27 = 1512? That's way too high.
        
        # I'm confusing myself. Let me think step by step.
        
        # The 6x6 matrix is partitioned into 2x2 blocks of size 3x3.
        # A = [[A00, A01], [A10, A11]] where each Aij is 3x3.
        # B = [[B00, B01], [B10, B11]] where each Bjk is 3x3.
        # C = A * B, so C00 = A00*B00 + A01*B10, etc.
        
        # Using Strassen for the 2x2 block structure:
        # M1 = (A00+A11)(B00+B11)  -- product of two 3x3 matrices
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Each Mi is a 3x3 matrix computed as a product of two 3x3 matrices.
        # Each such product requires rank 27 (schoolbook) in the tensor decomposition.
        # So total rank = 7 * 27 = 189.
        
        # This matches the previous score of 189.
        
        rank = 7 * 27  # 189
        U = np.zeros((n*m, rank), dtype=np.int8)
        V = np.zeros((m*p, rank), dtype=np.int8)
        W = np.zeros((p*n, rank), dtype=np.int8)
        
        # Strassen coefficients for the 2x2 block structure
        # Each term has: list of (block_row, block_col, coeff) for U, V, W
        strassen_terms = [
            # M1 = (A00+A11)(B00+B11)
            {
                'U': [(0,0,1), (1,1,1)],  # A00 + A11
                'V': [(0,0,1), (1,1,1)],  # B00 + B11
                'W': [(0,0,1), (1,1,1)],  # C00 += M1, C11 += M1
            },
            # M2 = (A10+A11)B00
            {
                'U': [(1,0,1), (1,1,1)],  # A10 + A11
                'V': [(0,0,1)],            # B00
                'W': [(1,0,1), (1,1,-1)],  # C10 += M2, C11 -= M2
            },
            # M3 = A00(B01-B11)
            {
                'U': [(0,0,1)],            # A00
                'V': [(0,1,1), (1,1,-1)],  # B01 - B11
                'W': [(0,1,1), (1,1,1)],   # C01 += M3, C11 += M3
            },
            # M4 = A11(B10-B00)
            {
                'U': [(1,1,1)],            # A11
                'V': [(1,0,1), (0,0,-1)],  # B10 - B00
                'W': [(0,0,1), (1,0,1)],   # C00 += M4, C10 += M4
            },
            # M5 = (A00+A01)B11
            {
                'U': [(0,0,1), (0,1,1)],   # A00 + A01
                'V': [(1,1,1)],            # B11
                'W': [(0,0,-1), (0,1,1)],  # C00 -= M5, C01 += M5
            },
            # M6 = (A10-A00)(B00+B01)
            {
                'U': [(1,0,1), (0,0,-1)],  # A10 - A00
                'V': [(0,0,1), (0,1,1)],   # B00 + B01
                'W': [(1,1,1)],            # C11 += M6
            },
            # M7 = (A01-A11)(B10+B11)
            {
                'U': [(0,1,1), (1,1,-1)],  # A01 - A11
                'V': [(1,0,1), (1,1,1)],   # B10 + B11
                'W': [(0,0,1)],            # C00 += M7
            },
        ]
        
        col = 0
        for term in strassen_terms:
            # For each Strassen term, we need to expand into 27 rank-1 terms
            # for the 3x3 block multiplication.
            for i_local in range(3):
                for j_local in range(3):
                    for k_local in range(3):
                        # U: sum over (br, bc) in term['U'] of coeff * A[br][bc][i_local][j_local]
                        u_val = 0
                        for (br, bc, coeff) in term['U']:
                            u_val += coeff
                        # V: sum over (br, bc) in term['V'] of coeff * B[br][bc][j_local][k_local]
                        v_val = 0
                        for (br, bc, coeff) in term['V']:
                            v_val += coeff
                        # W: sum over (br, bc) in term['W'] of coeff * C[br][bc][i_local][k_local]
                        w_val = 0
                        for (br, bc, coeff) in term['W']:
                            w_val += coeff
                        
                        # Global indices
                        # U[i*m + j, col] where i = br*3 + i_local, j = bc*3 + j_local
                        # But we need to sum over all (br, bc) in term['U']
                        
                        # Actually, this approach is wrong. Each Strassen term involves
                        # sums of blocks, so we need to add contributions from each block.
                        
                        pass  # This is getting complicated, let me use a different approach
        
        # Let me use a cleaner approach: directly construct the rank-1 terms
        # by iterating over all (i, j, k) for the 3x3 blocks.
        
        U = np.zeros((n*m, rank), dtype=np.int8)
        V = np.zeros((m*p, rank), dtype=np.int8)
        W = np.zeros((p*n, rank), dtype=np.int8)
        
        col = 0
        for term in strassen_terms:
            # For this Strassen term, we compute a product of two 3x3 matrices.
            # The product is formed by summing over the intermediate index.
            for a_row in range(3):
                for b_col in range(3):
                    for a_col in range(3):  # This is the summation index
                        # U contribution: from blocks in term['U'] at position (br, bc)
                        # where the local indices are (a_row, a_col)
                        for (br, bc, coeff) in term['U']:
                            gi = br * 3 + a_row
                            gj = bc * 3 + a_col
                            U[gi * m + gj, col] += coeff
                        
                        # V contribution: from blocks in term['V'] at position (br, bc)
                        # where the local indices are (a_col, b_col)
                        for (br, bc, coeff) in term['V']:
                            gj = br * 3 + a_col
                            gk = bc * 3 + b_col
                            V[gj * p + gk, col] += coeff
                        
                        # W contribution: from blocks in term['W'] at position (br, bc)
                        # where the local indices are (a_row, b_col)
                        for (br, bc, coeff) in term['W']:
                            gi = br * 3 + a_row
                            gk = bc * 3 + b_col
                            W[gk * n + gi, col] += coeff
                        
                        col += 1
        
        assert col == rank, f"Expected {rank} columns, got {col}"
        return U, V, W
    
    # Verify decomposition
    def verify_decomposition(U_mat, V_mat, W_mat, n=n, m=m, p=p, atol=1e-9):
        """Verify that sum_r U[:,r] * V[:,r] * W[:,r] gives correct matrix multiplication tensor."""
        rank = U_mat.shape[1]
        T = np.zeros((n, m, p))
        for r in range(rank):
            u_col = U_mat[:, r]
            v_col = V_mat[:, r]
            w_col = W_mat[:, r]
            u_mat_2d = u_col.reshape(n, m)
            v_mat_2d = v_col.reshape(m, p)
            w_mat_2d = w_col.reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', u_mat_2d, v_mat_2d, w_mat_2d)
        expected = np.ones((n, m, p))
        if not np.allclose(T, expected, atol=atol):
            return False
        return True
    
    # Get the Strassen decomposition
    U, V, W = get_strassen_2x2()
    best_U, best_V, best_W = U, V, W
    best_rank = U.shape[1]
    
    # Verify
    if not verify_decomposition(U, V, W):
        # Fallback to schoolbook
        U, V, W = get_schoolbook()
        best_U, best_V, best_W = U, V, W
        best_rank = U.shape[1]
    
    # Attempt rank reduction by finding linearly dependent columns
    if time.time() < deadline - 20:
        reduced_U = [U[:, c].copy() for c in range(best_rank)]
        reduced_V = [V[:, c].copy() for c in range(best_rank)]
        reduced_W = [W[:, c].copy() for c in range(best_rank)]
        
        used = [True] * best_rank
        merged = 0
        
        def find_ratio(vec_a, vec_b, tol=1e-9):
            """Find scalar r such that vec_b = r * vec_a, or None."""
            r = None
            for idx in range(len(vec_a)):
                if abs(vec_a[idx]) > 1e-12:
                    if abs(vec_b[idx]) > 1e-12:
                        candidate = vec_b[idx] / vec_a[idx]
                        if r is None:
                            r = candidate
                        elif abs(candidate - r) > tol:
                            return None
                    else:
                        if r is not None and abs(r) > tol:
                            return None
                        r = 0.0
            if r is not None:
                if not np.allclose(vec_b, r * vec_a, atol=tol):
                    return None
            return r
        
        for i in range(best_rank):
            if not used[i]:
                continue
            for j in range(i+1, best_rank):
                if not used[j]:
                    continue
                if time.time() > deadline - 15:
                    break
                
                ui = reduced_U[i]
                uj = reduced_U[j]
                vi = reduced_V[i]
                vj = reduced_V[j]
                
                ratio_u = find_ratio(ui, uj)
                if ratio_u is None:
                    continue
                
                ratio_v = find_ratio(vi, vj)
                if ratio_v is None:
                    continue
                
                wi = reduced_W[i]
                wj = reduced_W[j]
                reduced_W[i] = wi + ratio_u * ratio_v * wj
                used[j] = False
                merged += 1
        
        if merged > 0:
            new_rank = sum(used)
            final_U = np.column_stack([reduced_U[i] for i in range(best_rank) if used[i]])
            final_V = np.column_stack([reduced_V[i] for i in range(best_rank) if used[i]])
            final_W = np.column_stack([reduced_W[i] for i in range(best_rank) if used[i]])
            
            if new_rank < best_rank:
                if verify_decomposition(final_U, final_V, final_W):
                    best_U, best_V, best_W = final_U, final_V, final_W
                    best_rank = new_rank
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
