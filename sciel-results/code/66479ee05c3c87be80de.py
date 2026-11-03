"""Matrix multiplication 2,4,6 — rank-minimal exact tensor decomposition."""
import time
import itertools
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 4, 6]

def _mul(A, B):
    """Exact integer matrix multiply using numpy int64."""
    return A.astype(np.int64) @ B.astype(np.int64)

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 * 4x6 matrix multiplication.
    Returns {'U', 'V', 'W'} with shapes (8, rank), (24, rank), (12, rank).
    Score is the rank (lower is better).
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s limit

    # The multiplication tensor T[i,j,k] = e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    # We seek rank-r factors U (n*m x r), V (m*p x r), W (p*n x r) such that:
    # For all (i,j,k): T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # This is equivalent to: the tensor T = sum_c u_c ⊗ v_c ⊗ w_c
    # where u_c is a vector of length n*m, v_c of length m*p, w_c of length p*n.
    #
    # Known lower bound: rank >= max(n*m, m*p, p*n) / something... actually for matrix mult
    # the rank is at least n*m*p / (some bound). The trivial rank is n*m*p = 48.
    #
    # For 2x4 * 4x6, we can try to find a decomposition with rank < 48.
    #
    # Strategy: Use the fact that we can factor through smaller intermediate dimensions.
    # A 2x4 times 4x6 = 2x6. The "natural" decomposition has rank 2*4*6 = 48.
    #
    # We can try to exploit structure. Let's think about it differently:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # One approach: Strassen-like or other bilinear algorithms.
    # For 2x2 * 2x2, Strassen gives rank 7 < 8.
    #
    # For 2x4 * 4x6, we could pad to 4x4 * 4x4 (rank 49 via Strassen-based) or use
    # other methods. But we need EXACT integer/half-integer entries.
    #
    # Let's try a direct approach: search for a low-rank decomposition.
    #
    # Key insight: The tensor T has entries that are 0 or 1. Specifically,
    # T[i,j,k] = 1 if the indices are "compatible" in the sense that the (i,j) input,
    # (j,k) middle, and (k,i) output form a valid multiplication term.
    #
    # Actually, T is a 3-tensor of size (n*m) x (m*p) x (p*n) = 8 x 24 x 12.
    # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i.
    # T[a,b,c] = 1 if the indices are consistent, i.e., if a = i*m+j, b = j*p+k, c = k*n+i
    # for some valid i,j,k. Since a determines (i,j), b determines (j,k), c determines (k,i),
    # the consistency condition is: j from a equals j from b, and k from b equals k from c,
    # and i from a equals i from c.
    #
    # So T[a,b,c] = 1 iff (a mod m) == (b // p) and (b mod p) == (c // n) and (a // m) == (c % n).
    #
    # Let's build this tensor and try to find its rank.

    # Build the tensor T as a 3D array
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    # The rank of this tensor is what we want to minimize.
    # Trivial rank: n*m*p = 48.
    #
    # Let's try to find a better decomposition using a greedy/optimization approach.
    #
    # One idea: use the fact that the tensor is a "matrix multiplication tensor" and
    # known results. For 2x4 * 4x6, the minimum rank is not known to be below 48 in general,
    # but let's try to find something.
    #
    # Actually, let's think about this more carefully. We can use block decomposition.
    # Write A as a 2x4 matrix (2 rows, 4 cols), B as 4x6 (4 rows, 6 cols).
    # C = A * B is 2x6.
    #
    # We can split B into two 4x3 blocks: B = [B1 | B2] where B1 is 4x3, B2 is 4x3.
    # Then C = [A*B1 | A*B2].
    # Each A*B1 and A*B2 is 2x3 = 6 entries, computed with rank 2*4*3 = 24 each.
    # Total rank = 48. No improvement.
    #
    # What if we use a different splitting? A is 2x4. We can think of A as having 4 columns.
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k].
    #
    # Let's try to use a "fast" algorithm. For 2xN * NxM, the rank is at most 2*N*M.
    # Can we do better?
    #
    # One approach: use the identity that for 2x2 * 2x2, Strassen uses 7 multiplications.
    # We can embed our 2x4 * 4x6 into a larger problem.
    #
    # Actually, let's just try to search for a low-rank decomposition directly.
    # We'll use a randomized approach: pick random vectors and try to decompose.
    #
    # Better idea: use the fact that the tensor is sparse and has structure.
    # Let's try to find a decomposition of rank r by solving a system of equations.
    #
    # For rank r, we have r vectors u_c (length 8), v_c (length 24), w_c (length 12).
    # The condition is: for all (a,b,c) in [8]x[24]x[12]:
    #   T[a,b,c] = sum_{d=0}^{r-1} U[a,d] * V[b,d] * W[c,d]
    #
    # This is a system of 8*24*12 = 2304 equations in 8*r + 24*r + 12*r = 44*r unknowns.
    # For r = 48: 44*48 = 2112 unknowns, 2304 equations. Overdetermined but the trivial solution works.
    # For r < 48: even more overdetermined.
    #
    # Let's try a different approach: use the structure of the problem.
    #
    # Key observation: The tensor T can be written as:
    # T = sum_{i,j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    #
    # This is exactly the sum of n*m*p = 48 rank-1 tensors. The question is whether
    # we can combine some of them to reduce the rank.
    #
    # For a sum of rank-1 tensors, the rank is at most the number of terms, but can be less
    # if there are linear dependencies.
    #
    # Let's check: are the 48 rank-1 tensors linearly independent?
    # Each term is u_{i,j} ⊗ v_{j,k} ⊗ w_{k,i} where:
    #   u_{i,j} = e_{i*m+j} (standard basis vector of length 8)
    #   v_{j,k} = e_{j*p+k} (standard basis vector of length 24)
    #   w_{k,i} = e_{k*n+i} (standard basis vector of length 12)
    #
    # These are all distinct standard basis vectors in their respective spaces, so the 48
    # rank-1 tensors are linearly independent. Thus the rank is exactly 48? No, that's not
    # necessarily true. The rank of a tensor is the minimum number of rank-1 tensors whose
    # sum equals the tensor. Linear independence of a particular decomposition doesn't mean
    # it's minimal.
    #
    # However, for matrix multiplication tensors, it's known that the rank is at least
    # n*m*p / (something). For our case, let's just try to find the minimum rank.
    #
    # Let's use a computational approach: try to find a decomposition of rank r for
    # decreasing r starting from 48.
    #
    # For small r, we can use a greedy approach: at each step, find the best rank-1
    # approximation to the residual tensor, subtract it, and repeat.
    #
    # But this is NP-hard in general. For our specific tensor, let's try a structured approach.
    #
    # Alternative: use the fact that we can "share" computations. For example, if we compute
    # A[i,j] * B[j,k] for all j, we can factor this.
    #
    # Let's try a specific construction. Consider:
    # For each j in 0..3, we have A[:,j] (2x1) and B[j,:] (1x6).
    # C = sum_j A[:,j] * B[j,:]
    # This is 4 outer products of 2x1 and 1x6, each contributing 2*6=12 entries.
    # Total: 4 * 12 = 48 scalar multiplications. This is the rank-48 decomposition.
    #
    # To do better, we need to combine terms. For example, if we could compute
    # (A[:,j1] + A[:,j2]) * (B[j1,:] + B[j2,:]) and then correct, we might save multiplications.
    # But this introduces cross terms that we need to handle.
    #
    # Let's try a Strassen-like approach. Split A into two 2x2 blocks: A = [A1 | A2]
    # where A1 is 2x2 (first 2 cols), A2 is 2x2 (last 2 cols).
    # Split B into two 2x6 blocks... wait, B is 4x6. Split B into B_top (2x6) and B_bot (2x6).
    # Then C = A1 * B_top + A2 * B_bot.
    # Each 2x2 * 2x6 can be done with rank 2*2*6 = 24, total 48. No improvement.
    #
    # What if we use Strassen for the 2x2 * 2x2 part? But B is 2x6, not 2x2.
    # We can split B_top into three 2x2 blocks: B_top = [B11 | B12 | B13].
    # Then A1 * B_top = A1*B11 + A1*B12 + A1*B13.
    # Each A1*B1i is 2x2 * 2x2, which can be done with Strassen rank 7.
    # Total for A1*B_top: 3*7 = 21. Similarly for A2*B_bot: 21. Total: 42.
    #
    # Wait, but Strassen's algorithm uses 7 multiplications for 2x2 * 2x2, but the
    # intermediate results are sums/differences of entries, not just products. The rank
    # of the 2x2 * 2x2 tensor is 7 (Strassen), so we can decompose it into 7 rank-1 terms.
    #
    # Let's verify: for 2x2 * 2x2, the tensor T_222 has rank 7. So we can write:
    # T_222 = sum_{c=1}^{7} u_c ⊗ v_c ⊗ w_c
    # where u_c is 4-dim, v_c is 4-dim, w_c is 4-dim.
    #
    # For our problem, we have:
    # C = A1 * B11 + A1 * B12 + A1 * B13 + A2 * B21 + A2 * B22 + A2 * B23
    # where A1, A2 are 2x2, B11, B12, B13, B21, B22, B23 are 2x2.
    #
    # Each term A_i * B_jk can be computed with rank 7. Total rank: 6 * 7 = 42.
    #
    # But wait, we need to be careful. The "rank" here refers to the number of scalar
    # multiplications. In the tensor decomposition sense, each 2x2 * 2x2 multiplication
    # tensor has rank 7, so we can decompose it into 7 rank-1 tensors. The total tensor
    # for our problem would be a sum of 6 such tensors, each of rank 7, giving total rank 42.
    #
    # However, we need to make sure the entries are integer/half-integer. Strassen's algorithm
    # uses additions and subtractions, so the coefficients are integers (-1, 0, 1). This should
    # work.
    #
    # Let's construct this decomposition explicitly.

    # Strassen decomposition for 2x2 * 2x2:
    # Given A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = A * B = [[c11, c12], [c21, c22]]
    #
    # Strassen's 7 products:
    # m1 = (a11 + a22) * (b11 + b22)
    # m2 = (a21 + a22) * b11
    # m3 = a11 * (b12 - b22)
    # m4 = a22 * (b21 - b11)
    # m5 = (a11 + a12) * b22
    # m6 = (a21 - a11) * (b11 + b12)
    # m7 = (a12 - a22) * (b21 + b22)
    #
    # Then:
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    #
    # In tensor form, the 2x2 * 2x2 tensor T_222 (size 4x4x4) can be decomposed as:
    # T_222 = sum_{c=1}^{7} u_c ⊗ v_c ⊗ w_c
    # where:
    # u_c are vectors of length 4 (for A entries), v_c of length 4 (for B entries), w_c of length 4 (for C entries).

    # Let's define the Strassen factors.
    # Indexing: A = [[a0, a1], [a2, a3]], B = [[b0, b1], [b2, b3]], C = [[c0, c1], [c2, c3]]
    # So A indices: 0=a11, 1=a12, 2=a21, 3=a22
    # B indices: 0=b11, 1=b12, 2=b21, 3=b22
    # C indices: 0=c11, 1=c12, 2=c21, 3=c22

    def strassen_factors():
        """Return (U, V, W) for 2x2 * 2x2 with rank 7."""
        # U: 4x7, V: 4x7, W: 4x7
        U = np.zeros((4, 7), dtype=np.int64)
        V = np.zeros((4, 7), dtype=np.int64)
        W = np.zeros((4, 7), dtype=np.int64)
        
        # m1 = (a0 + a3) * (b0 + b3)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # m2 = (a2 + a3) * b0
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1
        
        # m3 = a0 * (b1 - b3)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # m4 = a3 * (b2 - b0)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # m5 = (a0 + a1) * b3
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # m6 = (a2 - a0) * (b0 + b1)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[2, 5] = 1; W[3, 5] = 1
        
        # m7 = (a1 - a3) * (b2 + b3)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[2, 6] = 1
        
        return U, V, W
    
    # Now, for our 2x4 * 4x6 problem:
    # A is 2x4: A = [A1 | A2] where A1 = [[a0, a1], [a2, a3]], A2 = [[a4, a5], [a6, a7]]
    # B is 4x6: B = [B_top; B_bot] where B_top is 2x6, B_bot is 2x6.
    # B_top = [B11 | B12 | B13], B_bot = [B21 | B22 | B23], each 2x2.
    #
    # C = A1 * B_top + A2 * B_bot
    #   = A1*B11 + A1*B12 + A1*B13 + A2*B21 + A2*B22 + A2*B23
    #
    # Each term is a 2x2 * 2x2 multiplication, decomposable with rank 7.
    # Total rank: 6 * 7 = 42.
    #
    # Let's construct the full U, V, W matrices.
    #
    # U has shape (8, 42): rows indexed by A entries (0..7)
    # V has shape (24, 42): rows indexed by B entries (0..23)
    # W has shape (12, 42): rows indexed by C entries (0..11)
    #
    # B entries: B is 4x6, so B[j,k] with j=0..3, k=0..5.
    # B index = j*6 + k.
    # B_top = rows 0,1 of B; B_bot = rows 2,3 of B.
    # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4], B13 = B[0:2, 4:6]
    # B21 = B[2:4, 0:2], B22 = B[2:4, 2:4], B23 = B[2:4, 4:6]
    #
    # C entries: C is 2x6, so C[i,k] with i=0..1, k=0..5.
    # C index = k*2 + i (following the trace convention W[k*n+i, r]).
    #
    # For each 2x2 * 2x2 block, we need to map the local indices to global indices.
    #
    # Let's define the mapping for each of the 6 blocks.
    # Block (block_idx): (A_block, B_block, C_block)
    # A_block: which 2x2 submatrix of A (A1 or A2)
    # B_block: which 2x2 submatrix of B (B11, B12, B13, B21, B22, B23)
    # C_block: which 2x2 submatrix of C

    # A1 = A[0:2, 0:2], A2 = A[0:2, 2:4]
    # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4], B13 = B[0:2, 4:6]
    # B21 = B[2:4, 0:2], B22 = B[2:4, 2:4], B23 = B[2:4, 4:6]
    # C1 = C[0:2, 0:2], C2 = C[0:2, 2:4], C3 = C[0:2, 4:6]
    #
    # C = A1*B11 + A1*B12 + A1*B13 + A2*B21 + A2*B22 + A2*B23
    # C1 = A1*B11 + A2*B21
    # C2 = A1*B12 + A2*B22
    # C3 = A1*B13 + A2*B23
    #
    # So C1, C2, C3 are each 2x2 matrices that are sums of two 2x2 products.
    #
    # For C1 = A1*B11 + A2*B21:
    #   A1*B11 uses A entries 0,1,2,3 and B entries 0,1,6,7 (B[0,0],B[0,1],B[1,0],B[1,1])
    #   A2*B21 uses A entries 4,5,6,7 and B entries 12,13,18,19 (B[2,0],B[2,1],B[3,0],B[3,1])
    #   Both contribute to C entries 0,1,2,3 (C[0,0],C[0,1],C[1,0],C[1,1])
    #
    # Let's build the full decomposition.

    # Define the 6 blocks: (A_row_offset, A_col_offset, B_row_offset, B_col_offset, C_col_offset)
    # A is 2x4, B is 4x6, C is 2x6.
    # Block 0: A1*B11: A rows 0-1, cols 0-1; B rows 0-1, cols 0-1; C cols 0-1
    # Block 1: A1*B12: A rows 0-1, cols 0-1; B rows 0-1, cols 2-3; C cols 2-3
    # Block 2: A1*B13: A rows 0-1, cols 0-1; B rows 0-1, cols 4-5; C cols 4-5
    # Block 3: A2*B21: A rows 0-1, cols 2-3; B rows 2-3, cols 0-1; C cols 0-1
    # Block 4: A2*B22: A rows 0-1, cols 2-3; B rows 2-3, cols 2-3; C cols 2-3
    # Block 5: A2*B23: A rows 0-1, cols 2-3; B rows 2-3, cols 4-5; C cols 4-5

    blocks = [
        # (a_col_off, b_row_off, b_col_off, c_col_off)
        (0, 0, 0, 0),  # A1*B11
        (0, 0, 2, 2),  # A1*B12
        (0, 0, 4, 4),  # A1*B13
        (2, 2, 0, 0),  # A2*B21
        (2, 2, 2, 2),  # A2*B22
        (2, 2, 4, 4),  # A2*B23
    ]

    # Get Strassen factors
    S_U, S_V, S_W = strassen_factors()

    # Build full U, V, W
    rank = 6 * 7  # 42
    U = np.zeros((8, rank), dtype=np.int64)
    V = np.zeros((24, rank), dtype=np.int64)
    W = np.zeros((12, rank), dtype=np.int64)

    for block_idx, (a_col_off, b_row_off, b_col_off, c_col_off) in enumerate(blocks):
        col_start = block_idx * 7
        for c in range(7):
            col = col_start + c
            # A local indices: 0,1,2,3 map to global A indices:
            # A is 2x4, A[i,j] with i=0,1, j=0..3.
            # Local A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
            # Global A index = i*4 + j
            for local_i in range(4):
                # Local A index local_i corresponds to:
                # local_i = 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
                li = local_i % 2  # local row
                lj = local_i // 2  # local col
                # Wait, let me re-check the Strassen indexing.
                # In strassen_factors, A = [[a0, a1], [a2, a3]]
                # So a0 = A[0,0], a1 = A[0,1], a2 = A[1,0], a3 = A[1,1]
                # Local index 0 -> (0,0), 1 -> (0,1), 2 -> (1,0), 3 -> (1,1)
                # So li = local_i % 2, lj = local_i // 2? No:
                # local_i = 0: li=0, lj=0
                # local_i = 1: li=1, lj=0 -> but we want li=0, lj=1
                # So: li = local_i % 2 gives 0,1,0,1 and lj = local_i // 2 gives 0,0,1,1
                # That's not right. Let me just use explicit mapping.
                pass
            
            # Let me use explicit mappings.
            # Local A index -> (row, col) in the 2x2 block
            # 0 -> (0,0), 1 -> (0,1), 2 -> (1,0), 3 -> (1,1)
            a_local_to_rc = [(0,0), (0,1), (1,0), (1,1)]
            b_local_to_rc = [(0,0), (0,1), (1,0), (1,1)]
            c_local_to_rc = [(0,0), (0,1), (1,0), (1,1)]
            
            for local_idx in range(4):
                ar, ac = a_local_to_rc[local_idx]
                global_a = ar * 4 + (a_col_off + ac)
                U[global_a, col] += S_U[local_idx, c]
                
                br, bc = b_local_to_rc[local_idx]
                global_b = (b_row_off + br) * 6 + (b_col_off + bc)
                V[global_b, col] += S_V[local_idx, c]
                
                cr, cc = c_local_to_rc[local_idx]
                global_c = (c_col_off + cc) * 2 + cr
                W[global_c, col] += S_W[local_idx, c]

    # Verify the decomposition
    # T[a,b,c] = sum_d U[a,d] * V[b,d] * W[c,d]
    # Let's check a few entries.
    def verify(U, V, W, T, n, m, p):
        """Check if the decomposition is correct."""
        nm = n * m
        mp = m * p
        pn = p * n
        U = np.array(U, dtype=np.int64)
        V = np.array(V, dtype=np.int64)
        W = np.array(W, dtype=np.int64)
        T = np.array(T, dtype=np.int64)
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    val = sum(U[a, d] * V[b, d] * W[c, d] for d in range(U.shape[1]))
                    if val != T[a, b, c]:
                        return False, (a, b, c, val, T[a, b, c])
        return True, None

    is_valid, err = verify(U, V, W, T, n, m, p)
    
    if not is_valid:
        # Fall back to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.int64)
        V = np.zeros((m * p, rank), dtype=np.int64)
        W = np.zeros((p * n, rank), dtype=np.int64)
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
    else:
        rank = 42

    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
