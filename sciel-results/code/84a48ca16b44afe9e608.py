import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # The tensor equation is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s deadline, 10s safety margin
    
    # Schoolbook fallback: rank 72
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best = {"U": U, "V": V, "W": W}
    best_rank = rank_schoolbook
    
    # Strategy: Direct construction of a rank-66 decomposition using block structure.
    # We partition the 3x4 matrix A into blocks and 4x6 matrix B into blocks,
    # then apply Strassen-like recurrences to reduce the number of multiplications.
    #
    # A is 3x4: partition as A = [A11 A12; A21 A22] where A11,A12 are 2x2, A21,A22 are 1x2
    # B is 4x6: partition as B = [B11 B12; B21 B22] where B11,B21 are 2x3, B12,B22 are 2x3
    #
    # Actually, let's use a more systematic approach:
    # Partition A (3x4) into 3 row blocks of 1x4? No, that's too fine.
    # Let's try: A = [A_top (2x4); A_bottom (1x4)]
    # B = [B_left (4x3); B_right (4x3)]
    #
    # Then C = A*B is 3x6.
    # C_top = A_top * B  (2x6)
    # C_bottom = A_bottom * B  (1x6)
    #
    # This doesn't immediately help. Let's use the known result that rank(3,4,6) <= 66.
    # We'll construct a rank-66 decomposition by combining:
    # - 2 Strassen 2x2x2 blocks for the 2x2x2 subproblems
    # - Additional terms for the remaining dimensions
    #
    # A cleaner approach: Use the fact that we can embed the 3x4x6 problem into a larger
    # structured problem and use known low-rank decompositions for sub-blocks.
    #
    # Let's use a direct algebraic construction:
    # Partition A (3x4) as:
    #   A = [[a11 a12 a13 a14],
    #        [a21 a22 a23 a24],
    #        [a31 a32 a33 a34]]
    #
    # Partition B (4x6) as:
    #   B = [[b11 b12 b13 b14 b15 b16],
    #        [b21 b22 b23 b24 b25 b26],
    #        [b31 b32 b33 b34 b35 b36],
    #        [b41 b42 b43 b44 b45 b46]]
    #
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # We'll use a decomposition based on splitting the 4 columns of A / 4 rows of B
    # into 2+2, and the 6 columns of B / 3 rows of C into 3+3.
    #
    # Let A = [A1 A2] where A1 is 3x2, A2 is 3x2
    # Let B = [B1; B2] where B1 is 2x6, B2 is 2x6
    # Then C = A1*B1 + A2*B2
    #
    # Each A_i is 3x2, B_i is 2x6, so each product is 3x6.
    # Rank of 3x2x6: schoolbook is 18. But we can do better.
    #
    # For a 3x2x6 multiplication:
    #   A (3x2) * B (2x6) = C (3x6)
    #   C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    #   Schoolbook rank = 3*2*6 = 36? No, that's overcounting.
    #   Actually, for fixed i, C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    #   This is a sum of 2 rank-1 terms in the (k) dimension for each i.
    #   Total rank = 3 * 2 = 6? No, the tensor is T[i,j,k] = delta_j0*B[j,k]*A[i,j]... 
    #   Let me think more carefully.
    #
    # The multiplication tensor for m x n x p has rank at most m*n*p (schoolbook).
    # For 3x2x6: schoolbook rank = 3*2*6 = 36. But we can do much better.
    #
    # Actually, a 3x2 matrix times a 2x6 matrix:
    #   Let A = [a1 a2; a3 a4; a5 a6] (3x2)
    #   Let B = [b11 b12 ... b16; b21 b22 ... b26] (2x6)
    #   C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    #
    # We can write this as:
    #   C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k]
    #
    # This is already a rank-2 decomposition in the j dimension, but we need to express
    # it as a sum of rank-1 tensors in (i,j,k).
    #
    # A rank-1 tensor is u_i * v_j * w_k.
    # The multiplication tensor is T[i,j,k] = delta_{j,0} * 1 * 1 + delta_{j,1} * 1 * 1? No.
    # T[i,j,k] = 1 if the product involves A[i,j] and B[j,k].
    # Actually, the tensor for matrix multiplication is:
    #   T[i,j,k] = 1 for all i,j,k (it's the identity tensor in the middle dimension)
    #   No, that's not right either.
    #
    # Let me re-derive. The matrix multiplication tensor T satisfies:
    #   C = A * B  <=>  C[i,k] = sum_j A[i,j] * B[j,k]
    #   This can be written as:  C[i,k] = sum_{j,c} A[i,j] * B[j,k] * W[i,k,c] * U[i,j,c] * V[j,k,c]
    #   where the decomposition is: sum_c U[i,j,c] * V[j,k,c] * W[i,k,c] = delta_{j,j} = 1
    #   Wait, no. The tensor is:  T[i,j,k] such that C[i,k] = sum_j A[i,j] B[j,k] T[i,j,k]
    #   Here T[i,j,k] = 1 for all i,j,k? No, T is just 1, and the multiplication is:
    #   C[i,k] = sum_j A[i,j] * B[j,k]
    #   So the tensor is T[i,j,k] = 1 for all i,j,k.
    #
    # A rank-1 decomposition is: 1 = U[i,j] * V[j,k] * W[i,k]
    # This is impossible for a single term unless we use the structure.
    #
    # The standard schoolbook decomposition uses rank = m*n*p:
    #   For each (i,j,k): U[i*m+j, c] = delta_{i,j}, V[j*p+k, c] = delta_{j,k}, W[k*n+i, c] = delta_{i,k}
    #   Wait, that's not right. Let me use the correct indexing.
    #
    # For multiplying A (n x m) by B (m x p) to get C (n x p):
    #   C[i,k] = sum_j A[i,j] * B[j,k]
    #   Schoolbook: for each (i,j,k), one term:
    #     U[i*m + j, c] = 1, V[j*p + k, c] = 1, W[k*n + i, c] = 1
    #   Then sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for the matching (i,j,k)
    #   and 0 otherwise? No, that gives 1 for every (i,j,k) combination, which is correct
    #   because the tensor T[i,j,k] = 1 for all i,j,k.
    #
    # OK so the schoolbook rank is n*m*p = 3*4*6 = 72.
    #
    # To reduce the rank, we need to find a set of (U_c, V_c, W_c) such that:
    #   sum_c U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i] = 1  for all i,j,k
    #
    # This is a system of n*m*p = 72 equations in 3*rank*... variables.
    #
    # Let's use a known low-rank decomposition. For 3x4x6, the best known rank is 66.
    # I'll construct it using a block-Strassen approach.
    #
    # Partition A (3x4) into [A1; A2] where A1 is 2x4, A2 is 1x4
    # Partition B (4x6) into [B1 B2] where B1 is 4x3, B2 is 4x3
    #
    # C = A*B = [A1; A2] * [B1 B2] = [A1*B1  A1*B2; A2*B1  A2*B2]
    #
    # A1*B1: 2x4 * 4x3 = 2x3, schoolbook rank 24
    # A1*B2: 2x4 * 4x3 = 2x3, schoolbook rank 24
    # A2*B1: 1x4 * 4x3 = 1x3, schoolbook rank 12
    # A2*B2: 1x4 * 4x3 = 1x3, schoolbook rank 12
    # Total schoolbook: 24+24+12+12 = 72
    #
    # Now apply Strassen to A1*B1 and A1*B2:
    # A1 is 2x4, B1 is 4x3. Split A1 = [A11 A12] (2x2, 2x2), B1 = [B11; B12] (2x3, 2x3)
    # A1*B1 = A11*B11 + A12*B12
    # Each is 2x2 * 2x3 = 2x3, schoolbook rank 12.
    # Apply Strassen to 2x2x3: 
    #   2x2 * 2x3: C = A*B where A is 2x2, B is 2x3.
    #   C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    #   Schoolbook rank = 2*2*3 = 12.
    #   Can we do better? 
    #   Let's use: M1 = (A00+A11)*B00, M2 = (A00+A11)*B10, M3 = A00*(B00-B10), M4 = A11*(B01-B11)... 
    #   This is getting complex. Let me just use a known rank-66 construction.
    #
    # I'll use a pre-computed rank-66 decomposition. Since I can't search for it in 140s,
    # I'll use a structured approach that is known to achieve rank 66.
    #
    # Alternative: Use the fact that we can reduce the rank by exploiting the rectangular shape.
    # For 3x4x6, we can use 6 Strassen 2x2x2 blocks (each rank 7) = 42, plus additional terms.
    #
    # Let me try a different partition:
    # A (3x4) = [A11 A12; A21 A22; A31 A32] where each block is 1x2? No, that's 3x2 blocks.
    # Let's do: A (3x4) split into 3 rows of 1x4.
    # B (4x6) split into 4 rows of 1x6.
    # This gives 3*4 = 12 products of 1x4 * 4x6 = 1x6, each rank 4 (schoolbook).
    # Total: 12*4 = 48? No, that's not how it works.
    #
    # I think the cleanest approach is to use a pre-computed set of factors.
    # Since I cannot reliably compute a rank-66 decomposition on the fly in 140s,
    # I'll use a rank-68 decomposition (which is easier to construct) and hope it scores better than 72.
    #
    # Actually, let me try to construct a rank-66 decomposition using the following method:
    # 1. Start with schoolbook rank 72.
    # 2. Use 6 Strassen reductions, each reducing rank by 1 (from 8 to 7 for 2x2x2).
    #    6 * 1 = 6 reduction, giving rank 66.
    #
    # To do this, I need to identify 6 disjoint 2x2x2 subproblems within the 3x4x6 problem.
    #
    # Let's partition:
    # A (3x4) into blocks: rows {0,1}, {2}, cols {0,1}, {2,3}
    #   A11 (2x2), A12 (2x2), A21 (1x2), A22 (1x2)
    # B (4x6) into blocks: rows {0,1}, {2,3}, cols {0,1,2}, {3,4,5}
    #   B11 (2x3), B12 (2x3), B21 (2x3), B22 (2x3)
    #
    # C = A*B (3x6)
    # C11 = A11*B11 (2x2 * 2x3 = 2x3)
    # C12 = A11*B12 + A12*B22? No, C12 = A11*B12 + A12*B21? Let me be more careful.
    #
    # A = [[A11 A12],
    #      [A21 A22]]  where A11 is 2x2, A12 is 2x2, A21 is 1x2, A22 is 1x2
    # B = [[B11 B12],
    #      [B21 B22]]  where B11 is 2x3, B12 is 2x3, B21 is 2x3, B22 is 2x3
    #
    # C = [[A11*B11 + A12*B21,  A11*B12 + A12*B22],
    #      [A21*B11 + A22*B21,  A21*B12 + A22*B22]]
    #
    # C11: 2x2 * 2x3 + 2x2 * 2x3 = two 2x2x3 products
    # C12: 2x2 * 2x3 + 2x2 * 2x3 = two 2x2x3 products
    # C21: 1x2 * 2x3 + 1x2 * 2x3 = two 1x2x3 products
    # C22: 1x2 * 2x3 + 1x2 * 2x3 = two 1x2x3 products
    #
    # For each 2x2x3 product, schoolbook rank is 12. Can we do rank 11?
    # For 2x2x3: C[i,k] = sum_j A[i,j] B[j,k]
    #   = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    #   Schoolbook: 2*2*3 = 12 terms.
    #   Strassen for 2x2x2 gives rank 7. For 2x2x3, we can use:
    #   M1 = (A00+A11)*B00, M2 = (A00+A11)*B10, M3 = A00*(B00-B10), M4 = A11*(B01-B11),
    #   M5 = A01*(B10+B11), M6 = A10*(B00-B01), M7 = (A10+A11)*B11, M8 = ... 
    #   This is getting too complex to derive on the fly.
    #
    # Given the time constraints, I'll use a rank-68 decomposition which is easier to construct:
    # 8 Strassen 2x2x2 blocks (rank 7 each) = 56, plus 12 additional terms = 68.
    #
    # Let me construct a rank-68 decomposition:
    # Partition A (3x4) into 2x2, 2x2, 1x2, 1x2 blocks (rows 0-1, 2; cols 0-1, 2-3)
    # Partition B (4x6) into 2x3, 2x3, 2x3, 2x3 blocks (rows 0-1, 2-3; cols 0-2, 3-5)
    #
    # This gives 4 subproblems:
    # C11 = A11*B11 + A12*B21 (2x2*2x3 + 2x2*2x3)
    # C12 = A11*B12 + A12*B22 (2x2*2x3 + 2x2*2x3)
    # C21 = A21*B11 + A22*B21 (1x2*2x3 + 1x2*2x3)
    # C22 = A21*B12 + A22*B22 (1x2*2x3 + 1x2*2x3)
    #
    # For the 2x2x3 products, use schoolbook rank 12 each: 4*2*12 = 96? No, that's too much.
    #
    # I think I'm overcomplicating this. Let me just use a known rank-66 decomposition
    # by hardcoding the factors. Since I can't derive them reliably, I'll use a simpler
    # approach: try to find a rank-66 decomposition using a more targeted search.
    #
    # Actually, let me use a different strategy: use the fact that the problem is small
    # enough to use a direct algebraic construction. I'll use the following approach:
    #
    # 1. Start with schoolbook rank 72.
    # 2. Apply 6 Strassen reductions to get rank 66.
    #
    # Each Strassen reduction replaces 8 multiplications with 7, reducing rank by 1.
    # I need to identify 6 disjoint 2x2x2 subproblems.
    #
    # For 3x4x6, I can identify the following 2x2x2 subproblems:
    # - (rows 0-1 of A, cols 0-1 of A) x (rows 0-1 of B, cols 0-1 of B) -> 2x2x2
    # - (rows 0-1 of A, cols 0-1 of A) x (rows 0-1 of B, cols 2-3 of B) -> 2x2x2
    # - (rows 0-1 of A, cols 2-3 of A) x (rows 0-1 of B, cols 0-1 of B) -> 2x2x2
    # - (rows 0-1 of A, cols 2-3 of A) x (rows 0-1 of B, cols 2-3 of B) -> 2x2x2
    # - (rows 0-1 of A, cols 0-1 of A) x (rows 2-3 of B, cols 0-1 of B) -> 2x2x2
    # - (rows 0-1 of A, cols 0-1 of A) x (rows 2-3 of B, cols 2-3 of B) -> 2x2x2
    #
    # But these overlap, so I can't just apply Strassen to each independently.
    #
    # Given the complexity, I'll use a pragmatic approach: construct a rank-68 decomposition
    # using 8 Strassen 2x2x2 blocks and 12 additional terms, which should be achievable.
    #
    # For now, I'll return the schoolbook decomposition (rank 72) as a safe fallback,
    # and try to improve it if time permits.
    
    # Try to construct a rank-66 decomposition using a direct approach
    # We'll use a combination of Strassen and schoolbook terms
    
    # For a 2x2x2 Strassen block, the 7 rank-1 terms are:
    # M1 = (A00+A11)*B00
    # M2 = (A00+A11)*B10  
    # M3 = A00*(B00-B10)
    # M4 = A11*(B01-B11)
    # M5 = A01*(B10+B11)
    # M6 = A10*(B00-B01)
    # M7 = (A10+A11)*B11
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let me construct a rank-66 decomposition by combining:
    # - 6 Strassen 2x2x2 blocks (6*7 = 42 terms)
    # - 24 additional schoolbook terms for the remaining parts
    # Total: 42 + 24 = 66
    
    # I'll define the 6 Strassen blocks and 24 schoolbook terms explicitly.
    
    # This is complex, so I'll use a simpler approach: just return schoolbook for now
    # and note that a rank-66 decomposition exists but requires more careful construction.
    
    # For the purpose of this task, I'll return the schoolbook decomposition.
    return best

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s deadline, 10s safety margin
    
    # Schoolbook fallback: rank 72
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best = {"U": U, "V": V, "W": W}
    best_rank = rank_schoolbook
    
    # Strategy: Direct construction of a rank-66 decomposition using block-Strassen.
    # We'll use 6 Strassen 2x2x2 blocks (each rank 7) = 42 terms,
    # plus 24 schoolbook terms for the remaining parts = 66 total.
    
    # Define the 6 Strassen blocks. Each block covers a 2x2x2 subproblem.
    # Block (r1, r2, c1, c2, k1, k2) covers:
    #   A rows r1,r2; A cols c1,c2; B rows c1,c2; B cols k1,k2
    #   This contributes to C rows r1,r2 and C cols k1,k2.
    
    # The 6 Strassen blocks:
    # Block 0: A rows 0-1, A cols 0-1, B rows 0-1, B cols 0-1
    # Block 1: A rows 0-1, A cols 0-1, B rows 0-1, B cols 2-3
    # Block 2: A rows 0-1, A cols 2-3, B rows 0-1, B cols 0-1
    # Block 3: A rows 0-1, A cols 2-3, B rows 0-1, B cols 2-3
    # Block 4: A rows 0-1, A cols 0-1, B rows 2-3, B cols 0-1
    # Block 5: A rows 0-1, A cols 0-1, B rows 2-3, B cols 2-3
    
    # Wait, this doesn't cover all parts. Let me reconsider.
    
    # A is 3x4, B is 4x6.
    # Let's partition A into 2x2, 2x2, 1x2, 1x2 (rows 0-1, 2; cols 0-1, 2-3)
    # Let's partition B into 2x3, 2x3, 2x3, 2x3 (rows 0-1, 2-3; cols 0-2, 3-5)
    
    # C11 = A11*B11 + A12*B21 (2x2*2x3 + 2x2*2x3)
    # C12 = A11*B12 + A12*B22 (2x2*2x3 + 2x2*2x3)
    # C21 = A21*B11 + A22*B21 (1x2*2x3 + 1x2*2x3)
    # C22 = A21*B12 + A22*B22 (1x2*2x3 + 1x2*2x3)
    
    # For each 2x2x3 product, we can use a rank-11 decomposition (schoolbook is 12).
    # For each 1x2x3 product, we can use a rank-3 decomposition (schoolbook is 6? No, 1*2*3=6).
    
    # This is getting too complex. Let me just use a known rank-66 decomposition
    # by hardcoding the factors. I'll use a pre-computed set of factors.
    
    # Since I can't reliably compute a rank-66 decomposition on the fly,
    # I'll use a rank-68 decomposition which is easier to construct.
    
    # For now, I'll return the schoolbook decomposition (rank 72) as a safe fallback.
    return best
# EVOLVE-BLOCK-END
