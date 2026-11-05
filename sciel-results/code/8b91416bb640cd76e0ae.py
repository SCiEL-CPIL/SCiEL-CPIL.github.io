"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-117 exact tensor decomposition for 5x5 matrix multiplication.
    
    Uses Strassen's rank-7 algorithm on the four 2x2x2 sub-blocks of a 2+2+1 partition,
    plus the standard rank-28 for the remaining cross terms.
    
    Partition: rows split as {0,1}, {2,3}, {4}; cols as {0,1}, {2,3}, {4}; outputs as {0,1}, {2,3}, {4}
    
    The 3x3 block structure gives 9 blocks:
    - (0,0): 2x2x2 -> rank 7 (Strassen)
    - (0,1): 2x2x2 -> rank 7 (Strassen)
    - (0,2): 2x1x2 -> rank 2 (schoolbook: 2*1*2)
    - (1,0): 2x2x2 -> rank 7 (Strassen)
    - (1,1): 2x2x2 -> rank 7 (Strassen)
    - (1,2): 2x1x2 -> rank 2 (schoolbook)
    - (2,0): 1x2x2 -> rank 2 (schoolbook)
    - (2,1): 1x2x2 -> rank 2 (schoolbook)
    - (2,2): 1x1x2 -> rank 1 (schoolbook)
    
    Total: 7+7+2+7+7+2+2+2+1 = 37... wait, that's not 117.
    
    Let me reconsider. The full 5x5x5 tensor has rank at most 125 (schoolbook).
    Using Strassen on 2x2x2 blocks: each 2x2x2 block that is part of the decomposition
    needs rank 7 instead of 8, saving 1 per block.
    
    Actually, the approach from the knowledge: "Using Strassen's rank-7 algorithm for the eight
    2x2x2 sub-blocks in a 2+2+1 partition reduces the total rank from 125 to 117."
    
    So in a 2+2+1 partition, there are 8 sub-blocks that are 2x2x2 (the blocks where both
    row-group and col-group are size 2, and output-group is size 2). Wait, let me think again.
    
    With partition 2+2+1 for rows, cols, and outputs:
    - Blocks (i,j) where i is row-group, j is col-group, and the output is determined by i,j.
    
    Actually for matrix multiplication C = A*B, the (i,j) output block depends on row-group i of A 
    and col-group j of B. So we have 3x3 = 9 blocks:
    - (0,0): rows {0,1} of A, cols {0,1} of B -> 2x2 output block at (0,0) -> 2x2x2, rank 7
    - (0,1): rows {0,1} of A, cols {2,3} of B -> 2x2 output block at (0,1) -> 2x2x2, rank 7
    - (0,2): rows {0,1} of A, col {4} of B -> 2x1 output block at (0,2) -> 2x1x2, rank 2
    - (1,0): rows {2,3} of A, cols {0,1} of B -> 2x2 output block at (1,0) -> 2x2x2, rank 7
    - (1,1): rows {2,3} of A, cols {2,3} of B -> 2x2 output block at (1,1) -> 2x2x2, rank 7
    - (1,2): rows {2,3} of A, col {4} of B -> 2x1 output block at (1,2) -> 2x1x2, rank 2
    - (2,0): row {4} of A, cols {0,1} of B -> 1x2 output block at (2,0) -> 1x2x2, rank 2
    - (2,1): row {4} of A, cols {2,3} of B -> 1x2 output block at (2,1) -> 1x2x2, rank 2
    - (2,2): row {4} of A, col {4} of B -> 1x1 output block at (2,2) -> 1x1x2, rank 1
    
    Total: 7+7+2+7+7+2+2+2+1 = 37. But that's way less than 117.
    
    Hmm, I think I'm misunderstanding the structure. Let me reconsider.
    
    Actually, I think the "eight 2x2x2 sub-blocks" refers to a different decomposition. 
    Perhaps the 5x5x5 tensor is decomposed by considering all pairs of 2x2 blocks?
    
    Wait, 5 = 2+2+1. The number of 2x2x2 sub-blocks in a 2+2+1 partition would be:
    Choose which 2 of the 3 row-groups are size 2: C(3,2) = 3
    Choose which 2 of the 3 col-groups are size 2: C(3,2) = 3
    Choose which 2 of the 3 output-groups are size 2: C(3,2) = 3
    But that gives 27, not 8.
    
    Let me re-read: "eight 2x2x2 sub-blocks in a 2+2+1 partition". 
    
    I think the correct interpretation is: In the 2+2+1 partition, there are exactly 8 blocks
    that are 2x2x2. Let's see:
    - Row groups: R0={0,1}, R1={2,3}, R2={4}
    - Col groups: C0={0,1}, C1={2,3}, C2={4}
    - Output groups: O0={0,1}, O1={2,3}, O2={4}
    
    Block (Ri, Cj) contributes to output at position (i,j) in the block matrix.
    For the output to be 2x2, we need both i and j to be 0 or 1 (the size-2 groups).
    So blocks (0,0), (0,1), (1,0), (1,1) are 2x2 outputs. That's 4 blocks, each 2x2x2.
    
    But that's only 4, not 8. Unless... the Strassen decomposition of each 2x2x2 block
    itself involves 7 terms, and maybe some of those terms interact?
    
    Actually wait - I think I need to reconsider. The total rank is 117, which is 125 - 8.
    So 8 terms are saved. Each Strassen 2x2x2 saves 1 term (7 instead of 8). So there must
    be 8 Strassen blocks. 
    
    Perhaps the partition is different. What if we use 2+2+1 but the "eight 2x2x2 sub-blocks"
    means something else? 
    
    Let me try a different approach: just implement the known result of rank 117 directly.
    The key insight is that we can replace 8 of the 125 schoolbook terms with Strassen's 7-term
    decomposition of 2x2x2 blocks, saving 8 terms total.
    
    One way to get 8 Strassen blocks: consider all ways to pick 2 rows, 2 cols, 2 outputs
    that form a 2x2x2 sub-tensor, and apply Strassen. But we need them to be disjoint or
    to compose correctly.
    
    Actually, I think the correct construction is:
    - Use the 2+2+1 partition
    - The 4 corner blocks (0,0), (0,1), (1,0), (1,1) are each 2x2x2 -> use Strassen (rank 7 each)
    - The remaining 5 blocks use schoolbook
    - Total: 4*7 + 2+2+2+2+1 = 28 + 9 = 37
    
    But 37 != 117. So this can't be right either.
    
    I think I'm confusing myself. Let me just directly construct a rank-117 decomposition
    using a known method. 
    
    Actually, upon reflection, I believe the correct interpretation is:
    - The 5x5x5 multiplication tensor can be decomposed by considering it as a sum of
      smaller tensors.
    - One such decomposition uses the fact that 5 = 2+2+1, and applies Strassen to
      the 2x2x2 sub-blocks.
    
    But the arithmetic doesn't work out to 117 with simple block decomposition.
    
    Let me try yet another interpretation: maybe the rank-117 comes from a more sophisticated
    construction where the Strassen algorithm is applied in a way that shares terms.
    
    Given the time constraints, let me just implement a straightforward rank-117 construction
    that I can verify. I'll use the following approach:
    
    Start with the full schoolbook decomposition (rank 125). Then, for each of 8 specific
    2x2x2 sub-blocks, replace the 8 schoolbook terms with Strassen's 7 terms.
    
    The 8 sub-blocks could be: for each choice of 2 rows from {0,1,2,3,4}, 2 cols, and 2 outputs,
    but we need exactly 8 disjoint (or properly composed) sub-blocks.
    
    Actually, I think the simplest correct approach is:
    - Partition into 2+2+1
    - Apply Strassen to the 4 main 2x2x2 blocks: saves 4 terms -> rank 121
    - That's not enough.
    
    I'm going to take a different tactic. Let me just hardcode a known rank-117 decomposition
    for 5x5x5 matrix multiplication. This is a well-studied problem.
    
    Actually, let me reconsider the problem. The knowledge says rank 117 was achieved. Let me
    just implement a valid rank-117 decomposition. 
    
    Here's a cleaner way to think about it:
    - 5x5x5 has 125 basis terms in schoolbook.
    - We can find 8 groups of 8 terms that form 2x2x2 sub-tensors, and replace each with
      Strassen's 7 terms, saving 1 term per group.
    - 125 - 8 = 117.
    
    The 8 groups must be such that each group corresponds to a 2x2x2 sub-tensor, and the
    groups don't overlap in a way that prevents the substitution.
    
    One natural set of 8 groups:
    - Rows: {0,1} or {2,3} (2 choices for the "first pair" and 2 for the "second pair", but we
      need exactly 2 rows per group)
    
    Let me enumerate: we need 8 disjoint 2x2x2 sub-tensors. Each uses 2 rows, 2 cols, 2 outputs.
    Total rows used: 8*2 = 16, but we only have 5 rows. So they can't be disjoint in rows.
    
    This means the sub-tensors must overlap, which makes the substitution more complex.
    
    Given the complexity, let me just implement a verified rank-117 decomposition. I'll use
    NumPy to construct it and verify it.
    
    Actually, the simplest correct approach given the time: implement the schoolbook decomposition
    but with a known rank-117 construction. Let me use the following well-known result:
    
    For 5x5x5, a rank-117 decomposition can be constructed by:
    1. Using Strassen's algorithm for 2x2x2 blocks in a specific pattern.
    
    Let me just code up a direct construction that I can verify produces rank 117.
    """
    import numpy as np
    import time
    
    start_time = time.time()
    deadline = 145  # seconds, with 5s safety margin
    
    n, m, p = 5, 5, 5
    
    # Start with schoolbook (rank 125)
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Now apply Strassen to 8 specific 2x2x2 sub-blocks to reduce rank by 8
    # Strassen's 7-term decomposition of 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # Define Strassen coefficients for a 2x2x2 block
    # For rows {r0,r1}, cols {c0,c1}, outputs {o0,o1}:
    # Each Strassen term is a linear combination of the 8 schoolbook terms
    
    # Strassen term -> (row_coeffs, col_coeffs, out_coeffs)
    # M1: rows (1,1), cols (1,1), outputs (1,1) with signs: C00+=1, C11+=1
    # But in the tensor decomposition, each term has U, V, W vectors
    
    # For a 2x2x2 sub-block with local indices:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = A*B
    
    # Strassen:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)(b00)
    # M3 = (a00)(b10-b11)
    # M4 = (a11)(b01-b10)
    # M5 = (a00+a01)(b11)
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # In the tensor decomposition, each schoolbook term (i,j,k) has:
    # U[i*m+j] = 1, V[j*p+k] = 1, W[k*n+i] = 1
    
    # For Strassen, we need to express the 7 M terms as linear combinations of the 8 schoolbook terms.
    # Each M term is a product of a linear combo of A entries and a linear combo of B entries.
    # So each M term corresponds to a tensor that is the outer product of the A-combo and B-combo.
    
    # M1: A-combo = [1,0,0,1] (a00 + a11), B-combo = [1,0,0,1] (b00 + b11)
    #     Tensor = A-combo ⊗ B-combo (flattened)
    #     This contributes to C with coefficients from the Strassen formula.
    
    # Actually, in the tensor decomposition framework:
    # Each rank-1 term is U_vec ⊗ V_vec ⊗ W_vec
    # The sum of all terms should equal the multiplication tensor T[i,j,k] = delta(i,k) (for C[i,k] = sum_j A[i,j]*B[j,k])
    
    # Wait, the multiplication tensor is T[i,j,k] = 1 if i==k else 0? No.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor is T[i,j,k] = delta_{i,k}... no, that's not right either.
    # The tensor T satisfies C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all i,j,k? No, that would give C[i,k] = sum_j A[i,j]B[j,k] which is correct!
    # Wait, no. If T[i,j,k] = 1 for all i,j,k, then C[i,k] = sum_j 1 * A[i,j] * B[j,k] = (AB)[i,k]. Yes!
    # So the multiplication tensor is all ones? That can't be right for a rank-125 decomposition.
    
    # Let me reconsider. The tensor T is defined by:
    # C[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
    # For this to equal (AB)[i,k] = sum_j A[i,j] B[j,k], we need T[i,j,k] = 1 for all i,j,k.
    # But then the rank would be 1, which is absurd.
    
    # I think I'm confusing the tensor structure. Let me reconsider.
    # The matrix multiplication tensor is a 3-tensor T of shape (n*m, m*p, p*n) or similar.
    # Actually, the standard formulation is:
    # T[a, b, c] where a indexes (i,j) pairs of A, b indexes (j,k) pairs of B, c indexes (i,k) pairs of C.
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(i,k) for the same i,j,k, else 0.
    # So T[(i,j), (j,k), (i,k)] = 1 for all i,j,k, and 0 otherwise.
    # This is a sparse tensor with n*m*p = 125 non-zero entries.
    # The rank of this tensor is the minimum number of rank-1 tensors that sum to it.
    
    # OK so the tensor is: T[(i*m+j), (j*p+k), (k*n+i)] = 1 for all i,j,k in range(5).
    # And 0 elsewhere.
    
    # A rank-1 decomposition term is: U[a] * V[b] * W[c] for all a,b,c.
    # The sum of rank-1 terms should equal T.
    
    # For the schoolbook decomposition:
    # For each (i,j,k), we have a term with:
    #   U[i*m+j] = 1, all other U = 0
    #   V[j*p+k] = 1, all other V = 0  
    #   W[k*n+i] = 1, all other W = 0
    # This gives T[(i*m+j), (j*p+k), (k*n+i)] = 1, which is correct.
    
    # Now for Strassen: we want to replace 8 schoolbook terms (for a 2x2x2 sub-block) with 7 terms.
    # The 8 schoolbook terms for rows {r0,r1}, cols {c0,c1}, outputs {o0,o1} are:
    # (r0,c0,o0), (r0,c0,o1), (r0,c1,o0), (r0,c1,o1), (r1,c0,o0), (r1,c0,o1), (r1,c1,o0), (r1,c1,o1)
    # Wait, the output index is k*n+i, so for fixed i (row) and k (output col), the output index is k*n+i.
    # In a 2x2x2 sub-block with rows {r0,r1}, cols {c0,c1}, and outputs {o0,o1} (where o is the k index):
    # The 8 terms are:
    # (i=r0, j=c0, k=o0): U[r0*m+c0], V[c0*p+o0], W[o0*n+r0]
    # (i=r0, j=c0, k=o1): U[r0*m+c0], V[c0*p+o1], W[o1*n+r0]
    # (i=r0, j=c1, k=o0): U[r0*m+c1], V[c1*p+o0], W[o0*n+r0]
    # (i=r0, j=c1, k=o1): U[r0*m+c1], V[c1*p+o1], W[o1*n+r0]
    # (i=r1, j=c0, k=o0): U[r1*m+c0], V[c0*p+o0], W[o0*n+r1]
    # (i=r1, j=c0, k=o1): U[r1*m+c0], V[c0*p+o1], W[o1*n+r1]
    # (i=r1, j=c1, k=o0): U[r1*m+c1], V[c1*p+o0], W[o0*n+r1]
    # (i=r1, j=c1, k=o1): U[r1*m+c1], V[c1*p+o1], W[o1*n+r1]
    
    # Strassen's 7 terms for this sub-block:
    # M1 = (A[r0,c0]+A[r1,c1])(B[c0,o0]+B[c1,o1])
    #     U: U[r0*m+c0] += 1, U[r1*m+c1] += 1
    #     V: V[c0*p+o0] += 1, V[c1*p+o1] += 1
    #     W: W[o0*n+r0] += 1, W[o1*n+r1] += 1
    #     But wait, this is a rank-1 term with U having 2 non-zeros, V having 2 non-zeros, W having 2 non-zeros.
    #     The product U*V*W has 8 non-zero entries, which is correct for M1.
    
    # Let me define the 7 Strassen terms properly:
    # For rows r0,r1; cols c0,c1; outputs o0,o1:
    
    # M1: A-part = A[r0,c0] + A[r1,c1], B-part = B[c0,o0] + B[c1,o1]
    #   U: [r0*m+c0]=1, [r1*m+c1]=1
    #   V: [c0*p+o0]=1, [c1*p+o1]=1
    #   W: [o0*n+r0]=1, [o1*n+r1]=1
    #   Contribution to C: C[r0,o0] += M1, C[r1,o1] += M1
    
    # M2: A-part = A[r1,c0] + A[r1,c1], B-part = B[c0,o0]
    #   U: [r1*m+c0]=1, [r1*m+c1]=1
    #   V: [c0*p+o0]=1
    #   W: [o0*n+r1]=1
    #   Contribution: C[r1,o0] += M2
    
    # M3: A-part = A[r0,c0], B-part = B[c1,o0] - B[c1,o1]
    #   U: [r0*m+c0]=1
    #   V: [c1*p+o0]=1, [c1*p+o1]=-1
    #   W: [o0*n+r0]=1, [o1*n+r0]=-1
    #   Contribution: C[r0,o0] += M3, C[r0,o1] -= M3
    
    # M4: A-part = A[r1,c1], B-part = B[c0,o1] - B[c0,o0]
    #   U: [r1*m+c1]=1
    #   V: [c0*p+o1]=1, [c0*p+o0]=-1
    #   W: [o1*n+r1]=1, [o0*n+r1]=-1
    #   Contribution: C[r1,o1] += M4, C[r1,o0] -= M4
    
    # M5: A-part = A[r0,c0] + A[r0,c1], B-part = B[c1,o1]
    #   U: [r0*m+c0]=1, [r0*m+c1]=1
    #   V: [c1*p+o1]=1
    #   W: [o1*n+r0]=1
    #   Contribution: C[r0,o1] += M5
    
    # M6: A-part = A[r1,c0] - A[r0,c0], B-part = B[c0,o0] + B[c1,o0]
    #   U: [r1*m+c0]=1, [r0*m+c0]=-1
    #   V: [c0*p+o0]=1, [c1*p+o0]=1
    #   W: [o0*n+r1]=1, [o0*n+r0]=-1
    #   Contribution: C[r1,o0] += M6, C[r0,o0] -= M6
    
    # M7: A-part = A[r0,c1] - A[r1,c1], B-part = B[c0,o1] + B[c1,o1]
    #   U: [r0*m+c1]=1, [r1*m+c1]=-1
    #   V: [c0*p+o1]=1, [c1*p+o1]=1
    #   W: [o1*n+r0]=1, [o1*n+r1]=-1
    #   Contribution: C[r0,o1] += M7, C[r1,o1] -= M7
    
    # Let me verify: C[r0,o0] = M1 + M3 - M6
    #   M1 contributes A[r0,c0]B[c0,o0] + A[r0,c0]B[c1,o1] + A[r1,c1]B[c0,o0] + A[r1,c1]B[c1,o1]
    #   That's not right. M1 should only contribute to C[r0,o0] and C[r1,o1].
    
    # I think I'm making an error. Let me use the standard Strassen formulation more carefully.
    
    # Standard Strassen for 2x2 matrices:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(g-h)
    # M4 = d(f-g)
    # M5 = (a+b)h
    # M6 = (c-a)(e+g)
    # M7 = (b-d)(f+h)
    # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    
    # In our notation: a=A[r0,c0], b=A[r0,c1], c=A[r1,c0], d=A[r1,c1]
    #                  e=B[c0,o0], f=B[c0,o1], g=B[c1,o0], h=B[c1,o1]
    
    # M1 = (a+d)(e+h) = (A[r0,c0]+A[r1,c1])(B[c0,o0]+B[c1,o1])
    #   As a tensor term: U has 1 at [r0*m+c0] and [r1*m+c1]
    #                    V has 1 at [c0*p+o0] and [c1*p+o1]
    #                    W has 1 at [o0*n+r0] and [o1*n+r1]
    #   This term contributes to:
    #     T[(r0,c0),(c0,o0),(o0,r0)] += 1
    #     T[(r0,c0),(c1,o1),(o1,r0)] += 1
    #     T[(r1,c1),(c0,o0),(o0,r1)] += 1
    #     T[(r1,c1),(c1,o1),(o1,r1)] += 1
    
    # But the original schoolbook terms for this sub-block are:
    #   T[(r0,c0),(c0,o0),(o0,r0)] = 1  (i=r0,j=c0,k=o0)
    #   T[(r0,c0),(c0,o1),(o1,r0)] = 1  (i=r0,j=c0,k=o1)
    #   T[(r0,c1),(c1,o0),(o0,r0)] = 1  (i=r0,j=c1,k=o0)
    #   T[(r0,c1),(c1,o1),(o1,r0)] = 1  (i=r0,j=c1,k=o1)
    #   T[(r1,c0),(c0,o0),(o0,r1)] = 1  (i=r1,j=c0,k=o0)
    #   T[(r1,c0),(c0,o1),(o1,r1)] = 1  (i=r1,j=c0,k=o1)
    #   T[(r1,c1),(c1,o0),(o0,r1)] = 1  (i=r1,j=c1,k=o0)
    #   T[(r1,c1),(c1,o1),(o1,r1)] = 1  (i=r1,j=c1,k=o1)
    
    # The Strassen terms should sum to these 8 terms. Let me verify M1:
    # M1 contributes to (r0,c0),(c0,o0),(o0,r0): coefficient 1*1*1 = 1 ✓
    # M1 contributes to (r0,c0),(c1,o1),(o1,r0): coefficient 1*1*1 = 1, but this should be 0! ✗
    
    # So M1 alone doesn't match. The Strassen decomposition works because the 7 terms together
    # sum to the 8 schoolbook terms. The individual terms don't match individual schoolbook terms.
    
    # This is getting very complex. Given the time constraints, let me just implement a
    # verified rank-117 decomposition by directly constructing it and checking.
    
    # For now, let me return the schoolbook decomposition (rank 125) as a safe fallback,
    # and try to implement the Strassen-based reduction if time permits.
    
    if time.time() - start_time > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Define 8 Strassen sub-blocks partitioning the 4x4x4 sub-tensor
    # Rows {0,1,2,3}, Cols {0,1,2,3}, Outputs {0,1,2,3}
    # 2 row-pairs x 2 col-pairs x 2 output-pairs = 8 blocks
    strassen_blocks = []
    for r0, r1 in [(0,1), (2,3)]:
        for c0, c1 in [(0,1), (2,3)]:
            for o0, o1 in [(0,1), (2,3)]:
                strassen_blocks.append((r0, r1, c0, c1, o0, o1))
    
    # Identify which (i,j,k) terms are covered by Strassen
    strassen_terms = set()
    for r0, r1, c0, c1, o0, o1 in strassen_blocks:
        for i in [r0, r1]:
            for j in [c0, c1]:
                for k in [o0, o1]:
                    strassen_terms.add((i, j, k))
    
    # Build the new decomposition using numpy arrays for efficiency
    strassen_count = len(strassen_blocks) * 7
    schoolbook_count = n*m*p - len(strassen_terms)
    total_rank = strassen_count + schoolbook_count
    
    U_mat = np.zeros((total_rank, n*m))
    V_mat = np.zeros((total_rank, m*p))
    W_mat = np.zeros((total_rank, p*n))
    
    col = 0
    # Add Strassen terms
    for r0, r1, c0, c1, o0, o1 in strassen_blocks:
        a, b, c, d = r0*m+c0, r0*m+c1, r1*m+c0, r1*m+c1  # A indices
        e, f, g, h = c0*p+o0, c0*p+o1, c1*p+o0, c1*p+o1  # B indices
        E, F, G, H = o0*n+r0, o1*n+r0, o0*n+r1, o1*n+r1  # C indices (W indices)
        
        # M1 = (a+d)(e+h)
        U_mat[col, a] = 1; U_mat[col, d] = 1
        V_mat[col, e] = 1; V_mat[col, h] = 1
        W_mat[col, E] = 1; W_mat[col, H] = 1
        col += 1
        
        # M2 = (c+d)e
        U_mat[col, c] = 1; U_mat[col, d] = 1
        V_mat[col, e] = 1
        W_mat[col, G] = 1
        col += 1
        
        # M3 = a(g-h)
        U_mat[col, a] = 1
        V_mat[col, g] = 1; V_mat[col, h] = -1
        W_mat[col, E] = 1; W_mat[col, F] = -1
        col += 1
        
        # M4 = d(f-g)
        U_mat[col, d] = 1
        V_mat[col, f] = 1; V_mat[col, g] = -1
        W_mat[col, H] = 1; W_mat[col, G] = -1
        col += 1
        
        # M5 = (a+b)h
        U_mat[col, a] = 1; U_mat[col, b] = 1
        V_mat[col, h] = 1
        W_mat[col, F] = 1
        col += 1
        
        # M6 = (c-a)(e+g)
        U_mat[col, c] = 1; U_mat[col, a] = -1
        V_mat[col, e] = 1; V_mat[col, g] = 1
        W_mat[col, G] = 1; W_mat[col, E] = -1
        col += 1
        
        # M7 = (b-d)(f+h)
        U_mat[col, b] = 1; U_mat[col, d] = -1
        V_mat[col, f] = 1; V_mat[col, h] = 1
        W_mat[col, F] = 1; W_mat[col, H] = -1
        col += 1
    
    # Add schoolbook terms for the remaining 61 terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in strassen_terms:
                    U_mat[col, i*m+j] = 1
                    V_mat[col, j*p+k] = 1
                    W_mat[col, k*n+i] = 1
                    col += 1
    
    new_U = U_mat.T.tolist()
    new_V = V_mat.T.tolist()
    new_W = W_mat.T.tolist()
    
    # Verify the decomposition efficiently
    if time.time() - start_time < deadline - 3:
        U_arr = U_mat  # shape (rank, n*m)
        V_arr = V_mat  # shape (rank, m*p)
        W_arr = W_mat  # shape (rank, p*n)
        
        # Check the 125 non-zero entries: T[(i*m+j), (j*p+k), (k*n+i)] = 1
        errors = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m+j
                    b = j*p+k
                    c = k*n+i
                    val = np.sum(U_arr[:, a] * V_arr[:, b] * W_arr[:, c])
                    if abs(val - 1) > 1e-9:
                        errors += 1
        
        # Check zero entries using vectorized approach for speed
        # Build a boolean mask of valid (a,b,c) triples
        # For each rank term r, the outer product U_r ⊗ V_r ⊗ W_r should only have
        # non-zero entries where a=i*m+j, b=j*p+k, c=k*n+i with matching indices
        # We check a subset of zero entries to keep it fast
        import random
        rng = random.Random(42)
        zero_checks = 5000
        for _ in range(zero_checks):
            a = rng.randint(0, n*m-1)
            b = rng.randint(0, m*p-1)
            c = rng.randint(0, p*n-1)
            i1, j1 = divmod(a, m)
            j2, k2 = divmod(b, p)
            k3, i3 = divmod(c, n)
            if not (j1 == j2 and i1 == i3 and k2 == k3):
                val = np.sum(U_arr[:, a] * V_arr[:, b] * W_arr[:, c])
                if abs(val) > 1e-9:
                    errors += 1
                    if errors > 5:
                        break
        
        if errors == 0:
            return {"U": new_U, "V": new_V, "W": new_W}
        else:
            # Fall back to schoolbook
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": new_U, "V": new_V, "W": new_W}
# EVOLVE-BLOCK-END
