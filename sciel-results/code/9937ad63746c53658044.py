"""Matrix multiplication 2,3,5 — rank-30 decomposition via Strassen 2x2x3 blocks."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 2x3 * 3x5 matrix multiplication using rank-30 ansatz.
    
    We split the 3x5 input into two blocks: 3x2 and 3x3, or more naturally,
    we view the computation as:
      C[2,5] = A[2,3] * B[3,5]
    
    Strategy: Use the known rank-6 decomposition for 2x2x2 (Strassen) for
    sub-blocks. We decompose the 2x3x5 tensor into:
      - A 2x2x5 block (rows 0,1 of A; cols 0,1 of B... wait, let me think differently)
    
    Actually, let's use a direct structured approach:
    
    The tensor T[i,j,k] = delta (for multiplication: T[i,j,k] = 1 if A[i,j]*B[j,k])
    
    We use rank-30 = 2*3*5 (schoolbook) but with Strassen-style reductions.
    
    Better approach: We know that 2x2x2 has rank 7 (Strassen). 
    2x2x3 has rank 8 (can be shown).
    2x3x3 has rank 8.
    3x3x3 has rank 19 (or 23, unknown exact).
    
    For 2x3x5: Split the 5 columns into 2+3.
    C = A * B = A * [B1 | B2] where B1 is 3x2, B2 is 3x3.
    C1 = A * B1 (2x2), C2 = A * B2 (2x3).
    
    Rank(C1) for 2x3x2: This is like multiplying 2x3 by 3x2.
    We can split A into rows: A = [A1 | A2 | A3] where A1,A2,A3 are 2x1.
    C1 = A1*B1[0,:] + A2*B1[1,:] + A3*B1[2,:]
    
    This is getting complex. Let me use a known result:
    The rank of matrix multiplication M_{2,3,5} is at most 30 (schoolbook).
    With Strassen-like tricks, we can potentially do better.
    
    Known bounds: 18 <= rank(M_{2,3,5}) <= 30.
    
    Let me implement a rank-30 solution first (correct, score = 1.0) and then
    try to find a lower-rank solution via structured ansatz + linear algebra.
    
    For the FRESH DESIGN: I'll implement a rank-29 ansatz based on splitting
    and using one Strassen reduction.
    
    Actually, let me implement the known rank-30 schoolbook but with a 
    structured approach that can be optimized. Then I'll attempt a rank-29
    or rank-28 decomposition.
    
    Let me think about this more carefully. The multiplication tensor for
    2x3 * 3x5 has 30 entries. The schoolbook uses rank 30.
    
    Key insight: We can group terms. Consider:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    For i in {0,1}, j in {0,1,2}, k in {0,1,2,3,4}.
    
    That's 2*3*5 = 30 terms, each of rank 1.
    
    To reduce rank, we need to find linear combinations. For example,
    if we can write some product terms as combinations of others.
    
    One approach: Use the fact that for 2x2 blocks, Strassen saves 1 rank.
    Split the 3 rows of A into (2,1) and the 5 columns of B into (2,3):
    
    C = [A_top(2x2) A_bot(2x1)] * [B_tl(2x2) B_tr(2x3); B_bl(1x2) B_br(1x3)]
      = A_top*B_tl + A_bot*B_bl  (2x2 result, cols 0,1)
      + A_top*B_tr + A_bot*B_br  (2x3 result, cols 2,3,4)
    
    Wait, that's not quite right. Let me be more careful.
    
    A is 2x3: A = [[a00,a01,a02],[a10,a11,a12]]
    B is 3x5: B = [[b00,b01,b02,b03,b04],
                    [b10,b11,b12,b13,b14],
                    [b20,b21,b22,b23,b24]]
    
    C[i,k] = a_i0*b_0k + a_i1*b_1k + a_i2*b_2k
    
    Schoolbook: 2*3*5 = 30 rank-1 terms.
    
    Now, can we reduce? Consider the 2x2x2 sub-problem for (i in {0,1}, j in {0,1}, k in {0,1}):
    C[i,k] for i=0,1 and k=0,1 gets contribution from j=0,1:
    C[i,k] += a_i0*b_0k + a_i1*b_1k
    
    This 2x2x2 tensor has rank 7 (Strassen) instead of 4. Wait, that's WORSE.
    No: schoolbook for 2x2x2 is rank 4. Strassen is rank 7? No, Strassen is rank 7 for 2x2x2?
    
    Wait no. Strassen's algorithm computes 2x2 * 2x2 using 7 multiplications instead of 8.
    So the tensor rank of M_{2,2,2} is 7, not 4! I had it backwards.
    
    So schoolbook (rank 4) is BETTER than Strassen (rank 7) for 2x2x2?
    
    No wait, that can't be right either. Let me reconsider.
    
    The matrix multiplication tensor M_{n,m,p} has entries T[i,j,k] = delta_{...}
    For M_{2,2,2}, there are 8 entries, and the schoolbook decomposition has rank 8.
    Strassen shows rank 7. So Strassen IS better: 7 < 8.
    
    I was confusing myself. The "rank" here is the tensor rank (number of rank-1 terms),
    not the matrix rank. Schoolbook for 2x2x2 gives rank 8 (one term per (i,j,k)).
    Strassen gives rank 7.
    
    OK so for 2x3x5: schoolbook rank = 30.
    
    Can we do better? Known: rank(M_{2,3,5}) <= 28 (from various constructions).
    
    Let me implement a rank-28 ansatz. The strategy from the directive mentions:
    "decomposing the 2x3x5 tensor into a 2x2x5 block and a 2x1x5 block"
    
    2x2x5: split into 2x2x2 + 2x2x3. Rank(2x2x2)=7, Rank(2x2x3)=8. Total: 15? Or maybe 14.
    2x1x5: This is just 1*5 = 5 rank-1 terms. Or maybe less.
    
    Total: ~15 + 5 = 20? That seems too optimistic.
    
    Actually, let me look at this differently. I'll implement a rank-30 solution that's
    clearly correct, and then try to find a lower-rank solution using linear algebra.
    
    For the implementation, I'll use the following approach:
    1. Start with a structured ansatz of rank R (try R=28, 29, 30).
    2. Set up the linear system: for each (i,j,k) in the tensor,
       sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
       where T[i,j,k] = 1 if this is a valid multiplication entry, 0 otherwise.
       
       Wait, the tensor for matrix multiplication is:
       T[i, j, k] = 1 for all (i,j,k) since C[i,k] = sum_j A[i,j]*B[j,k]
       The tensor has a 1 at every (i,j,k) position.
       
       Actually no. The multiplication tensor is defined as:
       sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] = sum_{i,k} C[i,k]
       
       So T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor in this case? No...)
       
       Let me re-read the problem. The tensor decomposition means:
       C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
       
       No, that's not right either. The standard formulation is:
       
       C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
       
       No, that's a bilinear form, not trilinear.
       
       Actually, the matrix multiplication tensor is trilinear in (A, B, C):
       sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C[i,k] = 0 (for the "dual" formulation)
       
       Or equivalently:
       C[i,k] = sum_j A[i,j] B[j,k]
       
       The tensor rank decomposition:
       C[i,k] = sum_c (sum_j U_c[i,j] A[i,j]) (sum_j V_c[j,k] B[j,k])
       
       No, that's not right. Let me think again.
       
       The correct formulation for a rank-r decomposition of matrix multiplication:
       
       C[i,k] = sum_{c=1}^{r} W_c[i,k] * (sum_j U_c[i,j] * A[i,j]) * (sum_j V_c[j,k] * B[j,k])
       
       No, that would be rank r where each term is a product of three linear forms.
       
       Actually, I think the formulation is:
       
       We want: for all A, B, C[i,k] = sum_j A[i,j] B[j,k]
       
       This can be written as: sum_c lambda_c * L_A_c(A) * L_B_c(B) = C
       
       Where L_A_c and L_B_c are linear forms in A and B respectively, and lambda_c * L_A_c * L_B_c
       contributes to C. But C is a matrix, so we need:
       
       C[i,k] = sum_c W_c[i,k] * L_A_c(A) * L_B_c(B)
       
       Where W_c[i,k] are the output weights. This is the standard formulation.
       
       In the flattened notation:
       U has shape (n*m, rank): U[i*m+j, c] is the coefficient of A[i,j] in L_A_c
       V has shape (m*p, rank): V[j*p+k, c] is the coefficient of B[j,k] in L_B_c  
       W has shape (p*n, rank): W[k*n+i, c] is the coefficient for C[i,k] in the output
       
       And the constraint is:
       C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,j])
       
       Wait, I need to be more careful. Let me use the standard formulation:
       
       C[i,k] = sum_{c=1}^{r} (sum_j U_{i,j,c} A[i,j]) * (sum_j V_{j,k,c} B[j,k]) * W_{i,k,c}
       
       No, that's not standard either. The standard rank-r decomposition of the matrix
       multiplication tensor T in C^{n x m} * C^{m x p} -> C^{n x p} is:
       
       T = sum_{c=1}^{r} u_c * v_c * w_c
       
       where u_c in C^{n x m}, v_c in C^{m x p}, w_c in C^{n x p} (or transposed).
       
       And the action is: <T, A, B> = <C, ...> where <T, A, B> = sum T[i,j,k] A[i,j] B[j,k]
       
       Hmm, let me just look at what the code does and match it.
       
       From the seed code:
       U[i*m+j, column] = 1
       V[j*p+k, column] = 1  
       W[k*n+i, column] = 1
       
       And the verification would be:
       For the tensor T[i,j,k] (which is 1 for all i,j,k in the multiplication tensor):
       T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
       
       Wait, but that's the "flat" tensor formulation. Let me verify with schoolbook:
       For a specific (i,j,k), the schoolbook has exactly one term where U[i*m+j,c]=1,
       V[j*p+k,c]=1, W[k*n+i,c]=1 simultaneously. So T[i,j,k] = 1. ✓
       
       But what IS T[i,j,k] for matrix multiplication? 
       
       The matrix multiplication tensor is:
       T[i,j,k] = 1 for all (i,j,k)
       
       No! That would mean sum_{i,j,k} A[i,j] B[j,k] * 1, which doesn't make sense.
       
       I think the correct statement is:
       C[i,k] = sum_j A[i,j] B[j,k]
       
       And the tensor T is defined such that:
       sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,k} C[i,k] (some contraction)
       
       Actually, I think for this problem, the tensor is simply:
       T[i,j,k] = 1 for all valid (i,j,k), and the decomposition requires:
       1 = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for all (i,j,k)
       
       And the "output" C is recovered as:
       C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
       
       Let me verify this gives correct matrix multiplication for the schoolbook case.
       For schoolbook, for a fixed c corresponding to (i',j',k'):
       U[i*m+j, c] = 1 if (i,j)=(i',j') else 0
       V[j'*p+k, c] = 1 if (j',k)=(k')... 
       
       Hmm, I'm getting confused. Let me just implement the schoolbook correctly and verify.
       
       Actually, you know what, let me just carefully implement a known correct rank-30
       decomposition, verify it, and then try to reduce the rank.
       
       The schoolbook decomposition:
       For each (i,j,k) in {0..1} x {0..2} x {0..4}:
         c = i*15 + j*5 + k
         U[i*3+j, c] = 1
         V[j*5+k, c] = 1
         W[k*2+i, c] = 1
       
       This gives rank 30.
       
       Now, to reduce rank, I need to find a set of R < 30 triples (u_r, v_r, w_r)
       such that for all (i,j,k): sum_r u_r[i,j] * v_r[j,k] * w_r[i,k] = 1
       
       This is a system of 30 equations in 30*R unknowns (with R*30 parameters for U, V, W).
       
       Let me try R = 28. I'll use a structured ansatz.
       
       One idea: Use the fact that for 2x2x2, we can use rank 7 instead of 8.
       Split the 3x5 into blocks:
       - Block 1: rows 0,1 of j; cols 0,1 of k -> 2x2x2 sub-tensor
       - Block 2: rows 0,1 of j; cols 2,3,4 of k -> 2x2x3 sub-tensor
       - Block 3: row 2 of j; cols 0,1,2,3,4 of k -> 1x2x5 sub-tensor (trivial, rank 10)
       
       Wait, this doesn't quite work because the "i" dimension is shared.
       
       Let me try a different approach. I'll implement the schoolbook rank-30 and
       then try to find linear dependencies to reduce the rank.
       
       Actually, for the purpose of this task, let me just implement a clean
       rank-30 solution. The score will be 1.0 (same as seed), but at least it's
       correct and clean. Then I can iterate.
       
       But the directive says to try for lower rank. Let me implement a rank-29
       ansatz using a single Strassen-like reduction.
       
       Here's my plan:
       1. Start with schoolbook rank 30.
       2. Find one pair of terms that can be combined using a Strassen-like trick.
       
       For 2x2x2: the 8 schoolbook terms can be replaced by 7 Strassen terms.
       In our 2x3x5 problem, consider the sub-tensor for i in {0,1}, j in {0,1}, k in {0,1}.
       This is a 2x2x2 sub-tensor. The remaining terms are for j=2 or k>=2.
       
       Total terms: 30.
       Terms with j in {0,1} and k in {0,1}: 2*2*2 = 8 terms.
       Replace these 8 with 7 Strassen terms: save 1 rank.
       New total: 29.
       
       Let me implement this!
       
       The 8 schoolbook terms for (i,j,k) with i in {0,1}, j in {0,1}, k in {0,1}:
       (0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)
       
       Strassen for 2x2x2:
       m1 = (a00+a11)*(b00+b11)
       m2 = (a10+a11)*b00
       m3 = a00*(b01-b11)
       m4 = a11*(b10-b00)
       m5 = (a00+a01)*b11
       m6 = (a10-a00)*(b00+b01)
       m7 = (a01-a11)*(b10+b11)
       
       c00 = m1 + m4 - m5 + m7
       c01 = m3 + m5
       c10 = m2 + m4
       c11 = m1 - m2 + m3 + m6
       
       In tensor form, each m_r is a rank-1 term:
       m_r = L_r(A) * L_r(B) * (coefficients for C)
       
       For m1: L_r(A) = a00 + a11, L_r(B) = b00 + b11
       Contribution to C: c00 += 1, c11 += 1
       So: U[0*3+0, c]=1, U[1*3+1, c]=1, V[0*5+0, c]=1, V[1*5+1, c]=1, W[0*2+0, c]=1, W[1*2+1, c]=1
       
       For m2: L_r(A) = a10 + a11, L_r(B) = b00
       Contribution to C: c10 += 1, c11 += -1
       So: U[1*3+0, c]=1, U[1*3+1, c]=1, V[0*5+0, c]=1, W[0*2+1, c]=1, W[1*2+1, c]=-1
       
       For m3: L_r(A) = a00, L_r(B) = b01 - b11
       Contribution to C: c01 += 1, c11 += 1
       So: U[0*3+0, c]=1, V[0*5+1, c]=1, V[1*5+1, c]=-1, W[1*2+0, c]=1, W[1*2+1, c]=1
       
       For m4: L_r(A) = a11, L_r(B) = b10 - b00
       Contribution to C: c00 += 1, c10 += 1
       So: U[1*3+1, c]=1, V[1*5+0, c]=1, V[0*5+0, c]=-1, W[0*2+0, c]=1, W[0*2+1, c]=1
       
       For m5: L_r(A) = a00 + a01, L_r(B) = b11
       Contribution to C: c00 += -1, c01 += 1
       So: U[0*3+0, c]=1, U[0*3+1, c]=1, V[1*5+1, c]=1, W[0*2+0, c]=-1, W[1*2+0, c]=1
       
       For m6: L_r(A) = a10 - a00, L_r(B) = b00 + b01
       Contribution to C: c11 += 1
       So: U[1*3+0, c]=1, U[0*3+0, c]=-1, V[0*5+0, c]=1, V[0*5+1, c]=1, W[1*2+1, c]=1
       
       For m7: L_r(A) = a01 - a11, L_r(B) = b10 + b11
       Contribution to C: c00 += 1
       So: U[0*3+1, c]=1, U[1*3+1, c]=-1, V[1*5+0, c]=1, V[1*5+1, c]=1, W[0*2+0, c]=1
       
       Now, the remaining 22 schoolbook terms (for j=2 or k>=2):
       (i, 2, k) for i in {0,1}, k in {0,1,2,3,4}: 10 terms
       (i, j, k) for i in {0,1}, j in {0,1}, k in {2,3,4}: 6 terms
       Total: 16 terms.
       
       Wait: 8 (Strassen) + 16 (remaining) = 24. That's not 29.
       
       Oh I see, I double-counted. The 8 terms I'm replacing are part of the 30.
       So: 30 - 8 + 7 = 29. Yes, 29 total.
       
       The 16 remaining terms are:
       - (i, 2, k) for i in {0,1}, k in {0,1,2,3,4}: 10 terms
       - (i, j, k) for i in {0,1}, j in {0,1}, k in {2,3,4}: 6 terms
       Total: 16. ✓
       
       So total rank = 7 + 16 = 23? No wait, that's not right either.
       
       30 total schoolbook terms.
       8 of them are for (i in {0,1}, j in {0,1}, k in {0,1}).
       Replace those 8 with 7 Strassen terms.
       Remaining: 30 - 8 = 22 terms.
       Total: 7 + 22 = 29. ✓
       
       Great! So rank 29 is achievable with this simple modification.
       
       Let me implement this.
       """
    import time
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin
    
    n, m, p = 2, 3, 5
    rank = 29
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    c = 0
    
    # Strassen 2x2x2 block for i in {0,1}, j in {0,1}, k in {0,1}
    # m1 = (a00+a11)*(b00+b11), contributes to c00(+1), c11(+1)
    U[0*3+0][c] = 1; U[1*3+1][c] = 1
    V[0*5+0][c] = 1; V[1*5+1][c] = 1
    W[0*2+0][c] = 1; W[1*2+1][c] = 1
    c += 1
    
    # m2 = (a10+a11)*b00, contributes to c10(+1), c11(-1)
    U[1*3+0][c] = 1; U[1*3+1][c] = 1
    V[0*5+0][c] = 1
    W[0*2+1][c] = 1; W[1*2+1][c] = -1
    c += 1
    
    # m3 = a00*(b01-b11), contributes to c01(+1), c11(+1)
    U[0*3+0][c] = 1
    V[0*5+1][c] = 1; V[1*5+1][c] = -1
    W[1*2+0][c] = 1; W[1*2+1][c] = 1
    c += 1
    
    # m4 = a11*(b10-b00), contributes to c00(+1), c10(+1)
    U[1*3+1][c] = 1
    V[1*5+0][c] = 1; V[0*5+0][c] = -1
    W[0*2+0][c] = 1; W[0*2+1][c] = 1
    c += 1
    
    # m5 = (a00+a01)*b11, contributes to c00(-1), c01(+1)
    U[0*3+0][c] = 1; U[0*3+1][c] = 1
    V[1*5+1][c] = 1
    W[0*2+0][c] = -1; W[1*2+0][c] = 1
    c += 1
    
    # m6 = (a10-a00)*(b00+b01), contributes to c11(+1)
    U[1*3+0][c] = 1; U[0*3+0][c] = -1
    V[0*5+0][c] = 1; V[0*5+1][c] = 1
    W[1*2+1][c] = 1
    c += 1
    
    # m7 = (a01-a11)*(b10+b11), contributes to c00(+1)
    U[0*3+1][c] = 1; U[1*3+1][c] = -1
    V[1*5+0][c] = 1; V[1*5+1][c] = 1
    W[0*2+0][c] = 1
    c += 1
    
    # Remaining 22 schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip the 8 terms we already handled with Strassen
                if i in (0, 1) and j in (0, 1) and k in (0, 1):
                    continue
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    
    assert c == rank, f"Expected rank {rank}, got {c}"
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
