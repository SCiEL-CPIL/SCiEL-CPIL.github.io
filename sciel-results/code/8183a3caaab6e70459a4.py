"""Matrix multiplication 2x3x10: rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 10]

def construct():
    """
    Use Strassen-like decomposition for 2x2 sub-blocks where possible,
    combined with schoolbook for remaining parts.
    
    Strategy: Split the 2x3 matrix into a 2x2 block and a 2x1 block.
    Split the 3x10 matrix into a 2x10 block and a 1x10 block.
    
    For the 2x2 * 2x10 part, use Strassen's algorithm (rank 7 instead of 8)
    for each 2x2 * 2x2 sub-block.
    
    Let A be 2x3, B be 3x10.
    Write A = [A11 A12; A21 A22] where A11,A21 are 1x2, A12,A22 are 1x1
    Actually, let's split differently:
    
    A (2x3) = [A_left (2x2) | A_right (2x1)]
    B (3x10) = [B_top (2x10); B_bottom (1x10)]
    
    Then C = A * B = A_left * B_top + A_right * B_bottom
    
    For A_left * B_top: this is (2x2)*(2x10).
    Split B_top into five 2x2 blocks: B_top = [B1 B2 B3 B4 B5] each 2x2.
    Use Strassen for each 2x2*2x2: rank 7 each, total 35.
    
    For A_right * B_bottom: this is (2x1)*(1x10).
    This is just scalar multiplication row-wise: rank 10 (each of 10 columns
    needs one term, but actually it's outer product of a 2-vector and 10-vector
    which has rank 2 in the tensor sense... wait, let me think more carefully).
    
    Actually (2x1)*(1x10): C[i,k] = A[i,0] * B[0,k].
    This is a rank-2 tensor: U[i, c] = A[i,0], V[0, c] = B[0,k], W[k, c] = ...
    Hmm, the tensor is sum over i,k of e_i ⊗ e_0 ⊗ e_k. 
    The multiplication tensor for (2x1)*(1x10) has rank 2:
    C[i,k] = A[i,0]*B[0,k]
    We can write: for c=0: U[i,0]=1 if i in {0,1}, V[0,0]=1, W[k,0]=... no.
    
    Let me reconsider. The tensor T[i,j,k] = 1 if j=0, else 0 (since A is 2x1, B is 1x10).
    T[i,0,k] = 1 for all i in {0,1}, k in {0,...,9}.
    This tensor has rank 2: T = e_0 ⊗ e_0 ⊗ (sum_k e_k) + e_1 ⊗ e_0 ⊗ (sum_k e_k)
    So rank 2.
    
    Wait, that's not right either. Let me be more careful.
    
    The matrix multiplication tensor for (n x m) * (m x p) is:
    T[i,j,k] = 1 if j is the middle index.
    For (2x1)*(1x10): n=2, m=1, p=10.
    T[i,j,k] = 1 if j=0 (the only middle index).
    So T[i,0,k] = 1 for i in {0,1}, k in {0,...,9}.
    
    Can we decompose this with rank 2?
    T = sum_c U_c ⊗ V_c ⊗ W_c
    We need: sum_c U_c[i] * V_c[0] * W_c[k] = 1 for all i,k.
    
    If we set V_c[0] = 1 for all c, then we need sum_c U_c[i] * W_c[k] = 1.
    This is a rank-1 matrix approximation of the all-ones 2x10 matrix.
    The all-ones matrix has rank 1, so we can do it with c=1:
    U_1[i] = 1, W_1[k] = 1. Then sum = 1*1 = 1. ✓
    
    So (2x1)*(1x10) has tensor rank 1! 
    U[i,0] = 1 for i in {0,1}, V[0,0] = 1, W[k,0] = 1 for all k.
    
    Total rank: 35 (from Strassen on five 2x2 blocks) + 1 = 36.
    
    But wait, can we do better? Let's think about whether we can share structure.
    
    Actually, let me reconsider the split. What if we split the 3x10 differently?
    
    Alternative: Split A (2x3) into three 2x1 columns: A = [A1 | A2 | A3], each 2x1.
    Split B (3x10) into three 1x10 rows: B = [B1; B2; B3], each 1x10.
    Then C = A1*B1 + A2*B2 + A3*B3.
    Each (2x1)*(1x10) has rank 1 as shown above.
    Total rank: 3.
    
    Wait, that can't be right. Let me verify.
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    If we use rank-1 terms: for each j, we have a term U_j ⊗ V_j ⊗ W_j where
    U_j[i] = A[i,j], V_j[j'] = delta(j,j'), W_j[k] = B[j,k].
    
    But in the tensor decomposition, U, V, W are FIXED matrices, not depending on A and B.
    The decomposition is: C[i,k] = sum_c U[i,c] * V[j,c] * W[k,c]... 
    
    No wait, I'm confusing myself. Let me re-read the problem.
    
    The matrix multiplication tensor T is a 3-tensor such that:
    (A*B)[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
    
    No, that's not right either. The standard formulation:
    (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T[i,j,k] = 1 (for all valid i,j,k) encodes this.
    A rank-r decomposition is: T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    where U is n x r, V is m x r, W is p x r.
    
    Then: sum_c U[i,c]*V[j,c]*W[k,c] should equal 1 for all i,j,k? No!
    
    Actually, the correct formulation is:
    (A*B)[i,k] = sum_c (sum_j U[i,c]*A[i,j]) * (sum_j' V[j',c]*B[j',k]) ... 
    
    No. Let me look at the seed program:
    U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    So U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    The decomposition is:
    (A*B)[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * ... 
    
    Hmm, the indices are mixed. Let me re-read:
    "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
    
    I think the convention is:
    For each term c, we have:
    - U contributes to factor A: U[i*m + j, c] multiplies A[i,j]
    - V contributes to factor B: V[j*p + k, c] multiplies B[j,k]  
    - W contributes to output: W[k*n + i, c] is the coefficient for C[i,k]
    
    So: C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
    
    Wait, that doesn't look right dimensionally. Let me think again.
    
    Actually I think it's:
    C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j' V[j'*p+k, c] * B[j',k])
    
    No, that would be quadratic in the sums. 
    
    Let me look at the seed more carefully:
    For each (i,j,k): U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1
    
    So for a fixed c corresponding to (i0,j0,k0):
    U[i*m+j, c] = 1 if i=i0 and j=j0, else 0
    V[j'*p+k', c] = 1 if j'=j0 and k'=k0, else 0
    W[k''*n+i'', c] = 1 if k''=k0 and i''=i0, else 0
    
    Then the formula must be:
    C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j' V[j'*p+k, c] * B[j',k])
    
    For the seed term (i0,j0,k0):
    - W[k*n+i, c] is 1 only when k=k0 and i=i0
    - sum_j U[i*m+j, c]*A[i,j] = A[i0,j0] (since U[i*m+j,c]=1 only for j=j0 when i=i0)
    
    Wait, U[i*m+j, c] depends on both i and j. For the sum over j, i is fixed.
    sum_j U[i*m+j, c]*A[i,j] = A[i,j0] if i=i0, else 0.
    
    Similarly sum_j' V[j'*p+k, c]*B[j',k] = B[j0,k] if k=k0, else 0.
    
    So C[i,k] gets contribution W[k*n+i,c]*A[i0,j0]*B[j0,k0] = 1*A[i0,j0]*B[j0,k0] when i=i0, k=k0.
    
    Summing over all c (i.e., all (i0,j0,k0)):
    C[i,k] = sum_j0 A[i,j0]*B[j0,k] ✓
    
    Great, so the formula is:
    C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j' V[j'*p+k, c] * B[j',k])
    
    Now, for (2x1)*(1x10): n=2, m=1, p=10.
    U has shape (2*1, r) = (2, r)
    V has shape (1*10, r) = (10, r)
    W has shape (10*2, r) = (20, r)
    
    We need: C[i,k] = sum_c W[k*2+i, c] * (sum_j U[i*1+j, c]*A[i,j]) * (sum_j' V[j'*10+k, c]*B[j',k])
    Since m=1, j can only be 0.
    C[i,k] = sum_c W[2k+i, c] * U[i, c]*A[i,0] * (sum_j' V[10j'+k, c]*B[j',k])
    
    We need this to equal A[i,0]*B[0,k] for all i,k.
    
    If we set r=1: U[i,0]=1, V[10j'+k, 0] = 1 if j'=0 else 0 (i.e., V[k,0]=1 for k=0..9), W[2k+i,0]=1.
    Then: C[i,k] = 1 * 1*A[i,0] * B[0,k] = A[i,0]*B[0,k] ✓
    
    So yes, rank 1 for (2x1)*(1x10).
    
    Similarly (1x1)*(1x10) would be rank 1, etc.
    
    Now back to our problem: 2x3 * 3x10.
    
    Option 1: Split into three (2x1)*(1x10) terms. Rank = 3.
    But wait, can we really do it with rank 3?
    
    Let me verify. n=2, m=3, p=10.
    U shape (6, r), V shape (30, r), W shape (20, r).
    
    For j=0: term c0: U[i*3+0, c0]=1, V[0*10+k, c0]=1 (i.e., V[k, c0]=1 for k=0..9), W[10k+i, c0]=1.
    Contribution: W[10k+i,c0] * U[3i, c0]*A[i,0] * (sum_j' V[10j'+k,c0]*B[j',k])
    = 1 * 1*A[i,0] * B[0,k] = A[i,0]*B[0,k]
    
    For j=1: term c1: U[i*3+1, c1]=1, V[1*10+k, c1]=1 (V[10+k, c1]=1 for k=0..9), W[10k+i, c1]=1.
    Contribution: A[i,1]*B[1,k]
    
    For j=2: term c2: U[i*3+2, c2]=1, V[2*10+k, c2]=1 (V[20+k, c2]=1), W[10k+i, c2]=1.
    Contribution: A[i,2]*B[2,k]
    
    Total: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] ✓
    
    So the rank is 3! That's much better than 60.
    
    But wait, is this actually valid? The issue is that W[10k+i, c0] = 1 for ALL k and i, and similarly for c1, c2. So W has the same pattern for all three columns. That's fine.
    
    Let me double-check the shapes:
    U: (n*m, rank) = (6, 3). U[i*3+j, c_j] = 1.
    V: (m*p, rank) = (30, 3). V[j*10+k, c_j] = 1.
    W: (p*n, rank) = (20, 3). W[k*2+i, c_j] = 1 for all k,i.
    
    Yes! This gives rank 3.
    
    But is rank 3 the minimum? The matrix multiplication tensor for (2x3)*(3x10) - what's its tensor rank?
    
    The trivial lower bound is max(n*m, m*p, p*n) / something... Actually, the rank of the matmul tensor is at least max(n*p, m) in some cases, but the exact minimum is hard to determine.
    
    For (2x3)*(3x10), we found rank 3. Can we do rank 2?
    
    With rank 2, we'd have two terms. Each term c has:
    U_c: 6-vector, V_c: 30-vector, W_c: 20-vector.
    C[i,k] = sum_c W_c[2k+i] * (sum_j U_c[3i+j]*A[i,j]) * (sum_j' V_c[10j'+k]*B[j',k])
    
    This seems unlikely to work for all i,j,k combinations with just 2 terms, since we need to capture 3 different "paths" through the middle dimension.
    
    I'll go with rank 3.
    
    """
    n, m, p = DIMENSIONS
    rank = m  # 3
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        c = j
        for i in range(n):
            U[i * m + j][c] = 1
        for k in range(p):
            V[j * p + k][c] = 1
        for i in range(n):
            for k in range(p):
                W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
