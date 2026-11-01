"""Strassen-based rank-56 decomposition for 2x2 @ 2x16 matrix multiplication."""
import time

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    """
    Build a rank-56 exact tensor decomposition for 2x2 @ 2x16.
    
    Strategy: The 2x16 matrix B has 4 independent 2x2 blocks.
    We use Strassen's rank-7 algorithm for each 2x2 @ 2x2 block.
    Since the A matrix (2x2) is shared across all 4 blocks, we can
    share the Strassen factorizations of A.
    
    For Strassen 2x2:
      M1 = (a00+a11)(b00+b11)
      M2 = (a10+a11)b00
      M3 = a00(b01-b11)
      M4 = a11(b10-b00)
      M5 = (a00+a01)b11
      M6 = (a10-a00)(b00+b01)
      M7 = (a01-a11)(b10+b11)
    
    c00 = M1 + M4 - M5 + M7
    c01 = M3 + M5
    c10 = M2 + M4
    c11 = M1 - M2 + M3 + M6
    
    For 2x2 @ 2x16, B = [B0 | B1 | B2 | B3] where each Bi is 2x2.
    C = [C0 | C1 | C2 | C3] where Ci = A @ Bi.
    
    Each Ci uses the same 7 Strassen products M1..M7 (since A is shared).
    So total rank = 7 (for A's Strassen factors) + 4*4 (for combining) = wait,
    let me think more carefully.
    
    Actually, the tensor decomposition rank is the number of rank-1 terms.
    
    For 2x2 @ 2x2, Strassen gives rank 7:
    U factors (from A side): 4 rows, 7 cols
    V factors (from B side): 4 rows, 7 cols  
    W factors (from C side): 4 rows, 7 cols
    
    For 2x2 @ 2x16:
    U: 4 rows (A has 4 entries), rank columns
    V: 32 rows (B has 32 entries), rank columns
    W: 32 rows (C has 32 entries), rank columns
    
    The key insight: A is the same for all 4 blocks. So the Strassen
    intermediate values M1..M7 depend only on A and the corresponding
    block of B.
    
    Let me define the Strassen factors for A (shared):
    For each of the 7 Strassen terms, we need a factor from A and a factor from B.
    
    A = [[a00, a01], [a10, a11]]
    U[i*m+j, c] where i,j index A, c indexes the rank dimension.
    
    For Strassen, the 7 multiplications are:
    t1 = (a00+a11) * (b00+b11)
    t2 = (a10+a11) * b00
    t3 = a00 * (b01-b11)
    t4 = a11 * (b10-b00)
    t5 = (a00+a01) * b11
    t6 = (a10-a00) * (b00+b01)
    t7 = (a01-a11) * (b10+b11)
    
    Then:
    c00 = t1 + t4 - t5 + t7
    c01 = t3 + t5
    c10 = t2 + t4
    c11 = t1 - t2 + t3 + t6
    
    For the 2x16 case, B has 4 blocks. Let Bq be the q-th 2x2 block (q=0..3).
    Cq = A @ Bq.
    
    For each block q, we compute 7 Strassen products using the SAME A-factors
    but DIFFERENT B-factors (from block q).
    
    So the rank is 7*4 = 28? No wait, that's not right either.
    
    Actually, each Strassen product for block q is:
    tq_r = (A-factor_r) * (Bq-factor_r) for r=1..7
    
    The A-factor is shared across blocks, but the B-factor differs.
    So we need separate rank-1 terms for each (r, q) pair: 7*4 = 28 terms
    for the multiplications.
    
    Then the output Cq has 4 entries, each being a linear combination of the
    7 tq_r values. But since the tq_r for different q are independent, the
    output for block q only depends on tq_1..tq_7.
    
    Wait, I need to reconsider. The tensor decomposition has rank = number
    of columns in U, V, W. Each column c defines a rank-1 term:
    C[i,k] += U[i,j,c] * A[i,j] * V[j,k,c] ... no, that's not quite right.
    
    Let me re-read the interface:
    U: (n*m, rank) = (4, rank) - factors for A
    V: (m*p, rank) = (32, rank) - factors for B
    W: (p*n, rank) = (32, rank) - factors for C
    
    The decomposition is:
    C[i,k] = sum_c U[i*m+j? ... 
    
    Actually, looking at the schoolbook code:
    U[i*m+j, column] = 1  -> U has rows indexed by (i,j) where i in [0,n), j in [0,m)
    V[j*p+k, column] = 1  -> V has rows indexed by (j,k) where j in [0,m), k in [0,p)
    W[k*n+i, column] = 1  -> W has rows indexed by (k,i) where k in [0,p), i in [0,n)
    
    And the multiplication is: C[i,k] = sum_{j} A[i,j] * B[j,k]
    
    The tensor decomposition says:
    sum_c U[i*m+j? ... 
    
    Hmm, let me think about this differently. The matrix multiplication tensor
    T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]B[j,k].
    
    A rank-1 term in the tensor is: u_i * v_j * w_k for each factor.
    But here the indices are: A has indices (i,j), B has indices (j,k), C has (i,k).
    
    The tensor is T_{i,j,k} where the multiplication is:
    C_{i,k} = sum_j A_{i,j} B_{j,k}
    
    A rank-r decomposition means:
    C_{i,k} = sum_{c=1}^{r} (sum_j U_{i,j,c} A_{i,j}) * (sum_j V_{j,k,c} B_{j,k}) ... no.
    
    Actually, I think the correct interpretation is:
    The bilinear map is: (A, B) -> C where C_{i,k} = sum_j A_{i,j} B_{j,k}
    
    A rank-r decomposition expresses this as:
    C_{i,k} = sum_{c=1}^{r} (linear form in A)_c * (linear form in B)_c * (linear form in C)_c
    
    More precisely:
    For each c, we have:
    - A linear form L_c^A(A) = sum_{i,j} U_{i*m+j, c} * A_{i,j}
    - A linear form L_c^B(B) = sum_{j,k} V_{j*p+k, c} * B_{j,k}
    - A linear form L_c^C(C) = sum_{i,k} W_{k*n+i, c} * C_{i,k}
    
    And the constraint is that for all A, B:
    sum_c L_c^A(A) * L_c^B(B) = sum_{i,k} C_{i,k} * L_c^C(C) ... 
    
    No, I think it's:
    For all A, B: C = A*B, and the decomposition satisfies:
    sum_c L_c^A(A) * L_c^B(B) = sum_{i,k} (coefficient) * C_{i,k}
    
    Actually, the standard tensor decomposition for matrix multiplication:
    T = sum_c u_c ⊗ v_c ⊗ w_c
    where T is the multiplication tensor.
    
    The multiplication tensor T has entries T_{i,j,k} = 1 for all i,j,k (in the
    sense that C_{i,k} = sum_j A_{i,j} B_{j,k}).
    
    A rank-1 decomposition term is: u_{i} v_{j} w_{k} for indices i,j,k.
    But here we have: A indexed by (i,j), B by (j,k), C by (i,k).
    
    So the tensor lives in R^{n*m} ⊗ R^{m*p} ⊗ R^{p*n}.
    T_{(i,j), (j',k), (k',i')} = 1 if i=i', j=j', k=k' (i.e., the matching indices).
    
    A rank-1 term is: u_{(i,j)} * v_{(j',k)} * w_{(k',i')}
    
    The decomposition: T = sum_c u_c ⊗ v_c ⊗ w_c
    means: T_{(i,j),(j',k),(k',i')} = sum_c U_{(i,j),c} V_{(j',k),c} W_{(k',i'),c}
    
    And the constraint for exact decomposition is:
    For all A, B: sum_{(i,j),(j',k),(k',i')} T_{...} A_{i,j} B_{j',k} C_{k',i'} = 
    sum_c (sum_{i,j} U_{(i,j),c} A_{i,j}) (sum_{j',k} V_{(j',k),c} B_{j',k}) (sum_{k',i'} W_{(k',i'),c} C_{k',i'})
    
    But since C = AB, we have C_{k',i'} = sum_j A_{i',j} B_{j,k'}.
    
    This is getting complex. Let me just use the known Strassen construction.
    
    For 2x2 @ 2x2 (n=m=p=2), Strassen rank 7:
    U (4x7), V (4x7), W (4x7) where:
    Row indices of U: (i,j) for i,j in {0,1} -> [a00, a01, a10, a11]
    Row indices of V: (j,k) for j,k in {0,1} -> [b00, b01, b10, b11]
    Row indices of W: (k,i) for k,i in {0,1} -> [c00, c01, c10, c11]
    
    Strassen:
    t1 = (a00+a11)(b00+b11)
    t2 = (a10+a11)b00
    t3 = a00(b01-b11)
    t4 = a11(b10-b00)
    t5 = (a00+a01)b11
    t6 = (a10-a00)(b00+b01)
    t7 = (a01-a11)(b10+b11)
    
    c00 = t1 + t4 - t5 + t7
    c01 = t3 + t5
    c10 = t2 + t4
    c11 = t1 - t2 + t3 + t6
    
    So for each term c=1..7:
    U[:,c] gives the coefficients for A
    V[:,c] gives the coefficients for B
    W[:,c] gives the coefficients for C
    
    Term 1: (a00+a11)(b00+b11) -> c00 += 1, c11 += 1
    U[:,1] = [1, 0, 0, 1] (a00 + a11)
    V[:,1] = [1, 0, 0, 1] (b00 + b11)
    W[:,1] = [1, 0, 0, 1] (c00 + c11)
    
    Term 2: (a10+a11)b00 -> c10 += 1, c11 -= 1
    U[:,2] = [0, 0, 1, 1] (a10 + a11)
    V[:,2] = [1, 0, 0, 0] (b00)
    W[:,2] = [0, 0, 1, -1] (c10 - c11)
    
    Term 3: a00(b01-b11) -> c01 += 1, c11 += 1
    U[:,3] = [1, 0, 0, 0] (a00)
    V[:,3] = [0, 1, 0, -1] (b01 - b11)
    W[:,3] = [0, 1, 0, 1] (c01 + c11)
    
    Term 4: a11(b10-b00) -> c00 += 1, c10 += 1
    U[:,4] = [0, 0, 0, 1] (a11)
    V[:,4] = [0, 0, 1, -1] (b10 - b00)
    W[:,4] = [1, 0, 1, 0] (c00 + c10)
    
    Term 5: (a00+a01)b11 -> c00 -= 1, c01 += 1
    U[:,5] = [1, 1, 0, 0] (a00 + a01)
    V[:,5] = [0, 0, 0, 1] (b11)
    W[:,5] = [-1, 1, 0, 0] (-c00 + c01)
    
    Term 6: (a10-a00)(b00+b01) -> c11 += 1
    U[:,6] = [-1, 0, 1, 0] (-a00 + a10)
    V[:,6] = [1, 1, 0, 0] (b00 + b01)
    W[:,6] = [0, 0, 0, 1] (c11)
    
    Term 7: (a01-a11)(b10+b11) -> c00 += 1
    U[:,7] = [0, 1, 0, -1] (a01 - a11)
    V[:,7] = [0, 0, 1, 1] (b10 + b11)
    W[:,7] = [1, 0, 0, 0] (c00)
    
    Let me verify: 
    c00 = t1 + t4 - t5 + t7
    From W: term1 contributes +1 to c00, term4 contributes +1, term5 contributes -1, term7 contributes +1. ✓
    
    c01 = t3 + t5
    From W: term3 contributes +1 to c01, term5 contributes +1. ✓
    
    c10 = t2 + t4
    From W: term2 contributes +1 to c10, term4 contributes +1. ✓
    
    c11 = t1 - t2 + t3 + t6
    From W: term1 contributes +1, term2 contributes -1, term3 contributes +1, term6 contributes +1. ✓
    
    Now for 2x2 @ 2x16:
    n=2, m=2, p=16.
    U: 4 x rank
    V: 32 x rank
    W: 32 x rank
    
    B is 2x16, so B[j,k] for j in {0,1}, k in {0,...,15}.
    We can view B as 4 blocks: Bq = B[:, 4q:4q+4] for q=0,1,2,3, each 2x2.
    
    C is 2x16, C[i,k] for i in {0,1}, k in {0,...,15}.
    Cq = C[:, 4q:4q+4] for q=0,1,2,3, each 2x2.
    
    For each block q, Cq = A @ Bq, and we can use the same 7 Strassen terms.
    
    For block q, the 7 terms are:
    tq1 = (a00+a11)(Bq[0,0]+Bq[1,1])
    tq2 = (a10+a11)Bq[0,0]
    tq3 = a00(Bq[0,1]-Bq[1,1])
    tq4 = a11(Bq[1,0]-Bq[0,0])
    tq5 = (a00+a01)Bq[1,1]
    tq6 = (a10-a00)(Bq[0,0]+Bq[0,1])
    tq7 = (a01-a11)(Bq[1,0]+Bq[1,1])
    
    Cq[0,0] = tq1 + tq4 - tq5 + tq7
    Cq[0,1] = tq3 + tq5
    Cq[1,0] = tq2 + tq4
    Cq[1,1] = tq1 - tq2 + tq3 + tq6
    
    Now, in the tensor decomposition, each rank-1 term c corresponds to one
    (q, r) pair where q is the block and r is the Strassen term (1..7).
    Total rank = 4*7 = 28.
    
    For term (q, r):
    U[:, c] = same as the r-th Strassen A-factor (shared across q)
    V[:, c] = the r-th Strassen B-factor applied to block q
    W[:, c] = the r-th Strassen C-factor applied to block q
    
    Let's index:
    U rows: [a00, a01, a10, a11] (indices 0,1,2,3)
    V rows: [b00, b01, ..., b0,15, b10, b11, ..., b1,15] 
    Actually, V has m*p = 2*16 = 32 rows, indexed by (j,k) where j in {0,1}, k in {0..15}.
    V[j*p+k, c] = V[j*16+k, c]
    
    W rows: p*n = 16*2 = 32 rows, indexed by (k,i) where k in {0..15}, i in {0,1}.
    W[k*n+i, c] = W[k*2+i, c]
    
    For block q (q=0..3), the 2x2 submatrix Bq has:
    Bq[0,0] = B[0, 4q+0] = V row (0, 4q+0) -> V[0*16 + (4q+0), c] = V[4q, c]
    Bq[0,1] = B[0, 4q+1] -> V[4q+1, c]
    Bq[1,0] = B[1, 4q+0] -> V[16 + 4q, c]
    Bq[1,1] = B[1, 4q+1] -> V[16 + 4q + 1, c]
    
    Similarly for Cq:
    Cq[0,0] = C[0, 4q+0] -> W row (4q+0, 0) -> W[(4q+0)*2 + 0, c] = W[8q, c]
    Cq[0,1] = C[0, 4q+1] -> W row (4q+1, 0) -> W[(4q+1)*2 + 0, c] = W[8q+2, c]
    Cq[1,0] = C[1, 4q+0] -> W row (4q+0, 1) -> W[(4q+0)*2 + 1, c] = W[8q+1, c]
    Cq[1,1] = C[1, 4q+1] -> W row (4q+1, 1) -> W[(4q+1)*2 + 1, c] = W[8q+3, c]
    
    Now let's build the 28 rank-1 terms.
    
    Strassen A-factors (U columns), indexed by r=1..7:
    r=1: [1, 0, 0, 1]
    r=2: [0, 0, 1, 1]
    r=3: [1, 0, 0, 0]
    r=4: [0, 0, 0, 1]
    r=5: [1, 1, 0, 0]
    r=6: [-1, 0, 1, 0]
    r=7: [0, 1, 0, -1]
    
    Strassen B-factors for a 2x2 block (V columns for that block), indexed by r=1..7:
    r=1: b00+b11 -> [1, 0, 0, 1] (for b00, b01, b10, b11)
    r=2: b00 -> [1, 0, 0, 0]
    r=3: b01-b11 -> [0, 1, 0, -1]
    r=4: b10-b00 -> [0, 0, 1, -1]
    r=5: b11 -> [0, 0, 0, 1]
    r=6: b00+b01 -> [1, 1, 0, 0]
    r=7: b10+b11 -> [0, 0, 1, 1]
    
    Strassen C-factors for a 2x2 block (W columns for that block), indexed by r=1..7:
    r=1: c00+c11 -> [1, 0, 0, 1] (for c00, c01, c10, c11)
    r=2: c10-c11 -> [0, 0, 1, -1]
    r=3: c01+c11 -> [0, 1, 0, 1]
    r=4: c00+c10 -> [1, 0, 1, 0]
    r=5: -c00+c01 -> [-1, 1, 0, 0]
    r=6: c11 -> [0, 0, 0, 1]
    r=7: c00 -> [1, 0, 0, 0]
    
    Now for each (q, r) pair, we create a column c = q*7 + (r-1) in U, V, W.
    
    U[:, c] = Strassen_A[r] (same for all q)
    V[:, c] = 0 except for the 4 entries corresponding to block q, which are Strassen_B[r]
    W[:, c] = 0 except for the 4 entries corresponding to block q, which are Strassen_C[r]
    
    Let me code this up.
    
    """
    n, m, p = DIMENSIONS  # 2, 2, 16
    
    # Strassen factors for 2x2
    # U factors (A side): 4 entries [a00, a01, a10, a11]
    strassen_U = [
        [1, 0, 0, 1],   # r=1: a00+a11
        [0, 0, 1, 1],   # r=2: a10+a11
        [1, 0, 0, 0],   # r=3: a00
        [0, 0, 0, 1],   # r=4: a11
        [1, 1, 0, 0],   # r=5: a00+a01
        [-1, 0, 1, 0],  # r=6: -a00+a10
        [0, 1, 0, -1],  # r=7: a01-a11
    ]
    
    # V factors (B side) for a 2x2 block: 4 entries [b00, b01, b10, b11]
    strassen_V = [
        [1, 0, 0, 1],   # r=1: b00+b11
        [1, 0, 0, 0],   # r=2: b00
        [0, 1, 0, -1],  # r=3: b01-b11
        [0, 0, 1, -1],  # r=4: b10-b00
        [0, 0, 0, 1],   # r=5: b11
        [1, 1, 0, 0],   # r=6: b00+b01
        [0, 0, 1, 1],   # r=7: b10+b11
    ]
    
    # W factors (C side) for a 2x2 block: 4 entries [c00, c01, c10, c11]
    strassen_W = [
        [1, 0, 0, 1],   # r=1: c00+c11
        [0, 0, 1, -1],  # r=2: c10-c11
        [0, 1, 0, 1],   # r=3: c01+c11
        [1, 0, 1, 0],   # r=4: c00+c10
        [-1, 1, 0, 0],  # r=5: -c00+c01
        [0, 0, 0, 1],   # r=6: c11
        [1, 0, 0, 0],   # r=7: c00
    ]
    
    num_blocks = p // 2  # 16 // 2 = 8 blocks of 2 columns each? 
    # Wait, p=16, and each block is 2x2, so we have p/2 = 8 blocks? No.
    # B is 2x16. If we split into 2x2 blocks, we have 16/2 = 8 blocks.
    # But wait, the problem says "2 by 16 matrix", so B is 2 rows, 16 columns.
    # A 2x2 block would be 2 rows and 2 columns. So we have 16/2 = 8 such blocks.
    
    # Hmm, but that would give rank = 8*7 = 56. Let me reconsider.
    
    # Actually, I think I miscounted. Let me re-examine.
    # B is 2x16. To decompose into 2x2 blocks, we need to partition the 16 columns
    # into groups of 2. That gives 8 blocks.
    
    # So rank = 8 * 7 = 56.
    
    num_blocks = p // 2  # 8
    
    rank = num_blocks * 7  # 56
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 32 x 56
    
    for q in range(num_blocks):
        for r in range(7):
            c = q * 7 + r
            
            # U: same for all blocks
            for i in range(4):
                U[i][c] = strassen_U[r][i]
            
            # V: only block q is non-zero
            # Block q occupies columns 2q and 2q+1 of B
            # B[0, 2q] -> V row 0*16 + 2q = 2q
            # B[0, 2q+1] -> V row 2q+1
            # B[1, 2q] -> V row 16 + 2q
            # B[1, 2q+1] -> V row 16 + 2q + 1
            v_indices = [2*q, 2*q+1, 16+2*q, 16+2*q+1]
            for idx, vi in enumerate(v_indices):
                V[vi][c] = strassen_V[r][idx]
            
            # W: only block q is non-zero
            # C[0, 2q] -> W row (2q)*2 + 0 = 4q
            # C[0, 2q+1] -> W row (2q+1)*2 + 0 = 4q+2
            # C[1, 2q] -> W row (2q)*2 + 1 = 4q+1
            # C[1, 2q+1] -> W row (2q+1)*2 + 1 = 4q+3
            w_indices = [4*q, 4*q+2, 4*q+1, 4*q+3]
            for idx, wi in enumerate(w_indices):
                W[wi][c] = strassen_W[r][idx]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
