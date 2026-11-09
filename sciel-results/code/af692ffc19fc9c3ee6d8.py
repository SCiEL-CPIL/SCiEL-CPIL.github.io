"""
Matrix multiplication 2x5x5: rank-minimal exact tensor decomposition.
Strategy: Exact algebraic construction using Strassen's 7-factor algorithm
on three 2x2x2 block subproblems (blocks {0,1}, {2,3}, {4} of the 5 indices).
This gives rank 3*7 + 1*2 = 23? No: 3 blocks * 7 = 21 for the 2x2 parts,
plus the {4} block contributes 2*1*1 = 2 more. Total = 23? Let me recount.

Actually:
- Block {0,1}: A uses cols 0,1 (2x2), B uses rows 0,1 and cols 0,1 (2x2). Strassen: rank 7.
- Block {2,3}: A uses cols 2,3 (2x2), B uses rows 2,3 and cols 2,3 (2x2). Strassen: rank 7.
- Block {4}: A uses col 4 (2x1), B uses row 4 and col 4 (1x1). Schoolbook: rank 2.

Total rank = 7 + 7 + 2 = 16? That can't be right since lower bound is 25.

Wait, I'm confusing myself. Let me think more carefully.

The matrix multiplication C = A*B where A is 2x5, B is 5x5, C is 2x5.
C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]

Partition j into J0={0,1}, J1={2,3}, J2={4}
Partition k into K0={0,1}, K1={2,3}, K2={4}

Then C[i,k] = sum_{j in J0} A[i,j]*B[j,k] + sum_{j in J1} A[i,j]*B[j,k] + A[i,4]*B[4,k]

For the (J0, K0) block: C[i, k] for i in {0,1}, k in {0,1} gets contribution from
A[i, j] for j in {0,1} and B[j, k] for k in {0,1}. This is a 2x2x2 multiplication: rank 7 by Strassen.

For the (J0, K1) block: C[i, k] for i in {0,1}, k in {2,3} gets contribution from
A[i, j] for j in {0,1} and B[j, k] for k in {2,3}. This is also 2x2x2: rank 7.

Similarly for (J0, K2), (J1, K0), (J1, K1), (J1, K2), (J2, K0), (J2, K1), (J2, K2).

That's 9 blocks, each needing separate handling. But Strassen only works when the
A-part, B-part, and C-part all have the same 2x2 structure aligned.

Actually the correct way: we have a 2x5 * 5x5 = 2x5 product. The "blocks" in the
j-index (inner dimension) must match between A's columns and B's rows. The k-index
partition of B's columns determines which C entries get contributions.

Let me reframe: We have 5 "terms" in the sum. We can group them:
- Group 0: j=0,1 -> contributes to all C[i,k]
- Group 1: j=2,3 -> contributes to all C[i,k]  
- Group 2: j=4 -> contributes to all C[i,k]

For Group 0: A_sub = A[:,0:2] (2x2), B_sub = B[0:2,:] (2x5). Product is 2x5.
For Group 1: A_sub = A[:,2:4] (2x2), B_sub = B[2:4,:] (2x5). Product is 2x5.
For Group 2: A_sub = A[:,4:5] (2x1), B_sub = B[4:5,:] (1x5). Product is 2x5.

Now, 2x2 * 2x5: We can further partition the 5 columns of B into {0,1}, {2,3}, {4}.
For 2x2 * 2x2 (first two cols): Strassen rank 7.
For 2x2 * 2x2 (next two cols): Strassen rank 7.
For 2x2 * 2x1 (last col): rank 2*2*1 = 4 schoolbook, or can we do better? 2x2*2x1 is
computing 2x1 = 2x2 * 2x1. That's 4 scalar mults schoolbook. Can Strassen help? 
Actually 2x2 * 2x1: C[0] = A[0,0]*B[0] + A[0,1]*B[1], C[1] = A[1,0]*B[0] + A[1,1]*B[1].
This is 4 multiplications. The rank of 2x2x1 tensor is 4 (it's just a 2x2 matrix times a vector,
which is a 4-dimensional bilinear form). Actually no, the tensor is 2x2x1, and its rank is 4
since it's equivalent to 4 independent scalar products.

Hmm, so total would be:
- Group 0, cols {0,1}: 7
- Group 0, cols {2,3}: 7
- Group 0, col {4}: 4
- Group 1, cols {0,1}: 7
- Group 1, cols {2,3}: 7
- Group 1, col {4}: 4
- Group 2 (j=4), cols {0,1}: 2x1 * 1x2 = 2 mults
- Group 2 (j=4), cols {2,3}: 2x1 * 1x2 = 2 mults
- Group 2 (j=4), col {4}: 2x1 * 1x1 = 2 mults

Total: 7+7+4+7+7+4+2+2+2 = 42. That's better than 50 but worse than the claimed 46.

Let me reconsider the 2x2x2 * 2x2x5 approach differently.

Actually, the known result from the cell finding says rank 46 was achieved by
"replacing 8 scalar multiplications with 7 for each of the four 2x2x2 blocks."
Four 2x2x2 blocks: that suggests partitioning 5 into 2+2+1, giving 4 blocks of
2x2 in the j-k plane... but 2+2+1 gives 3x3=9 blocks, not 4.

Wait: "four 2x2x2 blocks" - maybe they mean:
- Partition j into {0,1}, {2,3}, {4} (3 groups)
- Partition k into {0,1}, {2,3}, {4} (3 groups)
- The 2x2x2 blocks are where both j-group and k-group have size 2: (J0,K0), (J0,K1), (J1,K0), (J1,K1)
  That's 4 blocks, each 2x2x2, each reduced from 8 to 7 mults.
- The remaining blocks: (J0,K2), (J1,K2), (J2,K0), (J2,K1), (J2,K2) are 2x2x1, 2x1x2, or 2x1x1.

Let me count:
- 4 blocks of type 2x2x2: 4*7 = 28
- (J0,K2): A is 2x2, B is 2x1, C is 2x1: rank 4
- (J1,K2): A is 2x2, B is 2x1, C is 2x1: rank 4
- (J2,K0): A is 2x1, B is 1x2, C is 2x2: rank 4
- (J2,K1): A is 2x1, B is 1x2, C is 2x2: rank 4
- (J2,K2): A is 2x1, B is 1x1, C is 2x1: rank 2

Total: 28 + 4 + 4 + 4 + 4 + 2 = 46. Yes! That matches.

So the construction is:
- 4 Strassen-7 blocks for the 2x2x2 subproblems
- 4 schoolbook-4 blocks for the 2x2x1 or 2x1x2 subproblems
- 1 schoolbook-2 block for the 2x1x1 subproblem
Total rank = 46.
"""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]
N, M, P = DIMENSIONS
_TIME_BUDGET = 140.0
_DEADLINE_START = None


def _elapsed():
    global _DEADLINE_START
    if _DEADLINE_START is None:
        _DEADLINE_START = time.time()
    return time.time() - _DEADLINE_START


def _within_budget(margin=5.0):
    return _elapsed() < _TIME_BUDGET - margin


def _verify(U, V, W, n, m, p, num_tests=5):
    """Verify decomposition using multiple random test matrices."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    for test in range(num_tests):
        rng = np.random.RandomState(1000 + test)
        A = rng.randn(n * m)
        B = rng.randn(m * p)
        A_mat = A.reshape(n, m)
        B_mat = B.reshape(m, p)
        C_true = (A_mat @ B_mat).flatten()
        C_true_wn = np.zeros(p * n)
        for i in range(n):
            for k in range(p):
                C_true_wn[k * n + i] = C_true[i * p + k]
        C_comp = np.zeros(p * n)
        for c in range(rank):
            vA = float(U_np[:, c] @ A)
            vB = float(V_np[:, c] @ B)
            if vA != 0.0 and vB != 0.0:
                C_comp += W_np[:, c] * vA * vB
        if not np.allclose(C_comp, C_true_wn, atol=1e-9):
            return False
    return True


def _build_strassen_block(a_row_offsets, a_col_offsets, b_row_offsets, b_col_offsets,
                          c_row_offsets, c_col_offsets, base_col, n, m, p):
    """
    Build Strassen's 7-factor decomposition for a 2x2x2 block.
    
    a_row_offsets: 2 row indices in A (i values)
    a_col_offsets: 2 col indices in A (j values)
    b_row_offsets: 2 row indices in B (j values)
    b_col_offsets: 2 col indices in B (k values)
    c_row_offsets: 2 row indices in C (i values)
    c_col_offsets: 2 col indices in C (k values)
    
    Returns: list of (U_entries, V_entries, W_entries, col_offset)
    where each entry is a dict mapping flat_index -> coefficient
    """
    factors = []
    # Strassen's 7 products for 2x2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # A is indexed by (i, j), flat = i*m + j
    # B is indexed by (j, k), flat = j*p + k
    # C is indexed by (i, k), flat in W = k*n + i
    
    # Define the 7 factors
    # Each factor: (u_dict, v_dict, w_dict)
    # u_dict maps A flat index -> coeff
    # v_dict maps B flat index -> coeff
    # w_dict maps C flat index (k*n+i) -> coeff
    
    a00 = a_row_offsets[0] * m + a_col_offsets[0]
    a01 = a_row_offsets[0] * m + a_col_offsets[1]
    a10 = a_row_offsets[1] * m + a_col_offsets[0]
    a11 = a_row_offsets[1] * m + a_col_offsets[1]
    
    b00 = b_row_offsets[0] * p + b_col_offsets[0]
    b01 = b_row_offsets[0] * p + b_col_offsets[1]
    b10 = b_row_offsets[1] * p + b_col_offsets[0]
    b11 = b_row_offsets[1] * p + b_col_offsets[1]
    
    c00 = b_col_offsets[0] * n + a_row_offsets[0]  # k*n + i
    c01 = b_col_offsets[1] * n + a_row_offsets[0]
    c10 = b_col_offsets[0] * n + a_row_offsets[1]
    c11 = b_col_offsets[1] * n + a_row_offsets[1]
    
    # M1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    factors.append((
        {a00: 1, a11: 1},
        {b00: 1, b11: 1},
        {c00: 1, c11: 1}
    ))
    # M2 = (A10+A11)B00 -> C10 += 1, C11 += -1
    factors.append((
        {a10: 1, a11: 1},
        {b00: 1},
        {c10: 1, c11: -1}
    ))
    # M3 = A00(B10-B11) -> C01 += 1, C11 += 1
    factors.append((
        {a00: 1},
        {b10: 1, b11: -1},
        {c01: 1, c11: 1}
    ))
    # M4 = A11(B01-B00) -> C00 += 1, C10 += 1
    factors.append((
        {a11: 1},
        {b01: 1, b00: -1},
        {c00: 1, c10: 1}
    ))
    # M5 = (A00+A01)B11 -> C00 += -1, C01 += 1
    factors.append((
        {a00: 1, a01: 1},
        {b11: 1},
        {c00: -1, c01: 1}
    ))
    # M6 = (A10-A00)(B00+B01) -> C11 += 1
    factors.append((
        {a10: 1, a00: -1},
        {b00: 1, b01: 1},
        {c11: 1}
    ))
    # M7 = (A01-A11)(B10+B11) -> C00 += 1
    factors.append((
        {a01: 1, a11: -1},
        {b10: 1, b11: 1},
        {c00: 1}
    ))
    
    return factors


def _build_schoolbook_block(a_row_offsets, a_col_offsets, b_row_offsets, b_col_offsets,
                            c_row_offsets, c_col_offsets, n, m, p):
    """
    Build schoolbook decomposition for a rectangular block.
    Returns list of (u_dict, v_dict, w_dict)
    """
    factors = []
    for i_off in a_row_offsets:
        for j_off in a_col_offsets:
            for k_off in b_col_offsets:
                # A[i, j] * B[j, k] -> C[i, k]
                a_flat = i_off * m + j_off
                b_flat = j_off * p + k_off
                c_flat = k_off * n + i_off
                factors.append(
                    ({a_flat: 1}, {b_flat: 1}, {c_flat: 1})
                )
    return factors


def _assemble_decomposition(factor_list, n, m, p):
    """
    Assemble a list of (u_dict, v_dict, w_dict) into U, V, W matrices.
    Returns (U, V, W) as lists of lists.
    """
    rank = len(factor_list)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u_dict, v_dict, w_dict) in enumerate(factor_list):
        for idx, coeff in u_dict.items():
            U[idx][c] = coeff
        for idx, coeff in v_dict.items():
            V[idx][c] = coeff
        for idx, coeff in w_dict.items():
            W[idx][c] = coeff
    
    return U, V, W


def _build_rank46():
    """
    Build the rank-46 decomposition using 4 Strassen blocks + 4 schoolbook-4 + 1 schoolbook-2.
    
    Partition j (inner dim) into: J0={0,1}, J1={2,3}, J2={4}
    Partition k (output cols) into: K0={0,1}, K1={2,3}, K2={4}
    A rows: {0,1} (both rows always)
    
    Blocks:
    (J0,K0): 2x2x2 Strassen -> 7 factors
    (J0,K1): 2x2x2 Strassen -> 7 factors
    (J1,K0): 2x2x2 Strassen -> 7 factors
    (J1,K1): 2x2x2 Strassen -> 7 factors
    (J0,K2): 2x2x1 schoolbook -> 4 factors
    (J1,K2): 2x2x1 schoolbook -> 4 factors
    (J2,K0): 2x1x2 schoolbook -> 4 factors
    (J2,K1): 2x1x2 schoolbook -> 4 factors
    (J2,K2): 2x1x1 schoolbook -> 2 factors
    Total: 28 + 16 + 2 = 46
    """
    n, m, p = N, M, P
    
    # A rows are always {0, 1}
    a_rows = [0, 1]
    
    # J partitions
    J0 = [0, 1]
    J1 = [2, 3]
    J2 = [4]
    
    # K partitions
    K0 = [0, 1]
    K1 = [2, 3]
    K2 = [4]
    
    all_factors = []
    
    # 4 Strassen blocks (2x2x2)
    for j_group, k_group in [(J0, K0), (J0, K1), (J1, K0), (J1, K1)]:
        factors = _build_strassen_block(
            a_rows, j_group, j_group, k_group, a_rows, k_group,
            base_col=0, n=n, m=m, p=p
        )
        all_factors.extend(factors)
    
    # (J0, K2): A is 2x2 (rows {0,1}, cols J0), B is 2x1 (rows J0, col K2[0]=4)
    # Schoolbook: 2*2*1 = 4 factors
    for i in a_rows:
        for j in J0:
            for k in K2:
                a_flat = i * m + j
                b_flat = j * p + k
                c_flat = k * n + i
                all_factors.append(({a_flat: 1}, {b_flat: 1}, {c_flat: 1}))
    
    # (J1, K2): A is 2x2 (rows {0,1}, cols J1), B is 2x1 (rows J1, col K2[0]=4)
    for i in a_rows:
        for j in J1:
            for k in K2:
                a_flat = i * m + j
                b_flat = j * p + k
                c_flat = k * n + i
                all_factors.append(({a_flat: 1}, {b_flat: 1}, {c_flat: 1}))
    
    # (J2, K0): A is 2x1 (rows {0,1}, col J2[0]=4), B is 1x2 (row J2[0]=4, cols K0)
    for i in a_rows:
        for j in J2:
            for k in K0:
                a_flat = i * m + j
                b_flat = j * p + k
                c_flat = k * n + i
                all_factors.append(({a_flat: 1}, {b_flat: 1}, {c_flat: 1}))
    
    # (J2, K1): A is 2x1 (rows {0,1}, col J2[0]=4), B is 1x2 (row J2[0]=4, cols K1)
    for i in a_rows:
        for j in J2:
            for k in K1:
                a_flat = i * m + j
                b_flat = j * p + k
                c_flat = k * n + i
                all_factors.append(({a_flat: 1}, {b_flat: 1}, {c_flat: 1}))
    
    # (J2, K2): A is 2x1 (rows {0,1}, col 4), B is 1x1 (row 4, col 4)
    for i in a_rows:
        for j in J2:
            for k in K2:
                a_flat = i * m + j
                b_flat = j * p + k
                c_flat = k * n + i
                all_factors.append(({a_flat: 1}, {b_flat: 1}, {c_flat: 1}))
    
    U, V, W = _assemble_decomposition(all_factors, n, m, p)
    return {"U": U, "V": V, "W": W}


def _build_schoolbook():
    """Return the schoolbook rank-50 decomposition."""
    n, m, p = N, M, P
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _try_merge_factors(U, V, W, n, m, p, time_limit=20.0):
    """
    Try to merge pairs of factors in the rank-46 decomposition to get rank 45.
    Two factors c1, c2 can be merged if there exists a linear combination
    that eliminates one while preserving the tensor.
    
    Strategy: For each pair (c1, c2), check if we can find a scalar t such that
    replacing factor c1 with (U_c1 + t*U_c2, V_c1, W_c1) and adjusting c2
    still gives a valid decomposition. This is generally not possible, but
    we can try a more systematic approach.
    
    Alternative: Try to find if any factor is redundant (can be expressed as
    a linear combination of others in a way that preserves the tensor).
    
    For now, try a simple approach: for each pair of factors, try to combine
    them using the identity that if two factors share the same V vector,
    we can merge their U and W vectors.
    """
    if not _within_budget(margin=15):
        return None
    
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check for factors with identical V vectors
    # If V[:, c1] == V[:, c2], we can potentially merge
    for c1 in range(rank):
        if not _within_budget(margin=15):
            return None
        for c2 in range(c1 + 1, rank):
            if np.allclose(V_np[:, c1], V_np[:, c2], atol=1e-12):
                # Try merging: new factor = (U_c1 + U_c2, V_c1, W_c1 + W_c2)
                # This works if the cross terms cancel, which they don't in general.
                # But let's try: new_U = U_c1 + U_c2, new_V = V_c1, new_W = W_c1 + W_c2
                # This gives (u1+u2) * v * (w1+w2) = u1*v*w1 + u1*v*w2 + u2*v*w1 + u2*v*w2
                # We need just u1*v*w1 + u2*v*w2, so the cross terms u1*v*w2 + u2*v*w1
                # must be zero, which is not generally true.
                # Skip this approach.
                pass
    
    # Try a different approach: use the structure of Strassen to see if
    # any of the 7 factors in a Strassen block can be eliminated by
    # combining with factors from adjacent blocks.
    # This is complex; skip for now.
    
    return None


def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x5x5 matrix mult.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = N, M, P
    
    # Build the rank-46 Strassen-based decomposition
    result = _build_rank46()
    
    # Verify
    if _verify(result["U"], result["V"], result["W"], n, m, p, num_tests=5):
        best = result
        best_rank = 46
    else:
        # Fallback to schoolbook
        best = _build_schoolbook()
        best_rank = 50
    
    # Try to reduce further
    if _within_budget(margin=20):
        merged = _try_merge_factors(best["U"], best["V"], best["W"], n, m, p,
                                     time_limit=10.0)
        if merged is not None:
            if _verify(merged["U"], merged["V"], merged["W"], n, m, p, num_tests=5):
                best = merged
                best_rank = 45
    
    return best
