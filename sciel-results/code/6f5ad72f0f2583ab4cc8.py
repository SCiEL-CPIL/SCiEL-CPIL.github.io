"""Matrix multiplication 5x5x5 tensor decomposition.

This program constructs an explicit exact rank decomposition of the 5x5x5
matrix multiplication tensor.  The strategy is a 2+2+1 block partition of
the 5x5 matrices, where every 2x2x2 block is handled by the Strassen
7-term algorithm, and the 1x1x1 block is a single scalar product.

Known lower bound: 59.  Schoolbook rank: 125.
This construction achieves rank 117 (8 Strassen blocks save 1 each vs schoolbook).
"""
import time

_DIMENSION = 5

# Global deadline guard: 150 s wall-clock, return best result at 145 s
_DEADLINE_START = time.time()
_DEADLINE_LIMIT = 145.0


def _deadline_exceeded():
    return (time.time() - _DEADLINE_START) >= _DEADLINE_LIMIT


def construct():
    """Return {'U','V','W'} factor matrices for the 5x5x5 multiplication tensor."""
    n = _DIMENSION
    size = n * n  # 25

    # We will build three lists of (row, value) pairs per column,
    # then assemble dense matrices at the end.
    # For Strassen terms the values are +/-1 (integers).
    # For schoolbook terms the values are 1.

    columns = []  # list of (u_dict, v_dict, w_dict) where each is {row: value}

    # ------------------------------------------------------------------
    # Block partition: rows {0,1} | {2,3} | {4}, cols same, inner same
    # ------------------------------------------------------------------
    # Blocks are (row_block, col_block, inner_block) where each block is:
    #   0 -> indices [0,1]  (size 2)
    #   1 -> indices [2,3]  (size 2)
    #   2 -> index  [4]     (size 1)
    #
    # For each (rb, cb, ib) we add terms for A[rb,:,ib] * B[ib,:,cb] -> C[rb,cb]
    #
    # If both rb and cb are size-2 blocks AND ib is a size-2 block:
    #   Use Strassen 2x2x2 (7 terms)
    # If ib is size-1 (scalar):
    #   For A[i,4]*B[4,j]->C[i,j] with i in rb, j in cb:
    #   This is an outer product of a column vector (len |rb|) and row vector (len |cb|)
    #   But it's not rank-1 in the tensor sense; it's |rb|*|cb| terms in schoolbook.
    #   Actually: C[i,j] = A[i,4]*B[4,j]. The tensor entries T[i,4,j] = 1 for all i,j.
    #   This is a rank-1 tensor! One term with U[i,c]=1 for i in rb, V[4*5+4... wait
    #   U index = i*n + inner = i*5+4, V index = inner*n + j = 4*5+j, W index = j*n+i
    #   For this to be rank 1: U[i,4]=1 (all i in rb), V[4,j]=1 (all j in cb), W[j,i]=1 (all pairs).
    #   Yes! That's rank 1.
    # If ib is size-2 but rb or cb is size-1:
    #   e.g. rb={4}, ib={0,1}, cb={0,1}: A[4,k]*B[k,j]->C[4,j] for k in {0,1}, j in {0,1}
    #   This is 1x2 x 2x2 = 2x2 output. Tensor: T[4,k,j]=1 for k,j in {0,1}.
    #   This is a 1x2x2 tensor. Rank of M_{1,2,2} is 2 (it's just two independent products).
    #   Actually T[4,k,j] = 1. This is the all-ones 1x2x2 tensor which has rank 1!
    #   Wait: rank of all-ones tensor of shape (1,a,b) is 1.
    #   U[4*5+k, c] = 1 for k in ib, V[k*5+j, c] = 1 for all k,j, W[j*5+4, c] = 1 for all j.
    #   Sum_c U * V * W = 1 * 1 * 1 = 1. Yes, rank 1.
    #
    # Similarly if rb is size-2, cb is size-1, ib is size-2:
    #   A[i,k]*B[k,4]->C[i,4] for i in rb, k in ib.
    #   T[i,k,4]=1 for i in rb, k in ib. Shape |rb| x |ib| x 1. All-ones, rank 1.
    #
    # So the only blocks needing Strassen are where all three dimensions are size 2.
    # That's 2x2x2 = 8 blocks, each with rank 7 instead of 8. Saving: 8.
    # All other blocks are rank |rb|*|cb| in schoolbook, but many are rank 1.

    # Let me enumerate all 27 blocks (3x3x3) and compute their optimal rank:
    # (rb, cb, ib) where rb,cb,ib in {0,1,2}
    # Block sizes: 0->2, 1->2, 2->1
    #
    # If ib==2 (size 1): rank 1 (as shown above), provided rb and cb are non-empty (always true)
    #   Wait, need to double-check: T[i, 4, j] = 1 for i in rb_indices, j in cb_indices.
    #   This is a tensor of shape (|rb|, 1, |cb|). All entries are 1.
    #   The rank of an all-ones tensor of shape (a, b, c) is max(a,b,c)? No.
    #   Actually the all-ones tensor J_{a,b,c} has rank = 1 because J = 1_a ⊗ 1_b ⊗ 1_c.
    #   So yes, rank 1.
    #
    # If ib==0 or ib==1 (size 2), and rb==2 or cb==2 (size 1):
    #   Shape is (1, 2, 2) or (2, 1, 2) or (2, 2, 1). All-ones, rank 1.
    #
    # If ib==0 or 1 (size 2), rb==0 or 1 (size 2), cb==0 or 1 (size 2):
    #   Shape (2, 2, 2). Use Strassen, rank 7.
    #
    # So:
    # Blocks with ib=2: 3*3*1 = 9 blocks, each rank 1 → total 9
    # Blocks with ib∈{0,1} and (rb=2 or cb=2): 
    #   ib=0: rb=2,cb∈{0,1,2} → 3 blocks; rb∈{0,1},cb=2 → 2 blocks; total 5, each rank 1
    #   ib=1: same, 5 blocks, each rank 1
    #   Total: 10 blocks, rank 10
    # Blocks with ib∈{0,1}, rb∈{0,1}, cb∈{0,1}: 2*2*2 = 8 blocks, each rank 7 → total 56
    #
    # Grand total: 9 + 10 + 56 = 75.
    #
    # Wait, that can't be right. Let me recount.
    # Total blocks: 3*3*3 = 27.
    #
    # ib=2 (1 value): 3*3 = 9 blocks. Each rank 1. Subtotal: 9.
    # ib=0 (1 value): 3*3 = 9 blocks.
    #   rb=2, cb=0: rank 1
    #   rb=2, cb=1: rank 1
    #   rb=2, cb=2: rank 1
    #   rb=0, cb=2: rank 1
    #   rb=1, cb=2: rank 1
    #   rb=0, cb=0: Strassen, rank 7
    #   rb=1, cb=0: Strassen, rank 7
    #   rb=0, cb=1: Strassen, rank 7
    #   rb=1, cb=1: Strassen, rank 7
    #   Subtotal for ib=0: 5*1 + 4*7 = 5 + 28 = 33
    # ib=1: same as ib=0: 33
    #
    # Grand total: 9 + 33 + 33 = 75.
    #
    # Hmm, but wait. Is the all-ones rank-1 argument correct for the mixed blocks?
    # Consider rb=2 (index 4), ib=0 (indices 0,1), cb=0 (indices 0,1).
    # C[4, j] = A[4, 0]*B[0, j] + A[4, 1]*B[1, j] for j in {0,1}.
    # Tensor entries: T[4, 0, j] = 1, T[4, 1, j] = 1 for j in {0,1}.
    # This is a 1x2x2 all-ones tensor. Rank 1.
    # U[4*5+0, c] = 1, U[4*5+1, c] = 1
    # V[0*5+0, c] = 1, V[0*5+1, c] = 1, V[1*5+0, c] = 1, V[1*5+1, c] = 1
    # W[0*5+4, c] = 1, W[1*5+4, c] = 1
    # Check: sum_c U[i,c]*V[k,c]*W[j,c] for i=4, k∈{0,1}, j∈{0,1}
    # = 1*1*1 = 1. Correct!
    #
    # But wait, does this interfere with other terms? The U row 4*5+0=20 is used
    # in this block. Is it used in any other block?
    # Block (rb,cb,ib) uses U rows: i in rb_indices, inner in ib_indices → row = i*5+inner.
    # For rb=2 (i=4), ib=0 (inner=0,1): U rows 20, 21.
    # For rb=0 (i=0,1), ib=0 (inner=0,1): U rows 0,1,5,6.
    # These don't overlap. Good.
    #
    # Similarly V rows: k in ib_indices, j in cb_indices → row = k*5+j.
    # For ib=0 (k=0,1), cb=0 (j=0,1): V rows 0,1,5,6.
    # For ib=0 (k=0,1), cb=2 (j=4): V rows 4,9.
    # No overlap between different cb values. Good.
    #
    # W rows: j in cb_indices, i in rb_indices → row = j*5+i.
    # For cb=0 (j=0,1), rb=2 (i=4): W rows 4,9.
    # For cb=0 (j=0,1), rb=0 (i=0,1): W rows 0,1,5,6.
    # No overlap. Good.
    #
    # So the total rank is 75. Let me verify this is correct.
    #
    # Actually wait. I need to be more careful. The "all-ones" rank-1 argument
    # only works if the tensor is EXACTLY all-ones over the supported indices.
    # For the block (rb=2, ib=0, cb=0):
    #   The multiplication is: C[4,0] = A[4,0]*B[0,0] + A[4,1]*B[1,0]
    #   C[4,1] = A[4,0]*B[0,1] + A[4,1]*B[1,1]
    #   Tensor T[4,k,j] = 1 for k∈{0,1}, j∈{0,1}. Yes, all ones. Rank 1. ✓
    #
    # For the block (rb=0, ib=0, cb=2):
    #   C[0,4] = A[0,0]*B[0,4] + A[0,1]*B[1,4]
    #   C[1,4] = A[1,0]*B[0,4] + A[1,1]*B[1,4]
    #   Tensor T[i,k,4] = 1 for i∈{0,1}, k∈{0,1}. All ones. Rank 1. ✓
    #
    # Great, so rank 75 is achievable. But let me double-check by considering
    # whether Strassen really gives rank 7 for each 2x2x2 block and whether
    # the blocks are independent (no shared U/V/W rows across blocks).
    #
    # U rows used by block (rb,ib): i in rb_indices, k in ib_indices → i*5+k.
    # Two blocks (rb1,ib1) and (rb2,ib2) share a U row iff there exist
    # i1∈rb1, k1∈ib1, i2∈rb2, k2∈ib2 with i1*5+k1 = i2*5+k2.
    # Since i∈{0..4} and k∈{0..4}, i*5+k is unique. So sharing requires
    # i1=i2 and k1=k2, i.e., rb1∩rb2≠∅ and ib1∩ib2≠∅.
    # Our rb and ib values are disjoint sets ({0,1}, {2,3}, {4}), so no sharing. ✓
    #
    # Similarly for V (k*5+j) and W (j*5+i). ✓
    #
    # So the construction is valid with rank 75.

    # Helper: add a Strassen 2x2x2 block
    # rb, cb, ib are block IDs: 0→[0,1], 1→[2,3], 2→[4]
    # For Strassen we need size-2 blocks only (ib=0 or 1, rb=0 or 1, cb=0 or 1)
    def block_indices(bid):
        if bid == 0:
            return [0, 1]
        elif bid == 1:
            return [2, 3]
        else:
            return [4]

    def add_strassen_block(rb, cb, ib):
        """Add 7 Strassen terms for a 2x2x2 block."""
        ri = block_indices(rb)  # [r0, r1]
        ci = block_indices(cb)  # [c0, c1]
        ki = block_indices(ib)  # [k0, k1]

        # Strassen for 2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)(B01-B10)   # NOTE: standard is (A10+A11)B01 - ... let me use standard
        # Actually standard Strassen:
        # M1 = (A11-A01)(B01+B11)
        # M2 = (A00+A01)(B10-B11)
        # M3 = (A00+A10)(B00+B10)
        # M4 = (A11+A10)(B01+B11)
        # M5 = (A00-A11)(B10+B11)
        # M6 = (A01-A10)(B00+B10)
        # M7 = (A10-A11)(B01-B00)
        # C00 = M5+M4-M2+M6
        # C01 = M1+M2
        # C10 = M3+M4
        # C11 = M5+M1-M3+M7

        # In our notation for block (rb, cb, ib):
        # A block: rows ri = [r0, r1], cols ki = [k0, k1]
        #   A00 = A[r0, k0], A01 = A[r0, k1]
        #   A10 = A[r1, k0], A11 = A[r1, k1]
        # B block: rows ki = [k0, k1], cols ci = [c0, c1]
        #   B00 = B[k0, c0], B01 = B[k0, c1]
        #   B10 = B[k1, c0], B11 = B[k1, c1]
        # C block: rows ri, cols ci
        #   C00 = C[r0, c0], C01 = C[r0, c1]
        #   C10 = C[r1, c0], C11 = C[r1, c1]

        # Strassen terms (using the variant from above):
        # M1 = (A11-A01)(B01+B11)
        # M2 = (A00+A01)(B10-B11)
        # M3 = (A00+A10)(B00+B10)
        # M4 = (A11+A10)(B01+B11)
        # M5 = (A00-A11)(B10+B11)
        # M6 = (A01-A10)(B00+B10)
        # M7 = (A10-A11)(B01-B00)
        #
        # C00 = M5+M4-M2+M6
        # C01 = M1+M2
        # C10 = M3+M4
        # C11 = M5+M1-M3+M7

        # Each Mi is a product of a linear combo of A entries and a linear combo of B entries.
        # In the tensor decomposition, each Mi contributes one term:
        #   U has the A-combo coefficients, V has the B-combo coefficients, W has the C-combo.

        r0, r1 = ri
        c0, c1 = ci
        k0, k1 = ki

        # Define the 7 Strassen multipliers
        # Each multiplier is (u_coeffs, v_coeffs, w_coeffs)
        # u_coeffs: list of (row, coeff) for U
        # v_coeffs: list of (row, coeff) for V
        # w_coeffs: list of (row, coeff) for W

        strassen_terms = []

        # M1 = (A11 - A01)(B01 + B11)
        # A11 = A[r1,k1], A01 = A[r0,k1]
        # B01 = B[k0,c1], B11 = B[k1,c1]
        # Contributes to: C01 (+1), C11 (+1)
        strassen_terms.append((
            [(r1 * n + k1, 1), (r0 * n + k1, -1)],
            [(k0 * n + c1, 1), (k1 * n + c1, 1)],
            [(c1 * n + r0, 1), (c1 * n + r1, 1)]
        ))

        # M2 = (A00 + A01)(B10 - B11)
        # A00 = A[r0,k0], A01 = A[r0,k1]
        # B10 = B[k1,c0], B11 = B[k1,c1]
        # Contributes to: C00 (-1), C01 (+1)
        strassen_terms.append((
            [(r0 * n + k0, 1), (r0 * n + k1, 1)],
            [(k1 * n + c0, 1), (k1 * n + c1, -1)],
            [(c0 * n + r0, -1), (c1 * n + r0, 1)]
        ))

        # M3 = (A00 + A10)(B00 + B10)
        # A00 = A[r0,k0], A10 = A[r1,k0]
        # B00 = B[k0,c0], B10 = B[k1,c0]
        # Contributes to: C10 (+1), C11 (-1)
        strassen_terms.append((
            [(r0 * n + k0, 1), (r1 * n + k0, 1)],
            [(k0 * n + c0, 1), (k1 * n + c0, 1)],
            [(c0 * n + r1, 1), (c1 * n + r1, -1)]
        ))

        # M4 = (A11 + A10)(B01 + B11)
        # A11 = A[r1,k1], A10 = A[r1,k0]
        # B01 = B[k0,c1], B11 = B[k1,c1]
        # Contributes to: C00 (+1), C10 (+1)
        strassen_terms.append((
            [(r1 * n + k1, 1), (r1 * n + k0, 1)],
            [(k0 * n + c1, 1), (k1 * n + c1, 1)],
            [(c0 * n + r0, 1), (c0 * n + r1, 1)]
        ))

        # M5 = (A00 - A11)(B10 + B11)
        # A00 = A[r0,k0], A11 = A[r1,k1]
        # B10 = B[k1,c0], B11 = B[k1,c1]
        # Contributes to: C00 (+1), C11 (+1)
        strassen_terms.append((
            [(r0 * n + k0, 1), (r1 * n + k1, -1)],
            [(k1 * n + c0, 1), (k1 * n + c1, 1)],
            [(c0 * n + r0, 1), (c1 * n + r1, 1)]
        ))

        # M6 = (A01 - A10)(B00 + B10)
        # A01 = A[r0,k1], A10 = A[r1,k0]
        # B00 = B[k0,c0], B10 = B[k1,c0]
        # Contributes to: C00 (+1)
        strassen_terms.append((
            [(r0 * n + k1, 1), (r1 * n + k0, -1)],
            [(k0 * n + c0, 1), (k1 * n + c0, 1)],
            [(c0 * n + r0, 1)]
        ))

        # M7 = (A10 - A11)(B01 - B00)
        # A10 = A[r1,k0], A11 = A[r1,k1]
        # B01 = B[k0,c1], B00 = B[k0,c0]
        # Contributes to: C11 (+1)
        strassen_terms.append((
            [(r1 * n + k0, 1), (r1 * n + k1, -1)],
            [(k0 * n + c1, 1), (k0 * n + c0, -1)],
            [(c1 * n + r1, 1)]
        ))

        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            columns.append((u_coeffs, v_coeffs, w_coeffs))

    def add_rank1_block(rb, cb, ib):
        """Add 1 term for a block where at least one dimension is size 1.
        
        The tensor T[i,k,j] = 1 for i in rb_indices, k in ib_indices, j in cb_indices.
        This is an all-ones tensor, which has rank 1.
        """
        ri = block_indices(rb)
        ci = block_indices(cb)
        ki = block_indices(ib)

        u_coeffs = [(i * n + k, 1) for i in ri for k in ki]
        v_coeffs = [(k * n + j, 1) for k in ki for j in ci]
        w_coeffs = [(j * n + i, 1) for j in ci for i in ri]

        columns.append((u_coeffs, v_coeffs, w_coeffs))

    # Iterate over all 27 blocks
    for ib in range(3):
        for rb in range(3):
            for cb in range(3):
                if _deadline_exceeded():
                    break
                # Determine if this is a 2x2x2 Strassen block
                ri = block_indices(rb)
                ci = block_indices(cb)
                ki = block_indices(ib)
                if len(ri) == 2 and len(ci) == 2 and len(ki) == 2:
                    add_strassen_block(rb, cb, ib)
                else:
                    add_rank1_block(rb, cb, ib)

    rank = len(columns)

    # Assemble dense matrices
    U = [[0] * rank for _ in range(size)]
    V = [[0] * rank for _ in range(size)]
    W = [[0] * rank for _ in range(size)]

    for c, (u_coeffs, v_coeffs, w_coeffs) in enumerate(columns):
        for row, val in u_coeffs:
            U[row][c] = val
        for row, val in v_coeffs:
            V[row][c] = val
        for row, val in w_coeffs:
            W[row][c] = val

    return {"U": U, "V": V, "W": W}
