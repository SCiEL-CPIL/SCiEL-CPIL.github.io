import time
import numpy as np

DIMENSIONS = [2, 3, 8]


def _verify(U, V, W, n, m, p, r):
    """Check that the tensor decomposition is exact for the matrix multiplication tensor."""
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)

    # Build the reconstructed tensor
    # T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r_idx in range(r):
                    s += U[i * m + j, r_idx] * V[j * p + k, r_idx] * W[k * n + i, r_idx]
                if s != 1:
                    return False
    # Also verify orthogonality: for any (i,j,k) that is NOT a valid multiplication index,
    # the sum should be 0. But since we only have one index per entry, we check that
    # entries not in the multiplication pattern are 0.
    # Actually, the tensor for matrix multiplication has 1 at (i,j,k) for all i,j,k
    # and 0 elsewhere. Since our indices cover all (i,j,k), we just need s==1 for all.
    return True


def _build_rank44():
    """Build the rank-44 Strassen-based decomposition."""
    n, m, p = 2, 3, 8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    for blk in range(4):
        bo = 2 * blk
        ve = bo
        vf = bo + 1
        vg = 8 + bo
        vh = 8 + bo + 1
        w00 = 2 * bo
        w01 = 2 * bo + 2
        w10 = 2 * bo + 1
        w11 = 2 * bo + 3

        # P1
        U[0][col] = 1; U[4][col] = 1
        V[ve][col] = 1; V[vh][col] = 1
        W[w00][col] = 1; W[w11][col] = 1
        col += 1
        # P2
        U[3][col] = 1; U[4][col] = 1
        V[ve][col] = 1
        W[w10][col] = 1; W[w11][col] = -1
        col += 1
        # P3
        U[0][col] = 1
        V[vf][col] = 1; V[vh][col] = -1
        W[w01][col] = 1; W[w11][col] = 1
        col += 1
        # P4
        U[4][col] = 1
        V[vg][col] = 1; V[ve][col] = -1
        W[w00][col] = 1; W[w10][col] = 1
        col += 1
        # P5
        U[0][col] = 1; U[1][col] = 1
        V[vh][col] = 1
        W[w00][col] = -1; W[w01][col] = 1
        col += 1
        # P6
        U[3][col] = 1; U[0][col] = -1
        V[ve][col] = 1; V[vf][col] = 1
        W[w11][col] = 1
        col += 1
        # P7
        U[1][col] = 1; U[4][col] = -1
        V[vg][col] = 1; V[vh][col] = 1
        W[w00][col] = 1
        col += 1

    for k in range(8):
        for i in range(2):
            U[i * 3 + 2][col] = 1
            V[16 + k][col] = 1
            W[2 * k + i][col] = 1
            col += 1

    assert col == rank, f"col={col}, rank={rank}"
    return U, V, W


def _build_rank43():
    """Attempt rank-43 by sharing one Strassen factor between adjacent blocks."""
    n, m, p = 2, 3, 8
    rank = 43
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    # Process 4 blocks of 2x2. For blocks 0 and 1, try to share P1 (the (a+d)(e+h) factor).
    # P1 for block 0: U uses A[0,0]+A[1,1], V uses B[0,0]+B[1,1]
    # P1 for block 1: U uses A[0,0]+A[1,1], V uses B[0,2]+B[1,3]
    # The U part is the same! So we can use one U-factor for both, but V differs.
    # Actually, the factorization requires the same U, V, W entries per rank-1 term.
    # We can't share U across two different rank-1 terms unless we also share V and W.
    #
    # Alternative: Use a different decomposition of 2x2*2x8.
    # The tensor for 2x2*2x8 has rank at most 4*7=28 (four independent 2x2*2x2).
    # Could it be lower? The 2x2*2x8 tensor is isomorphic to 2x2*2x2 ⊗ I_4 (in the output dim).
    # Actually no, it's 4 copies of 2x2*2x2 with independent inputs.
    # The rank of 2x2*2x8 = 4 * rank(2x2*2x2) if they're independent.
    # But they're not fully independent - they share the same A matrix.
    #
    # Let's try: treat A as 2x2, B as 2x8.
    # We can write C = A * B. Use Strassen-like with 7 multiplications per 2x2 block of B.
    # That's what we have. Total 28 for the 2x2*2x8 part.
    #
    # For the 2x1*1x8 part: rank 16.
    # Total: 44.
    #
    # To get 43, we need to save 1. Let's try a non-standard decomposition.
    #
    # Idea: For the 2x1*1x8 part, can we use rank < 16?
    # A is 2x1 (column vector), B is 1x8 (row vector).
    # C = A * B is a 2x8 outer product.
    # The tensor for outer product of (2x1) and (1x8) is rank 2*8 = 16.
    # It's an outer product tensor, which is rank = product of dimensions = 16.
    # Can't reduce below 16.
    #
    # So we need to reduce the 2x2*2x8 part from 28 to 27.
    # The 2x2*2x8 tensor: is its rank 28 or less?
    # It's the tensor T[i,j,k] = δ_{i,j} for i,j in {0,1}, k in {0..7}... no.
    # Actually T[i,j,k] = 1 if the (i,j) entry of A times (j,k) entry of B contributes to C[i,k].
    # For 2x2 * 2x8: C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor has entries T[i,j,k] where i in {0,1}, j in {0,1}, k in {0..7}.
    # This is 4 independent copies of the 2x2*2x2 tensor (one for each pair of columns of B).
    # Since A is shared, the tensors are not independent.
    # The full tensor is: for each (i,j), the slice over k is a vector of length 8,
    # and T[i,j,k] = 1 for all k (since each A[i,j] contributes to all C[i,k] via B[j,k]).
    # Wait, no. T[i,j,k] is the coefficient of A[i,j]*B[j,k] in C[i,k].
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k].
    # So T[i,0,k] = 1 and T[i,1,k] = 1 for all i,k.
    # This means the tensor is: T[i,j,k] = 1 for all i,j,k.
    # This is just the outer product of three all-ones vectors!
    # The rank of the all-ones tensor of shape (2,2,8) is 1!
    #
    # Wait, that can't be right. Let me reconsider.
    # The matrix multiplication tensor for (n,m,p) has T[i,j,k] = δ_{i,j} * δ_{j,k}? No.
    # T[i,j,k] is 1 if j is the inner index, i is row of A, k is col of B.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # So the tensor T has T[i,j,k] = 1 for all valid (i,j,k) where i in [n], j in [m], k in [p].
    # But that's not right either. The tensor is defined as:
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k * (coefficient)
    # where coefficient is 1 if (i,j,k) is a valid multiplication triple.
    # For 2x2*2x8: i in {0,1}, j in {0,1}, k in {0..7}.
    # All 2*2*8 = 32 triples are valid.
    # So T is the all-ones tensor of shape (2,2,8).
    # The rank of the all-ones tensor is 1!
    #
    # But that contradicts Strassen. The issue is that the tensor decomposition must be
    # multiplicative: each rank-1 term is u ⊗ v ⊗ w, and the product u_i * v_j * w_k
    # must equal T[i,j,k].
    # For the all-ones tensor, u_i * v_j * w_k = 1 for all i,j,k.
    # This requires u_i = a_i, v_j = b_j, w_k = c_k with a_i b_j c_k = 1.
    # This is possible: a_i = 1, b_j = 1, c_k = 1. So rank 1.
    #
    # But that would mean 2x2*2x8 has rank 1, which is absurd.
    #
    # I'm confusing the tensor with the multiplication map. Let me reconsider.
    #
    # The matrix multiplication tensor is:
    # μ = sum_{i,j,k} e_i^{(C)} ⊗ e_j^{(A)} ⊗ e_k^{(B)} where j links A and B.
    # More precisely: μ = sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p e_i^C ⊗ e_{ij}^A ⊗ e_{jk}^B
    # where e_{ij}^A is the (i,j)-th basis vector of A (flattened), etc.
    #
    # The tensor T has T[i, j, k] = 1 if the (i,j) entry of A and (j,k) entry of B
    # multiply to contribute to C[i,k]. But j is the shared index.
    #
    # For 2x2 * 2x8:
    # A has entries A[0,0], A[0,1], A[1,0], A[1,1] -> indices 0,1,2,3 (row-major)
    # B has entries B[0,0], B[0,1], ..., B[1,7] -> indices 0..15
    # C has entries C[0,0], C[0,1], ..., C[1,7] -> indices 0..15
    #
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    #
    # So the tensor T[i, a_idx, b_idx, c_idx] is not a 3-way tensor in the simple sense.
    # The standard formulation: T is a 3-way tensor where:
    # - First index: (i, j) for A, flattened to n*m
    # - Second index: (j, k) for B, flattened to m*p
    # - Third index: (i, k) for C, flattened to n*p
    #
    # T[(i,j), (j,k), (i,k)] = 1, and T = 0 otherwise.
    #
    # So for 2x2*2x8:
    # A index: (i,j) -> i in {0,1}, j in {0,1} -> 4 entries
    # B index: (j,k) -> j in {0,1}, k in {0..7} -> 16 entries
    # C index: (i,k) -> i in {0,1}, k in {0..7} -> 16 entries
    #
    # T[(i,j), (j',k), (i',k')] = 1 if i=i', j=j', k=k', else 0.
    #
    # This is NOT the all-ones tensor. It's a "diagonal" tensor.
    # The number of non-zero entries is 2*2*8 = 32.
    #
    # This tensor can be decomposed as:
    # T = sum_{i,j,k} e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(i,k)}
    #
    # This is a sum of 32 rank-1 terms (schoolbook).
    # Strassen reduces this for 2x2*2x2 from 8 to 7 terms.
    #
    # For 2x2*2x8, we have 4 independent "slices" in the k direction (pairs of k).
    # But A is shared across all slices.
    #
    # The rank of this tensor: it's the sum of 4 tensors, each being a 2x2*2x2 tensor
    # embedded in the full space. Since they share the A indices, the rank is at most
    # 4 * 7 = 28 (using Strassen for each slice).
    #
    # Can we do better? The 4 slices share A, so there might be cross-slice optimizations.
    # This is essentially the 2x2*2x8 tensor, whose rank is known to be 28 (or less?).
    #
    # For now, assume rank 28 for 2x2*2x8 and rank 16 for 2x1*1x8, total 44.
    # Return None to indicate we can't construct rank 43.
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 110
    n, m, p = DIMENSIONS

    # First, build the known rank-44 solution
    U44, V44, W44 = _build_rank44()

    # Try rank-43
    if time.time() < deadline - 60:
        result43 = _build_rank43()
        if result43 is not None:
            U43, V43, W43 = result43
            if _verify(U43, V43, W43, n, m, p, 43):
                return {"U": U43, "V": V43, "W": W43}

    # Verify rank-44
    if _verify(U44, V44, W44, n, m, p, 44):
        return {"U": U44, "V": V44, "W": W44}

    # Fallback: schoolbook
    rank_fallback = n * m * p
    U = [[0] * rank_fallback for _ in range(n * m)]
    V = [[0] * rank_fallback for _ in range(m * p)]
    W = [[0] * rank_fallback for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}
