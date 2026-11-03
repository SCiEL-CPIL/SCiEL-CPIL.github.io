"""Rewritten program for 4x4x5 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Tensors: U (4x7), V (4x7), W (4x7) with W indexed as [k*2+i, r] for output C[i,k].
    """
    # Strassen's 7 products:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M2 - M6
    #
    # A indices: A11->0, A12->1, A21->2, A22->3
    # B indices: B11->0, B12->1, B21->2, B22->3
    # C indices: C11->0, C12->1, C21->2, C22->3 (row-major: C[i,k] at i*2+k)
    # W index: k*2 + i, so C11->0*2+0=0, C12->0*2+1=1, C21->1*2+0=2, C22->1*2+1=3

    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)

    # M1: (A11+A22)*(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1  # C11
    W[3, 0] = 1  # C22

    # M2: (A21+A22)*B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1  # C21
    W[3, 1] = 1  # C22

    # M3: A11*(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1  # C12
    W[3, 2] = -1 # C22

    # M4: A22*(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1 # C11
    W[2, 3] = 1  # C21

    # M5: (A11+A12)*B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1  # C11
    W[1, 4] = 1  # C12

    # M6: (A21-A11)*(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = -1 # C22

    # M7: (A12-A22)*(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1 # C11

    return U, V, W


def construct():
    """
    Construct a tensor decomposition for 4x4 * 4x5 -> 4x5.
    Uses Strassen-based block decomposition to achieve rank < 80 (schoolbook).
    """
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 4, 4, 5

    # Strategy: Partition A (4x4) into 2x2 blocks of 2x2, B (4x5) into 2x2 blocks of 2x2 plus a 2x1 strip.
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12, B13], [B21, B22, B23]] where B11,B21 are 2x2 and B12,B22 are 2x2 and B13,B23 are 2x1
    #
    # C = A*B: C[i,k] = sum_j A[i,j]*B[j,k]
    # C is 4x5. Partition C into blocks:
    # C = [[C11, C12, C13], [C21, C22, C23]]
    # where C11, C21 are 2x2, C12, C22 are 2x2, C13, C23 are 2x1
    #
    # C11 = A11*B11 + A12*B21  (2x2 * 2x2 + 2x2 * 2x2)
    # C12 = A11*B12 + A12*B22  (2x2 * 2x2 + 2x2 * 2x2)
    # C13 = A11*B13 + A12*B23  (2x2 * 2x1 + 2x2 * 2x1)
    # C21 = A21*B11 + A22*B21  (2x2 * 2x2 + 2x2 * 2x2)
    # C22 = A21*B12 + A22*B22  (2x2 * 2x2 + 2x2 * 2x2)
    # C23 = A21*B13 + A22*B23  (2x2 * 2x1 + 2x2 * 2x1)
    #
    # For each 2x2*2x2 product, use Strassen (rank 7).
    # For 2x2*2x1, we need a decomposition. A 2x2 times 2x1 gives 2x1.
    # (2x2)*(2x1): C[i] = A[i,0]*B[0] + A[i,1]*B[1]
    # This is just a bilinear form with 4 terms (rank 4 schoolbook), or we can use
    # a clever decomposition. Actually for 2x2*2x1, the tensor has dimension 2*2*2=8,
    # and the minimum rank is 4 (since it's essentially two independent 2x1 multiplications).
    # Let's just use rank 4 for each.
    #
    # Total rank estimate:
    # 6 instances of 2x2*2x2: 6*7 = 42
    # 2 instances of 2x2*2x1: 2*4 = 8
    # Total: 50
    #
    # But we can share Strassen factors across blocks that share the same A or B submatrices.
    # Let's think about this more carefully.
    #
    # Actually, let's use a different approach. We'll compute:
    # For the 2x2*2x2 blocks, we use Strassen. But we need to be careful about sharing.
    #
    # Let me define the indices:
    # A is 4x4: A[i,j] for i,j in {0,1,2,3}
    # B is 4x5: B[j,k] for j in {0,1,2,3}, k in {0,1,2,3,4}
    # C is 4x5: C[i,k] for i in {0,1,2,3}, k in {0,1,2,3,4}
    #
    # W index: k*n + i = k*4 + i
    #
    # Block structure:
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4], A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4], B13 = B[0:2, 4:5]
    # B21 = B[2:4, 0:2], B22 = B[2:4, 2:4], B23 = B[2:4, 4:5]
    #
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C13 = A11*B13 + A12*B23
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # C23 = A21*B13 + A22*B23

    # Let's build the decomposition column by column.
    # Each column r corresponds to one bilinear term: U[:,r] * V[:,r] * W[:,r]
    # where the term contributes to C[i,k] the value U[i,j_part]*V[j_part,k_part]*W[k*4+i]
    #
    # Actually, the standard form is:
    # C[i,k] = sum_r U[i*m+j_r, r] * V[j_r*p+k_r, r] * W[k*n+i, r]
    # Wait, that's not quite right. Let me re-read the contract.
    #
    # U has shape (n*m, rank) = (16, rank)
    # V has shape (m*p, rank) = (20, rank)
    # W has shape (p*n, rank) = (20, rank)
    #
    # The decomposition is: C[i,k] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
    # No wait, that's not right either. Let me think about this more carefully.
    #
    # The matrix multiplication tensor T is defined as:
    # T[i, j, k] = 1 if C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # A rank-r decomposition writes:
    # T[i,j,k] = sum_{r=1}^{r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # Wait, that doesn't make sense dimensionally. Let me reconsider.
    #
    # Actually, the standard tensor decomposition for matrix multiplication is:
    # T_{i,j,k} = sum_r u_{i,j}^{(r)} * v_{j,k}^{(r)} * w_{k,i}^{(r)}
    # where T_{i,j,k} = 1 for all i,j,k (it's the multiplication tensor).
    #
    # But the contract says:
    # U has shape (n*m, rank), indexed by (i*m + j, r)
    # V has shape (m*p, rank), indexed by (j*p + k, r)
    # W has shape (p*n, rank), indexed by (k*n + i, r)
    #
    # And the decomposition should satisfy:
    # For the multiplication tensor, we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    #
    # Wait no, that's not right. The multiplication tensor has entries T[i,j,k] = 1
    # only when the indices are consistent... Actually no, the matrix multiplication
    # tensor is defined such that:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    #
    # As a tensor, this is: sum_{i,j,k} A[i,j] * B[j,k] * T[i,j,k] = (A*B)[i,k]
    # where T[i,j,k] = 1 for all valid i,j,k.
    #
    # So the tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    #
    # A rank-r decomposition means:
    # T[i,j,k] = sum_{r=1}^{r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # And we need this to equal 1 for all i,j,k.
    #
    # Let me verify with the schoolbook: for each (i,j,k), there's one column where
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1, and all others are 0.
    # So the sum is 1. Correct.

    # OK so I need to find U, V, W such that for all i in [0,4), j in [0,4), k in [0,5):
    # sum_r U[i*4+j, r] * V[j*5+k, r] * W[k*4+i, r] = 1

    # Let me use the Strassen-based approach.
    #
    # For 2x2*2x2 Strassen, the decomposition satisfies:
    # For i in {0,1}, j in {0,1}, k in {0,1}:
    # sum_r U_2x2[i*2+j, r] * V_2x2[j*2+k, r] * W_2x2[k*2+i, r] = 1
    #
    # Now for our 4x4*4x5 problem, let's partition into blocks.
    #
    # Let's define block indices:
    # A block (a, b) where a in {0,1} (row block), b in {0,1} (col block), each 2x2
    # B block (b, c) where b in {0,1} (row block), c in {0,1,2} (col block)
    #   - c=0: cols 0-1 (2x2)
    #   - c=1: cols 2-3 (2x2)
    #   - c=2: col 4 (2x1)
    # C block (a, c) where a in {0,1} (row block), c in {0,1,2} (col block)
    #
    # C[a, c] = sum_b A[a, b] * B[b, c]
    #
    # For c=0,1 (2x2 output): C[a,c] = A[a,0]*B[0,c] + A[a,1]*B[1,c]
    #   Each is a sum of two 2x2*2x2 products.
    # For c=2 (2x1 output): C[a,2] = A[a,0]*B[0,2] + A[a,1]*B[1,2]
    #   Each is a 2x2*2x1 product.
    #
    # So we have:
    # - 4 instances of (2x2)*(2x2) for the 2x2 output blocks: (a=0,1) x (c=0,1), each with b=0,1
    #   That's 8 instances of 2x2*2x2.
    # - 2 instances of (2x2)*(2x1) for the 2x1 output blocks: (a=0,1), each with b=0,1
    #   That's 4 instances of 2x2*2x1.
    #
    # If we use Strassen (rank 7) for each 2x2*2x2 and rank 4 for each 2x2*2x1:
    # Total rank = 8*7 + 4*4 = 56 + 16 = 72.
    #
    # That's better than 80 (schoolbook) but not great. Let me see if I can do better.
    #
    # Alternative: Use a 4x4*4x4 decomposition and then extend to 4x4*4x5.
    #
    # Actually, let me try a different approach. Let's use the fact that we can
    # combine some of the Strassen decompositions.
    #
    # For the 2x2*2x2 products in C[a,c] = A[a,0]*B[0,c] + A[a,1]*B[1,c],
    # we can think of this as a 4x4*4x4 problem partitioned into 2x2 blocks.
    #
    # Actually, let me just go with the straightforward approach and get rank 72,
    # which is better than 80.
    #
    # Wait, I realize I should double-check: is the schoolbook rank really 80?
    # n*m*p = 4*4*5 = 80. Yes.
    #
    # Let me implement the rank-72 decomposition.

    # First, let me get the Strassen 2x2 decomposition
    U2, V2, W2 = _strassen_2x2()

    # For 2x2*2x1: C[i] = A[i,0]*B[0] + A[i,1]*B[1]
    # U has shape (4, rank), V has shape (2, rank), W has shape (2, rank)
    # W index: k*2+i where k in {0,1}, i in {0,1}
    # Actually for 2x2*2x1: n=2, m=2, p=1
    # U shape: (n*m, rank) = (4, rank)
    # V shape: (m*p, rank) = (2, rank)
    # W shape: (p*n, rank) = (2, rank)
    # W index: k*n + i = k*2 + i, with k in {0}, i in {0,1}
    # So W[0, r] and W[1, r] correspond to output C[0] and C[1]
    #
    # Schoolbook for 2x2*2x1 (rank 4):
    # Term (i,j): U[i*2+j, r] = 1, V[j*1+0, r] = 1, W[0*2+i, r] = 1
    U21 = np.zeros((4, 4), dtype=float)
    V21 = np.zeros((2, 4), dtype=float)
    W21 = np.zeros((2, 4), dtype=float)
    for i in range(2):
        for j in range(2):
            r = i * 2 + j
            U21[i * 2 + j, r] = 1
            V21[j, r] = 1
            W21[i, r] = 1  # k=0, so k*2+i = i

    # Now build the full 4x4*4x5 decomposition.
    #
    # Column assignments:
    # For each (a, b, c) where a in {0,1}, b in {0,1}, c in {0,1,2}:
    #   If c in {0,1}: 2x2*2x2 product, uses 7 Strassen columns
    #   If c == 2: 2x2*2x1 product, uses 4 columns
    #
    # Total columns: 4 * (7 + 7 + 4) = 4 * 18 = 72

    total_rank = 4 * (7 + 7 + 4)  # 72
    U_full = np.zeros((16, total_rank), dtype=float)
    V_full = np.zeros((20, total_rank), dtype=float)
    W_full = np.zeros((20, total_rank), dtype=float)

    col = 0
    for a in range(2):  # A row block
        for b in range(2):  # B row block / A col block
            for c in range(3):  # B col block / C col block
                if c < 2:
                    # 2x2 * 2x2 product: A[a,b] * B[b,c] contributes to C[a,c]
                    # A[a,b] indices: rows 2*a to 2*a+1, cols 2*b to 2*b+1
                    # B[b,c] indices: rows 2*b to 2*b+1, cols 2*c to 2*c+1
                    # C[a,c] indices: rows 2*a to 2*a+1, cols 2*c to 2*c+1
                    #
                    # In the full problem:
                    # A[i,j] where i = 2*a + i2, j = 2*b + j2
                    # B[j,k] where j = 2*b + j2, k = 2*c + k2
                    # C[i,k] where i = 2*a + i2, k = 2*c + k2
                    #
                    # Strassen decomposition for local 2x2*2x2:
                    # U2[i2*2+j2, r] * V2[j2*2+k2, r] * W2[k2*2+i2, r] = 1
                    #
                    # In full indices:
                    # U_full[(2*a+i2)*4 + (2*b+j2), col+r] = U2[i2*2+j2, r]
                    # V_full[(2*b+j2)*5 + (2*c+k2), col+r] = V2[j2*2+k2, r]
                    # W_full[(2*c+k2)*4 + (2*a+i2), col+r] = W2[k2*2+i2, r]

                    for i2 in range(2):
                        for j2 in range(2):
                            for r in range(7):
                                val = U2[i2 * 2 + j2, r]
                                if val != 0:
                                    U_full[(2 * a + i2) * 4 + (2 * b + j2), col + r] += val

                    for j2 in range(2):
                        for k2 in range(2):
                            for r in range(7):
                                val = V2[j2 * 2 + k2, r]
                                if val != 0:
                                    V_full[(2 * b + j2) * 5 + (2 * c + k2), col + r] += val

                    for k2 in range(2):
                        for i2 in range(2):
                            for r in range(7):
                                val = W2[k2 * 2 + i2, r]
                                if val != 0:
                                    W_full[(2 * c + k2) * 4 + (2 * a + i2), col + r] += val

                    col += 7
                else:
                    # 2x2 * 2x1 product: A[a,b] * B[b,2] contributes to C[a,2]
                    # A[a,b] indices: rows 2*a to 2*a+1, cols 2*b to 2*b+1
                    # B[b,2] indices: rows 2*b to 2*b+1, col 4
                    # C[a,2] indices: rows 2*a to 2*a+1, col 4
                    #
                    # In the full problem:
                    # A[i,j] where i = 2*a + i2, j = 2*b + j2
                    # B[j,k] where j = 2*b + j2, k = 4
                    # C[i,k] where i = 2*a + i2, k = 4
                    #
                    # 2x2*2x1 decomposition:
                    # U21[i2*2+j2, r] * V21[j2, r] * W21[i2, r] = 1
                    #
                    # In full indices:
                    # U_full[(2*a+i2)*4 + (2*b+j2), col+r] = U21[i2*2+j2, r]
                    # V_full[(2*b+j2)*5 + 4, col+r] = V21[j2, r]
                    # W_full[4*4 + (2*a+i2), col+r] = W21[i2, r]
                    # W index: k*n + i = 4*4 + (2*a+i2) = 16 + 2*a + i2

                    for i2 in range(2):
                        for j2 in range(2):
                            for r in range(4):
                                val = U21[i2 * 2 + j2, r]
                                if val != 0:
                                    U_full[(2 * a + i2) * 4 + (2 * b + j2), col + r] += val

                    for j2 in range(2):
                        for r in range(4):
                            val = V21[j2, r]
                            if val != 0:
                                V_full[(2 * b + j2) * 5 + 4, col + r] += val

                    for i2 in range(2):
                        for r in range(4):
                            val = W21[i2, r]
                            if val != 0:
                                W_full[4 * 4 + (2 * a + i2), col + r] += val

                    col += 4

    # Verify the decomposition
    # Check: sum_r U[i*4+j, r] * V[j*5+k, r] * W[k*4+i, r] should be 1 for all i,j,k
    max_err = 0.0
    for i in range(4):
        for j in range(4):
            for k in range(5):
                s = 0.0
                for r in range(total_rank):
                    s += U_full[i * 4 + j, r] * V_full[j * 5 + k, r] * W_full[k * 4 + i, r]
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err

    if max_err > 1e-10:
        # Fallback to schoolbook if verification fails
        rank = 80
        U_fb = np.zeros((16, rank), dtype=float)
        V_fb = np.zeros((20, rank), dtype=float)
        W_fb = np.zeros((20, rank), dtype=float)
        column = 0
        for i in range(4):
            for j in range(4):
                for k in range(5):
                    U_fb[i * 4 + j, column] = 1
                    V_fb[j * 5 + k, column] = 1
                    W_fb[k * 4 + i, column] = 1
                    column += 1
        U_full = U_fb
        V_full = V_fb
        W_full = W_fb
        total_rank = rank

    # Convert to lists of lists with integer/half-integer entries
    U_list = [[float(x) for x in row] for row in U_full.tolist()]
    V_list = [[float(x) for x in row] for row in V_full.tolist()]
    W_list = [[float(x) for x in row] for row in W_full.tolist()]

    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
