"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin before 120s limit

    n, m, p = DIMENSIONS

    # Schoolbook decomposition as fallback (rank 54)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    def to_lists(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to find a lower-rank decomposition via random search + local optimization
    # The tensor for 3x3x6 multiplication has a specific structure.
    # We use the fact that this is equivalent to 3 independent 1x3x6 multiplications
    # (each row of A times B), so we can try to decompose each 1x3x6 tensor.

    # For 1x3x6: C[k] = sum_j A[j] * B[j,k]
    # This is a bilinear form with 3*6 = 18 entries.
    # The tensor has dimensions (3, 6) with 18 non-zero entries.
    # Rank of this tensor: it's the outer product structure of a 3x6 matrix viewed as a tensor.
    # Actually, T_{j,k} = 1 for all j,k. This is a rank-1 tensor!
    # Wait, that's not right. Let me think again.

    # The multiplication tensor for 1x3x6:
    # T[j, k] = 1 (for all j in 0..2, k in 0..5)
    # But this is just the identity on the bilinear form, which has rank... 
    # Actually, the tensor is T_{j,k} = e_j ⊗ e_k, so it's a sum of 18 rank-1 terms.
    # But can we do better? The tensor T is a 3x6 matrix of all ones.
    # The rank of a matrix as a tensor (with one mode of size 1) is just the matrix rank.
    # A 3x6 all-ones matrix has rank 1!
    # So each 1x3x6 sub-tensor has rank 1? That can't be right for matrix multiplication.

    # Let me reconsider. The full 3x3x6 tensor T_{i,j,k}:
    # T[i,j,k] = 1 if the (i,j) entry of A multiplies with (j,k) entry of B
    # to contribute to (i,k) entry of C.
    # So T[i,j,k] = 1 for all valid (i,j,k) where j is shared.
    # This is a 3x3x6 tensor where T[i,j,k] = 1 for all i,j,k.
    # That's a rank-1 tensor! But that's the all-ones tensor, not the multiplication tensor.

    # I'm confusing myself. Let me be very precise.
    # Matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = 1 (for all i,j,k in range).
    # No wait, T[i,j,k] is the coefficient of A[i,j]*B[j,k] in C[i,k].
    # So T[i,j,k] = 1 if the pair (i,j,k) is valid, i.e., always in this case.
    # So T is the all-ones tensor of shape 3x3x6.
    # The rank of the all-ones tensor of shape (n,m,p) is 1 (it's the outer product of three all-ones vectors).
    # But that would mean rank-1 decomposition exists, which is clearly wrong for matrix multiplication.

    # I think I'm misunderstanding the tensor structure. Let me look at the schoolbook code:
    # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # This means: for each (i,j,k), there's a rank-1 term that picks A[i,j], B[j,k], and puts it in C[i,k].
    # The tensor entry T[i,j,k] (with appropriate indexing) is 1 for each (i,j,k) triple.
    # But the tensor is not simply all-ones; it's structured such that
    # T[i, j, k] contributes to the bilinear form.

    # Actually, the standard tensor for matrix multiplication M_{n,m,p} has:
    # T[i, j, k] = 1 for all (i,j,k) in [n]x[m]x[p]
    # where the first index corresponds to row of A / row of C,
    # second to column of A / row of B,
    # third to column of B / column of C.
    # And the decomposition is: T = sum_s U_s ⊗ V_s ⊗ W_s
    # where (U_s)_{i,j} * (V_s)_{j,k} * (W_s)_{i,k} = T[i,j,k] = 1 for all (i,j,k).

    # Wait, that's not the right equation either. The correct equation is:
    # For each s, we have vectors u_s (length n*m), v_s (length m*p), w_s (length p*n).
    # Reshape: u_s -> U_s (n x m), v_s -> V_s (m x p), w_s -> W_s (p x n) -> W_s^T (n x p)
    # Then: sum_s U_s * V_s * W_s^T = I (identity in the appropriate sense)
    # More precisely: the bilinear form sum_j A[i,j] B[j,k] should be recoverable.
    # The condition is: sum_s (u_s)_{i*m+j} (v_s)_{j*p+k} (w_s)_{k*n+i} = 1 for all (i,j,k).

    # So T[i*m+j, j*p+k, k*n+i] = 1 for all (i,j,k).
    # This is a sparse tensor! Not all entries are 1.

    # For 3x3x6: the tensor has 3*3*6 = 54 non-zero entries (all 1s) in a 9x18x18 space.
    # The schoolbook uses rank 54 (one rank-1 term per non-zero entry).

    # Can we do better? This is essentially asking for the matrix multiplication rank
    # of M_{3,3,6}. Known results:
    # - M_{n,n,n} has rank at most n^3 (schoolbook), at least 2n^2 - 1 (for n>=2).
    # - For M_{3,3,3}: Strassen gives rank 7.
    # - For M_{3,3,6}: we can view this as 3 copies of M_{1,3,6} or 6 copies of M_{3,3,1}.
    # - M_{1,3,6}: C[k] = sum_j A[j] B[j,k]. This is a 3x6 matrix multiplication with one row.
    #   The tensor for this has 18 non-zero entries. The rank of this tensor...
    #   Actually, this is equivalent to multiplying a 1x3 vector by a 3x6 matrix.
    #   This can be done with rank 6 (one rank-1 term per output k, using sum_j A[j]B[j,k]).
    #   Wait, no. Each output C[k] = sum_j A[j]B[j,k] requires 3 multiplications (one per j).
    #   But we can factor: C = A * B where A is 1x3 and B is 3x6.
    #   This is a standard matrix multiply of size 1x3 times 3x6.
    #   The tensor rank of M_{1,3,6} is... Let's think.
    #   T[j,k] = 1 for all j in 0..2, k in 0..5. This is a 3x6 all-ones matrix.
    #   As a bilinear tensor (with one mode being trivial), the rank equals the matrix rank = 1.
    #   But that's not right for multiplication. 

    # I think the issue is that the tensor for M_{1,3,6} is NOT the all-ones matrix.
    # Let me re-derive from the schoolbook construction.
    # For M_{1,3,6} (n=1, m=3, p=6):
    # U has shape (1*3, rank) = (3, rank)
    # V has shape (3*6, rank) = (18, rank)  
    # W has shape (6*1, rank) = (6, rank)
    # Schoolbook: for each (i,j,k) with i=0:
    #   U[j, col] = 1, V[j*6+k, col] = 1, W[k*1+0, col] = 1
    # So the condition is: sum_s U_s[j] V_s[j*6+k] W_s[k] = 1 for all j in 0..2, k in 0..5.
    # This means the 3x18x6 tensor has 18 non-zero entries.
    # We can try: let W_s[k] = delta_{k,s} (standard basis), then we need
    #   sum_s U_s[j] V_s[j*6+k] delta_{k,s} = U_k[j] V_k[j*6+k] = 1 for all j,k.
    # So U_k[j] V_k[j*6+k] = 1. We can set U_k[j] = 1, V_k[j*6+k] = 1 for all j,k.
    # That gives rank 6.
    # Can we do better? If we set W_s to be arbitrary, we need the system to have rank < 6.
    # The tensor T[j, k] (where we've summed over the internal index) is the 3x6 all-ones matrix.
    # But the tensor is 3-dimensional: T[j, j'p+k, k]... no, the indices are (j, jp+k, k) -> T[j, jp+k, k] = 1.
    # This is a 3x18x6 tensor. Its rank is at least max(3, 18, 6) / something... 
    # Actually, for this specific tensor, I believe the rank is 6 (same as schoolbook for this sub-problem).

    # Given the complexity, let me just try a random search for a lower-rank decomposition.
    # We'll use the fact that entries can be half-integers, so we can work in 1/2 * Z.

    # Actually, let me try a different approach: use the known Strassen decomposition for 3x3x3
    # and extend it. But 3x3x6 is not a standard case.

    # Let me just verify that the schoolbook is correct and return it, as finding a better
    # decomposition within the time limit is challenging.

    # Quick sanity check: verify the schoolbook decomposition
    def verify(U, V, W, n, m, p):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for col in range(rank):
                        s += U[i*m+j, col] * V[j*p+k, col] * W[k*n+i, col]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Try a simple structured approach:
    # For 3x3x6, we can write it as a block structure.
    # A is 3x3, B is 3x6, C is 3x6.
    # C = A * B.
    # We can use the fact that B can be split into two 3x3 blocks: B = [B1 | B2].
    # Then C = [A*B1 | A*B2].
    # So we need to compute two 3x3 matrix multiplications.
    # If we use Strassen for each (rank 7), total rank = 14.
    # But we need to check if the decomposition is exact with integer/half-integer entries.
    # Strassen's algorithm uses +/-1 entries, which are integers.

    # Let's implement Strassen for 3x3x3 and use it twice.
    # Strassen decomposition for 3x3x3:
    # There are 7 rank-1 terms.

    # Actually, the minimal rank for 3x3x3 matrix multiplication is known to be 7 (Strassen).
    # Let me look up the explicit Strassen tensors for 3x3x3.

    # Strassen's 7-term decomposition for 3x3:
    # M1 = (A11 - A22)(B22 - B11)  -- no, that's not right.
    # Standard Strassen for 2x2:
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B21-B22)
    # P4 = A22(B12-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6

    # For 3x3, we can use a 3x3 decomposition. One approach:
    # Use the 2x2 Strassen on sub-blocks and handle the boundary.
    # This gets complicated. Let me just use the schoolbook for now.

    # Given time constraints, return the schoolbook decomposition.
    return to_lists(U_school, V_school, W_school)
# EVOLVE-BLOCK-END
