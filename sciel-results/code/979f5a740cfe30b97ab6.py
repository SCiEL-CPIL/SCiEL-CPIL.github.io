import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115

    n, m, p = DIMENSIONS  # n=2, m=2, p=12

    # Rank 24 decomposition
    # The tensor for 2x2x12 multiplication is a direct sum of 12 copies of the 2x2x1 tensor.
    # The 2x2x1 tensor (matrix-vector product) has rank 2.
    # Therefore, the total rank is 12 * 2 = 24.
    #
    # Decomposition:
    # For each column b (0..11) of B (and C):
    #   C[0, b] = A[0, 0] * B[0, b] + A[0, 1] * B[1, b]
    #   C[1, b] = A[1, 0] * B[0, b] + A[1, 1] * B[1, b]
    #
    # We can represent this with 2 rank-1 terms per column b.
    #
    # Term 1 (for column b):
    #   Multiplies A[0, 0] and B[0, b] to contribute to C[0, b].
    #   Wait, this only covers part of C[0, b].
    #   Let's re-examine the rank of 2x2x1.
    #
    #   y_0 = a_00 x_0 + a_01 x_1
    #   y_1 = a_10 x_0 + a_11 x_1
    #
    #   Rank 2 decomposition:
    #   M_1 = (a_00 + a_10) * (x_0 + x_1) ? No, that gives cross terms.
    #   
    #   Let's look at the standard rank-2 decomposition for 2x2x1.
    #   Actually, the rank of 2x2x1 is 2.
    #   T_{i, j, k} = 1 if k is the output index corresponding to row i?
    #   No, T_{i, j, k} where i is row of A, j is col of A, k is row of Y (output).
    #   Y is 2x1, so k=0,1.
    #   T_{i, j, k} is non-zero only if k=i?
    #   y_k = sum_j a_{k, j} x_{j, 0}.
    #   So T_{i, j, k} = 1 if i=k, else 0.
    #   This is a block diagonal tensor.
    #   T = T^{(0)} \oplus T^{(1)}.
    #   T^{(0)}_{i, j, 0} = 1 if i=0. (Row 0 of A, any col j, output 0)
    #   T^{(1)}_{i, j, 1} = 1 if i=1. (Row 1 of A, any col j, output 1)
    #   
    #   Rank of T^{(0)}:
    #   It's a 2x2x1 tensor with entries:
    #   (0,0,0)=1, (0,1,0)=1
    #   (1,0,0)=0, (1,1,0)=0
    #   This is u \otimes v \otimes w where u=[1,0], v=[1,1], w=[1].
    #   So rank is 1.
    #   
    #   Rank of T^{(1)}:
    #   Entries:
    #   (0,0,1)=0, (0,1,1)=0
    #   (1,0,1)=1, (1,1,1)=1
    #   This is u \otimes v \otimes w where u=[0,1], v=[1,1], w=[1].
    #   So rank is 1.
    #   
    #   Total rank = 1 + 1 = 2.
    #
    #   So for each column b, we need 2 terms.
    #   Total rank = 12 * 2 = 24.
    #
    #   Let's construct the 24 terms.
    #   For each b in 0..11:
    #   Term 1:
    #     U: 1 at index 0 (A[0,0])? No, we need to capture the whole row.
    #     The rank-1 term for T^{(0)} is u=[1,0], v=[1,1], w=[1].
    #     This means:
    #     U factor: 1 at row 0 of A. (Indices 0 and 1? No, U is size n*m = 4).
    #     U[0] = 1, U[1] = 0?
    #     Wait, u is a vector of length n=2.
    #     In the flattened U matrix (size n*m=4), the index for A[i, j] is i*m + j.
    #     So u=[1,0] corresponds to U[0]=1, U[1]=0, U[2]=0, U[3]=0?
    #     No, u is the factor for the A-index.
    #     The term is \sum_{i,j} u_i v_j w_k A_{i,j} B_{j,k} ?
    #     No, the tensor decomposition is T = \sum u^{(r)} \otimes v^{(r)} \otimes w^{(r)}.
    #     Then y_k = \sum_{i,j} (\sum_r u^{(r)}_i v^{(r)}_j w^{(r)}_k) A_{i,j} B_{j,k}.
    #     Here B is 2x12. So B_{j, b}.
    #     The index k in w is the output row index (0 or 1).
    #     The index b is the column index.
    #     So we have a separate tensor for each b.
    #     T_b = u \otimes v \otimes w.
    #     Contribution to C_{i, b} is \sum_{j} u_i v_j w_i A_{i,j} B_{j,b}.
    #     Wait, w is a vector of length n=2 (output rows).
    #     So w_i is the coefficient for output row i.
    #     
    #     For T^{(0)} (output row 0):
    #     u = [1, 0] (only row 0 of A matters)
    #     v = [1, 1] (both cols of A matter)
    #     w = [1, 0] (only output row 0 matters)
    #     
    #     For T^{(1)} (output row 1):
    #     u = [0, 1] (only row 1 of A matters)
    #     v = [1, 1] (both cols of A matter)
    #     w = [0, 1] (only output row 1 matters)
    #
    #     So for each b:
    #     Term 1:
    #       U: [1, 0, 0, 0] -> U[0]=1
    #       V: [1, 1, 0, 0, ...] -> V[0]=1, V[1]=1 (B[0,b], B[1,b])
    #       Wait, V is size m*p = 24. Index is j*p + b.
    #       So V[0*p + b] = 1, V[1*p + b] = 1.
    #       W: [1, 0, 0, 0, ...] -> W[0*p + b] = 1. (C[0,b])
    #       
    #     Term 2:
    #       U: [0, 0, 1, 0] -> U[2]=1
    #       V: [0, 0, 1, 1, ...] -> V[0*p + b] = 1, V[1*p + b] = 1.
    #       Wait, V is the same for both terms?
    #       Yes, v=[1,1] for both.
    #       W: [0, 1, 0, 0, ...] -> W[1*p + b] = 1. (C[1,b])
    #
    #     Let's verify the contribution.
    #     Term 1 contributes to C[0,b]:
    #     \sum_j U[0, r] V[j*p+b, r] W[0*p+b, r] A[0,j] B[j,b]
    #     = 1 * (1*A[0,0]B[0,b] + 1*A[0,1]B[1,b]) * 1
    #     = A[0,0]B[0,b] + A[0,1]B[1,b]. Correct.
    #     
    #     Term 2 contributes to C[1,b]:
    #     \sum_j U[2, r] V[j*p+b, r] W[1*p+b, r] A[1,j] B[j,b]
    #     = 1 * (1*A[1,0]B[0,b] + 1*A[1,1]B[1,b]) * 1
    #     = A[1,0]B[0,b] + A[1,1]B[1,b]. Correct.
    #
    #     So we just need 2 terms per column.
    #     Total 24 terms.
    
    rank = 24
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    for b in range(p):
        r1 = 2 * b
        r2 = 2 * b + 1
        
        # Term 1: Row 0 of A, Row 0 of C
        U[0, r1] = 1.0
        V[0 * p + b, r1] = 1.0
        V[1 * p + b, r1] = 1.0
        W[0 * p + b, r1] = 1.0
        
        # Term 2: Row 1 of A, Row 1 of C
        U[2, r2] = 1.0
        V[0 * p + b, r2] = 1.0
        V[1 * p + b, r2] = 1.0
        W[1 * p + b, r2] = 1.0

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
