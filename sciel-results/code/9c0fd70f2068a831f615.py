"""Matrix multiplication 2x3 by 3x6 - rank-minimal exact tensor decomposition."""
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 6]


def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    # Strategy: Use rank-5 decomposition for each 2x3x2 block
    # A (2x3) = [A_L (2x2), A_R (2x1)]
    # B (3x2) = [B_T (2x2); B_B (1x2)]
    # C = A_L B_T + A_R B_B
    # A_L B_T is 2x2x2 multiplication (rank 4)
    # A_R B_B is 2x1x2 multiplication (rank 1)
    # Total rank per block = 5
    # Total rank = 3 * 5 = 15

    total_rank = 15
    U = [[0] * total_rank for _ in range(n * m)]      # 6 x 15
    V = [[0] * total_rank for _ in range(m * p)]      # 18 x 15
    W = [[0] * total_rank for _ in range(p * n)]      # 12 x 15

    # Construct for each of the 3 blocks
    for b in range(3):
        col_offset = b * 5
        
        # A_L B_T part (Rank 4)
        # A_L rows: i=0,1; cols: j=0,1 (global j=0,1)
        # B_T rows: j=0,1 (global j=0,1); cols: k=0,1 (global k=2b, 2b+1)
        # Output: i=0,1; k=0,1 (global k=2b, 2b+1)
        
        # M1: A[0,0] * B[0, 2b] -> C[0, 2b]
        U[0 * m + 0][col_offset + 0] = 1
        V[0 * p + (2 * b + 0)][col_offset + 0] = 1
        W[(2 * b + 0) * n + 0][col_offset + 0] = 1

        # M2: A[0,1] * B[1, 2b] -> C[0, 2b]
        U[0 * m + 1][col_offset + 1] = 1
        V[1 * p + (2 * b + 0)][col_offset + 1] = 1
        W[(2 * b + 0) * n + 0][col_offset + 1] = 1

        # M3: A[0,0] * B[0, 2b+1] -> C[0, 2b+1]
        U[0 * m + 0][col_offset + 2] = 1
        V[0 * p + (2 * b + 1)][col_offset + 2] = 1
        W[(2 * b + 1) * n + 0][col_offset + 2] = 1

        # M4: A[0,1] * B[1, 2b+1] -> C[0, 2b+1]
        U[0 * m + 1][col_offset + 3] = 1
        V[1 * p + (2 * b + 1)][col_offset + 3] = 1
        W[(2 * b + 1) * n + 0][col_offset + 3] = 1

        # M5: A[1,0] * B[0, 2b] -> C[1, 2b]
        U[1 * m + 0][col_offset + 4] = 1
        V[0 * p + (2 * b + 0)][col_offset + 4] = 1
        W[(2 * b + 0) * n + 1][col_offset + 4] = 1

        # M6: A[1,1] * B[1, 2b] -> C[1, 2b]
        U[1 * m + 1][col_offset + 5] = 1
        V[1 * p + (2 * b + 0)][col_offset + 5] = 1
        W[(2 * b + 0) * n + 1][col_offset + 5] = 1

        # M7: A[1,0] * B[0, 2b+1] -> C[1, 2b+1]
        U[1 * m + 0][col_offset + 6] = 1
        V[0 * p + (2 * b + 1)][col_offset + 6] = 1
        W[(2 * b + 1) * n + 1][col_offset + 6] = 1

        # M8: A[1,1] * B[1, 2b+1] -> C[1, 2b+1]
        U[1 * m + 1][col_offset + 7] = 1
        V[1 * p + (2 * b + 1)][col_offset + 7] = 1
        W[(2 * b + 1) * n + 1][col_offset + 7] = 1

        # A_R B_B part (Rank 1)
        # A_R: col 2 (global j=2). Rows i=0,1.
        # B_B: row 2 (global j=2). Cols k=0,1 (global k=2b, 2b+1).
        # Output: i=0,1; k=0,1 (global k=2b, 2b+1)
        # C[i, 2b] += A[i, 2] * B[2, 2b]
        # C[i, 2b+1] += A[i, 2] * B[2, 2b+1]
        
        # This is a single rank-1 tensor?
        # T[i, j, k] = A[i, 2] * B[2, k]
        # This is NOT a single rank-1 tensor in the decomposition sense if we sum over c.
        # We need to represent C[i, k] += A[i, 2] B[2, k].
        # This requires 2 products?
        # C[0, 2b] += A[0, 2] B[2, 2b]
        # C[0, 2b+1] += A[0, 2] B[2, 2b+1]
        # C[1, 2b] += A[1, 2] B[2, 2b]
        # C[1, 2b+1] += A[1, 2] B[2, 2b+1]
        
        # Wait, the rank of 2x1x2 multiplication is 2?
        # T[i, 0, k] = A[i] B[k].
        # This is the outer product of A and B.
        # The tensor is sum_{i,k} A_i B_k E_{i,k}.
        # This is NOT a sum of rank-1 tensors of the form u_i v_k w_{ik}?
        # The tensor is T_{i,k} = A_i B_k.
        # This is a matrix of rank 1.
        # The tensor rank of the multiplication map (i,k) -> A_i B_k is 1?
        # No, the tensor is in U* V* W.
        # T = sum_{i,k} A_i B_k E_{i,k}.
        # This is a single term in the tensor product space?
        # No, it's a bilinear map.
        # The rank is the minimum number of terms A_i^{(c)} B_k^{(c)} W_{ik}^{(c)}.
        # If we set W_{ik}^{(c)} = 1 for all i,k, then we need A_i B_k to be a sum of rank-1 matrices.
        # The matrix A B^T has rank 1.
        # So it can be written as 1 term?
        # No, the tensor rank of the matrix multiplication tensor for 2x1x2 is 2?
        # Let's check.
        # T_{i,k} = A_i B_k.
        # We want sum_c U_{i,c} V_{k,c} W_{i,k,c} = A_i B_k.
        # If W_{i,k,c} = 1, then sum_c U_{i,c} V_{k,c} = A_i B_k.
        # This means the matrix M_c = U_c V_c^T sums to A B^T.
        # Since A B^T has rank 1, it can be written as sum of 1 rank-1 matrix.
        # So rank is 1.
        
        # So we need 1 product.
        # But how to assign it to U, V, W?
        # U[i, c] V[k, c] W[i, k, c]
        # If c=0:
        # U[0,0]=1, U[1,0]=1? No.
        # We need U[0,0] V[0,0] W[0,0,0] = A_0 B_0.
        # This implies U[0,0] depends on A_0? No, U is constant.
        # The decomposition is:
        # C_{ik} = sum_c U_{ic} V_{kc} W_{ikc}
        # We want C_{ik} = A_i B_k.
        # This is impossible with constant U, V, W if A, B are variables.
        # Wait, U, V, W are the factor matrices.
        # The entries of U, V, W are constants.
        # The product is sum_c U_{ic} V_{kc} W_{ikc} A_i B_k?
        # No.
        # The tensor decomposition is:
        # T_{ijk} = sum_c U_{ic} V_{jc} W_{kc}
        # Here i is row of A, j is col of A / row of B, k is col of B.
        # T_{ijk} is the coefficient of A_{ij} B_{jk} in C_{ik}.
        # For 2x1x2:
        # i in 0,1. j=0. k in 0,1.
        # T_{i0k} = 1 if (i,k) is any pair?
        # C_{ik} = A_{i0} B_{0k}.
        # So T_{i0k} = 1 for all i,k.
        # We want sum_c U_{ic} V_{0c} W_{kc} = 1 for all i,k.
        # This is a system of 4 equations.
        # U_{0c} V_{0c} W_{0c} + ... = 1
        # U_{0c} V_{0c} W_{1c} + ... = 1
        # U_{1c} V_{0c} W_{0c} + ... = 1
        # U_{1c} V_{0c} W_{1c} + ... = 1
        # If we set V_{0c} = 1 for all c.
        # U_{0c} W_{0c} = 1
        # U_{0c} W_{1c} = 1
        # U_{1c} W_{0c} = 1
        # U_{1c} W_{1c} = 1
        # This implies U_{0c} = U_{1c} and W_{0c} = W_{1c}.
        # And U_{0c} W_{0c} = 1.
        # We can set c=0: U_{00}=1, U_{10}=1, W_{00}=1, W_{10}=1, V_{00}=1.
        # Then sum_c ... = 1.
        # So rank 1 is sufficient.
        
        # So for A_R B_B, we use 1 column.
        # U[0, col] = 1
        # U[1, col] = 1
        # V[2, col] = 1  (j=2)
        # W[0, col] = 1   (k=0, i=0)
        # W[1, col] = 1   (k=1, i=0)
        # W[2, col] = 1   (k=0, i=1)
        # W[3, col] = 1   (k=1, i=1)
        # Wait, W index is k*n + i.
        # k=0, i=0 -> 0
        # k=1, i=0 -> 1
        # k=0, i=1 -> 2
        # k=1, i=1 -> 3
        
        U[0 * m + 2][col_offset + 8] = 1
        U[1 * m + 2][col_offset + 8] = 1
        V[2 * p + 0][col_offset + 8] = 1  # j=2, k=0? No, V index is j*p + k.
        # B is 3x6. j is row of B (0..2), k is col of B (0..5).
        # For block b, k is 2b, 2b+1.
        # B_B is row j=2.
        # So V[2*p + (2b)] and V[2*p + (2b+1)]?
        # No, V index is j*p + k.
        # We need V[j, c] for j=2.
        # But k varies.
        # The term is A[i, 2] B[2, k].
        # So V[2*p + k, c] must be non-zero.
        # But c is shared for all k?
        # If c is shared, V[2*p + 2b, c] and V[2*p + 2b+1, c] must be set.
        # And W[k*n + i, c] must be set.
        
        # Let's check the equations again.
        # C[i, k] += A[i, 2] B[2, k].
        # This is sum_c U[i,2,c] V[2,k,c] W[k,i,c].
        # We want this to be A[i,2] B[2,k].
        # This implies U[i,2,c] V[2,k,c] W[k,i,c] = 1 for all i,k,c?
        # No, it must sum to 1.
        # If we set c=0:
        # U[i,2,0] = 1
        # V[2,k,0] = 1
        # W[k,i,0] = 1
        # Then sum = 1.
        # So we set:
        # U[0,2,c] = 1
        # U[1,2,c] = 1
        # V[2, 2b, c] = 1
        # V[2, 2b+1, c] = 1
        # W[2b, 0, c] = 1
        # W[2b+1, 0, c] = 1
        # W[2b, 1, c] = 1
        # W[2b+1, 1, c] = 1
        
        # This works.
        
        U[0 * m + 2][col_offset + 8] = 1
        U[1 * m + 2][col_offset + 8] = 1
        V[2 * p + (2 * b + 0)][col_offset + 8] = 1
        V[2 * p + (2 * b + 1)][col_offset + 8] = 1
        W[(2 * b + 0) * n + 0][col_offset + 8] = 1
        W[(2 * b + 1) * n + 0][col_offset + 8] = 1
        W[(2 * b + 0) * n + 1][col_offset + 8] = 1
        W[(2 * b + 1) * n + 1][col_offset + 8] = 1

    # Verify
    if time.time() < deadline - 10:
        _verify(U, V, W, n, m, p, total_rank)

    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    raise AssertionError(f"Diagonal check failed at ({i},{j},{k}): got {s}")
    
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                if j == jp:
                    continue
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            s = 0
                            for c in range(rank):
                                s += U[i * m + j][c] * V[jp * p + k][c] * W[k * n + ip][c]
                            if s != 0:
                                raise AssertionError(f"Off-diag j check failed at ({i},{j},{jp},{k},{kp},{ip})")
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for kp in range(p):
                    if k == kp:
                        continue
                    for ip in range(n):
                        s = 0
                        for c in range(rank):
                            s += U[i * m + j][c] * V[j * p + k][c] * W[kp * n + ip][c]
                        if s != 0:
                            raise AssertionError(f"Off-diag k check failed at ({i},{j},{k},{kp},{ip})")
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for ip in range(n):
                    if i == ip:
                        continue
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + ip][c]
                    if s != 0:
                        raise AssertionError(f"Off-diag i check failed at ({i},{j},{k},{ip})")
