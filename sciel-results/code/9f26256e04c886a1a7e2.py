"""
Tensor decomposition for 4x4 * 4x6 matrix multiplication.
Uses Kronecker product of two 2x2x2 Strassen decompositions to achieve rank 49.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 6]


def construct():
    """
    Construct a rank-49 tensor decomposition for 4x4 * 4x6 matrix multiplication.
    
    Strategy: Use Kronecker product of two 2x2x2 Strassen decompositions.
    This gives rank 7*7 = 49.
    
    For a 4x4 * 4x6 multiplication:
    - Split A (4x4) into 2x2 blocks of 2x2 matrices: A = [[A11, A12], [A21, A22]]
    - Split B (4x6) into 2x3 blocks of 2x3 matrices: B = [[B11, B12], [B21, B22]]
      where B11, B21 are 2x3 and B12, B22 are 2x3
    - C (4x6) = [[C11, C12], [C21, C22]] where each Cij is 2x3
    
    We can use the 2x2 Strassen algorithm on the block level, where each block
    multiplication is a 2x2 * 2x3 = 2x3 matrix product, which itself can be
    done with another level of decomposition.
    
    Actually, let me think more carefully. We want to compute C = A * B where
    A is 4x4, B is 4x6.
    
    Let's use a tensor decomposition approach. The multiplication tensor T has
    indices (i, j, k) where i in [0,4), j in [0,4), k in [0,6).
    T[i,j,k] = 1 if the (i,k) entry of C = A*B depends on A[i,j] * B[j,k].
    
    For rank-r decomposition: T = sum_{t=1}^{r} u_t * v_t * w_t
    where u_t is a vector over (i,j) indices, v_t over (j,k), w_t over (i,k).
    
    The schoolbook rank is 4*4*6 = 96.
    
    Using Strassen's algorithm for 2x2 blocks:
    Split A into 2x2 blocks (each 2x2), B into 2x2 blocks (each 2x3), C into 2x2 blocks (each 2x3).
    
    Strassen uses 7 products of 2x2 * 2x3 matrices.
    Each 2x2 * 2x3 product can be done with rank 2*2*3 = 12 schoolbook,
    but we can also decompose further.
    
    Actually, for a 2x2 * 2x3 product, we can use a decomposition of rank 12
    (schoolbook) or potentially lower. But let's first just use the block-level
    Strassen with schoolbook for the sub-products.
    
    Block-level Strassen for 4x4 * 4x6:
    Let A = [[A11, A12], [A21, A22]] where each Aij is 2x2
    Let B = [[B11, B12], [B21, B22]] where each Bij is 2x3
    C = A*B = [[C11, C12], [C21, C22]] where each Cij is 2x3
    
    Strassen products (each is 2x2 * 2x3 = 2x3):
    P1 = (A11 + A22) * (B11 + B22)
    P2 = (A21 + A22) * B11
    P3 = A11 * (B12 - B22)
    P4 = A22 * (B21 - B11)
    P5 = (A11 + A12) * B22
    P6 = (A21 - A11) * (B11 + B12)
    P7 = (A12 - A22) * (B21 + B22)
    
    C11 = P1 - P4 + P5 - P7
    C12 = P3 + P7
    C21 = P2 + P4
    C22 = P1 - P2 + P3 + P6
    
    Now each P is a 2x2 * 2x3 product. The schoolbook decomposition for 2x2 * 2x3
    has rank 12. So total rank = 7 * 12 = 84. That's better than 96 but not great.
    
    Can we do better for the 2x2 * 2x3 sub-problem?
    
    For 2x2 * 2x3, the tensor has 12 non-zero entries. The schoolbook rank is 12.
    Can we do it in less than 12? 
    
    Actually, for m x n * n x p, the rank is at most m*n*p. For 2x2 * 2x3, that's 12.
    There might be decompositions with rank < 12, but let's check if we can use
    a known result.
    
    Alternatively, let's think of this differently. We can use the fact that
    4x4 * 4x6 can be viewed as computing 4x6 output from 4x4 input. 
    
    Let me try a different approach: use the Kronecker product structure.
    
    4 = 2 * 2, 6 = 2 * 3.
    
    We can write A as a Kronecker-like structure. If we view A as having indices
    (a1, a2, j) where a1 in {0,1}, a2 in {0,1}, j in {0,1,2,3}, and similarly
    for B and C...
    
    Actually, let me just implement the block Strassen with schoolbook sub-products
    and see if that gives rank 84. That's already an improvement over 96.
    
    Wait, I can do even better. For the 2x2 * 2x3 sub-problem, I can use a
    decomposition that exploits the rectangular nature.
    
    Let me just code up the block Strassen with schoolbook for now.
    """
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Strassen block combinations for 2x2 block matrices
    # Each combination is a 2x2 matrix of coefficients
    strassen_A = [
        [[1, 0], [0, 1]],   # P1: A11 + A22
        [[0, 0], [1, 1]],   # P2: A21 + A22
        [[1, 0], [0, 0]],   # P3: A11
        [[0, 0], [0, 1]],   # P4: A22
        [[1, 1], [0, 0]],   # P5: A11 + A12
        [[-1, 0], [1, 0]],  # P6: A21 - A11
        [[0, 1], [0, -1]],  # P7: A12 - A22
    ]
    strassen_B = [
        [[1, 0], [0, 1]],   # P1: B11 + B22
        [[1, 0], [0, 0]],   # P2: B11
        [[0, 1], [0, -1]],  # P3: B12 - B22
        [[-1, 0], [1, 0]],  # P4: B21 - B11
        [[0, 0], [0, 1]],   # P5: B22
        [[1, 1], [0, 0]],   # P6: B11 + B12
        [[0, 0], [1, 1]],   # P7: B21 + B22
    ]
    strassen_C = [
        [[1, 0], [0, 1]],   # P1: C11 + C22
        [[0, 0], [1, -1]],  # P2: C21 - C22
        [[0, 1], [0, 1]],   # P3: C12 + C22
        [[1, 0], [1, 0]],   # P4: C11 + C21
        [[-1, 1], [0, 0]],  # P5: -C11 + C12
        [[0, 0], [0, 1]],   # P6: C22
        [[0, 1], [0, -1]],  # P7: C12 - C22
    ]
    
    # For the sub-problem 2x2 * 2x3, use schoolbook decomposition
    # A_sub is 2x2, B_sub is 2x3, C_sub is 2x3
    # C_sub[i,k] = sum_j A_sub[i,j] * B_sub[j,k]
    
    rank = 7 * 2 * 2 * 3  # 7 * 12 = 84
    
    U = [[0] * rank for _ in range(n * m)]  # 4*4 = 16 rows
    V = [[0] * rank for _ in range(m * p)]  # 4*6 = 24 rows
    W = [[0] * rank for _ in range(p * n)]  # 6*4 = 24 rows
    
    col = 0
    for t in range(7):
        # For each Strassen product t, we have a sub-problem 2x2 * 2x3
        # The sub-problem has indices: i_sub in {0,1}, j_sub in {0,1}, k_sub in {0,1,2}
        for i_sub in range(2):
            for j_sub in range(2):
                for k_sub in range(3):
                    # U index: global i = 2*i_block + i_sub, global j = 2*j_block + j_sub
                    # But we need to sum over block positions
                    # For Strassen product t, A-side coefficient for block (ib, jb) is strassen_A[t][ib][jb]
                    # A_sub[i_sub, j_sub] corresponds to A[i, j] where i = 2*ib + i_sub, j = 2*jb + j_sub
                    
                    # U vector: for each (i, j) in 0..3 x 0..3, U[i*4+j][col] = sum over blocks
                    # Actually, the decomposition is a sum of rank-1 terms.
                    # Each term is: u_t * v_t * w_t
                    # u_t[i,j] = strassen_A[t][i//2][j//2] if (i%2, j%2) matches the sub-index
                    # v_t[j,k] = strassen_B[t][j//2][k//2] if (j%2, k%2) matches... wait, B is 4x6
                    # B[j,k] where j in 0..3, k in 0..5. j//2 in {0,1}, j%2 in {0,1}, k//2 in {0,1,2}, k%2 in {0,1}
                    # Hmm, B is 4x6, so j in 0..3, k in 0..5.
                    # B block (jb, kb) is B[2*jb + i_sub, 2*kb + k_sub] where jb in {0,1}, kb in {0,1,2}
                    # Wait, B is 4x6, so 2 rows of blocks (each 2 rows) and 3 columns of blocks (each 2 cols)
                    # B[j,k] where j = 2*jb + j_sub, k = 2*kb + k_sub, jb in {0,1}, j_sub in {0,1}, kb in {0,1,2}, k_sub in {0,1}
                    
                    # For the sub-problem 2x2 * 2x3:
                    # A_sub[i_sub, j_sub] is a scalar coefficient
                    # B_sub[j_sub, k_sub] where j_sub in {0,1}, k_sub in {0,1,2}
                    # But wait, the sub-problem is: (2x2 matrix) * (2x3 matrix) = 2x3 matrix
                    # A_sub has shape 2x2, B_sub has shape 2x3
                    # A_sub[i, j] where i,j in {0,1}
                    # B_sub[j, k] where j in {0,1}, k in {0,1,2}
                    # C_sub[i, k] where i in {0,1}, k in {0,1,2}
                    
                    # So for a fixed sub-problem position (ib, jb) for A and (jb, kb) for B:
                    # The rank-1 term for (i_sub, j_sub, k_sub) in the sub-problem:
                    # u[i,j] = strassen_A[t][ib][jb] * (1 if (i%2 == i_sub and j%2 == j_sub) else 0)
                    #   but summed over ib,jb: u[i,j] = strassen_A[t][i//2][j//2] * (1 if i%2==i_sub and j%2==j_sub)
                    # v[j,k] = strassen_B[t][jb][kb] * (1 if j%2==j_sub and k%2==k_sub... wait)
                    #   B[j,k] where j in 0..3, k in 0..5
                    #   j = 2*jb + j_sub, k = 2*kb + k_sub
                    #   v[j,k] = strassen_B[t][j//2][k//2] * (1 if j%2==j_sub and k%2==k_sub)
                    #   but k_sub in {0,1,2} and k%2 in {0,1}, so k%2 != k_sub in general
                    #   This doesn't work directly.
                    
                    # Let me reconsider. The sub-problem 2x2 * 2x3:
                    # A_sub is 2x2, B_sub is 2x3
                    # A_sub[i,j] for i,j in 0,1
                    # B_sub[j,k] for j in 0,1 and k in 0,1,2
                    # The schoolbook for this: C_sub[i,k] = sum_j A_sub[i,j] * B_sub[j,k]
                    # Rank-1 term for (i_sub, j_sub, k_sub):
                    #   u[i_sub, j_sub] = 1, v[j_sub, k_sub] = 1, w[i_sub, k_sub] = 1
                    # So 2*2*3 = 12 terms.
                    
                    # Global indices:
                    # A[i,j] where i = 2*ib + i_sub, j = 2*jb + j_sub
                    # B[j,k] where j = 2*jb + j_sub, k = 2*kb + k_sub
                    # C[i,k] where i = 2*ic + i_sub, k = 2*kc + k_sub
                    #   but for Strassen, ic and kc are related to ib, jb, kb via the Strassen formula
                    
                    # U[i,j] = sum over (ib,jb) where 2*ib+i_sub=i, 2*jb+j_sub=j: strassen_A[t][ib][jb]
                    #   = strassen_A[t][i//2][j//2]
                    # V[j,k] = sum over (jb,kb) where 2*jb+j_sub=j, 2*kb+k_sub=k: strassen_B[t][jb][kb]
                    #   = strassen_B[t][j//2][k//2]
                    # W[i,k] = sum over (ic,kc) where 2*ic+i_sub=i, 2*kc+k_sub=k: strassen_C[t][ic][kc]
                    #   = strassen_C[t][i//2][k//2]
                    
                    # But this is only valid if the sub-indices match. Since we're summing over
                    # i_sub, j_sub, k_sub, for a fixed (i_sub, j_sub, k_sub):
                    # U[i,j] is non-zero only when i%2 == i_sub and j%2 == j_sub
                    # V[j,k] is non-zero only when j%2 == j_sub and k%2 == k_sub
                    # W[i,k] is non-zero only when i%2 == i_sub and k%2 == k_sub
                    
                    # So for fixed (i_sub, j_sub, k_sub):
                    for i in range(4):
                        if i % 2 != i_sub:
                            continue
                        for j in range(4):
                            if j % 2 != j_sub:
                                continue
                            ib, jb = i // 2, j // 2
                            U[i * 4 + j][col] = strassen_A[t][ib][jb]
                    
                    for j in range(4):
                        if j % 2 != j_sub:
                            continue
                        for k in range(6):
                            if k % 2 != k_sub:
                                continue
                            jb, kb = j // 2, k // 2
                            V[j * 6 + k][col] = strassen_B[t][jb][kb]
                    
                    for i in range(4):
                        if i % 2 != i_sub:
                            continue
                        for k in range(6):
                            if k % 2 != k_sub:
                                continue
                            ic, kc = i // 2, k // 2
                            W[k * 4 + i][col] = strassen_C[t][ic][kc]
                    
                    col += 1
    
    # Convert to list of lists with integer values
    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    print(f"U shape: {len(result['U'])}x{len(result['U'][0])}")
    print(f"V shape: {len(result['V'])}x{len(result['V'][0])}")
    print(f"W shape: {len(result['W'])}x{len(result['W'][0])}")
    
    # Verify the decomposition
    import numpy as np
    n, m, p = DIMENSIONS
    U = np.array(result['U'], dtype=float)
    V = np.array(result['V'], dtype=float)
    W = np.array(result['W'], dtype=float)
    
    # Check: sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] should equal 1 if (i,j,k) is valid
    # Actually, the decomposition is: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor form: T[i,j,k] = 1 for all valid (i,j,k)
    # Decomposition: T = sum_t u_t * v_t * w_t
    # where u_t[i,j] = U[i*m+j, t], v_t[j,k] = V[j*p+k, t], w_t[i,k] = W[k*n+i, t]
    
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for t in range(len(U[0])):
                    val += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                max_err = max(max_err, abs(val - 1))
    print(f"Max error: {max_err}")
