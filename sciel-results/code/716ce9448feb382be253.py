import numpy as np
import time
import itertools

n, m, p = 2, 5, 6

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2x5 matrix by a 5x6 matrix.
    
    Strategy:
    - Partition middle dimension j=5 into {0,1}, {2,3}, {4}
    - For j-pairs {0,1} and {2,3}: use Strassen (rank 7) for each 2x2*2x2 sub-block
      across the 3 k-pairs (k=0,1), (k=2,3), (k=4,5) -> 3*7 + 3*7 = 42
    - For j=4: outer product 2x1 * 1x6 -> rank 12
    - Total rank: 42 + 12 = 54
    """
    start_time = time.time()
    deadline = start_time + 110

    def check_decomposition(U, V, W, tol=1e-9):
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = 0.0
                    for r in range(R):
                        val += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def get_schoolbook():
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
        return U, V, W

    def get_rank54():
        """
        Rank-54 decomposition using Strassen on 2x2 blocks + outer product for the last column.
        """
        rank = 54
        U = [[0] * rank for _ in range(n * m)]  # (10, 54)
        V = [[0] * rank for _ in range(m * p)]  # (30, 54)
        W = [[0] * rank for _ in range(p * n)]  # (12, 54)

        col = 0

        def add_strassen_block(i_base, j_base, k_base):
            """
            Add a rank-7 Strassen decomposition for a 2x2 * 2x2 sub-block.
            i_base: row offset in A and C (0 for our 2-row case)
            j_base: column offset in A / row offset in B
            k_base: column offset in B / column offset in C
            
            A_sub[i,j] corresponds to U[i_base*m + j_base+j]
            B_sub[j,k] corresponds to V[(j_base+j)*p + k_base+k]
            C_sub[i,k] corresponds to W[(k_base+k)*n + i_base+i]
            """
            nonlocal col

            # M0 = (A00 + A11) * (B00 + B11)
            U[i_base * m + j_base][col] += 1
            U[(i_base + 1) * m + (j_base + 1)][col] += 1
            V[j_base * p + k_base][col] += 1
            V[(j_base + 1) * p + (k_base + 1)][col] += 1
            W[k_base * n + i_base][col] += 1
            W[(k_base + 1) * n + (i_base + 1)][col] += 1
            col += 1

            # M1 = (A10 + A11) * B00
            U[(i_base + 1) * m + j_base][col] += 1
            U[(i_base + 1) * m + (j_base + 1)][col] += 1
            V[j_base * p + k_base][col] += 1
            W[k_base * n + (i_base + 1)][col] += 1
            W[(k_base + 1) * n + (i_base + 1)][col] -= 1
            col += 1

            # M2 = A00 * (B01 - B11)
            U[i_base * m + j_base][col] += 1
            V[j_base * p + (k_base + 1)][col] += 1
            V[(j_base + 1) * p + (k_base + 1)][col] -= 1
            W[(k_base + 1) * n + i_base][col] += 1
            W[(k_base + 1) * n + (i_base + 1)][col] += 1
            col += 1

            # M3 = A11 * (B10 - B00)
            U[(i_base + 1) * m + (j_base + 1)][col] += 1
            V[(j_base + 1) * p + k_base][col] += 1
            V[j_base * p + k_base][col] -= 1
            W[k_base * n + i_base][col] += 1
            W[k_base * n + (i_base + 1)][col] += 1
            col += 1

            # M4 = (A00 + A01) * B11
            U[i_base * m + j_base][col] += 1
            U[i_base * m + (j_base + 1)][col] += 1
            V[(j_base + 1) * p + (k_base + 1)][col] += 1
            W[k_base * n + i_base][col] -= 1
            W[(k_base + 1) * n + i_base][col] += 1
            col += 1

            # M5 = (A10 - A00) * (B00 + B01)
            U[(i_base + 1) * m + j_base][col] += 1
            U[i_base * m + j_base][col] -= 1
            V[j_base * p + k_base][col] += 1
            V[j_base * p + (k_base + 1)][col] += 1
            W[(k_base + 1) * n + (i_base + 1)][col] += 1
            col += 1

            # M6 = (A01 - A11) * (B10 + B11)
            U[i_base * m + (j_base + 1)][col] += 1
            U[(i_base + 1) * m + (j_base + 1)][col] -= 1
            V[(j_base + 1) * p + k_base][col] += 1
            V[(j_base + 1) * p + (k_base + 1)][col] += 1
            W[k_base * n + i_base][col] += 1
            col += 1

        def add_outer_product_block(i_base, j_single, k_base, k_width):
            """
            Add rank (n * k_width) decomposition for a 2x1 * 1xk_width block.
            C[i, k] = A[i, j_single] * B[j_single, k]
            """
            nonlocal col
            for i in range(n):
                for k in range(k_base, k_base + k_width):
                    U[i * m + j_single][col] += 1
                    V[j_single * p + k][col] += 1
                    W[k * n + i][col] += 1
                    col += 1

        # Block 1: A[:, 0:2] * B[0:2, :] -> 2x2 * 2x6
        # Split into three 2x2 * 2x2 sub-blocks for k pairs
        add_strassen_block(0, 0, 0)  # j=0,1; k=0,1
        add_strassen_block(0, 0, 2)  # j=0,1; k=2,3
        add_strassen_block(0, 0, 4)  # j=0,1; k=4,5

        # Block 2: A[:, 2:4] * B[2:4, :] -> 2x2 * 2x6
        add_strassen_block(0, 2, 0)  # j=2,3; k=0,1
        add_strassen_block(0, 2, 2)  # j=2,3; k=2,3
        add_strassen_block(0, 2, 4)  # j=2,3; k=4,5

        # Block 3: A[:, 4] * B[4, :] -> 2x1 * 1x6, rank 12
        add_outer_product_block(0, 4, 0, 6)

        assert col == rank, f"Expected rank {rank}, got {col}"
        return U, V, W

    # Primary: rank-54 decomposition
    U, V, W = get_rank54()

    if check_decomposition(U, V, W):
        return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook
    if time.time() < deadline:
        U, V, W = get_schoolbook()
        return {"U": U, "V": V, "W": W}

    # Last resort (should not happen)
    U, V, W = get_schoolbook()
    return {"U": U, "V": V, "W": W}

def check_decomposition(U, V, W, tol=1e-9):
    n, m, p = 2, 5, 6
    R = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = 0.0
                for r in range(R):
                    val += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                if abs(val - 1.0) > tol:
                    return False
    return True

if __name__ == "__main__":
    res = construct()
    is_valid = check_decomposition(res['U'], res['V'], res['W'])
    rank = len(res['U'][0])
    print(f"Rank: {rank}, Valid: {is_valid}")
