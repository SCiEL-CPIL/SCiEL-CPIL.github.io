"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 by 3x6 matrix multiplication.
    Uses the even/odd split of the output dimension p=6 into two independent 2x3x3 sub-problems.
    Each 2x3x3 sub-problem is decomposed using Strassen (2x2x2, rank 7) + standard (2x1x2, rank 4) = rank 11.
    Total rank = 2 * 11 = 22.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety

    n, m, p = DIMENSIONS
    dim_a = n * m  # 6
    dim_b = m * p  # 18
    dim_c = p * n  # 12

    # Build the target tensor T[a, b, c]
    T = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def verify_decomposition(U, V, W):
        """Verify that U, V, W exactly decompose the tensor T."""
        rank = U.shape[1]
        result = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
        for c in range(rank):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            result += np.einsum('i,j,k->ijk', uc, vc, wc)
        return np.array_equal(result, T)

    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return U, V, W

    def build_rank11_for_2x3x3():
        """
        Build rank-11 decomposition for 2x3x3 matrix multiplication.
        A is 2x3, B is 3x3, C is 3x2.
        
        Split j into 0,1 (Strassen 2x2x2 part) and j=2 (standard 2x1x2 part).
        Strassen: rank 7
        Standard: rank 4
        Total: rank 11
        """
        rank = 11
        U_sub = np.zeros((6, rank), dtype=np.int64)  # 2*3 = 6
        V_sub = np.zeros((9, rank), dtype=np.int64)  # 3*3 = 9
        W_sub = np.zeros((6, rank), dtype=np.int64)  # 3*2 = 6
        
        # Strassen 2x2x2 factors
        S_U = [
            [[1, 0], [0, 1]],
            [[0, 0], [1, 1]],
            [[1, 0], [0, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[-1, 0], [1, 0]],
            [[0, 1], [0, -1]]
        ]
        S_V = [
            [[1, 0], [0, 1]],
            [[1, 0], [0, 0]],
            [[0, 1], [0, -1]],
            [[-1, 0], [1, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[0, 0], [1, 1]]
        ]
        S_W = [
            [[1, 0], [0, 1]],
            [[0, 1], [0, -1]],
            [[0, 0], [1, 1]],
            [[1, 1], [0, 0]],
            [[-1, 0], [1, 0]],
            [[0, 0], [0, 1]],
            [[1, 0], [0, 0]]
        ]
        
        # Standard 2x1x2 factors
        Std_U = [
            [[1], [0]],
            [[1], [0]],
            [[0], [1]],
            [[0], [1]]
        ]
        Std_V = [
            [[1, 0]],
            [[0, 1]],
            [[1, 0]],
            [[0, 1]]
        ]
        Std_W = [
            [[1, 0], [0, 0]],
            [[0, 0], [1, 0]],
            [[0, 1], [0, 0]],
            [[0, 0], [0, 1]]
        ]
        
        col = 0
        # Strassen part: handles j=0,1
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U_sub[i * 3 + j, col] = S_U[r][i][j]
            for j in range(2):
                for k in range(3):
                    V_sub[j * 3 + k, col] = S_V[r][j][k]
            for k in range(3):
                for i in range(2):
                    W_sub[k * 2 + i, col] = S_W[r][k][i]
            col += 1
        
        # Standard part: handles j=2
        for r in range(4):
            for i in range(2):
                U_sub[i * 3 + 2, col] = Std_U[r][i][0]
            for k in range(3):
                V_sub[2 * 3 + k, col] = Std_V[r][0][k]
            for k in range(3):
                for i in range(2):
                    W_sub[k * 2 + i, col] = Std_W[r][k][i]
            col += 1
        
        return U_sub, V_sub, W_sub

    def build_rank22_decomposition():
        """
        Build rank-22 decomposition for 2x3x6 matrix multiplication.
        Split k into even (0,2,4) and odd (1,3,5).
        Even part: 2x3x3 sub-problem, rank 11
        Odd part: 2x3x3 sub-problem, rank 11
        Total: rank 22
        """
        rank = 22
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)
        
        U_sub, V_sub, W_sub = build_rank11_for_2x3x3()
        rank_sub = 11
        
        # Even part: k = 0, 2, 4 (s = 0, 1, 2)
        for r in range(rank_sub):
            for a in range(dim_a):
                U[a, r] = U_sub[a, r]
            for j in range(3):
                for s in range(3):
                    b_sub = j * 3 + s
                    b_full = j * 6 + 2 * s
                    V[b_full, r] = V_sub[b_sub, r]
            for s in range(3):
                for i in range(2):
                    c_sub = s * 2 + i
                    c_full = (2 * s) * 2 + i
                    W[c_full, r] = W_sub[c_sub, r]
        
        # Odd part: k = 1, 3, 5 (s = 0, 1, 2)
        for r in range(rank_sub):
            r_full = rank_sub + r
            for a in range(dim_a):
                U[a, r_full] = U_sub[a, r]
            for j in range(3):
                for s in range(3):
                    b_sub = j * 3 + s
                    b_full = j * 6 + 2 * s + 1
                    V[b_full, r_full] = V_sub[b_sub, r]
            for s in range(3):
                for i in range(2):
                    c_sub = s * 2 + i
                    c_full = (2 * s + 1) * 2 + i
                    W[c_full, r_full] = W_sub[c_sub, r]
        
        return U, V, W

    # Start with the rank-22 decomposition
    best_U, best_V, best_W = build_rank22_decomposition()
    best_rank = 22
    
    # Verify the decomposition
    if not verify_decomposition(best_U, best_V, best_W):
        # Fallback to schoolbook if construction is wrong
        best_U, best_V, best_W = schoolbook()
        best_rank = n * m * p

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
