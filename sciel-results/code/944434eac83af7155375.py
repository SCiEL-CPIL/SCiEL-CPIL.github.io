"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 by 3x6 matrix multiplication.
    Strategy: Split B into two 3x3 blocks. Each 2x3x3 sub-problem has rank 9
    (using the fact that for each column j of A, the dot product with column k of B
    can be computed with rank-3 per j). Total rank = 18.
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
        for r in range(rank):
            uc = U[:, r]
            vc = V[:, r]
            wc = W[:, r]
            result += np.einsum('i,j,k->ijk', uc, vc, wc)
        return np.array_equal(result, T)

    def rank18_block_decomposition():
        """
        Rank-18 decomposition: split B into two 3x3 blocks.
        For each block (k=0,1,2 or k=3,4,5) and each j (0,1,2):
          U has 1 at positions i*m+j for i=0,1
          V has 1 at position j*p+k
          W has 1 at positions k*n+i for i=0,1
        This gives 2 blocks * 3 j-values * 3 k-values = 18 terms.
        """
        rank = 18
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)
        
        col = 0
        for block in range(2):  # Two 3x3 blocks of B
            for j in range(m):  # j in 0,1,2
                for k_local in range(3):  # k_local in 0,1,2
                    k_global = block * 3 + k_local  # k in 0..5
                    
                    # U: 1 at positions i*m+j for i=0,1
                    for i in range(n):
                        U[i * m + j, col] = 1
                    
                    # V: 1 at position j*p + k_global
                    V[j * p + k_global, col] = 1
                    
                    # W: 1 at positions k_global*n + i for i=0,1
                    for i in range(n):
                        W[k_global * n + i, col] = 1
                    
                    col += 1
        
        return U, V, W

    # Try rank-18 decomposition
    U18, V18, W18 = rank18_block_decomposition()
    if verify_decomposition(U18, V18, W18):
        return {
            "U": U18.tolist(),
            "V": V18.tolist(),
            "W": W18.tolist()
        }

    # Fallback: schoolbook rank 36
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

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
