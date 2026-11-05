"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will build a block-diagonal decomposition using:
    # - 3x3x3 block: Bini-Capovani-De Falco-Ronco (rank 16)
    # - 2x2x2 block: Strassen (rank 7)
    # The remaining cross terms are handled by schoolbook (rank 3*2*3 + 2*3*2 + ... etc.)
    # Actually, let's use the known 5x5x5 decomposition if available, or construct one.
    # A known rank for 5x5x5 is 38 (from Bini et al. or similar).
    # Let's try to construct a valid decomposition.
    # For now, let's use the schoolbook but optimized, or a known better one.
    
    # Known: rank(5,5,5) <= 38 (Bini et al. 1978, but that's for 5x5x5? No, 5x5x5 is 125 schoolbook).
    # Actually, the BCDR decomposition is for 3x3x3 with rank 16.
    # Strassen is 2x2x2 with rank 7.
    
    # Let's partition the 5x5 matrix into blocks:
    # A = [A11 A12; A21 A22] where A11 is 3x3, A12 is 3x2, A21 is 2x3, A22 is 2x2.
    # B = [B11 B12; B21 B22] where B11 is 3x3, B12 is 3x2, B21 is 2x3, B22 is 2x2.
    # C = A * B = [C11 C12; C21 C22]
    # C11 = A11*B11 + A12*B21 (3x3)
    # C12 = A11*B12 + A12*B22 (3x2)
    # C21 = A21*B11 + A22*B21 (2x3)
    # C22 = A21*B12 + A22*B22 (2x2)
    
    # This is complex. Let's use a simpler approach: use the known rank 38 decomposition if we can find it,
    # or just use schoolbook with some optimizations.
    
    # For now, let's implement the schoolbook but try to find a better one.
    # Actually, let's just return the schoolbook for now to ensure correctness,
    # and then try to improve it.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
