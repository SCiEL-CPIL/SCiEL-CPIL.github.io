"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Define the 2x2 Strassen factors
    # U2[i, r] where i is index in A (0:A00, 1:A01, 2:A10, 3:A11)
    # V2[j, r] where j is index in B (0:B00, 1:B01, 2:B10, 3:B11)
    # W2[k, r] where k is index in C (0:C00, 1:C01, 2:C10, 3:C11)
    # Note: The problem statement uses W[k*n+i, r]. For 2x2, n=2.
    # Index in C is k*2 + i.
    # My W2 above used 0:C00 (k=0,i=0), 1:C01 (k=0,i=1), 2:C10 (k=1,i=0), 3:C11 (k=1,i=1).
    # This matches the standard row-major indexing for C if C is [C00 C01; C10 C11].
    # However, the task says "output factor follows the official trace convention W[k*n+i,r]".
    # For matrix multiplication C = A * B, C[i,k] = sum_j A[i,j] B[j,k].
    # The task defines W[k*n+i, r]. This implies the index in W is k*n + i.
    # So for n=2, index 0 is k=0,i=0 (C00), index 1 is k=0,i=1 (C01), index 2 is k=1,i=0 (C10), index 3 is k=1,i=1 (C11).
    # This matches my W2 indexing.
    
    U2 = [
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ]
    
    V2 = [
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 0, 1, 0, 1, 0],  # B01
        [0, 0, 1, 0, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ]
    
    W2 = [
        [1, 0, 0, 1, -1, 0, 1], # C00 (k=0, i=0)
        [0, 0, 1, 0, 1, 0, 0],  # C01 (k=0, i=1)
        [0, 1, 0, 1, 0, 0, 0],  # C10 (k=1, i=0)
        [1, -1, 1, 0, 0, 1, 0]  # C11 (k=1, i=1)
    ]
    
    rank2 = 7
    rank4 = rank2 * rank2 # 49
    
    U = [[0] * rank4 for _ in range(n * m)]
    V = [[0] * rank4 for _ in range(m * p)]
    W = [[0] * rank4 for _ in range(p * n)]
    
    # Recursive construction
    # Index mapping:
    # i = 2*i1 + i0
    # j = 2*j1 + j0
    # k = 2*k1 + k0
    # r = 7*r1 + r0
    
    for i in range(n):
        for j in range(m):
            i1 = i // 2
            i0 = i % 2
            j1 = j // 2
            j0 = j % 2
            for r1 in range(rank2):
                u1 = U2[i1][r1]
                if u1 == 0:
                    continue
                for r0 in range(rank2):
                    u0 = U2[i0][r0]
                    if u0 == 0:
                        continue
                    r = r1 * rank2 + r0
                    U[i * m + j][r] = u1 * u0
                    
    for j in range(m):
        for k in range(p):
            j1 = j // 2
            j0 = j % 2
            k1 = k // 2
            k0 = k % 2
            for r1 in range(rank2):
                v1 = V2[j1][r1]
                if v1 == 0:
                    continue
                for r0 in range(rank2):
                    v0 = V2[j0][r0]
                    if v0 == 0:
                        continue
                    r = r1 * rank2 + r0
                    V[j * p + k][r] = v1 * v0

    for k in range(p):
        for i in range(n):
            k1 = k // 2
            k0 = k % 2
            i1 = i // 2
            i0 = i % 2
            for r1 in range(rank2):
                w1 = W2[k1 * 2 + i1][r1] # W2 index is k*n + i, for n=2 it is k1*2 + i1
                if w1 == 0:
                    continue
                for r0 in range(rank2):
                    w0 = W2[k0 * 2 + i0][r0]
                    if w0 == 0:
                        continue
                    r = r1 * rank2 + r0
                    W[k * n + i][r] = w1 * w0

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
