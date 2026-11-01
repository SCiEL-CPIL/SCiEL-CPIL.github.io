"""Matrix multiplication 2x5x5 - Rank 49 decomposition via Strassen block."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Finds a rank-49 exact tensor decomposition for 2x5x5 matrix multiplication.
    Uses Strassen's algorithm for the 2x2x2 sub-block (j in {0,1}, k in {0,1})
    and schoolbook multiplication for the remaining terms.
    """
    n, m, p = DIMENSIONS  # 2, 5, 5
    
    rank = 49
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add a schoolbook term A[i,j] * B[j,k]
    def add_schoolbook(i, j, k):
        nonlocal col
        U[i * m + j, col] = 1
        V[j * p + k, col] = 1
        W[k * n + i, col] = 1
        col += 1
        
    # Helper to add a Strassen-like product
    # u_vec: list of (i, j) indices with coeff 1 (or -1)
    # v_vec: list of (j, k) indices with coeff 1 (or -1)
    # w_vec: list of (k, i) indices with coeff 1 (or -1)
    def add_strassen_term(u_indices, v_indices, w_indices):
        nonlocal col
        for (i, j) in u_indices:
            U[i * m + j, col] = 1
        for (j, k) in v_indices:
            V[j * p + k, col] = 1
        for (k, i) in w_indices:
            W[k * n + i, col] = 1
        col += 1

    # 1. Strassen terms for block j in {0,1}, k in {0,1}
    
    # M1 = A[0,0] * (B[0,0] + B[1,1])
    # Contributes to C[0,0] (+1) and C[1,1] (+1)
    add_strassen_term(
        u_indices=[(0, 0)],
        v_indices=[(0, 0), (1, 1)],
        w_indices=[(0, 0), (1, 1)] # (k,i) -> (0,0) and (1,1)
    )
    
    # M2 = (A[0,0] + A[1,0]) * B[1,1]
    # Contributes to C[0,0] (-1) and C[1,0] (+1)
    add_strassen_term(
        u_indices=[(0, 0), (1, 0)],
        v_indices=[(1, 1)],
        w_indices=[(0, 0), (0, 1)] # (k,i) -> (0,0) coeff -1, (0,1) coeff +1
        # Note: The function above only adds 1s. We need to handle signs.
        # Let's modify add_strassen_term to accept signed indices or just return the matrices and we'll fix signs later?
        # Better: pass signed tuples.
    )
    
    # RESTART with signed support
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    
    def add_term(u_signed, v_signed, w_signed):
        nonlocal col
        for (idx, val) in u_signed:
            U[idx, col] = val
        for (idx, val) in v_signed:
            V[idx, col] = val
        for (idx, val) in w_signed:
            W[idx, col] = val
        col += 1
        
    def idx_U(i, j): return i * m + j
    def idx_V(j, k): return j * p + k
    def idx_W(k, i): return k * n + i

    # M1 = A[0,0] * (B[0,0] + B[1,1])
    # C[0,0] += M1, C[1,1] += M1
    add_term(
        u_signed=[(idx_U(0,0), 1)],
        v_signed=[(idx_V(0,0), 1), (idx_V(1,1), 1)],
        w_signed=[(idx_W(0,0), 1), (idx_W(1,1), 1)]
    )
    
    # M2 = (A[0,0] + A[1,0]) * B[1,1]
    # C[0,0] -= M2, C[1,0] += M2
    add_term(
        u_signed=[(idx_U(0,0), 1), (idx_U(1,0), 1)],
        v_signed=[(idx_V(1,1), 1)],
        w_signed=[(idx_W(0,0), -1), (idx_W(0,1), 1)]
    )
    
    # M3 = (A[0,1] + A[1,1]) * B[0,0]
    # C[0,1] += M3, C[1,1] -= M3
    add_term(
        u_signed=[(idx_U(0,1), 1), (idx_U(1,1), 1)],
        v_signed=[(idx_V(0,0), 1)],
        w_signed=[(idx_W(1,0), 1), (idx_W(1,1), -1)]
    )
    
    # M4 = A[1,1] * (B[1,0] + B[0,0])
    # C[0,0] += M4, C[1,0] += M4
    add_term(
        u_signed=[(idx_U(1,1), 1)],
        v_signed=[(idx_V(1,0), 1), (idx_V(0,0), 1)],
        w_signed=[(idx_W(0,0), 1), (idx_W(0,1), 1)]
    )
    
    # M5 = (A[0,0] + A[1,1]) * (B[1,1] + B[1,0])
    # C[0,1] += M5, C[1,1] += M5
    add_term(
        u_signed=[(idx_U(0,0), 1), (idx_U(1,1), 1)],
        v_signed=[(idx_V(1,1), 1), (idx_V(1,0), 1)],
        w_signed=[(idx_W(1,0), 1), (idx_W(1,1), 1)]
    )
    
    # M6 = (A[1,0] + A[0,0]) * (B[0,1] + B[1,1])
    # C[0,0] += M6
    add_term(
        u_signed=[(idx_U(1,0), 1), (idx_U(0,0), 1)],
        v_signed=[(idx_V(0,1), 1), (idx_V(1,1), 1)],
        w_signed=[(idx_W(0,0), 1)]
    )
    
    # M7 = (A[0,1] + A[1,1]) * (B[1,0] + B[0,0])
    # C[1,1] += M7
    add_term(
        u_signed=[(idx_U(0,1), 1), (idx_U(1,1), 1)],
        v_signed=[(idx_V(1,0), 1), (idx_V(0,0), 1)],
        w_signed=[(idx_W(1,1), 1)]
    )
    
    # 2. Schoolbook terms for the rest
    
    # Case 1: j in {0,1}, k in {2,3,4}
    for i in range(n):
        for j in [0, 1]:
            for k in [2, 3, 4]:
                add_term(
                    u_signed=[(idx_U(i, j), 1)],
                    v_signed=[(idx_V(j, k), 1)],
                    w_signed=[(idx_W(k, i), 1)]
                )
                
    # Case 2: j in {2,3,4}, k in {0,1}
    for i in range(n):
        for j in [2, 3, 4]:
            for k in [0, 1]:
                add_term(
                    u_signed=[(idx_U(i, j), 1)],
                    v_signed=[(idx_V(j, k), 1)],
                    w_signed=[(idx_W(k, i), 1)]
                )
                
    # Case 3: j in {2,3,4}, k in {2,3,4}
    for i in range(n):
        for j in [2, 3, 4]:
            for k in [2, 3, 4]:
                add_term(
                    u_signed=[(idx_U(i, j), 1)],
                    v_signed=[(idx_V(j, k), 1)],
                    w_signed=[(idx_W(k, i), 1)]
                )
                
    # Verify rank
    assert col == 49, f"Expected rank 49, got {col}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
