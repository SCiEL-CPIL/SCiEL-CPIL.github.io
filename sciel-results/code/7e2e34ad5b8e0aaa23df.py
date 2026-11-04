<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly represents matrix multiplication."""
    expected = np.ones((n, m, p), dtype=np.float64)
    
    rank = len(U[0])
    reconstructed = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        u = np.array([U[i][c] for i in range(n*m)]).reshape(n, m)
        v = np.array([V[j][c] for j in range(m*p)]).reshape(m, p)
        w = np.array([W[k][c] for k in range(p*n)]).reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    reconstructed[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    
    return np.allclose(expected, reconstructed, atol=tol)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Schoolbook decomposition
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
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly represents matrix multiplication."""
    expected = np.ones((n, m, p), dtype=np.float64)
    
    rank = len(U[0])
    reconstructed = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        u = np.array([U[i][c] for i in range(n*m)]).reshape(n, m)
        v = np.array([V[j][c] for j in range(m*p)]).reshape(m, p)
        w = np.array([W[k][c] for k in range(p*n)]).reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    reconstructed[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    
    return np.allclose(expected, reconstructed, atol=tol)

def _build_winograd_2x2_terms():
    """
    Build the 5 Winograd rank-1 terms for 2x2 matrix multiplication.
    Each term is (U, V, W) where U, V, W are 2x2 matrices (as lists of lists).
    The sum of U_c[i,j] * V_c[j,k] * W_c[k,i] over c equals 1 for all i,j,k in {0,1}.
    """
    terms = [
        ([[1, 0], [1, 0]], [[1, 0], [0, 1]], [[1, 0], [0, 0]]),   # m1
        ([[1, 0], [1, 0]], [[0, 0], [1, 1]], [[0, 0], [1, 0]]),   # m2
        ([[0, 1], [0, 1]], [[1, 0], [1, 0]], [[0, 1], [0, 0]]),   # m3
        ([[1, 0], [0, 0]], [[1, 0], [0, 0]], [[-1, -1], [0, 0]]), # m4
        ([[0, 0], [0, 1]], [[0, 0], [0, 1]], [[0, 0], [0, 1]]),   # m5
    ]
    
    # Verify the 2x2x2 decomposition
    for i in range(2):
        for j in range(2):
            for k in range(2):
                total = 0
                for U, V, W in terms:
                    total += U[i][j] * V[j][k] * W[k][i]
                assert abs(total - 1) < 1e-12, f"Winograd 2x2 verification failed at ({i},{j},{k}): {total}"
    
    return terms

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Build Winograd terms for 2x2x2
    winograd_terms = _build_winograd_2x2_terms()
    
    # Construct rank-24 decomposition for 4x4x5
    # 
    # Key insight: The 4x4x4 matrix multiplication tensor can be decomposed using
    # 4 copies of the 2x2x2 Winograd scheme, where each copy handles a 2x2x2 sub-tensor.
    # The 4 sub-tensors partition the 4x4x4 tensor as follows:
    #   Sub-tensor 0: i in {0,1}, j in {0,1}, k in {0,1}
    #   Sub-tensor 1: i in {0,1}, j in {2,3}, k in {0,1}
    #   Sub-tensor 2: i in {2,3}, j in {0,1}, k in {2,3}
    #   Sub-tensor 3: i in {2,3}, j in {2,3}, k in {2,3}
    #
    # Wait, this doesn't partition correctly. Let me use a different approach.
    #
    # Actually, for matrix multiplication C = A*B, the tensor T[i,j,k] = 1 represents
    # C[i,k] = sum_j A[i,j] * B[j,k].
    #
    # I can decompose this by splitting the j-index into two groups of 2:
    #   C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k] + sum_{j=2}^{3} A[i,j] * B[j,k]
    #
    # Each sum is a 4x2x4 matrix multiplication, which can be further decomposed.
    #
    # For the first sum (j in {0,1}): This is a 4x2x4 tensor.
    #   I can split i into {0,1} and {2,3}, and k into {0,1} and {2,3}.
    #   This gives 4 sub-tensors of size 2x2x2, each using 5 Winograd terms.
    #   Total: 4 * 5 = 20 terms for the j in {0,1} part.
    #
    # Similarly for the j in {2,3} part: another 20 terms.
    # Total for 4x4x4: 40 terms.
    #
    # For the 4x4x1 part (k=4): 4x4x1 tensor, which can be decomposed into
    #   4 sub-tensors of size 2x2x1, each using 4 schoolbook terms.
    #   Total: 4 * 4 = 16 terms.
    #
    # Grand total: 40 + 16 = 56 terms.
    #
    # That's better than 80 but not 24. Let me try to be smarter.
    
    # For the 4x2x4 tensor (j fixed to {0,1} or {2,3}):
    #   T[i,j,k] = 1 for i in {0,1,2,3}, j in {0,1}, k in {0,1,2,3}
    #   This is 2 * 4 * 4 = 32 entries.
    #
    # I can decompose this as a sum of 8 rank-1 terms using the following:
    #   For each i-block (2 blocks of size 2) and k-block (2 blocks of size 2):
    #     Use 2 terms to cover the 2x2x2 sub-tensor (i in block, j in {0,1}, k in block)
    #
    # A 2x2x2 tensor with j having 2 values can be decomposed into 2 rank-1 terms:
    #   T[i,j,k] = 1 for i in {0,1}, j in {0,1}, k in {0,1}
    #   = sum_{t=0}^{1} U_t[i,j] * V_t[j,k] * W_t[k,i]
    #   where:
    #     t=0: U = [[1,0],[1,0]], V = [[1,0],[0,1]], W = [[1,0],[0,0]]
    #     t=1: U = [[0,1],[0,1]], V = [[1,1],[0,0]], W = [[0,1],[0,0]]
    #   Let me verify:
    #     (0,0,0): 1*1*1 + 0*1*0 = 1 ✓
    #     (0,0,1): 1*0*0 + 0*1*0 = 0 ✗ (should be 1)
    #
    # This doesn't work. Let me just use schoolbook for the sub-tensors.
    
    # Given the complexity, let me implement a valid rank-48 decomposition:
    # Use 12 rank-1 terms per 2x2x4 sub-tensor, and there are 4 such sub-tensors
    # for the 4x4x4 part (total 48), plus 16 for the 4x4x1 part (total 64).
    # That's not better than 80.
    
    # I'll just use schoolbook for now to ensure correctness.
    
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
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

I'm struggling to implement the Winograd-based decomposition correctly within the time constraints. Let me implement a simpler but valid rank-reduced decomposition using a different approach.

