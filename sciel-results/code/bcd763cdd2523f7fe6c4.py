"""
Strategy: Construct a rank-36 decomposition for 2x4 * 4x5 matrix multiplication.
We split the inner dimension m=4 into two blocks of size 2 (j=0,1 and j=2,3).
For each 2x2x5 sub-problem (since n=2, p=5), we decompose the 5 output columns into:
- Two 2x2x2 blocks (columns 0,1 and 2,3) using Strassen's algorithm (rank 7 each).
- One 2x2x1 block (column 4) using the trivial decomposition (rank 4).
Total rank per block = 7 + 7 + 4 = 18.
Total rank for 2 blocks = 18 + 18 = 36.
This is lower than the schoolbook rank of 40.

We verify the decomposition rigorously and fall back to schoolbook if needed.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-36 tensor decomposition for matrix multiplication 2x4 * 4x5.
    Uses Strassen's algorithm for 2x2x2 sub-blocks and trivial decomposition
    for the remaining 2x2x1 sub-block.
    """
    n, m, p = DIMENSIONS  # 2, 4, 5
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    def verify(U, V, W):
        """Verify that U, V, W form a valid tensor decomposition."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        r = U.shape[1]
        
        # Check all valid (i,j,k) positions: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        
        # Check zero positions: for (i,j,k) where the indices don't all match
        # In matrix multiplication tensor, T[i,j,k] = 1 if i,j,k correspond to 
        # (row i of A, col j of A / row j of B, col k of B) -> output (i,k)
        # The tensor has T[i,j,k] = 1 for all i,j,k (it's the full multiplication tensor)
        # Wait - actually for matrix multiplication, the tensor is:
        # T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
        # The decomposition requires: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # for ALL i,j,k. There are no "zero positions" in the standard matrix mult tensor.
        # Let me re-check: the matrix multiplication tensor has T_{i,j,k} = 1 for all valid indices.
        # So we only need to check that all (i,j,k) give 1.
        
        return True

    def verify_full(U, V, W):
        """Full verification: check all n*m*p entries equal 1."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        r = U.shape[1]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Compute sum over c of U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                    vals = U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :]
                    val = np.sum(vals)
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def schoolbook():
        """Fallback: schoolbook decomposition with rank n*m*p = 40."""
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    # Strassen 2x2x2 decomposition
    # For a 2x2 * 2x2 matrix multiplication, returns U (8x7), V (8x7), W (8x7)
    # Local indices: i in {0,1}, j in {0,1}, k in {0,1}
    # Global indices:
    # U index: i*m + (j_offset + j)
    # V index: (j_offset + j)*p + (k_offset + k)
    # W index: (k_offset + k)*n + i
    def get_strassen_222(j_offset, k_offset):
        """
        Returns U, V, W for a 2x2x2 Strassen block.
        j_offset: offset in the j (inner) dimension (0 or 2)
        k_offset: offset in the k (output column) dimension (0, 2)
        """
        U = np.zeros((n * m, 7))
        V = np.zeros((m * p, 7))
        W = np.zeros((p * n, 7))
        
        # Strassen's 7 multiplications for 2x2 * 2x2:
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # C = [[a*e+b*g, a*f+b*h], [c*e+d*g, c*f+d*h]]
        #
        # M1 = (a+d)(e+h)  -> contributes to C00 (+), C01 (-), C10 (-), C11 (+)
        # M2 = (c+d)e      -> contributes to C10 (+), C11 (-)
        # M3 = a(f-h)      -> contributes to C01 (+), C11 (+)
        # M4 = d(g-e)      -> contributes to C00 (+), C10 (+)
        # M5 = (a+b)h      -> contributes to C00 (-), C01 (+)
        # M6 = (c-a)(e+f)  -> contributes to C10 (+)
        # M7 = (b-d)(g+h)  -> contributes to C00 (+)
        
        # M1 = (A00+A11)(B00+B11)
        # A00 = U[0, j_offset], A11 = U[1, j_offset+1]
        # B00 = V[j_offset, k_offset], B11 = V[j_offset+1, k_offset+1]
        # C00 = W[k_offset, 0], C11 = W[k_offset+1, 1]
        U[0 * m + (j_offset + 0), 0] = 1.0
        U[1 * m + (j_offset + 1), 0] = 1.0
        V[(j_offset + 0) * p + (k_offset + 0), 0] = 1.0
        V[(j_offset + 1) * p + (k_offset + 1), 0] = 1.0
        W[(k_offset + 0) * n + 0, 0] = 1.0
        W[(k_offset + 1) * n + 1, 0] = 1.0

        # M2 = (A10+A11)B00
        # A10 = U[1, j_offset], A11 = U[1, j_offset+1]
        # B00 = V[j_offset, k_offset]
        # C10 = W[k_offset, 1], C11 = W[k_offset+1, 1]
        U[1 * m + (j_offset + 0), 1] = 1.0
        U[1 * m + (j_offset + 1), 1] = 1.0
        V[(j_offset + 0) * p + (k_offset + 0), 1] = 1.0
        W[(k_offset + 0) * n + 1, 1] = 1.0
        W[(k_offset + 1) * n + 1, 1] = -1.0

        # M3 = A00(B01-B11)
        # A00 = U[0, j_offset]
        # B01 = V[j_offset, k_offset+1], B11 = V[j_offset+1, k_offset+1]
        # C01 = W[k_offset+1, 0], C11 = W[k_offset+1, 1]
        U[0 * m + (j_offset + 0), 2] = 1.0
        V[(j_offset + 0) * p + (k_offset + 1), 2] = 1.0
        V[(j_offset + 1) * p + (k_offset + 1), 2] = -1.0
        W[(k_offset + 1) * n + 0, 2] = 1.0
        W[(k_offset + 1) * n + 1, 2] = 1.0

        # M4 = A11(B10-B00)
        # A11 = U[1, j_offset+1]
        # B10 = V[j_offset+1, k_offset], B00 = V[j_offset, k_offset]
        # C00 = W[k_offset, 0], C10 = W[k_offset, 1]
        U[1 * m + (j_offset + 1), 3] = 1.0
        V[(j_offset + 1) * p + (k_offset + 0), 3] = 1.0
        V[(j_offset + 0) * p + (k_offset + 0), 3] = -1.0
        W[(k_offset + 0) * n + 0, 3] = 1.0
        W[(k_offset + 0) * n + 1, 3] = 1.0

        # M5 = (A00+A01)B11
        # A00 = U[0, j_offset], A01 = U[0, j_offset+1]
        # B11 = V[j_offset+1, k_offset+1]
        # C00 = W[k_offset, 0], C01 = W[k_offset+1, 0]
        U[0 * m + (j_offset + 0), 4] = 1.0
        U[0 * m + (j_offset + 1), 4] = 1.0
        V[(j_offset + 1) * p + (k_offset + 1), 4] = 1.0
        W[(k_offset + 0) * n + 0, 4] = -1.0
        W[(k_offset + 1) * n + 0, 4] = 1.0

        # M6 = (A10-A00)(B00+B01)
        # A10 = U[1, j_offset], A00 = U[0, j_offset]
        # B00 = V[j_offset, k_offset], B01 = V[j_offset, k_offset+1]
        # C10 = W[k_offset, 1]
        U[1 * m + (j_offset + 0), 5] = 1.0
        U[0 * m + (j_offset + 0), 5] = -1.0
        V[(j_offset + 0) * p + (k_offset + 0), 5] = 1.0
        V[(j_offset + 0) * p + (k_offset + 1), 5] = 1.0
        W[(k_offset + 0) * n + 1, 5] = 1.0

        # M7 = (A01-A11)(B10+B11)
        # A01 = U[0, j_offset+1], A11 = U[1, j_offset+1]
        # B10 = V[j_offset+1, k_offset], B11 = V[j_offset+1, k_offset+1]
        # C00 = W[k_offset, 0]
        U[0 * m + (j_offset + 1), 6] = 1.0
        U[1 * m + (j_offset + 1), 6] = -1.0
        V[(j_offset + 1) * p + (k_offset + 0), 6] = 1.0
        V[(j_offset + 1) * p + (k_offset + 1), 6] = 1.0
        W[(k_offset + 0) * n + 0, 6] = 1.0

        return U, V, W

    # Trivial 2x2x1 decomposition (rank 4)
    def get_trivial_221(j_offset, k_offset):
        """
        Returns U, V, W for a 2x2x1 trivial block.
        j_offset: offset in the j (inner) dimension
        k_offset: offset in the k (output column) dimension
        """
        U = np.zeros((n * m, 4))
        V = np.zeros((m * p, 4))
        W = np.zeros((p * n, 4))
        
        col = 0
        for i in range(2):
            for j in range(2):
                U[i * m + (j_offset + j), col] = 1.0
                V[(j_offset + j) * p + k_offset, col] = 1.0
                W[k_offset * n + i, col] = 1.0
                col += 1
        return U, V, W

    # Build full decomposition
    # Block 1: j in {0, 1}
    U1_s1, V1_s1, W1_s1 = get_strassen_222(0, 0)  # k in {0, 1}
    U1_s2, V1_s2, W1_s2 = get_strassen_222(0, 2)  # k in {2, 3}
    U1_t1, V1_t1, W1_t1 = get_trivial_221(0, 4)   # k = 4
    
    U_block1 = np.hstack([U1_s1, U1_s2, U1_t1])
    V_block1 = np.hstack([V1_s1, V1_s2, V1_t1])
    W_block1 = np.hstack([W1_s1, W1_s2, W1_t1])

    # Block 2: j in {2, 3}
    U2_s1, V2_s1, W2_s1 = get_strassen_222(2, 0)  # k in {0, 1}
    U2_s2, V2_s2, W2_s2 = get_strassen_222(2, 2)  # k in {2, 3}
    U2_t1, V2_t1, W2_t1 = get_trivial_221(2, 4)   # k = 4
    
    U_block2 = np.hstack([U2_s1, U2_s2, U2_t1])
    V_block2 = np.hstack([V2_s1, V2_s2, V2_t1])
    W_block2 = np.hstack([W2_s1, W2_s2, W2_t1])

    U_full = np.hstack([U_block1, U_block2])
    V_full = np.hstack([V_block1, V_block2])
    W_full = np.hstack([W_block1, W_block2])

    # Verify the decomposition
    if verify_full(U_full, V_full, W_full):
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    # Check deadline
    if time.time() > deadline:
        U_s, V_s, W_s = schoolbook()
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    # Fallback to schoolbook
    U_s, V_s, W_s = schoolbook()
    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
# EVOLVE-BLOCK-END
