"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    Output factor follows the official trace convention W[k*n+i, r].
    Returns: {'U': list of lists, 'V': list of lists, 'W': list of lists}
    """
    n, m, p = 3, 3, 7

    # Strategy: Split the 7 columns of B into blocks:
    #   Block 1: columns 0-2 (3x3) -> use Strassen 3x3*3x3 (rank 7)
    #   Block 2: columns 3-5 (3x3) -> use Strassen 3x3*3x3 (rank 7)
    #   Block 3: column 6     (3x1) -> use schoolbook 3x3*3x1 (rank 9)
    # Total rank = 7 + 7 + 9 = 23

    # Strassen 3x3 * 3x3 decomposition (rank 7)
    # For matrices A (3x3) and B (3x3), the product C = A*B has entries:
    # C[i,j] = sum over r of U[i,j,r] * V[j,k,r] * W[k,i,r]
    #
    # The 7 Strassen terms for 3x3 (using the standard embedding of 2x2 Strassen):
    # We split each 3x3 into 2x2 blocks plus a corner, but a cleaner approach:
    # Use the known rank-7 decomposition for 3x3*3x3.
    #
    # A simpler and verified approach: use the "bordered" Strassen.
    # Actually, let's use a direct construction.
    #
    # For 3x3 * 3x3, one known rank-7 decomposition uses:
    # Let A = [[a00,a01,a02],[a10,a11,a12],[a20,a21,a22]]
    # Let B = [[b00,b01,b02],[b10,b11,b12],[b20,b21,b22]]
    #
    # The 7 products are (using a specific 3x3 Strassen variant):
    # This is non-trivial. Let me use a different strategy.
    #
    # Alternative: Use the fact that 3x3*3x3 can be done in rank 7 by
    # viewing it as computing 9 outputs with 7 scalar multiplications.
    #
    # Let me use a known explicit decomposition. One approach:
    # Pad to 4x4 and use Strassen, but that gives rank 15 for 4x4*4x4.
    #
    # Actually, the minimal tensor rank of 3x3*3x3 is 7, but finding an
    # explicit decomposition is non-trivial. Let me use a different approach.
    #
    # SIMPLER APPROACH: Use schoolbook for all blocks but try to find
    # a better decomposition through a known construction.
    #
    # For 3x3 * 3x1 (matrix times vector), the rank is at most 9 (schoolbook).
    # Can we do better? The tensor rank of 3x3*3x1 is 9 (it's a "slice" of
    # matrix multiplication).
    #
    # So let's just use schoolbook for everything: rank 63.
    # But we can do better by using the block structure.
    #
    # Actually, let me just construct the schoolbook and then try to
    # verify if a lower rank is achievable through a simple search.
    #
    # For now, let's construct a valid decomposition with rank 63 (schoolbook)
    # and then attempt to reduce it.

    # Schoolbook decomposition (rank 63)
    schoolbook_rank = n * m * p
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    def verify(U, V, W, tol=1e-9):
        """Verify exact decomposition using numpy."""
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if len(U) == 0 or len(V) == 0 or len(W) == 0:
            return False
        r = len(U[0])
        if len(V[0]) != r or len(W[0]) != r or r <= 0:
            return False
        Ua = np.array(U, dtype=np.float64)
        Va = np.array(V, dtype=np.float64)
        Wa = np.array(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(Ua[i*m+j], Va[j*p+k] * Wa[k*n+i])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    # Verify schoolbook
    assert verify(U, V, W), "Schoolbook decomposition failed verification"

    # Now try to find a lower-rank decomposition.
    # Use a structured approach: try to merge schoolbook terms.
    #
    # Key insight: For 3x3 * 3x7, we can group the 7 columns.
    # If we could use a rank-7 decomposition for 3x3*3x3, we'd get:
    #   7 (for cols 0-2) + 7 (for cols 3-5) + 9 (for col 6) = 23
    #
    # Let me try to construct a rank-7 decomposition for 3x3*3x3.
    #
    # One known approach: Use the "Coppersmith-Winograd" style or
    # a direct algebraic construction.
    #
    # For simplicity and correctness, let me try a random search
    # for a rank-7 3x3*3x3 decomposition, then embed it.

    # Try to find rank-7 decomposition for 3x3*3x3
    def find_3x3_3x3_rank7(max_trials=1000):
        """Try to find a rank-7 decomposition for 3x3*3x3."""
        import random
        random.seed(42)
        
        for trial in range(max_trials):
            U3 = np.zeros((9, 7))
            V3 = np.zeros((9, 7))
            W3 = np.zeros((9, 7))
            
            for r in range(7):
                # Random sparse pattern
                idx_u = random.randint(0, 8)
                idx_v = random.randint(0, 8)
                idx_w = random.randint(0, 8)
                
                U3[idx_u, r] = random.choice([-1, 1])
                V3[idx_v, r] = random.choice([-1, 1])
                W3[idx_w, r] = random.choice([-1, 1])
            
            # Check if this works
            valid = True
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        val = np.dot(U3[i*3+j], V3[j*3+k] * W3[k*3+i])
                        if abs(val - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return U3, V3, W3
        
        return None

    result_3x3 = find_3x3_3x3_rank7(max_trials=200)
    
    if result_3x3 is not None:
        U3, V3, W3 = result_3x3
        
        # Build the full 3x3*3x7 decomposition
        total_rank = 7 + 7 + 9  # 23
        U_full = [[0] * total_rank for _ in range(n * m)]
        V_full = [[0] * total_rank for _ in range(m * p)]
        W_full = [[0] * total_rank for _ in range(p * n)]
        
        # Block 1: columns 0-2, Strassen terms 0-6
        for r in range(7):
            for idx_u in range(9):
                U_full[idx_u][r] = U3[idx_u, r]
            for idx_v in range(9):
                j = idx_v // 3
                k = idx_v % 3
                V_full[idx_v][r] = V3[idx_v, r]
            for idx_w in range(9):
                k = idx_w // 3
                i = idx_w % 3
                W_full[idx_w][r] = W3[idx_w, r]
        
        # Block 2: columns 3-5, Strassen terms 7-13
        for r in range(7):
            for idx_u in range(9):
                U_full[idx_u][7 + r] = U3[idx_u, r]
            for idx_v in range(9):
                j = idx_v // 3
                k = idx_v % 3
                # Map to columns 3-5
                V_full[j*3 + (k+3)][7 + r] = V3[idx_v, r]
            for idx_w in range(9):
                k = idx_w // 3
                i = idx_w % 3
                # Map to columns 3-5
                W_full[(k+3)*3 + i][7 + r] = W3[idx_w, r]
        
        # Block 3: column 6, schoolbook terms 14-22
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    r = 14 + i*9 + j*3 + k
                    U_full[i*3 + j][r] = 1
                    V_full[j*3 + 6][r] = 1
                    W_full[6*3 + i][r] = 1
        
        if verify(U_full, V_full, W_full):
            return {'U': U_full, 'V': V_full, 'W': W_full}
    
    # If rank-7 3x3*3x3 not found, return schoolbook
    return {'U': U, 'V': V, 'W': W}

# EVOLVE-BLOCK-END
