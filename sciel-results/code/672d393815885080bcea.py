"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 7]
SEED = 42

# EVOLVE-BLOCK-START
def _build_strassen_2x2x2():
    """
    Returns U, V, W for 2x2 * 2x2 -> 2x2 with rank 7.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i].
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # P1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # P2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # P3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # P4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # P5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # P6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # P7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_4x4x2_decomp():
    """
    Builds 4x4 * 4x2 -> 4x2 using Kronecker product of two 2x2 Strassen decompositions.
    Rank = 7 * 7 = 49.
    """
    U_s, V_s, W_s = _build_strassen_2x2x2()
    # U_total[ (i1*2+i2)*2 + (j1*2+j2), r1*7+r2 ] = U_s[i1*2+j1, r1] * U_s[i2*2+j2, r2]
    U = np.kron(U_s, U_s)
    V = np.kron(V_s, V_s)
    W = np.kron(W_s, W_s)
    return U, V, W

def _build_4x4x1_decomp():
    """
    Builds 4x4 * 4x1 -> 4x1 using schoolbook.
    Rank = 4 * 4 * 1 = 16.
    """
    n, m, p = 4, 4, 1
    rank = 16
    U = np.zeros((16, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            U[i*m + j, col] = 1
            V[j*p, col] = 1
            W[i, col] = 1
            col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition exactly reproduces matrix multiplication."""
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)
    
    if U_mat.shape != (n * m, U_mat.shape[1]):
        return False
    if U_mat.shape[1] != V_mat.shape[1] or V_mat.shape[1] != W_mat.shape[1]:
        return False
    if V_mat.shape[0] != m * p:
        return False
    if W_mat.shape[0] != p * n:
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_mat[i * m + j, :] * V_mat[j * p + k, :] * W_mat[k * n + i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # safety margin
    
    # 1. Build the 4x4x2 part (rank 49)
    U1, V1, W1 = _build_4x4x2_decomp()
    
    # 2. Build the 4x4x1 part (rank 16)
    U2, V2, W2 = _build_4x4x1_decomp()
    
    # 3. Combine them
    # Total rank = 49 + 16 = 65? 
    # Wait, the known rank is 93. Why?
    # 4x4x2 rank 49. 4x4x1 rank 16.
    # Total rank 65.
    # Is rank 65 valid?
    # Let's check the tensor.
    # T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
    # For k=0,1: U1, V1, W1 handle it.
    # For k=2..6: U2, V2, W2 handle it.
    # U1 shape (16, 49). V1 shape (8, 49). W1 shape (8, 49).
    # U2 shape (16, 16). V2 shape (4, 16). W2 shape (4, 16).
    
    # We need to pad V and W for the first part to match p=7.
    # V1 has shape (8, 49). We need (28, 65).
    # V1 covers j*2+k for k in {0,1}. 
    # Indices for V1: j*2+k. j in 0..3, k in 0..1. 
    # Wait, V index is j*p+k. For the subproblem 4x4x2, p=2.
    # So V1 indices are j*2+k.
    # In the full problem, p=7. V indices are j*7+k.
    # For k=0, V index is j*7.
    # For k=1, V index is j*7+1.
    # So V1[ j*2+k ] maps to V_full[ j*7+k ].
    
    # Similarly for W.
    # W1 indices are k*4+i. k in {0,1}.
    # In full problem, W indices are k*4+i. k in {0..6}.
    # So W1[ k*4+i ] maps to W_full[ k*4+i ].
    
    # U indices are i*4+j. Same for both.
    
    rank_total = U1.shape[1] + U2.shape[1]
    U_full = np.zeros((16, rank_total))
    V_full = np.zeros((28, rank_total))
    W_full = np.zeros((28, rank_total))
    
    # Copy part 1
    U_full[:, 0:49] = U1
    for j in range(4):
        for k in range(2):
            V_full[j*7 + k, 0:49] = V1[j*2 + k, 0:49]
            W_full[k*4 + i if False else k*4 + (j*4+k)//2 % 4, 0:49] # Wait, W index is k*n+i.
            # W1 index is k*4+i.
            # W_full index is k*4+i.
            # So W_full[ k*4 + i ] = W1[ k*4 + i ].
            # This is just a direct copy for the first 8 rows of W?
            # No, W1 has 8 rows (k=0,1; i=0..3).
            # W_full has 28 rows (k=0..6; i=0..3).
            # So rows 0..7 of W_full get W1.
            # Rows 8..27 are 0.
    
    # Let's do it properly.
    # V1 (8, 49) -> V_full (28, 49)
    # V1[j*2+k] -> V_full[j*7+k]
    for j in range(4):
        V_full[j*7, 0:49] = V1[j*2, 0:49]
        V_full[j*7+1, 0:49] = V1[j*2+1, 0:49]
        
    # W1 (8, 49) -> W_full (28, 49)
    # W1[k*4+i] -> W_full[k*4+i]
    # Since k is only 0,1, rows 0..7 are filled.
    W_full[0:8, 0:49] = W1
    
    # Copy part 2
    # U2 (16, 16) -> U_full[:, 49:65]
    U_full[:, 49:65] = U2
    
    # V2 (4, 16) -> V_full (28, 16)
    # V2[j] corresponds to j*1+0 = j.
    # In full problem, k=2..6.
    # V index j*7+k.
    # We need to map V2[j] to V_full[j*7 + k] for k in 2..6?
    # Wait. The 4x4x1 subproblem multiplies A (4x4) by B_col (4x1).
    # B_col corresponds to the 5th column of the full B (index 4? or 2?).
    # The decomposition splits B into B_left (4x2) and B_right (4x5)?
    # No, the known rank 93 comes from 4x4x2 (rank 49) + 4x4x5 (rank 80)? No.
    # 4x4x2 rank 49. 4x4x1 rank 16.
    # If we use this for 4x4x3, rank is 65.
    # For 4x4x7, we need to handle the remaining 5 columns.
    # Schoolbook for 4x4x5 is 4*4*5 = 80.
    # Total 49 + 80 = 129. Worse than 112.
    
    # How did the island get 93?
    # "Replacing the 4x4x3 schoolbook (rank 48) with a Strassen-based 4x4x2 (rank 28) + schoolbook 4x4x1 (rank 16) decomposition reduces total rank from 97 to 93"
    # Wait. 4x4x2 Strassen rank is 49. Why does it say 28?
    # Ah. 4x4x2 can be done with rank 49.
    # Maybe they used a different algorithm?
    # Or maybe 4x4x2 rank is lower?
    # Lower bound for 4x4x2 is not 49.
    # Actually, 4x4 matrix multiplication rank is 49? No, 4x4x4 is 49? No, 4x4x4 is 49?
    # Strassen for 4x4 is 49.
    # For 4x4x2, it's like 4x4x1 + 4x4x1. Rank 32?
    # If we use Strassen for 4x4x2, is it 49?
    # Kronecker product of 2x2x2 (rank 7) and 2x2x1 (rank 4)?
    # 2x2x1 rank is 4.
    # 7 * 4 = 28.
    # Yes! 4x4x2 can be decomposed using Kronecker product of 2x2x2 (Strassen, rank 7) and 2x2x1 (Schoolbook, rank 4).
    # Rank = 28.
    
    # So I need to build 4x4x2 using 2x2x2 (rank 7) x 2x2x1 (rank 4).
    # And 4x4x5 using Schoolbook (rank 80).
    # Total 28 + 80 = 108. Still worse than 112?
    # Wait. 97 to 93.
    # 97 was 4x4x3 schoolbook (48) + 4x4x4 schoolbook (64)? 48+64=112.
    # Where does 97 come from?
    # Maybe 4x4x2 (rank 32?) + 4x4x1 (rank 16) + 4x4x4 (rank 49?)?
    # 32 + 16 + 49 = 97.
    # And the improvement to 93:
    # 4x4x2 (rank 28) + 4x4x1 (rank 16) + 4x4x4 (rank 49) = 93.
    
    # So the strategy is:
    # 1. 4x4x2 using Kronecker(2x2x2 Strassen, 2x2x1 Schoolbook) -> Rank 28.
    # 2. 4x4x1 using Schoolbook -> Rank 16.
    # 3. 4x4x4 using Kronecker(2x2x2 Strassen, 2x2x2 Strassen) -> Rank 49.
    # Total Rank = 28 + 16 + 49 = 93.
    
    # Let's implement this.
    
    # Part 1: 4x4x2 (Rank 28)
    # U1 = kron(U_2x2x2, U_2x2x1)
    # V1 = kron(V_2x2x2, V_2x2x1)
    # W1 = kron(W_2x2x2, W_2x2x1)
    
    U_s2, V_s2, W_s2 = _build_strassen_2x2x2() # Rank 7
    # 2x2x1 Schoolbook
    U_s1 = np.eye(4).reshape(4, 4) # Wait. 2x2x1.
    # U shape (4, 4), V shape (2, 4), W shape (2, 4).
    # U[i*2+j, r] = 1 if r = i*2+j.
    # V[j*1+k, r] = 1 if r = i*2+j. (k=0)
    # W[k*2+i, r] = 1 if r = i*2+j.
    
    U_2x2x1 = np.zeros((4, 4))
    V_2x2x1 = np.zeros((2, 4))
    W_2x2x1 = np.zeros((2, 4))
    for i in range(2):
        for j in range(2):
            r = i*2+j
            U_2x2x1[i*2+j, r] = 1
            V_2x2x1[j, r] = 1
            W_2x2x1[i, r] = 1
            
    U_p1 = np.kron(U_s2, U_2x2x1) # (4,7) x (4,4) -> (16, 28)
    V_p1 = np.kron(V_s2, V_2x2x1) # (4,7) x (2,4) -> (8, 28)
    W_p1 = np.kron(W_s2, W_2x2x1) # (4,7) x (2,4) -> (8, 28)
    
    # Part 2: 4x4x1 (Rank 16)
    U_p2, V_p2, W_p2 = _build_4x4x1_decomp()
    
    # Part 3: 4x4x4 (Rank 49)
    U_p3 = np.kron(U_s2, U_s2)
    V_p3 = np.kron(V_s2, V_s2)
    W_p3 = np.kron(W_s2, W_s2)
    
    # Assemble
    rank_total = 28 + 16 + 49
    U_full = np.zeros((16, rank_total))
    V_full = np.zeros((28, rank_total))
    W_full = np.zeros((28, rank_total))
    
    # Part 1 (cols 0-27)
    U_full[:, 0:28] = U_p1
    # V_p1 (8, 28) -> V_full (28, 28)
    # V_p1[j*2+k] -> V_full[j*7+k] for k=0,1
    for j in range(4):
        V_full[j*7, 0:28] = V_p1[j*2, 0:28]
        V_full[j*7+1, 0:28] = V_p1[j*2+1, 0:28]
    # W_p1 (8, 28) -> W_full (28, 28)
    # W_p1[k*4+i] -> W_full[k*4+i] for k=0,1
    W_full[0:8, 0:28] = W_p1
    
    # Part 2 (cols 28-43)
    U_full[:, 28:44] = U_p2
    # V_p2 (4, 16) -> V_full (28, 16)
    # V_p2[j] -> V_full[j*7 + 2] (k=2)
    for j in range(4):
        V_full[j*7 + 2, 28:44] = V_p2[j, 0:16]
    # W_p2 (4, 16) -> W_full (28, 16)
    # W_p2[i] -> W_full[2*4 + i] (k=2)
    W_full[8:12, 28:44] = W_p2
    
    # Part 3 (cols 44-92)
    U_full[:, 44:93] = U_p3
    # V_p3 (16, 49) -> V_full (28, 49)
    # V_p3[j*4+k] -> V_full[j*7+k] for k=3,4,5,6
    for j in range(4):
        for k in range(4):
            V_full[j*7 + 3 + k, 44:93] = V_p3[j*4 + k, 0:49]
    # W_p3 (16, 49) -> W_full (28, 49)
    # W_p3[k*4+i] -> W_full[k*4+i] for k=3,4,5,6
    W_full[12:28, 44:93] = W_p3
    
    # Verify
    if _verify(U_full, V_full, W_full, n, m, p):
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
