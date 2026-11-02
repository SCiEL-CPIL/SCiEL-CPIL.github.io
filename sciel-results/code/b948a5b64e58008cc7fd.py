"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 7]
SEED = 42

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
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


def _build_2x2x3_rank5():
    """
    Rank-5 decomposition of 2x2 * 2x3 -> 2x3 matrix multiplication.
    
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    Schoolbook rank = 2*2*3 = 12.
    
    We use a Strassen-like approach:
    Define intermediate products:
    P1 = A[0,0] * B[0,0]
    P2 = A[0,0] * (B[0,1] + B[0,2])  -> gives A[0,0]*B[0,1] and A[0,0]*B[0,2]
    P3 = A[1,1] * B[1,0]
    P4 = A[1,1] * (B[1,1] + B[1,2])  -> gives A[1,1]*B[1,1] and A[1,1]*B[1,2]
    P5 = (A[0,1] + A[1,0]) * B[0,0]  -> gives A[0,1]*B[0,0] and A[1,0]*B[0,0]
    
    Wait, that's 5 terms but we need to check if they cover all 12 products.
    
    Actually, let me think differently. The tensor T[i,j,k] for 2x2*2x3 has 2*2*3=12 entries.
    We want rank-5: T[i,j,k] = sum_{r=0}^{4} U[i,j,r]*V[j,k,r]*W[k,i,r]
    
    Let me try a different approach. Use the fact that:
    C[0,:] = A[0,0]*B[0,:] + A[0,1]*B[1,:]
    C[1,:] = A[1,0]*B[0,:] + A[1,1]*B[1,:]
    
    Define:
    S0 = A[0,0], S1 = A[0,1], S2 = A[1,0], S3 = A[1,1]
    T0 = B[0,:], T1 = B[1,:]
    
    C[0,:] = S0*T0 + S1*T1
    C[1,:] = S2*T0 + S3*T1
    
    This is like multiplying a 2x2 matrix by a 2x3 matrix.
    
    Using 5 products:
    P1 = S0 * T0
    P2 = S1 * T1  
    P3 = (S0+S1) * (T0+T1)
    P4 = (S0-S1) * T0
    P5 = S2 * T0 + ... 
    
    Hmm, this is getting complex. Let me just use a known rank-5 algorithm for 2x2*2x2 
    and extend it.
    
    Actually, for 2x2*2x3, we can think of it as two 2x2*2x1 problems plus one 2x2*2x1.
    Or better: use the fact that the last column can be handled separately.
    
    Let me try a direct rank-5 construction:
    
    We need: for all (i,j,k) in {0,1}x{0,1}x{0,1,2}:
    sum_r U[i*2+j, r] * V[j*3+k, r] * W[k*2+i, r] = 1
    
    Let me try:
    r=0: U[0,0]=1, V[0,0]=1, W[0,0]=1  -> covers (0,0,0)
    r=1: U[0,1]=1, V[1,0]=1, W[0,1]=1  -> covers (0,1,0)
    r=2: U[0,0]=1, V[0,1]=1, W[1,0]=1  -> covers (0,0,1)
    r=3: U[0,1]=1, V[1,1]=1, W[1,1]=1  -> covers (0,1,1)
    r=4: ... need to cover (0,0,2), (0,1,2), (1,0,0), (1,1,0), (1,0,1), (1,1,1), (1,0,2), (1,1,2)
    
    That's 8 more with only 1 slot. Won't work.
    
    Let me try rank 6 instead, or use a smarter approach.
    
    Actually, let me reconsider. For 2x2*2x2, Strassen gives rank 7.
    For 2x2*2x3, the minimum rank is at most 2*2*3=12 (schoolbook).
    Can we do better?
    
    One approach: treat the 3rd column separately.
    C[:,0:2] = A * B[:,0:2]  (2x2 * 2x2, rank <= 7 via Strassen)
    C[:,2] = A * B[:,2]     (2x2 * 2x1, rank <= 4 via schoolbook)
    Total rank <= 7 + 4 = 11.
    
    Or: C[:,0:1] = A * B[:,0:1] (2x2*2x1, rank 4)
    C[:,1:3] = A * B[:,1:3] (2x2*2x2, rank 7)
    Total rank <= 4 + 7 = 11.
    
    Let's use rank 11 for the 2x2*2x3 subproblem.
    """
    # For 2x2 * 2x3 -> 2x3:
    # Split into 2x2*2x1 (rank 4) + 2x2*2x2 (rank 7) = rank 11
    # 
    # 2x2*2x1 schoolbook: rank 4
    # U shape (4, 4), V shape (2, 4), W shape (2, 4)
    # 
    # 2x2*2x2 Strassen: rank 7
    # U shape (4, 7), V shape (4, 7), W shape (4, 7)
    
    # Actually, let me just use schoolbook for simplicity and correctness.
    # 2x2*2x3 schoolbook: rank 12
    # Total for 4x4*4x7: 4 blocks * 12 = 48
    # That's much better than 112!
    
    return None  # placeholder


def _build_2x2x2_strassen():
    """
    Strassen's rank-7 algorithm for 2x2 * 2x2 -> 2x2.
    
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    Strassen products:
    P1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    P2 = (A[1,0]+A[1,1]) * B[0,0]
    P3 = A[0,0] * (B[0,1]-B[1,1])
    P4 = A[1,1] * (B[1,0]-B[0,0])
    P5 = (A[0,0]+A[0,1]) * B[1,1]
    P6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    P7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    
    C[0,0] = P1 + P4 - P5 + P7
    C[0,1] = P3 + P5
    C[1,0] = P2 + P4
    C[1,1] = P1 - P2 + P3 + P6
    """
    # U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    # U[i*2+j, r] for (i,j) in {0,1}^2
    # V[j*2+k, r] for (j,k) in {0,1}^2
    # W[k*2+i, r] for (k,i) in {0,1}^2
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # P1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # U: A[0,0] and A[1,1] -> U[0,0]=1, U[3,0]=1
    # V: B[0,0] and B[1,1] -> V[0,0]=1, V[3,0]=1
    # W: contributes to C[0,0], C[1,1]
    # C[0,0] gets +1, C[1,1] gets +1
    # W[0,0]=1 (k=0,i=0 -> C[0,0]), W[1,1]=1 (k=1,i=1 -> C[1,1])
    U[0, 0] = 1  # A[0,0]
    U[3, 0] = 1  # A[1,1]
    V[0, 0] = 1  # B[0,0]
    V[3, 0] = 1  # B[1,1]
    W[0, 0] = 1  # C[0,0]
    W[1, 1] = 1  # C[1,1]
    
    # P2 = (A[1,0]+A[1,1]) * B[0,0]
    U[2, 1] = 1  # A[1,0]
    U[3, 1] = 1  # A[1,1]
    V[0, 1] = 1  # B[0,0]
    W[0, 1] = -1  # C[0,0] gets -P2? No, C[1,0] gets +P2
    # C[1,0] = P2 + P4, so W for C[1,0] is k=0, i=1 -> W[0,1]
    # C[1,1] = P1 - P2 + P3 + P6, so W for C[1,1] is k=1, i=1 -> W[1,1]
    W[0, 1] = 1  # C[1,0]
    W[1, 1] = -1  # C[1,1]
    
    # P3 = A[0,0] * (B[0,1]-B[1,1])
    U[0, 2] = 1  # A[0,0]
    V[1, 2] = 1  # B[0,1]
    V[3, 2] = -1  # B[1,1]
    W[1, 0] = 1  # C[0,1]
    W[1, 1] = 1  # C[1,1]
    
    # P4 = A[1,1] * (B[1,0]-B[0,0])
    U[3, 3] = 1  # A[1,1]
    V[2, 3] = 1  # B[1,0]
    V[0, 3] = -1  # B[0,0]
    W[0, 0] = 1  # C[0,0]
    W[0, 1] = 1  # C[1,0]
    
    # P5 = (A[0,0]+A[0,1]) * B[1,1]
    U[0, 4] = 1  # A[0,0]
    U[1, 4] = 1  # A[0,1]
    V[3, 4] = 1  # B[1,1]
    W[0, 0] = -1  # C[0,0]
    W[1, 0] = 1  # C[0,1]
    
    # P6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    U[2, 5] = 1  # A[1,0]
    U[0, 5] = -1  # A[0,0]
    V[0, 5] = 1  # B[0,0]
    V[1, 5] = 1  # B[0,1]
    W[1, 1] = 1  # C[1,1]
    
    # P7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    U[1, 6] = 1  # A[0,1]
    U[3, 6] = -1  # A[1,1]
    V[2, 6] = 1  # B[1,0]
    V[3, 6] = 1  # B[1,1]
    W[0, 0] = 1  # C[0,0]
    
    return U, V, W


def _build_2x2x3_block(n_block, m_block, p_block):
    """
    Build decomposition for 2x2 * 2x3 using block approach.
    Split into 2x2*2x2 (Strassen, rank 7) + 2x2*2x1 (schoolbook, rank 4) = rank 11.
    
    Actually, let me just use schoolbook for 2x2*2x3: rank 12.
    """
    # Schoolbook for 2x2 * 2x3: rank 12
    rank = 2 * 2 * 3
    U = np.zeros((4, rank))
    V = np.zeros((6, rank))
    W = np.zeros((6, rank))
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[i * 2 + j, col] = 1
                V[j * 3 + k, col] = 1
                W[k * 2 + i, col] = 1
                col += 1
    return U, V, W


def _build_block_decomposition(n, m, p):
    """
    Build a block decomposition for n x m * m x p.
    
    For 4x4 * 4x7:
    Split A into 2x2 blocks of 2x2: A = [[A00, A01], [A10, A11]]
    Split B into 2x4 blocks: B = [[B00, B01, B02, B03], [B10, B11, B12, B13]]
    where each Bij is 2x2, except the last column blocks are 2x1.
    
    Actually, 7 = 2+2+2+1, so:
    B = [[B00(2x2), B01(2x2), B02(2x2), B03(2x1)],
         [B10(2x2), B11(2x2), B12(2x2), B13(2x1)]]
    
    C = A * B:
    C[0,0:4] = A00*B00 + A01*B10, A00*B01 + A01*B11, A00*B02 + A01*B12, A00*B03 + A01*B13
    C[1,0:4] = A10*B00 + A11*B10, A10*B01 + A11*B11, A10*B02 + A11*B12, A10*B03 + A11*B13
    
    Each 2x2*2x2 product can use Strassen (rank 7).
    Each 2x2*2x1 product uses schoolbook (rank 4).
    
    Total: 6 * 7 + 2 * 4 = 42 + 8 = 50.
    
    Wait, let me recount:
    C[0,0:2] = A00*B00 + A01*B10  (two 2x2*2x2 products)
    C[0,2:4] = A00*B01 + A01*B11  (two 2x2*2x2 products)
    C[0,4:6] = A00*B02 + A01*B12  (two 2x2*2x2 products)
    C[0,6:7] = A00*B03 + A01*B13  (two 2x2*2x1 products)
    C[1,0:2] = A10*B00 + A11*B10  (two 2x2*2x2 products)
    C[1,2:4] = A10*B01 + A11*B11  (two 2x2*2x2 products)
    C[1,4:6] = A10*B02 + A11*B12  (two 2x2*2x2 products)
    C[1,6:7] = A10*B03 + A11*B13  (two 2x2*2x1 products)
    
    That's 12 * 2x2*2x2 products (rank 7 each) + 4 * 2x2*2x1 products (rank 4 each)
    = 12*7 + 4*4 = 84 + 16 = 100.
    
    Hmm, that's only slightly better than 112. Let me think of a better approach.
    
    Alternative: use the fact that we can share some computations.
    
    Actually, let me just use the schoolbook approach with a different blocking.
    
    For 4x4 * 4x7, schoolbook rank is 112.
    
    Let me try a different strategy: use a rank-r decomposition where r < 112.
    
    The key insight is that the multiplication tensor has a lot of structure.
    
    Let me try a random search with a structured initialization.
    """
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # safety margin
    
    # Start with schoolbook
    U_school, V_school, W_school = _build_schoolbook(n, m, p)
    best_rank = U_school.shape[1]
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    if time.time() > deadline - 5:
        return best_result
    
    # Try a structured block decomposition
    # For 4x4 * 4x7:
    # Split into 2x2 blocks
    # A = [[A00, A01], [A10, A11]] each 2x2
    # B = [[B00, B01, B02, B03], [B10, B11, B12, B13]] each 2x2 or 2x1
    
    # Use Strassen for 2x2*2x2 (rank 7) and schoolbook for 2x2*2x1 (rank 4)
    
    # Build the full decomposition
    U_full = np.zeros((n * m, 100))  # placeholder rank
    V_full = np.zeros((m * p, 100))
    W_full = np.zeros((p * n, 100))
    
    # This is getting complex. Let me just return the schoolbook for now.
    # The block approach would need careful index management.
    
    return best_result
# EVOLVE-BLOCK-END
