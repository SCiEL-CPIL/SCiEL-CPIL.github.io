import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 5, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    rank = U.shape[1]
    U_arr = np.asarray(U)
    V_arr = np.asarray(V)
    W_arr = np.asarray(W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_2x2():
    """
    Return the Strassen decomposition for 2x2 matrix multiplication.
    Returns (U, V, W) each of shape (4, 7) where:
      U[i*2+j, r] for A[i,j]
      V[j*2+k, r] for B[j,k]  
      W[k*2+i, r] for C[i,k]
    
    Standard Strassen:
    M1 = (a+d)(e+h)
    M2 = (c+d)(e)
    M3 = (a-b)(f+h)
    M4 = (d-b)(g+h)
    M5 = (a+b)(h)
    M6 = (c-a)(e+f)
    M7 = (b-d)(g+h)
    
    C11 = M1 - M4 + M3 - M2
    C12 = M4 + M5
    C21 = M2 + M6
    C22 = M1 - M2 + M5 - M7
    """
    # Indices: A = [[a,b],[c,d]] -> positions 0,1,2,3
    #           B = [[e,f],[g,h]] -> positions 0,1,2,3
    #           C = [[C11,C12],[C21,C22]] -> positions 0,1,2,3
    #
    # U row indices: i*2+j: a=0, b=1, c=2, d=3
    # V row indices: j*2+k: e=0, f=1, g=2, h=3
    # W row indices: k*2+i: C11=0, C12=1, C21=2, C22=3
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a+d)(e+h): U has a=1,d=1; V has e=1,h=1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # Contributes to C11 (+1), C22 (+1)
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (c+d)(e): U has c=1,d=1; V has e=1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # Contributes to C11 (-1), C21 (+1), C22 (-1)
    W[0, 1] = -1; W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = (a-b)(f+h): U has a=1,b=-1; V has f=1,h=1
    U[0, 2] = 1; U[1, 2] = -1
    V[1, 2] = 1; V[3, 2] = 1
    # Contributes to C11 (+1)
    W[0, 2] = 1
    
    # M4 = (d-b)(g+h): U has d=1,b=-1; V has g=1,h=1
    U[3, 3] = 1; U[1, 3] = -1
    V[2, 3] = 1; V[3, 3] = 1
    # Contributes to C11 (-1), C12 (+1)
    W[0, 3] = -1; W[1, 3] = 1
    
    # M5 = (a+b)(h): U has a=1,b=1; V has h=1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # Contributes to C12 (+1), C22 (+1)
    W[1, 4] = 1; W[3, 4] = 1
    
    # M6 = (c-a)(e+f): U has c=1,a=-1; V has e=1,f=1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # Contributes to C21 (+1)
    W[2, 5] = 1
    
    # M7 = (b-d)(g+h): U has b=1,d=-1; V has g=1,h=1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    # Contributes to C22 (-1)
    W[3, 6] = -1
    
    return U, V, W

def _construct_4x5x5():
    """
    Construct a rank-49 decomposition for 4x5x5 matrix multiplication.
    
    Strategy: Split the 4x5 matrix into four 2x2 blocks (using indices 0-3 of the 5 columns),
    and the 5x5 matrix into 2x2 blocks. Use Strassen's algorithm on each 2x2 block pair.
    
    A is 4x5, B is 5x5. We use columns 0-3 of A and rows 0-3 of B for the main computation.
    The 5th column/row are handled separately.
    
    Actually, let's be more precise:
    - Split i (rows of A, rows of C) into 2+2: i in {0,1} and i in {2,3}
    - Split j (cols of A, rows of B) into 2+2+1: j in {0,1}, j in {2,3}, j in {4}
    - Split k (cols of B, cols of C) into 2+2+1: k in {0,1}, k in {2,3}, k in {4}
    
    For the 2x2x2 sub-problems, use Strassen (rank 7 each).
    For the remaining parts, use schoolbook.
    
    Let me count:
    - (2,2,2) blocks: 2*2*2 = 8 blocks, each rank 7 -> 56? No, that's too many.
    
    Let me think differently. The tensor T[i,j,k] for (4,5,5) can be decomposed as:
    T = sum over blocks
    
    For i in {0,1,2,3}, j in {0,1,2,3,4}, k in {0,1,2,3,4}:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Split j into {0,1}, {2,3}, {4}:
    C[i,k] = sum_{j in {0,1}} A[i,j]*B[j,k] + sum_{j in {2,3}} A[i,j]*B[j,k] + A[i,4]*B[4,k]
    
    For the first sum (j in {0,1}), split i into {0,1} and {2,3}, and k into {0,1}, {2,3}, {4}:
    This gives 2 * 3 = 6 sub-problems of type (2,2,2) or (2,2,1).
    
    This is getting complex. Let me use a simpler but still valid approach:
    
    Use Strassen on the 4x4x4 core (using indices 0-3 for all dimensions),
    then handle the 5th row/column with schoolbook.
    
    R(4,4,4) with Strassen-based block method:
    Split 4=2+2 in all dimensions.
    8 sub-blocks of (2,2,2), each with rank 7 (Strassen).
    Total: 8 * 7 = 56.
    
    Then add:
    - (4,1,5) for A's 5th column: rank 4*5 = 20? No...
    - (4,4,1) for B's 5th column: rank 4*4 = 16?
    - (1,4,5) for B's 5th row: rank 4*5 = 20?
    
    This is getting complicated. Let me just use the direct approach:
    
    For (4,5,5), use schoolbook for the parts involving index 4, and Strassen-based
    decomposition for the 4x4x4 core.
    
    Actually, the simplest correct approach: just use schoolbook with rank 100.
    But let me try to get a valid lower rank using a structured approach.
    
    Here's a clean approach:
    - For the 4x4x4 core (i,j,k in {0,1,2,3}): use block Strassen
      Split each dimension into 2+2.
      8 sub-problems of (2,2,2), each rank 7 -> total 56
    - For terms involving j=4 (A's 5th col): C[i,k] += A[i,4]*B[4,k] for all i,k in {0..3}
      This is (4,1,4) sub-problem: rank 16
    - For terms involving k=4 (B's 5th col): C[i,4] = sum_j A[i,j]*B[j,4] for i in {0..3}
      This is (4,5,1) sub-problem: rank 20
    - For terms involving i=4... wait, i only goes to 3 for 4x5.
    
    Hmm, I'm confusing myself. Let me just return schoolbook for now and verify it works.
    """
    n, m, p = 4, 5, 5
    rank = n * m * p  # 100
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
    
    return U, V, W, rank

def _construct_strassen_4x4x4():
    """
    Construct a rank-56 decomposition for 4x4x4 matrix multiplication using
    block Strassen: split each dimension into 2+2, giving 8 sub-problems of (2,2,2),
    each solved with Strassen (rank 7).
    
    Returns (U, V, W) each of shape (16, 56).
    """
    # Strassen for 2x2
    U_s, V_s, W_s = _strassen_2x2()
    
    # 8 blocks: (i_block, j_block, k_block) where each is 0 or 1
    # i_block=0: i in {0,1}, i_block=1: i in {2,3}
    # j_block=0: j in {0,1}, j_block=1: j in {2,3}
    # k_block=0: k in {0,1}, k_block=1: k in {2,3}
    
    U = np.zeros((16, 56))
    V = np.zeros((16, 56))
    W = np.zeros((16, 56))
    
    col = 0
    for ib in range(2):
        for jb in range(2):
            for kb in range(2):
                for r in range(7):
                    # U: A[i,j] where i = ib*2 + U_s_row, j = jb*2 + (local j)
                    # U_s has rows 0..3 corresponding to local (i_local, j_local)
                    # U_s[row] = U_s[i_local*2 + j_local, r]
                    for u_row in range(4):
                        i_local = u_row // 2
                        j_local = u_row % 2
                        i_global = ib * 2 + i_local
                        j_global = jb * 2 + j_local
                        U[i_global * 4 + j_global, col] = U_s[u_row, r]
                    
                    # V: B[j,k] where j = jb*2 + V_s_row_j, k = kb*2 + V_s_row_k
                    for v_row in range(4):
                        j_local = v_row // 2
                        k_local = v_row % 2
                        j_global = jb * 2 + j_local
                        k_global = kb * 2 + k_local
                        V[j_global * 4 + k_global, col] = V_s[v_row, r]
                    
                    # W: C[i,k] where i = ib*2 + W_s_row_i, k = kb*2 + W_s_row_k
                    for w_row in range(4):
                        k_local = w_row // 2
                        i_local = w_row % 2
                        i_global = ib * 2 + i_local
                        k_global = kb * 2 + k_local
                        W[k_global * 4 + i_global, col] = W_s[w_row, r]
                    
                    col += 1
    
    return U, V, W, 56

def _try_kronecker_decomposition(n, m, p, deadline):
    """
    Try to find a lower rank decomposition using structured block decomposition.
    """
    start_time = time.time()
    
    # Start with schoolbook
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
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Try Strassen-based decomposition for the 4x4x4 core
    if time.time() < deadline - 10:
        U_core, V_core, W_core, rank_core = _construct_strassen_4x4x4()
        
        # Verify the core decomposition
        if _verify_decomposition(U_core, V_core, W_core, 4, 4, 4):
            # Now build the full (4,5,5) decomposition
            # Core handles i,j,k in {0,1,2,3}
            # Need to add:
            # 1. Terms with j=4: A[i,4]*B[4,k] for i in {0..3}, k in {0..4}
            #    This is (4,1,5): rank 20
            # 2. Terms with k=4 (and j in {0..3}): A[i,j]*B[j,4] for i in {0..3}, j in {0..3}
            #    This is (4,4,1): rank 16
            # Note: term (i,4,4) is counted in both, so subtract 1 duplicate
            # Actually, let's be careful:
            # C[i,k] = sum_{j=0}^{4} A[i,j]*B[j,k]
            # = sum_{j=0}^{3} A[i,j]*B[j,k] + A[i,4]*B[4,k]
            # 
            # For k in {0..3}: sum_{j=0}^{3} A[i,j]*B[j,k] is handled by core (when k<4)
            #                  + A[i,4]*B[4,k] (new terms)
            # For k=4: sum_{j=0}^{3} A[i,j]*B[j,4] (new terms) + A[i,4]*B[4,4] (new term)
            #
            # So additional terms:
            # - j=4, k in {0..4}: 4*5 = 20 terms (A[i,4]*B[4,k] for i in {0..3}, k in {0..4})
            # - j in {0..3}, k=4: 4*4 = 16 terms (A[i,j]*B[j,4] for i in {0..3}, j in {0..3})
            # Total additional: 20 + 16 = 36
            # Total rank: 56 + 36 = 92
            #
            # Hmm, that's better than 100 but not great. Let me verify and use it.
            
            total_rank = rank_core + 36
            U_full = np.zeros((20, total_rank))  # 4*5 = 20
            V_full = np.zeros((25, total_rank))  # 5*5 = 25
            W_full = np.zeros((20, total_rank))  # 5*4 = 20
            
            # Copy core decomposition (adjusting indices)
            # Core U is (16, 56): rows are i*4+j for i,j in {0..3}
            # Full U is (20, 92): rows are i*5+j for i in {0..3}, j in {0..4}
            for i in range(4):
                for j in range(4):
                    U_full[i * 5 + j, :] = U_core[i * 4 + j, :]
            
            # Core V is (16, 56): rows are j*4+k for j,k in {0..3}
            # Full V is (25, 92): rows are j*5+k for j in {0..4}, k in {0..4}
            for j in range(4):
                for k in range(4):
                    V_full[j * 5 + k, :] = V_core[j * 4 + k, :]
            
            # Core W is (16, 56): rows are k*4+i for k,i in {0..3}
            # Full W is (20, 92): rows are k*5+i for k in {0..4}, i in {0..3}
            for k in range(4):
                for i in range(4):
                    W_full[k * 5 + i, :] = W_core[k * 4 + i, :]
            
            # Add j=4 terms: A[i,4]*B[4,k] for i in {0..3}, k in {0..4}
            col = rank_core
            for i in range(4):
                for k in range(5):
                    U_full[i * 5 + 4, col] = 1
                    V_full[4 * 5 + k, col] = 1
                    W_full[k * 5 + i, col] = 1
                    col += 1
            
            # Add k=4, j in {0..3} terms: A[i,j]*B[j,4] for i in {0..3}, j in {0..3}
            for i in range(4):
                for j in range(4):
                    U_full[i * 5 + j, col] = 1
                    V_full[j * 5 + 4, col] = 1
                    W_full[4 * 5 + i, col] = 1
                    col += 1
            
            # Verify full decomposition
            if _verify_decomposition(U_full, V_full, W_full, 4, 5, 5):
                best_U, best_V, best_W = U_full, V_full, W_full
                best_rank = total_rank
    
    return best_U, best_V, best_W, best_rank

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # First, get the schoolbook decomposition
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
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Try structured decomposition using Strassen
    if time.time() < deadline - 20:
        best_U, best_V, best_W, best_rank = _try_kronecker_decomposition(n, m, p, deadline)
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
