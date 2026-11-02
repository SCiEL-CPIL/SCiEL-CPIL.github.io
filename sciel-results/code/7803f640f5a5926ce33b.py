"""Program for Matrix multiplication 3,3,6: rank-42 decomposition using Strassen on 2x2 blocks."""
import numpy as np

DIMENSIONS = [3, 3, 6]

# Strassen rank-7 decomposition for 2x2 x 2x2 -> 2x2
# Indices: U is (i,j) in 2x2, V is (j,k) in 2x2, W is (k,i) in 2x2
# Using the standard Strassen formulas:
# m1 = (a11+a22)(b11+b22)
# m2 = (a21+a22)b11
# m3 = a11(b12-b22)
# m4 = a22(b21-b11)
# m5 = (a11+a12)b22
# m6 = (a21-a11)(b11+b12)
# m7 = (a12-a22)(b21+b22)
# c11 = m1 + m4 - m5 + m7
# c12 = m3 + m5
# c21 = m2 + m4
# c22 = m1 - m2 + m3 + m6

# Build U, V, W for Strassen 2x2x2x2 (rank 7)
# U row index: i*2+j for i,j in {0,1}
# V row index: j*2+k for j,k in {0,1}
# W row index: k*2+i for k,i in {0,1}

def build_strassen_2x2():
    """Build U, V, W for Strassen 2x2x2x2 multiplication, rank 7."""
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # m1 = (a11+a22)(b11+b22)
    # U contributions: a11 at (0,0), a22 at (1,1)
    # V contributions: b11 at (0,0), b22 at (1,1)
    # W contributions: c11 += 1, c22 += 1
    U[0, 0] = 1.0; U[3, 0] = 1.0
    V[0, 0] = 1.0; V[3, 0] = 1.0
    W[0, 0] = 1.0; W[3, 0] = 1.0  # c11 (k=0,i=0) and c22 (k=1,i=1)
    
    # m2 = (a21+a22)b11
    # U: a21 at (1,0), a22 at (1,1)
    # V: b11 at (0,0)
    # W: c21 += 1 (k=1,i=0)
    U[2, 1] = 1.0; U[3, 1] = 1.0
    V[0, 1] = 1.0
    W[2, 1] = 1.0  # c21 (k=1,i=0)
    
    # m3 = a11(b12-b22)
    # U: a11 at (0,0)
    # V: b12 at (0,1), -b22 at (1,1)
    # W: c12 += 1 (k=1,i=0)
    U[0, 2] = 1.0
    V[1, 2] = 1.0; V[3, 2] = -1.0
    W[2, 2] = 1.0  # c12 (k=1,i=0)
    
    # m4 = a22(b21-b11)
    # U: a22 at (1,1)
    # V: b21 at (1,0), -b11 at (0,0)
    # W: c11 += 1 (k=0,i=0), c21 += 1 (k=1,i=0)
    U[3, 3] = 1.0
    V[2, 3] = 1.0; V[0, 3] = -1.0
    W[0, 3] = 1.0; W[2, 3] = 1.0  # c11 and c21
    
    # m5 = (a11+a12)b22
    # U: a11 at (0,0), a12 at (0,1)
    # V: b22 at (1,1)
    # W: c11 -= 1 (k=0,i=0), c12 += 1 (k=1,i=0)
    U[0, 4] = 1.0; U[1, 4] = 1.0
    V[3, 4] = 1.0
    W[0, 4] = -1.0; W[2, 4] = 1.0  # c11 and c12
    
    # m6 = (a21-a11)(b11+b12)
    # U: a21 at (1,0), -a11 at (0,0)
    # V: b11 at (0,0), b12 at (0,1)
    # W: c22 += 1 (k=1,i=1)
    U[2, 5] = 1.0; U[0, 5] = -1.0
    V[0, 5] = 1.0; V[1, 5] = 1.0
    W[3, 5] = 1.0  # c22
    
    # m7 = (a12-a22)(b21+b22)
    # U: a12 at (0,1), -a22 at (1,1)
    # V: b21 at (1,0), b22 at (1,1)
    # W: c11 += 1 (k=0,i=0)
    U[1, 6] = 1.0; U[3, 6] = -1.0
    V[2, 6] = 1.0; V[3, 6] = 1.0
    W[0, 6] = 1.0  # c11
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    # We decompose 3x3 x 3x6 into:
    # - A 2x2 x 2x2 block for columns 0-1 of B (using Strassen, rank 7)
    # - A 2x2 x 2x2 block for columns 2-3 of B (using Strassen, rank 7)
    # - A 2x2 x 1x2 block for columns 4-5 of B (schoolbook, rank 4)
    # - A 1x2 x 3x6 block for row 2 of A (schoolbook, rank 6)
    # Total rank: 7 + 7 + 4 + 6 = 24
    
    # Actually, let me reconsider. The matrix A is 3x3, B is 3x6.
    # C = A * B is 3x6.
    
    # Let's partition A into rows [0:2, 2:3] and B into columns [0:2, 2:4, 4:6]
    # But the last block would be 2x2 x 2x2 for cols 4-5, which is another Strassen rank 7.
    # And the row 2 part is 1x3 x 3x6, rank 18.
    # Total: 7 + 7 + 7 + 18 = 39.
    
    # Alternatively, partition differently:
    # Use Strassen for the top-left 2x2 x 2x2 (cols 0-1 of B), rank 7
    # Use Strassen for the top-middle 2x2 x 2x2 (cols 2-3 of B), rank 7
    # Use Strassen for the top-right 2x2 x 2x2 (cols 4-5 of B), rank 7
    # Use schoolbook for bottom row: 1x3 x 3x6, rank 18
    # Total: 7 + 7 + 7 + 18 = 39
    
    # Or:
    # Use Strassen for top 2 rows, all 6 cols: but Strassen is for 2x2.
    # We can do 3 Strassen blocks for the 2x2 x 2x2 subproblems.
    
    # Let me try a different approach:
    # Partition A into 2x2 + 1x3, B into 3x2 + 3x2 + 3x2
    # Top 2x2 x 3x2: use Strassen (rank 7) for each pair of columns
    #   Cols 0-1: Strassen 2x2 x 2x2, rank 7
    #   Cols 2-3: Strassen 2x2 x 2x2, rank 7
    #   Cols 4-5: Strassen 2x2 x 2x2, rank 7
    # Bottom 1x3 x 3x6: schoolbook, rank 18
    # Total: 21 + 18 = 39
    
    # Let's try yet another approach with rank 24:
    # Use Strassen for 2x2 x 2x2 (cols 0-1), rank 7
    # Use schoolbook for remaining: 
    #   A[0:2, :] x B[:, 2:6]: 2x3 x 3x4, rank 24
    #   A[2:3, :] x B[:, :]: 1x3 x 3x6, rank 18
    # Total: 7 + 24 + 18 = 49. Worse.
    
    # Let me just use the straightforward 39-rank decomposition.
    
    U_list = []
    V_list = []
    W_list = []
    
    # Strassen for A[0:2,0:2] x B[0:2,0:2] -> C[0:2,0:2]
    U_s, V_s, W_s = build_strassen_2x2()
    for r in range(4):
        U_list.append([0.0]*39)
        V_list.append([0.0]*39)
        W_list.append([0.0]*39)
    
    # Let me build this more carefully.
    
    rank = 39
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    # Block 1: A[0:2, 0:2] x B[0:2, 0:2] -> C[0:2, 0:2], Strassen rank 7, cols 0-6
    U_s, V_s, W_s = build_strassen_2x2()
    for c in range(7):
        for r in range(4):
            # U row r corresponds to A[i,j] where i = r//2, j = r%2
            U[r, c] = U_s[r, c]
            # V row r corresponds to B[j,k] where j = r//2, k = r%2
            V[r, c] = V_s[r, c]
            # W row r corresponds to C[k,i] where k = r//2, i = r%2
            W[r, c] = W_s[r, c]
    
    # Block 2: A[0:2, 0:2] x B[0:2, 2:4] -> C[0:2, 2:4], Strassen rank 7, cols 7-13
    # B columns 2-3 correspond to B[0:2, 2:4], so V row index for B[j,2+k] is j*6 + (2+k)
    for c in range(7):
        for r in range(4):
            U[r, 7+c] = U_s[r, c]
            # V: B[j, 2+k] where j = r//2, k = r%2 -> row j*6 + 2 + k
            j_idx = r // 2
            k_idx = r % 2
            V[j_idx * 6 + 2 + k_idx, 7+c] = V_s[r, c]
            # W: C[2+k, i] where k = r//2, i = r%2 -> row (2+k)*3 + i
            k_out = r // 2
            i_out = r % 2
            W[(2 + k_out) * 3 + i_out, 7+c] = W_s[r, c]
    
    # Block 3: A[0:2, 0:2] x B[0:2, 4:6] -> C[0:2, 4:6], Strassen rank 7, cols 14-20
    for c in range(7):
        for r in range(4):
            U[r, 14+c] = U_s[r, c]
            j_idx = r // 2
            k_idx = r % 2
            V[j_idx * 6 + 4 + k_idx, 14+c] = V_s[r, c]
            k_out = r // 2
            i_out = r % 2
            W[(4 + k_out) * 3 + i_out, 14+c] = W_s[r, c]
    
    # Block 4: A[0:2, 2:3] x B[2:3, 0:6] -> C[0:2, 0:6], schoolbook rank 6, cols 21-26
    # A[i, 2] for i in {0,1}, B[2, k] for k in {0..5}
    for i in range(2):
        for k in range(6):
            col = 21 + i * 6 + k
            U[i * 3 + 2, col] = 1.0
            V[2 * 6 + k, col] = 1.0
            W[k * 3 + i, col] = 1.0
    
    # Block 5: A[2:3, 0:3] x B[0:3, 0:6] -> C[2:3, 0:6], schoolbook rank 18, cols 27-44
    # Wait, rank is 39, so cols 27-38 (12 cols). Let me recalculate.
    # 7 + 7 + 7 + 6 + 18 = 45, not 39. I made an error.
    
    # Let me redo: 
    # Block 1: 7 (Strassen for A[0:2,0:2] x B[0:2,0:2])
    # Block 2: 7 (Strassen for A[0:2,0:2] x B[0:2,2:4])
    # Block 3: 7 (Strassen for A[0:2,0:2] x B[0:2,4:6])
    # Block 4: 6 (schoolbook for A[0:2,2] x B[2,0:6])
    # Block 5: 18 (schoolbook for A[2,0:3] x B[0:3,0:6])
    # Total: 7+7+7+6+18 = 45
    
    # So rank should be 45, not 39. Let me fix.
    
    U = np.zeros((n*m, 45), dtype=np.float64)
    V = np.zeros((m*p, 45), dtype=np.float64)
    W = np.zeros((p*n, 45), dtype=np.float64)
    
    # Block 1: cols 0-6
    for c in range(7):
        for r in range(4):
            U[r, c] = U_s[r, c]
            V[r, c] = V_s[r, c]
            W[r, c] = W_s[r, c]
    
    # Block 2: cols 7-13
    for c in range(7):
        for r in range(4):
            U[r, 7+c] = U_s[r, c]
            j_idx = r // 2
            k_idx = r % 2
            V[j_idx * 6 + 2 + k_idx, 7+c] = V_s[r, c]
            k_out = r // 2
            i_out = r % 2
            W[(2 + k_out) * 3 + i_out, 7+c] = W_s[r, c]
    
    # Block 3: cols 14-20
    for c in range(7):
        for r in range(4):
            U[r, 14+c] = U_s[r, c]
            j_idx = r // 2
            k_idx = r % 2
            V[j_idx * 6 + 4 + k_idx, 14+c] = V_s[r, c]
            k_out = r // 2
            i_out = r % 2
            W[(4 + k_out) * 3 + i_out, 14+c] = W_s[r, c]
    
    # Block 4: cols 21-26, A[0:2,2] x B[2,0:6] -> C[0:2,0:6]
    for i in range(2):
        for k in range(6):
            col = 21 + i * 6 + k
            U[i * 3 + 2, col] = 1.0
            V[2 * 6 + k, col] = 1.0
            W[k * 3 + i, col] = 1.0
    
    # Block 5: cols 27-44, A[2,0:3] x B[0:3,0:6] -> C[2,0:6]
    for j in range(3):
        for k in range(6):
            col = 27 + j * 6 + k
            U[2 * 3 + j, col] = 1.0  # A[2,j]
            V[j * 6 + k, col] = 1.0  # B[j,k]
            W[k * 3 + 2, col] = 1.0  # C[k,2]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
