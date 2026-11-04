"""Matrix multiplication 4x4x4 via Kronecker product of Strassen rank-7."""
import time

DIMENSIONS = [4, 4, 4]

# Strassen decomposition for 2x2x2 matrix multiplication, rank 7.
# U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
# The 7 products are:
# M1 = (A11 - A22) * (B22 + B11)
# M2 = (A21 + A22) * B11
# M3 = A11 * (B12 - B22)
# M4 = A22 * (B21 - B11)
# M5 = (A11 + A12) * B22
# M6 = (A21 - A11) * (B11 + B12)
# M7 = (A12 - A22) * (B21 + B22)
#
# C11 = M1 + M4 - M5 + M7
# C12 = M3 + M5
# C21 = M2 + M4
# C22 = M1 - M2 + M3 + M6

def _strassen_2x2x2():
    """Return U, V, W (lists of lists) for 2x2x2 Strassen rank-7 decomposition."""
    # U has shape (2*2, 7) = (4, 7)
    # V has shape (2*2, 7) = (4, 7)
    # W has shape (2*2, 7) = (4, 7)
    #
    # Indexing: U[i*m + j, c] where m=2, so row index = i*2 + j
    # V[j*p + k, c] where p=2, so row index = j*2 + k
    # W[k*n + i, c] where n=2, so row index = k*2 + i
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A11 - A22) * (B22 + B11)
    # A11: i=0,j=0 -> U[0][0] = 1; A22: i=1,j=1 -> U[3][0] = -1
    # B11: j=0,k=0 -> V[0][0] = 1; B22: j=1,k=1 -> V[3][0] = 1
    # C11: k=0,i=0 -> W[0][0] = 1; C22: k=1,i=1 -> W[3][0] = 1
    U[0][0] = 1;  U[3][0] = -1
    V[0][0] = 1;  V[3][0] = 1
    W[0][0] = 1;  W[3][0] = 1
    
    # M2 = (A21 + A22) * B11
    # A21: i=1,j=0 -> U[2][1] = 1; A22: i=1,j=1 -> U[3][1] = 1
    # B11: j=0,k=0 -> V[0][1] = 1
    # C21: k=0,i=1 -> W[2][1] = 1
    U[2][1] = 1;  U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3 = A11 * (B12 - B22)
    # A11: i=0,j=0 -> U[0][2] = 1
    # B12: j=0,k=1 -> V[1][2] = 1; B22: j=1,k=1 -> V[3][2] = -1
    # C12: k=1,i=0 -> W[1][2] = 1
    U[0][2] = 1
    V[1][2] = 1;  V[3][2] = -1
    W[1][2] = 1
    
    # M4 = A22 * (B21 - B11)
    # A22: i=1,j=1 -> U[3][3] = 1
    # B11: j=0,k=0 -> V[0][3] = -1; B21: j=1,k=0 -> V[2][3] = 1
    # C11: k=0,i=0 -> W[0][3] = 1; C21: k=0,i=1 -> W[2][3] = 1
    U[3][3] = 1
    V[0][3] = -1;  V[2][3] = 1
    W[0][3] = 1;  W[2][3] = 1
    
    # M5 = (A11 + A12) * B22
    # A11: i=0,j=0 -> U[0][4] = 1; A12: i=0,j=1 -> U[1][4] = 1
    # B22: j=1,k=1 -> V[3][4] = 1
    # C11: k=0,i=0 -> W[0][4] = -1; C12: k=1,i=0 -> W[1][4] = 1
    U[0][4] = 1;  U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1;  W[1][4] = 1
    
    # M6 = (A21 - A11) * (B11 + B12)
    # A11: i=0,j=0 -> U[0][5] = -1; A21: i=1,j=0 -> U[2][5] = 1
    # B11: j=0,k=0 -> V[0][5] = 1; B12: j=0,k=1 -> V[1][5] = 1
    # C22: k=1,i=1 -> W[3][5] = 1
    U[0][5] = -1;  U[2][5] = 1
    V[0][5] = 1;  V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A12 - A22) * (B21 + B22)
    # A12: i=0,j=1 -> U[1][6] = 1; A22: i=1,j=1 -> U[3][6] = -1
    # B21: j=1,k=0 -> V[2][6] = 1; B22: j=1,k=1 -> V[3][6] = 1
    # C11: k=0,i=0 -> W[0][6] = 1
    U[1][6] = 1;  U[3][6] = -1
    V[2][6] = 1;  V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W


def construct():
    """Construct rank-49 decomposition for 4x4x4 via Kronecker product of Strassen."""
    deadline = time.time() + 145  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 4
    # Kronecker product: rank = 7 * 7 = 49
    # U shape: (n*m, rank) = (16, 49)
    # V shape: (m*p, rank) = (16, 49)
    # W shape: (p*n, rank) = (16, 49)
    
    U_s, V_s, W_s = _strassen_2x2x2()
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Kronecker product structure:
    # The 4x4 matrix A is partitioned into four 2x2 blocks:
    # A = [[A00, A01], [A10, A11]]
    # where Aij has entries A[i*2+a, j*2+b] for a,b in {0,1}
    #
    # In the schoolbook indexing U[i*m + j, c]:
    # For 4x4: row index = i*4 + j where i in {0,1,2,3}, j in {0,1,2,3}
    # Block row = i // 2, block col = j // 2
    # Local row = i % 2, local col = j % 2
    #
    # The Kronecker product of two rank-7 decompositions:
    # For each pair (c1, c2) of Strassen indices (0..6), the combined index is c1*7 + c2
    # U[i*4+j, c1*7+c2] = U_s[(i%2)*2 + (j%2), c1] * U_s[(i//2)*2 + (j//2), c2]
    # V[j*4+k, c1*7+c2] = V_s[(j%2)*2 + (k%2), c1] * V_s[(j//2)*2 + (k//2), c2]
    # W[k*4+i, c1*7+c2] = W_s[(k%2)*2 + (i%2), c1] * W_s[(k//2)*2 + (i//2), c2]
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            for i in range(4):
                for j in range(4):
                    # U
                    u_val = U_s[(i % 2) * 2 + (j % 2)][c1] * U_s[(i // 2) * 2 + (j // 2)][c2]
                    if u_val != 0:
                        U[i * 4 + j][c] = u_val
            for j in range(4):
                for k in range(4):
                    # V
                    v_val = V_s[(j % 2) * 2 + (k % 2)][c1] * V_s[(j // 2) * 2 + (k // 2)][c2]
                    if v_val != 0:
                        V[j * 4 + k][c] = v_val
            for k in range(4):
                for i in range(4):
                    # W
                    w_val = W_s[(k % 2) * 2 + (i % 2)][c1] * W_s[(k // 2) * 2 + (i // 2)][c2]
                    if w_val != 0:
                        W[k * 4 + i][c] = w_val
    
    if time.time() > deadline:
        # Should not happen, but return what we have
        pass
    
    return {"U": U, "V": V, "W": W}
