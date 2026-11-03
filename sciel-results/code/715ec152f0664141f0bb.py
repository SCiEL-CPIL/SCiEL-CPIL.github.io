"""Matrix multiplication 3x4x4: find rank-minimal tensor decomposition."""
import time

DIMENSIONS = [3, 4, 4]
n, m, p = DIMENSIONS

def construct():
    """
    Construct a tensor decomposition for 3x4 * 4x4 matmul.
    Uses Strassen's algorithm on 2x2 blocks for the 2x4 part (rank 28)
    and naive multiplication for the 1x4 part (rank 16).
    Total rank: 44.
    """
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a Strassen 2x2x2 decomposition
    # X is 2x2, Y is 2x2, M = XY
    # Adds to U, V, W at specified column offset
    # x_indices: list of 4 (row, col) indices in A for X
    # y_indices: list of 4 (row, col) indices in B for Y
    # c_indices: list of 4 (row, col) indices in C for M
    # offset: column offset in U, V, W
    def add_strassen(x_idx, y_idx, c_idx, offset):
        # x_idx: [[r0c0, r0c1], [r1c0, r1c1]] flattened or structured
        # Let's assume x_idx is [A_idx00, A_idx01, A_idx10, A_idx11]
        # y_idx is [B_idx00, B_idx01, B_idx10, B_idx11]
        # c_idx is [C_idx00, C_idx01, C_idx10, C_idx11]
        
        x00, x01, x10, x11 = x_idx
        y00, y01, y10, y11 = y_idx
        c00, c01, c10, c11 = c_idx
        
        # M1 = (x00 + x11)(y00 + y11)
        # M2 = (x10 + x11)y00
        # M3 = x00(y10 - y11)
        # M4 = x11(y01 - y00)
        # M5 = (x00 + x01)y11
        # M6 = (x10 - x00)(y00 + y01)
        # M7 = (x01 - x11)(y10 + y11)
        
        # M1
        U[x00][offset] += 1
        U[x11][offset] += 1
        V[y00][offset] += 1
        V[y11][offset] += 1
        W[c00][offset] += 1
        W[c01][offset] += 1
        W[c10][offset] += 1
        W[c11][offset] += 1
        
        # M2
        U[x10][offset+1] += 1
        U[x11][offset+1] += 1
        V[y00][offset+1] += 1
        W[c10][offset+1] += 1
        W[c11][offset+1] += -1
        
        # M3
        U[x00][offset+2] += 1
        V[y10][offset+2] += 1
        V[y11][offset+2] += -1
        W[c01][offset+2] += 1
        W[c11][offset+2] += -1
        
        # M4
        U[x11][offset+3] += 1
        V[y01][offset+3] += 1
        V[y00][offset+3] += -1
        W[c00][offset+3] += 1
        W[c10][offset+3] += 1
        
        # M5
        U[x00][offset+4] += 1
        U[x01][offset+4] += 1
        V[y11][offset+4] += 1
        W[c00][offset+4] += -1
        W[c01][offset+4] += 1
        
        # M6
        U[x10][offset+5] += 1
        U[x00][offset+5] += -1
        V[y00][offset+5] += 1
        V[y01][offset+5] += 1
        W[c11][offset+5] += -1
        
        # M7
        U[x01][offset+6] += 1
        U[x11][offset+6] += -1
        V[y10][offset+6] += 1
        V[y11][offset+6] += 1
        W[c00][offset+6] += 1
        W[c11][offset+6] += -1

    # Top part: 2x4 * 4x4
    # A_top rows 0, 1
    # A_00: A[0,0], A[0,1], A[1,0], A[1,1] -> indices 0, 1, 4, 5
    # A_01: A[0,2], A[0,3], A[1,2], A[1,3] -> indices 2, 3, 6, 7
    # B is 4x4
    # B_00: B[0,0], B[0,1], B[1,0], B[1,1] -> indices 0, 1, 4, 5
    # B_10: B[2,0], B[2,1], B[3,0], B[3,1] -> indices 8, 9, 12, 13
    # B_01: B[0,2], B[0,3], B[1,2], B[1,3] -> indices 2, 3, 6, 7
    # B_11: B[2,2], B[2,3], B[3,2], B[3,3] -> indices 10, 11, 14, 15
    
    # C_top is 2x4
    # C[0,0], C[0,1], C[0,2], C[0,3] -> indices 0, 1, 2, 3 (in W, p*n = 4*3=12, so W index is k*n+i? No, W[k*n+i,r].
    # W index for C[i,k] is k*n + i.
    # C[0,0] -> W[0*3+0] = W[0]
    # C[0,1] -> W[1*3+0] = W[3]
    # C[0,2] -> W[2*3+0] = W[6]
    # C[0,3] -> W[3*3+0] = W[9]
    # C[1,0] -> W[0*3+1] = W[1]
    # C[1,1] -> W[1*3+1] = W[4]
    # C[1,2] -> W[2*3+1] = W[7]
    # C[1,3] -> W[3*3+1] = W[10]
    
    # P1 = A_00 B_00
    # C_contrib_P1:
    # (P1)_00 adds to C[0,0] (W[0])
    # (P1)_01 adds to C[0,1] (W[3])
    # (P1)_10 adds to C[1,0] (W[1])
    # (P1)_11 adds to C[1,1] (W[4])
    
    # P2 = A_01 B_10
    # (P2)_00 adds to C[0,0] (W[0])
    # (P2)_01 adds to C[0,1] (W[3])
    # (P2)_10 adds to C[1,0] (W[1])
    # (P2)_11 adds to C[1,1] (W[4])
    
    # P3 = A_00 B_01
    # (P3)_00 adds to C[0,2] (W[6])
    # (P3)_01 adds to C[0,3] (W[9])
    # (P3)_10 adds to C[1,2] (W[7])
    # (P3)_11 adds to C[1,3] (W[10])
    
    # P4 = A_01 B_11
    # (P4)_00 adds to C[0,2] (W[6])
    # (P4)_01 adds to C[0,3] (W[9])
    # (P4)_10 adds to C[1,2] (W[7])
    # (P4)_11 adds to C[1,3] (W[10])
    
    # We need to combine the contributions.
    # The add_strassen function adds to specific W indices.
    # We cannot just call it 4 times because the W indices overlap.
    # We must manually construct the U, V, W entries for the 28 rank-1 tensors.
    
    # Let's define the 28 components explicitly.
    # Each component j (0..27) corresponds to one of the 7 Strassen steps for one of the 4 P's.
    
    # P1 (offset 0)
    # x: A[0,0]=0, A[0,1]=1, A[1,0]=4, A[1,1]=5
    # y: B[0,0]=0, B[0,1]=1, B[1,0]=4, B[1,1]=5
    # c: C[0,0]=0, C[0,1]=3, C[1,0]=1, C[1,1]=4
    
    # P2 (offset 7)
    # x: A[0,2]=2, A[0,3]=3, A[1,2]=6, A[1,3]=7
    # y: B[2,0]=8, B[2,1]=9, B[3,0]=12, B[3,1]=13
    # c: C[0,0]=0, C[0,1]=3, C[1,0]=1, C[1,1]=4
    
    # P3 (offset 14)
    # x: A[0,0]=0, A[0,1]=1, A[1,0]=4, A[1,1]=5
    # y: B[0,2]=2, B[0,3]=3, B[1,2]=6, B[1,3]=7
    # c: C[0,2]=6, C[0,3]=9, C[1,2]=7, C[1,3]=10
    
    # P4 (offset 21)
    # x: A[0,2]=2, A[0,3]=3, A[1,2]=6, A[1,3]=7
    # y: B[2,2]=10, B[2,3]=11, B[3,2]=14, B[3,3]=15
    # c: C[0,2]=6, C[0,3]=9, C[1,2]=7, C[1,3]=10
    
    # We will use a helper to fill U, V, W for a specific Strassen instance
    def fill_strassen(x_idx, y_idx, c_idx, base_col):
        # x_idx, y_idx, c_idx are lists of 4 indices
        # c_idx order: [C00, C01, C10, C11]
        # Strassen steps:
        # 1: (x00+x11)(y00+y11) -> C00+, C01+, C10+, C11+
        # 2: (x10+x11)y00 -> C10+, C11-
        # 3: x00(y10-y11) -> C01+, C11-
        # 4: x11(y01-y00) -> C00+, C10+
        # 5: (x00+x01)y11 -> C00-, C01+
        # 6: (x10-x00)(y00+y01) -> C11-
        # 7: (x01-x11)(y10+y11) -> C00+, C11-
        
        x00, x01, x10, x11 = x_idx
        y00, y01, y10, y11 = y_idx
        c00, c01, c10, c11 = c_idx
        
        # Step 1
        U[x00][base_col] += 1
        U[x11][base_col] += 1
        V[y00][base_col] += 1
        V[y11][base_col] += 1
        W[c00][base_col] += 1
        W[c01][base_col] += 1
        W[c10][base_col] += 1
        W[c11][base_col] += 1
        
        # Step 2
        U[x10][base_col+1] += 1
        U[x11][base_col+1] += 1
        V[y00][base_col+1] += 1
        W[c10][base_col+1] += 1
        W[c11][base_col+1] += -1
        
        # Step 3
        U[x00][base_col+2] += 1
        V[y10][base_col+2] += 1
        V[y11][base_col+2] += -1
        W[c01][base_col+2] += 1
        W[c11][base_col+2] += -1
        
        # Step 4
        U[x11][base_col+3] += 1
        V[y01][base_col+3] += 1
        V[y00][base_col+3] += -1
        W[c00][base_col+3] += 1
        W[c10][base_col+3] += 1
        
        # Step 5
        U[x00][base_col+4] += 1
        U[x01][base_col+4] += 1
        V[y11][base_col+4] += 1
        W[c00][base_col+4] += -1
        W[c01][base_col+4] += 1
        
        # Step 6
        U[x10][base_col+5] += 1
        U[x00][base_col+5] += -1
        V[y00][base_col+5] += 1
        V[y01][base_col+5] += 1
        W[c11][base_col+5] += -1
        
        # Step 7
        U[x01][base_col+6] += 1
        U[x11][base_col+6] += -1
        V[y10][base_col+6] += 1
        V[y11][base_col+6] += 1
        W[c00][base_col+6] += 1
        W[c11][base_col+6] += -1

    # Define indices
    # A indices: i*m + j
    # B indices: j*p + k
    # W indices: k*n + i
    
    # P1
    x1 = [0, 1, 4, 5] # A[0,0], A[0,1], A[1,0], A[1,1]
    y1 = [0, 1, 4, 5] # B[0,0], B[0,1], B[1,0], B[1,1]
    c1 = [0, 3, 1, 4] # C[0,0], C[0,1], C[1,0], C[1,1]
    fill_strassen(x1, y1, c1, 0)
    
    # P2
    x2 = [2, 3, 6, 7] # A[0,2], A[0,3], A[1,2], A[1,3]
    y2 = [8, 9, 12, 13] # B[2,0], B[2,1], B[3,0], B[3,1]
    c2 = [0, 3, 1, 4] # C[0,0], C[0,1], C[1,0], C[1,1]
    fill_strassen(x2, y2, c2, 7)
    
    # P3
    x3 = [0, 1, 4, 5]
    y3 = [2, 3, 6, 7] # B[0,2], B[0,3], B[1,2], B[1,3]
    c3 = [6, 9, 7, 10] # C[0,2], C[0,3], C[1,2], C[1,3]
    fill_strassen(x3, y3, c3, 14)
    
    # P4
    x4 = [2, 3, 6, 7]
    y4 = [10, 11, 14, 15] # B[2,2], B[2,3], B[3,2], B[3,3]
    c4 = [6, 9, 7, 10]
    fill_strassen(x4, y4, c4, 21)
    
    # Bottom part: 1x4 * 4x4
    # A[2,0]=8, A[2,1]=9, A[2,2]=10, A[2,3]=11
    # B rows:
    # B[0,0]=0, B[0,1]=1, B[0,2]=2, B[0,3]=3
    # B[1,0]=4, B[1,1]=5, B[1,2]=6, B[1,3]=7
    # B[2,0]=8, B[2,1]=9, B[2,2]=10, B[2,3]=11
    # B[3,0]=12, B[3,1]=13, B[3,2]=14, B[3,3]=15
    # C[2,0]=W[0*3+2]=W[2]
    # C[2,1]=W[1*3+2]=W[5]
    # C[2,2]=W[2*3+2]=W[8]
    # C[2,3]=W[3*3+2]=W[11]
    
    # For each k (0..3) and j (0..3):
    # Term: A[2,k] * B[k,j] contributes to C[2,j]
    # U[2*m+k] = U[8+k]
    # V[k*p+j] = V[4k+j]
    # W[j*n+2] = W[3j+2]
    
    col = 28
    for k in range(4):
        for j in range(4):
            U[8 + k][col] = 1
            V[4 * k + j][col] = 1
            W[3 * j + 2][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
