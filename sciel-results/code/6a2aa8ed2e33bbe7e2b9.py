"""Combined approach: structured Strassen-based decomposition for M_{4,4,7}."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=7
    start_time = time.time()
    deadline = start_time + 140  # Safety margin from 150s limit

    # Build Strassen decomposition for 2x2 x 2x2 (rank 7)
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[p,q],[r,s]]
    # Indices: a=0,b=1,c=2,d=3 for A; e=0,f=1,g=2,h=3 for B; p=0,q=1,r=2,s=3 for C
    def strassen_2x2():
        """Returns U, V, W for 2x2 x 2x2 Strassen decomposition, each 4x7."""
        U = np.zeros((4, 7), dtype=float)
        V = np.zeros((4, 7), dtype=float)
        W = np.zeros((4, 7), dtype=float)
        
        # p1 = (a+d)*(e+h)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # p2 = (c+d)*e
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1
        
        # p3 = a*(f-h)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # p4 = d*(g-e)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # p5 = (a+b)*h
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # p6 = (c-a)*(e+f)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # p7 = (b-d)*(g+h)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    # Build M_{4,4,4} using recursive Strassen (rank 49)
    # Split A into 4 blocks of 2x2: A11, A12, A21, A22
    # Split B into 4 blocks of 2x2: B11, B12, B21, B22
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # Use Strassen for each 2x2x2x2 multiplication
    
    def strassen_4x4x4():
        """Returns U, V, W for M_{4,4,4} using recursive Strassen, rank 49."""
        rank = 49
        U = np.zeros((16, rank), dtype=float)  # 4*4 = 16
        V = np.zeros((16, rank), dtype=float)  # 4*4 = 16
        W = np.zeros((16, rank), dtype=float)  # 4*4 = 16
        
        # Get base Strassen
        U2, V2, W2 = strassen_2x2()
        
        # Block structure:
        # A = [[A11, A12], [A21, A22]] where each block is 2x2
        # B = [[B11, B12], [B21, B22]] where each block is 2x2
        # C = [[C11, C12], [C21, C22]] where each block is 2x2
        #
        # A indices: A11 = rows 0-1, cols 0-1; A12 = rows 0-1, cols 2-3
        #           A21 = rows 2-3, cols 0-1; A22 = rows 2-3, cols 2-3
        # B indices: B11 = rows 0-1, cols 0-1; B12 = rows 0-1, cols 2-3
        #           B21 = rows 2-3, cols 0-1; B22 = rows 2-3, cols 2-3
        # C indices: C11 = rows 0-1, cols 0-1; C12 = rows 0-1, cols 2-3
        #           C21 = rows 2-3, cols 0-1; C22 = rows 2-3, cols 2-3
        #
        # In flat indices (row-major, 4x4):
        # A: A11 = [0,1,4,5], A12 = [2,3,6,7], A21 = [8,9,12,13], A22 = [10,11,14,15]
        # B: B11 = [0,1,4,5], B12 = [2,3,6,7], B21 = [8,9,12,13], B22 = [10,11,14,15]
        # C: C11 = [0,1,4,5], C12 = [2,3,6,7], C21 = [8,9,12,13], C22 = [10,11,14,15]
        
        # Define block index mappings
        # For a 4x4 matrix in row-major order, block (br, bc) where br,bc in {0,1}
        # has elements at positions: br*4 + bc*2 + r*2 + c for r,c in {0,1}
        def block_indices(br, bc, size=4):
            """Get flat indices for block (br, bc) in a size x size matrix."""
            indices = []
            for r in range(2):
                for c in range(2):
                    indices.append(br * size + bc * 2 + r * 2 + c)
            return indices
        
        # Wait, that's not right. For a 4x4 matrix:
        # Row br*2, col bc*2: index = br*2*4 + bc*2
        # Row br*2, col bc*2+1: index = br*2*4 + bc*2 + 1
        # Row br*2+1, col bc*2: index = (br*2+1)*4 + bc*2
        # Row br*2+1, col bc*2+1: index = (br*2+1)*4 + bc*2 + 1
        
        def block_indices_correct(br, bc, size=4):
            """Get flat indices for 2x2 block (br, bc) in a size x size matrix."""
            indices = []
            for r in range(2):
                for c in range(2):
                    row = br * 2 + r
                    col = bc * 2 + c
                    indices.append(row * size + col)
            return indices
        
        # A blocks
        A11_idx = block_indices_correct(0, 0)  # [0, 1, 4, 5]
        A12_idx = block_indices_correct(0, 1)  # [2, 3, 6, 7]
        A21_idx = block_indices_correct(1, 0)  # [8, 9, 12, 13]
        A22_idx = block_indices_correct(1, 1)  # [10, 11, 14, 15]
        
        # B blocks
        B11_idx = block_indices_correct(0, 0)  # [0, 1, 4, 5]
        B12_idx = block_indices_correct(0, 1)  # [2, 3, 6, 7]
        B21_idx = block_indices_correct(1, 0)  # [8, 9, 12, 13]
        B22_idx = block_indices_correct(1, 1)  # [10, 11, 14, 15]
        
        # C blocks
        C11_idx = block_indices_correct(0, 0)  # [0, 1, 4, 5]
        C12_idx = block_indices_correct(0, 1)  # [2, 3, 6, 7]
        C21_idx = block_indices_correct(1, 0)  # [8, 9, 12, 13]
        C22_idx = block_indices_correct(1, 1)  # [10, 11, 14, 15]
        
        # Now build the decomposition
        # C11 = A11*B11 + A12*B21
        # C12 = A11*B12 + A12*B22
        # C21 = A21*B11 + A22*B21
        # C22 = A21*B12 + A22*B22
        #
        # Each product (e.g., A11*B11) is a 2x2 x 2x2 multiplication
        # Using Strassen, each uses 7 rank-1 terms
        # Total: 8 products * 7 = 56 terms
        #
        # But we can share computations! The standard recursive Strassen
        # for 4x4 x 4x4 uses 7^2 = 49 terms by applying Strassen at the top level.
        
        # Let me use the top-level Strassen approach:
        # M = [[A11, A12], [A21, A22]] * [[B11, B12], [B21, B22]]
        # = [[C11, C12], [C21, C22]]
        #
        # Strassen:
        # P1 = (A11 + A22) * (B11 + B22)
        # P2 = (A21 + A22) * B11
        # P3 = A11 * (B12 - B22)
        # P4 = A22 * (B21 - B11)
        # P5 = (A11 + A12) * B22
        # P6 = (A21 - A11) * (B11 + B12)
        # P7 = (A12 - A22) * (B21 + B22)
        #
        # C11 = P1 + P4 - P5 + P7
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 + P3 + P6
        #
        # Each Pi is a 2x2 x 2x2 multiplication, which we decompose using Strassen (rank 7)
        # Total: 7 * 7 = 49 rank-1 terms
        
        col = 0
        # P1 = (A11 + A22) * (B11 + B22)
        # A11 + A22: elements at A11_idx and A22_idx
        # B11 + B22: elements at B11_idx and B22_idx
        # Output goes to C11, C12, C21, C22 with coefficients
        for r in range(7):
            # U contribution: A11 + A22
            for idx in A11_idx:
                U[idx, col] += U2[idx % 4, r]
            for idx in A22_idx:
                U[idx, col] += U2[idx % 4, r]
            # V contribution: B11 + B22
            for idx in B11_idx:
                V[idx, col] += V2[idx % 4, r]
            for idx in B22_idx:
                V[idx, col] += V2[idx % 4, r]
            # W contribution: P1 contributes to C11 (+1), C12 (0), C21 (0), C22 (-1)
            for idx in C11_idx:
                W[idx, col] += W2[idx % 4, r]
            for idx in C22_idx:
                W[idx, col] -= W2[idx % 4, r]
            col += 1
        
        # P2 = (A21 + A22) * B11
        for r in range(7):
            # U contribution: A21 + A22
            for idx in A21_idx:
                U[idx, col] += U2[idx % 4, r]
            for idx in A22_idx:
                U[idx, col] += U2[idx % 4, r]
            # V contribution: B11
            for idx in B11_idx:
                V[idx, col] += V2[idx % 4, r]
            # W contribution: P2 contributes to C21 (+1), C22 (-1)
            for idx in C21_idx:
                W[idx, col] += W2[idx % 4, r]
            for idx in C22_idx:
                W[idx, col] -= W2[idx % 4, r]
            col += 1
        
        # P3 = A11 * (B12 - B22)
        for r in range(7):
            # U contribution: A11
            for idx in A11_idx:
                U[idx, col] += U2[idx % 4, r]
            # V contribution: B12 - B22
            for idx in B12_idx:
                V[idx, col] += V2[idx % 4, r]
            for idx in B22_idx:
                V[idx, col] -= V2[idx % 4, r]
            # W contribution: P3 contributes to C12 (+1), C22 (+1)
            for idx in C12_idx:
                W[idx, col] += W2[idx % 4, r]
            for idx in C22_idx:
                W[idx, col] += W2[idx % 4, r]
            col += 1
        
        # P4 = A22 * (B21 - B11)
        for r in range(7):
            # U contribution: A22
            for idx in A22_idx:
                U[idx, col] += U2[idx % 4, r]
            # V contribution: B21 - B11
            for idx in B21_idx:
                V[idx, col] += V2[idx % 4, r]
            for idx in B11_idx:
                V[idx, col] -= V2[idx % 4, r]
            # W contribution: P4 contributes to C11 (+1), C21 (+1)
            for idx in C11_idx:
                W[idx, col] += W2[idx % 4, r]
            for idx in C21_idx:
                W[idx, col] += W2[idx % 4, r]
            col += 1
        
        # P5 = (A11 + A12) * B22
        for r in range(7):
            # U contribution: A11 + A12
            for idx in A11_idx:
                U[idx, col] += U2[idx % 4, r]
            for idx in A12_idx:
                U[idx, col] += U2[idx % 4, r]
            # V contribution: B22
            for idx in B22_idx:
                V[idx, col] += V2[idx % 4, r]
            # W contribution: P5 contributes to C11 (-1), C12 (+1)
            for idx in C11_idx:
                W[idx, col] -= W2[idx % 4, r]
            for idx in C12_idx:
                W[idx, col] += W2[idx % 4, r]
            col += 1
        
        # P6 = (A21 - A11) * (B11 + B12)
        for r in range(7):
            # U contribution: A21 - A11
            for idx in A21_idx:
                U[idx, col] += U2[idx % 4, r]
            for idx in A11_idx:
                U[idx, col] -= U2[idx % 4, r]
            # V contribution: B11 + B12
            for idx in B11_idx:
                V[idx, col] += V2[idx % 4, r]
            for idx in B12_idx:
                V[idx, col] += V2[idx % 4, r]
            # W contribution: P6 contributes to C22 (+1)
            for idx in C22_idx:
                W[idx, col] += W2[idx % 4, r]
            col += 1
        
        # P7 = (A12 - A22) * (B21 + B22)
        for r in range(7):
            # U contribution: A12 - A22
            for idx in A12_idx:
                U[idx, col] += U2[idx % 4, r]
            for idx in A22_idx:
                U[idx, col] -= U2[idx % 4, r]
            # V contribution: B21 + B22
            for idx in B21_idx:
                V[idx, col] += V2[idx % 4, r]
            for idx in B22_idx:
                V[idx, col] += V2[idx % 4, r]
            # W contribution: P7 contributes to C11 (+1)
            for idx in C11_idx:
                W[idx, col] += W2[idx % 4, r]
            col += 1
        
        assert col == 49, f"Expected 49 columns, got {col}"
        return U, V, W

    # Build M_{4,4,3} using schoolbook (rank 48)
    # A is 4x4, B is 4x3, C is 4x3
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Total terms: 4 * 4 * 3 = 48
    
    def schoolbook_4x4x3():
        """Returns U, V, W for M_{4,4,3} using schoolbook, rank 48."""
        rank = 48
        U = np.zeros((16, rank), dtype=float)  # 4*4 = 16
        V = np.zeros((12, rank), dtype=float)  # 4*3 = 12
        W = np.zeros((12, rank), dtype=float)  # 3*4 = 12
        
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    U[i * 4 + j, col] = 1
                    V[j * 3 + k, col] = 1
                    W[k * 4 + i, col] = 1
                    col += 1
        
        assert col == 48, f"Expected 48 columns, got {col}"
        return U, V, W

    # Combine M_{4,4,4} and M_{4,4,3}
    # A is 4x4, B is 4x7
    # Split B into B1 (4x4) and B2 (4x3)
    # C = A*B1 + A*B2
    # C1 (4x4) from A*B1, C2 (4x3) from A*B2
    
    U1, V1, W1 = strassen_4x4x4()
    U2, V2, W2 = schoolbook_4x4x3()
    
    # Combine into full matrices
    # U: 16 x (49 + 48) = 16 x 97
    # V: 28 x (49 + 48) = 28 x 97  (4*7 = 28)
    # W: 28 x (49 + 48) = 28 x 97  (7*4 = 28)
    
    rank_total = 49 + 48
    U_full = np.zeros((16, rank_total), dtype=float)
    V_full = np.zeros((28, rank_total), dtype=float)
    W_full = np.zeros((28, rank_total), dtype=float)
    
    # First 49 columns: M_{4,4,4} part
    U_full[:, :49] = U1
    V_full[:16, :49] = V1  # B1 is 4x4, indices 0-15
    W_full[:16, :49] = W1  # C1 is 4x4, indices 0-15
    
    # Next 48 columns: M_{4,4,3} part
    U_full[:, 49:] = U2
    V_full[16:28, 49:] = V2  # B2 is 4x3, indices 16-27
    W_full[16:28, 49:] = W2  # C2 is 4x3, indices 16-27
    
    # Convert to lists
    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()
    
    # Verify the solution
    def verify_solution(U, V, W):
        """Verify that U, V, W form an exact decomposition."""
        rank = len(U[0])
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check all target entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.sum(U_np[u_idx, :] * V_np[v_idx, :] * W_np[w_idx, :])
                    if abs(val - 1.0) > 1e-9:
                        return False, f"Target entry ({i},{j},{k}) = {val}, expected 1.0"
        
        # Check some non-target entries
        import random
        random.seed(42)
        for _ in range(1000):
            u_idx = random.randint(0, n * m - 1)
            v_idx = random.randint(0, m * p - 1)
            w_idx = random.randint(0, p * n - 1)
            
            # Check if this is a target entry
            i = u_idx // m
            j = u_idx % m
            j2 = v_idx // p
            k = v_idx % p
            k2 = w_idx // n
            i2 = w_idx % n
            
            if i == i2 and j == j2 and k == k2:
                continue  # This is a target entry
            
            val = np.sum(U_np[u_idx, :] * V_np[v_idx, :] * W_np[w_idx, :])
            if abs(val) > 1e-9:
                return False, f"Non-target entry ({u_idx},{v_idx},{w_idx}) = {val}, expected 0.0"
        
        return True, "OK"
    
    valid, msg = verify_solution(U_list, V_list, W_list)
    if not valid:
        print(f"Verification failed: {msg}")
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
