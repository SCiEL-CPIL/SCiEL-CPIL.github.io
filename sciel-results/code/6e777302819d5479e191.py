import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    # Target: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Need: sum_r U_full[i*m+j, r] * V_full[j*p+k, r] * W_full[k*n+i, r] = 1
    
    # Use Strassen-based construction for rank 42
    # Strategy: View 4x4 intermediate as 2x2 blocks, 6 columns as 2x3 blocks
    # Each Strassen multiplication on 2x2 blocks gives 7 scalar mults
    # With 2 row blocks and 3 col blocks: 7 * 2 * 3 = 42
    
    rank = 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen's 7 multiplications for 2x2 block matrices:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For our case: A is 2x4, B is 4x6
    # Split A into blocks: A = [A_top; A_bottom] where A_top is 1x4, A_bottom is 1x4
    # Actually, let's split the 4 intermediate dimension into 2x2:
    # A[i, j] where j = 0,1,2,3 -> split into j0=(0,1) and j1=(2,3)
    # B[j, k] where j = 0,1,2,3 -> same split, k = 0,1,2,3,4,5 -> split into k0=(0,1,2) and k1=(3,4,5)
    
    # So A has blocks: A[i, j0] (1x2) and A[i, j1] (1x2) for i=0,1
    # B has blocks: B[j0, k0] (2x3), B[j0, k1] (2x3), B[j1, k0] (2x3), B[j1, k1] (2x3)
    
    # C[i, k] = sum over j of A[i,j]*B[j,k]
    # = sum over j0 of A[i,j0]*B[j0,k] + sum over j1 of A[i,j1]*B[j1,k]
    
    # For each output block C[i, kb] where kb in {0,1} (each 3 wide):
    # C[i, kb] = A[i, j0] * B[j0, kb] + A[i, j1] * B[j1, kb]
    
    # This is a 1x2 times 2x3 multiplication for each (i, kb)
    # We can use Strassen-like decomposition on the 2 intermediate blocks
    
    # Let's define for each (i, kb):
    # Let a0 = A[i, 0:2] (1x2), a1 = A[i, 2:4] (1x2)
    # Let b0 = B[0:2, kb*3:(kb+1)*3] (2x3), b1 = B[2:4, kb*3:(kb+1)*3] (2x3)
    # C[i, kb] = a0 * b0 + a1 * b1 (1x3 result)
    
    # Using 3 multiplications (Strassen for 1x2 * 2x2 * 2x1 type):
    # Actually for 1x2 times 2x3, we need at least 3 mults (schoolbook is 6)
    # Let's use: M1 = a0 * b0, M2 = a1 * b1, M3 = (a0+a1) * (b0+b1)
    # Then C = M1 + M2 - M3/2? No, that's not right.
    # Actually: (a0+a1)(b0+b1) = a0*b0 + a0*b1 + a1*b0 + a1*b1
    # So M1 + M2 = a0*b0 + a1*b1, and M3 = a0*b0 + a0*b1 + a1*b0 + a1*b1
    # We can't get C from just these 3 without additional terms.
    
    # Let me use a different approach. Since n=2, m=4, p=6:
    # Schoolbook: 2*4*6 = 48 multiplications
    # We want to reduce this.
    
    # Alternative: Use the fact that we have 2 rows in A.
    # C[0,k] = sum_j A[0,j]*B[j,k]
    # C[1,k] = sum_j A[1,j]*B[j,k]
    
    # For each k, we compute two dot products of length 4.
    # We can share computations between the two rows.
    
    # Let's use a 2x4 x 4x6 specific decomposition.
    # Group the 4 intermediate indices into pairs: (0,1) and (2,3)
    # Group the 6 output indices into triples: (0,1,2) and (3,4,5)
    
    # For block (i, j_block, k_block) where j_block in {0,1}, k_block in {0,1}:
    # A_block[i, j_block] is 1x2, B_block[j_block, k_block] is 2x3
    # C_block[i, k_block] = A_block[i, j0] * B_block[j0, k_block] + A_block[i, j1] * B_block[j1, k_block]
    
    # For each (i, k_block), we have a 1x2 times 2x3 multiplication.
    # Minimum rank for 1x2 x 2x3 is... let's think.
    # The tensor is T[a, b, c] = 1 where a in {0,1} (A cols), b in {0,1} (B rows), c in {0,1,2} (B cols)
    # This is the same as multiplying a 1x2 by 2x3 matrix.
    # Schoolbook: 2*3 = 6 multiplications.
    # Can we do better? The border rank of 1x2 x 2x3 is 5 (I think).
    # But we need exact integer/half-integer decomposition.
    
    # Let's just use 6 per block: 2 i-values * 2 j-blocks * 2 k-blocks * 6 = 48 (schoolbook)
    # Or use 5 per block if possible: 2 * 2 * 2 * 5 = 40
    
    # Actually, let me try a direct construction with rank 42.
    # 42 = 7 * 6. Use Strassen's 7 mults for a 2x2 x 2x2 core, expanded to handle the 6 columns.
    
    # Let's try: split m=4 into 2x2, and handle p=6 as 3 groups of 2.
    # For each pair of output columns (k, k+1) where k=0,2,4:
    #   C[:, k:k+2] = A * B[:, k:k+2]
    #   This is 2x4 times 4x2.
    #   Split 4 into 2x2: A = [A11 A12; A21 A22] (2x2 blocks of 1x1... no, A is 2x4)
    #   A is 2x4, so blocks are 1x2 each. B[:, k:k+2] is 4x2, blocks are 2x1 each.
    
    # This is getting complex. Let me just use a verified rank-42 construction.
    
    # Simple approach: 7 Strassen multiplications on 2x2 blocks, each producing a 2x3 result.
    # Total: 7 * 2 * 3 = 42.
    
    # Define the Strassen matrices for the 2x2 block decomposition of the 4 intermediate dimension:
    # A blocks: A[i, 0:2] and A[i, 2:4] for i=0,1 -> 2x2 matrix of 1x2 blocks
    # B blocks: B[0:2, :] and B[2:4, :] -> 2x2 matrix of 2x6 blocks
    
    # Strassen multiplications (each is a product of a 1x2 block and a 2x6 block, giving 1x6):
    # M1 = (A[0,0:2] + A[1,2:4]) * (B[0:2,:] + B[2:4,:])
    # M2 = (A[1,0:2] + A[1,2:4]) * B[0:2,:]
    # M3 = A[0,0:2] * (B[0:2,:] - B[2:4,:])
    # M4 = A[1,2:4] * (B[2:4,:] - B[0:2,:])
    # M5 = (A[0,0:2] + A[0,2:4]) * B[2:4,:]
    # M6 = (A[1,0:2] - A[0,0:2]) * (B[0:2,:] + B[0:2,2:4])  # This doesn't work for non-square
    
    # The issue is Strassen assumes square blocks. Let me adapt.
    
    # For non-square blocks, we can still use 7 multiplications:
    # Let A = [A11 A12; A21 A22] where each block is 1x2
    # Let B = [B11 B12; B21 B22] where each block is 2x6
    # C = A*B = [A11*B11 + A12*B21, A11*B12 + A12*B22; A21*B11 + A22*B21, A21*B12 + A22*B22]
    
    # 7 multiplications:
    # M1 = (A11+A22) * (B11+B22)  -> 1x6
    # M2 = (A21+A22) * B11        -> 1x6
    # M3 = A11 * (B12-B22)        -> 1x6
    # M4 = A22 * (B21-B11)        -> 1x6
    # M5 = (A11+A12) * B22        -> 1x6
    # M6 = (A21-A11) * (B11+B12)  -> 1x6
    # M7 = (A12-A22) * (B21+B22)  -> 1x6
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each M is 1x6, so each M contributes 6 scalar multiplications.
    # Total: 7 * 6 = 42.
    
    # Now let's build the U, V, W matrices.
    # U[i*m+j, r] for i in {0,1}, j in {0,1,2,3}
    # V[j*p+k, r] for j in {0,1,2,3}, k in {0,1,2,3,4,5}
    # W[k*n+i, r] for k in {0,1,2,3,4,5}, i in {0,1}
    
    # For each Strassen multiplication M_s (s=1..7), we have 6 scalar mults (one per output column).
    # For M_s and output column k (0..5):
    #   U contribution: coefficients of A elements
    #   V contribution: coefficients of B elements  
    #   W contribution: 1 at position (k, i) for the appropriate i
    
    # M1 = (A11+A22) * (B11+B22)
    #   A11 = A[0,0:2] = A[0,0], A[0,1]
    #   A22 = A[1,2:4] = A[1,2], A[1,3]
    #   B11 = B[0:2,:] = B[0,:], B[1,:]
    #   B22 = B[2:4,:] = B[2,:], B[3,:]
    #   For output row i=0: C[0,:] gets M1
    #   For output row i=1: C[1,:] gets M1 (from C22 = M1 - M2 + M3 + M6)
    
    # Let me define the contributions:
    # For each (s, k) where s in 1..7, k in 0..5:
    #   r = (s-1)*6 + k
    
    strassen_A_coeffs = [
        # M1: A11 + A22 -> A[0,0]+A[1,2], A[0,1]+A[1,3]
        {0: 1, 1: 1, 6: 1, 7: 1},  # i*m+j: 0*4+0=0, 0*4+1=1, 1*4+2=6, 1*4+3=7
        # M2: A21 + A22 -> A[1,0]+A[1,2], A[1,1]+A[1,3]
        {4: 1, 5: 1, 6: 1, 7: 1},
        # M3: A11 -> A[0,0], A[0,1]
        {0: 1, 1: 1},
        # M4: A22 -> A[1,2], A[1,3]
        {6: 1, 7: 1},
        # M5: A11 + A12 -> A[0,0]+A[0,2], A[0,1]+A[0,3]
        {0: 1, 1: 1, 2: 1, 3: 1},
        # M6: A21 - A11 -> A[1,0]-A[0,0], A[1,1]-A[0,1]
        {0: -1, 1: -1, 4: 1, 5: 1},
        # M7: A12 - A22 -> A[0,2]-A[1,2], A[0,3]-A[1,3]
        {2: 1, 3: 1, 6: -1, 7: -1},
    ]
    
    
    # Let me build the rank-42 decomposition directly.
    # 7 Strassen mults, each expanded to 6 scalar mults.
    # For Strassen mult s (0-6) and scalar mult index t (0-5):
    #   r = s * 6 + t
    #   t corresponds to: which (a_elem, b_elem, c_elem) triple
    #   For a 1x2 times 2x3: a has 2 elements, b has 2*3=6 elements, c has 3 elements
    #   Schoolbook: for each of 3 output columns k, sum of 2 products = 6 mults
    #   So t = k * 2 + a_idx where a_idx in {0,1}, k in {0,1,2}
    #   Or t = a_idx * 3 + k
    
    # Let's use t = a_idx * 3 + k where a_idx in {0,1}, k in {0,1,2}
    
    # Strassen A coefficients (for each of 7 mults, which A elements with what sign):
    # A elements indexed as: a00=A[0,0], a01=A[0,1], a02=A[0,2], a03=A[0,3]
    #                       a10=A[1,0], a11=A[1,1], a12=A[1,2], a13=A[1,3]
    # In U: U[i*4+j, r]
    strassen_A = [
        # M1: A11+A22 = A[0,0]+A[1,2], A[0,1]+A[1,3]
        {0: 1, 1: 1, 6: 1, 7: 1},
        # M2: A21+A22 = A[1,0]+A[1,2], A[1,1]+A[1,3]
        {4: 1, 5: 1, 6: 1, 7: 1},
        # M3: A11 = A[0,0], A[0,1]
        {0: 1, 1: 1},
        # M4: A22 = A[1,2], A[1,3]
        {6: 1, 7: 1},
        # M5: A11+A12 = A[0,0]+A[0,2], A[0,1]+A[0,3]
        {0: 1, 1: 1, 2: 1, 3: 1},
        # M6: A21-A11 = A[1,0]-A[0,0], A[1,1]-A[0,1]
        {0: -1, 1: -1, 4: 1, 5: 1},
        # M7: A12-A22 = A[0,2]-A[1,2], A[0,3]-A[1,3]
        {2: 1, 3: 1, 6: -1, 7: -1},
    ]
    
    # Strassen B coefficients (for each of 7 mults, which B elements with what sign):
    # B elements: B[j,k] -> V[j*6+k, r]
    # B11 = B[0:2, 0:3], B12 = B[0:2, 3:6]
    # B21 = B[2:4, 0:3], B22 = B[2:4, 3:6]
    strassen_B = [
        # M1: B11+B22
        {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1},
        # M2: B11
        {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1},
        # M3: B12-B22
        {3: 1, 4: 1, 5: 1, 9: 1, 10: 1, 11: 1, 15: -1, 16: -1, 17: -1, 21: -1, 22: -1, 23: -1},
        # M4: B21-B11
        {0: -1, 1: -1, 2: -1, 6: -1, 7: -1, 8: -1, 12: 1, 13: 1, 14: 1, 18: 1, 19: 1, 20: 1},
        # M5: B22
        {12: 1, 13: 1, 14: 1, 18: 1, 19: 1, 20: 1},
        # M6: B11+B12
        {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1},
        # M7: B21+B22
        {12: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 18: 1, 19: 1, 20: 1, 21: 1, 22: 1, 23: 1},
    ]
    
    # W coefficients: for each (s, t), which C element gets what sign
    # t = a_idx * 3 + k where a_idx in {0,1}, k in {0,1,2}
    # The output column within the 3-wide block is k
    # For M_s, the output goes to specific C blocks:
    # M1 -> C11 and C22 (with signs +1 and +1)
    # M2 -> C21 and C22 (with signs +1 and -1)
    # M3 -> C12 and C22 (with signs +1 and +1)
    # M4 -> C11 and C21 (with signs +1 and +1)
    # M5 -> C11 and C12 (with signs -1 and +1)
    # M6 -> C22 (with sign +1)
    # M7 -> C11 (with sign +1)
    
    # C block to (i, k) mapping:
    # C11: i=0, k=0,1,2
    # C12: i=0, k=3,4,5
    # C21: i=1, k=0,1,2
    # C22: i=1, k=3,4,5
    
    # W[k*n+i, r] where k is the global column (0-5), i is the row (0-1)
    
    # For each Strassen mult s and scalar t:
    #   a_idx = t // 3, k_local = t % 3
    #   The output element is at C block position k_local within the 3-wide block
    #   Which C block(s) does M_s contribute to?
    
    w_contributions = [
        # M1: C11(+1), C22(+1)
        [(0, 1), (3, 1)],  # (block, sign) where block: 0=C11, 1=C12, 2=C21, 3=C22
        # M2: C21(+1), C22(-1)
        [(2, 1), (3, -1)],
        # M3: C12(+1), C22(+1)
        [(1, 1), (3, 1)],
        # M4: C11(+1), C21(+1)
        [(0, 1), (2, 1)],
        # M5: C11(-1), C12(+1)
        [(0, -1), (1, 1)],
        # M6: C22(+1)
        [(3, 1)],
        # M7: C11(+1)
        [(0, 1)],
    ]
    
    # Block to (i, k_offset) mapping:
    block_i_k = {
        0: (0, 0),  # C11: i=0, k starts at 0
        1: (0, 3),  # C12: i=0, k starts at 3
        2: (1, 0),  # C21: i=1, k starts at 0
        3: (1, 3),  # C22: i=1, k starts at 3
    }
    
    for s in range(7):
        for t in range(6):
            r = s * 6 + t
            a_idx = t // 3  # 0 or 1: which element of the 1x2 A-block
            k_local = t % 3  # 0, 1, or 2: which of the 3 output columns in the block
            
            # U contributions: for each A element in strassen_A[s],
            # determine its position within the 1x2 block (a_idx 0 or 1)
            for a_elem, coeff in strassen_A[s].items():
                j = a_elem % 4
                elem_a_idx = 0 if j < 2 else 1
                if elem_a_idx == a_idx:
                    U[a_elem, r] += coeff
            
            # V contributions: for each B element in strassen_B[s],
            # determine which of the 3 columns in its 3-wide block it belongs to
            for b_elem, coeff in strassen_B[s].items():
                k = b_elem % 6
                matched_k = k if k < 3 else k - 3
                if matched_k == k_local:
                    V[b_elem, r] += coeff
            
            # W contributions
            for block, sign in w_contributions[s]:
                i, k_offset = block_i_k[block]
                k_global = k_offset + k_local
                W[k_global * n + i, r] += sign
    
    # Verify
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                max_err = max(max_err, abs(val - 1))
    
    if max_err > 1e-6:
        # Fallback to schoolbook rank 48
        rank = 48
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, r] = 1
                    V[j*p+k, r] = 1
                    W[k*n+i, r] = 1
                    r += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to the original rank-42 decomposition
        # Fallback to the original rank-42 decomposition
        rank = 42
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        steps = [
            (
                {(0,0): 1, (0,1): 1, (1,2): 1, (1,3): 1},
                {(0,0): 1, (0,1): 1, (0,2): 1, (1,0): 1, (1,1): 1, (1,2): 1, (2,3): 1, (2,4): 1, (2,5): 1, (3,3): 1, (3,4): 1, (3,5): 1},
                [(0, 1), (3, 1)]
            ),
            (
                {(1,0): 1, (1,1): 1, (1,2): 1, (1,3): 1},
                {(0,0): 1, (0,1): 1, (0,2): 1, (1,0): 1, (1,1): 1, (1,2): 1},
                [(2, 1), (3, -1)]
            ),
            (
                {(0,0): 1, (0,1): 1},
                {(0,3): 1, (0,4): 1, (0,5): 1, (1,3): 1, (1,4): 1, (1,5): 1, (2,3): -1, (2,4): -1, (2,5): -1, (3,3): -1, (3,4): -1, (3,5): -1},
                [(1, 1), (3, 1)]
            ),
            (
                {(1,2): 1, (1,3): 1},
                {(2,0): 1, (2,1): 1, (2,2): 1, (3,0): 1, (3,1): 1, (3,2): 1, (0,0): -1, (0,1): -1, (0,2): -1, (1,0): -1, (1,1): -1, (1,2): -1},
                [(0, 1), (2, 1)]
            ),
            (
                {(0,0): 1, (0,1): 1, (0,2): 1, (0,3): 1},
                {(2,3): 1, (2,4): 1, (2,5): 1, (3,3): 1, (3,4): 1, (3,5): 1},
                [(0, -1), (1, 1)]
            ),
            (
                {(1,0): 1, (1,1): 1, (0,0): -1, (0,1): -1},
                {(0,0): 1, (0,1): 1, (0,2): 1, (0,3): 1, (0,4): 1, (0,5): 1, (1,0): 1, (1,1): 1, (1,2): 1, (1,3): 1, (1,4): 1, (1,5): 1},
                [(3, 1)]
            ),
            (
                {(0,2): 1, (0,3): 1, (1,2): -1, (1,3): -1},
                {(2,0): 1, (2,1): 1, (2,2): 1, (2,3): 1, (2,4): 1, (2,5): 1, (3,0): 1, (3,1): 1, (3,2): 1, (3,3): 1, (3,4): 1, (3,5): 1},
                [(0, 1)]
            ),
        ]
        
        col = 0
        for X_dict, Y_dict, C_targets in steps:
            x0_coeffs = {}
            x1_coeffs = {}
            for (i, j), c in X_dict.items():
                if j % 2 == 0:
                    x0_coeffs[(i, j)] = c
                else:
                    x1_coeffs[(i, j)] = c
                    
            y_coeffs = [[], [], [], [], [], []]
            for (j, k), c in Y_dict.items():
                r_block = j % 2
                c_block = k % 3
                idx = r_block * 3 + c_block
                y_coeffs[idx].append(((j, k), c))
                
            sub_terms = [
                (x0_coeffs, y_coeffs[0], 0),
                (x1_coeffs, y_coeffs[3], 0),
                (x0_coeffs, y_coeffs[1], 1),
                (x1_coeffs, y_coeffs[4], 1),
                (x0_coeffs, y_coeffs[2], 2),
                (x1_coeffs, y_coeffs[5], 2),
            ]
            
            for X_c, Y_c, z_idx in sub_terms:
                for (i, j), c in X_c.items():
                    U[i * m + j, col] += c
                for (j, k), c in Y_c:
                    V[j * p + k, col] += c
                
                for block_idx, sign in C_targets:
                    if block_idx == 0:
                        i_C = 0
                        k_C = z_idx
                    elif block_idx == 1:
                        i_C = 0
                        k_C = 3 + z_idx
                    elif block_idx == 2:
                        i_C = 1
                        k_C = z_idx
                    elif block_idx == 3:
                        i_C = 1
                        k_C = 3 + z_idx
                    
                    W[k_C * n + i_C, col] += sign
                
                col += 1
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Convert to the required format
    # U should be (n*m, rank), V should be (m*p, rank), W should be (p*n, rank)
    U_full = best_U.reshape(n, rank).T.reshape(n*m, rank)  # Wait, need to be careful
    # Actually, best_U is (n, rank), we need (n*m, rank)
    # U[i*m+j, r] = best_U[i, r] for all j? No, that's not right.
    
    # Let me re-examine. The decomposition is:
    # sum_r U[i,r]*V[j,r]*W[k,r] = 1 for all i,j,k
    # where U is (n,rank), V is (m,rank), W is (p,rank)
    # 
    # But the required format is:
    # U_full is (n*m, rank), V_full is (m*p, rank), W_full is (p*n, rank)
    # And the decomposition should be:
    # sum_r U_full[i*m+j, r] * V_full[j*p+k, r] * W_full[k*n+i, r] = 1
    # 
    # Wait, that doesn't match. Let me re-read the task.
    # "The output factor follows the official trace convention W[k*n+i,r]"
    # 
    # So W is indexed as W[k*n+i, r] where k is the column index and i is the row index.
    # 
    # The standard matrix multiplication tensor decomposition is:
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor T has T[i,j,k] = 1
    # A rank-r decomposition is: T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[i,k,r]
    # 
    # So U is (n*m, r), V is (m*p, r), W is (p*n, r) with W indexed as W[k*n+i, r]
    # 
    # My current decomposition has U as (n,rank), V as (m,rank), W as (p,rank)
    # with sum_r U[i,r]*V[j,r]*W[k,r] = 1
    # 
    # To convert: U_full[i*m+j, r] should be such that the product works out.
    # Actually, my decomposition is for a different tensor. Let me reconsider.
    
    # The tensor I'm decomposing is T[i,j,k] = 1 where i is row of A, j is column of A/row of B, k is column of B.
    # The standard decomposition is T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[i,k,r]
    # 
    # My decomposition: sum_r U[i,r]*V[j,r]*W[k,r] = 1
    # This is T'[i,j,k] = 1 where the factors are U[i,r], V[j,r], W[k,r]
    # 
    # To match the standard form, I need:
    # U_std[i*m+j, r] = U[i, r] (but this doesn't depend on j, which is wrong)
    
    # I think I made an error. Let me redo this.
    # 
    # The correct tensor for matrix multiplication C = A*B is:
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # The bilinear form is: (A, B) -> A*B
    # The tensor is T[i,j,k] = 1 (A[i,j] and B[j,k] contribute to C[i,k])
    # 
    # A rank-r decomposition is: T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[i,k,r]
    # where U is (n*m, r), V is (m*p, r), W is (n*p, r)
    # 
    # Note: W is indexed by (i,k) not (k,i), but the task says W[k*n+i,r], so W[k*n+i,r] = W_std[i,k,r]
    
    # So I need to decompose T[i,j,k] = 1 as sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    
    # My iterative method was solving a different problem. Let me fix it.
    
    # Actually, let me just use the fallback for now and return the rank-42 result.
    # The iterative method I wrote was solving sum_r U[i,r]*V[j,r]*W[k,r] = 1
    # which is NOT the same as the matrix multiplication tensor.
    
    # Let me just return the original rank-42 decomposition.
    rank = 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    steps = [
        (
            {(0,0): 1, (0,1): 1, (1,2): 1, (1,3): 1},
            {(0,0): 1, (0,1): 1, (0,2): 1, (1,0): 1, (1,1): 1, (1,2): 1, (2,3): 1, (2,4): 1, (2,5): 1, (3,3): 1, (3,4): 1, (3,5): 1},
            [(0, 1), (3, 1)]
        ),
        (
            {(1,0): 1, (1,1): 1, (1,2): 1, (1,3): 1},
            {(0,0): 1, (0,1): 1, (0,2): 1, (1,0): 1, (1,1): 1, (1,2): 1},
            [(2, 1), (3, -1)]
        ),
        (
            {(0,0): 1, (0,1): 1},
            {(0,3): 1, (0,4): 1, (0,5): 1, (1,3): 1, (1,4): 1, (1,5): 1, (2,3): -1, (2,4): -1, (2,5): -1, (3,3): -1, (3,4): -1, (3,5): -1},
            [(1, 1), (3, 1)]
        ),
        (
            {(1,2): 1, (1,3): 1},
            {(2,0): 1, (2,1): 1, (2,2): 1, (3,0): 1, (3,1): 1, (3,2): 1, (0,0): -1, (0,1): -1, (0,2): -1, (1,0): -1, (1,1): -1, (1,2): -1},
            [(0, 1), (2, 1)]
        ),
        (
            {(0,0): 1, (0,1): 1, (0,2): 1, (0,3): 1},
            {(2,3): 1, (2,4): 1, (2,5): 1, (3,3): 1, (3,4): 1, (3,5): 1},
            [(0, -1), (1, 1)]
        ),
        (
            {(1,0): 1, (1,1): 1, (0,0): -1, (0,1): -1},
            {(0,0): 1, (0,1): 1, (0,2): 1, (0,3): 1, (0,4): 1, (0,5): 1, (1,0): 1, (1,1): 1, (1,2): 1, (1,3): 1, (1,4): 1, (1,5): 1},
            [(3, 1)]
        ),
        (
            {(0,2): 1, (0,3): 1, (1,2): -1, (1,3): -1},
            {(2,0): 1, (2,1): 1, (2,2): 1, (2,3): 1, (2,4): 1, (2,5): 1, (3,0): 1, (3,1): 1, (3,2): 1, (3,3): 1, (3,4): 1, (3,5): 1},
            [(0, 1)]
        ),
    ]
    
    col = 0
    for X_dict, Y_dict, C_targets in steps:
        x0_coeffs = {}
        x1_coeffs = {}
        for (i, j), c in X_dict.items():
            if j % 2 == 0:
                x0_coeffs[(i, j)] = c
            else:
                x1_coeffs[(i, j)] = c
                
        y_coeffs = [[], [], [], [], [], []]
        for (j, k), c in Y_dict.items():
            r_block = j % 2
            c_block = k % 3
            idx = r_block * 3 + c_block
            y_coeffs[idx].append(((j, k), c))
            
        sub_terms = [
            (x0_coeffs, y_coeffs[0], 0),
            (x1_coeffs, y_coeffs[3], 0),
            (x0_coeffs, y_coeffs[1], 1),
            (x1_coeffs, y_coeffs[4], 1),
            (x0_coeffs, y_coeffs[2], 2),
            (x1_coeffs, y_coeffs[5], 2),
        ]
        
        for X_c, Y_c, z_idx in sub_terms:
            for (i, j), c in X_c.items():
                U[i * m + j, col] += c
            for (j, k), c in Y_c:
                V[j * p + k, col] += c
            
            for block_idx, sign in C_targets:
                if block_idx == 0:
                    i_C = 0
                    k_C = z_idx
                elif block_idx == 1:
                    i_C = 0
                    k_C = 3 + z_idx
                elif block_idx == 2:
                    i_C = 1
                    k_C = z_idx
                elif block_idx == 3:
                    i_C = 1
                    k_C = 3 + z_idx
                
                W[k_C * n + i_C, col] += sign
            
            col += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
