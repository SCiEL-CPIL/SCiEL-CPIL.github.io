import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Strassen 2x2x2 decomposition
    # Each entry: (U_dict, V_dict, W_dict)
    # A, B, C indexed as (row, col) for 2x2
    strassen_222 = [
        # M1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}),
        # M4
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
    ]
    
    # Build 4x4x4 decomposition using block Strassen
    # A is 4x4, B is 4x4, C is 4x4
    # Block structure:
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4]
    # A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # Similarly for B and C
    
    # Block Strassen formulas:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # Each Mj is a 2x2 matrix, computed via Strassen 2x2x2
    # So for each Mj, we have 7 rank-1 tensors
    
    # Let's build the full 4x4x4 decomposition
    # U has 16 rows (4x4), V has 16 rows (4x4), W has 16 rows (4x4)
    # Each rank-1 tensor contributes to U, V, W
    
    def build_444():
        """Build 4x4x4 decomposition. Returns list of (U_dict, V_dict, W_dict)"""
        tensors = []
        
        # Define block Strassen combinations
        # Each entry: (A_block_combos, B_block_combos, C_block_combos)
        # A_block_combos: dict mapping block (bi, bj) to coefficient
        # bi, bj in {0,1} for 2x2 block structure
        
        block_strassen = [
            # M1 = (A11+A22)*(B11+B22)
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, None),  # C handled separately
            # M2 = (A21+A22)*B11
            ({(1,0):1, (1,1):1}, {(0,0):1}, None),
            # M3 = A11*(B12-B22)
            ({(0,0):1}, {(0,1):1, (1,1):-1}, None),
            # M4 = A22*(B21-B11)
            ({(1,1):1}, {(1,0):1, (0,0):-1}, None),
            # M5 = (A11+A12)*B22
            ({(0,0):1, (0,1):1}, {(1,1):1}, None),
            # M6 = (A21-A11)*(B11+B12)
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, None),
            # M7 = (A12-A22)*(B21+B22)
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, None),
        ]
        
        # C block combinations for each Mj
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        c_combos = [
            # M1: contributes to C11 (+1), C22 (+1)
            {(0,0):1, (1,1):1},
            # M2: contributes to C21 (+1), C22 (-1)
            {(1,0):1, (1,1):-1},
            # M3: contributes to C12 (+1), C22 (+1)
            {(0,1):1, (1,1):1},
            # M4: contributes to C11 (+1), C21 (+1)
            {(0,0):1, (1,0):1},
            # M5: contributes to C11 (-1), C12 (+1)
            {(0,0):-1, (0,1):1},
            # M6: contributes to C22 (+1)
            {(1,1):1},
            # M7: contributes to C11 (+1), C22 (+1)
            {(0,0):1, (1,1):1},
        ]
        
        for j in range(7):
            a_blocks, b_blocks, _ = block_strassen[j]
            c_blocks = c_combos[j]
            
            # For this block multiplication Mj, we use Strassen 2x2x2
            # The "A" for this 2x2x2 is a 2x2 matrix whose entries are linear combos of A's blocks
            # The "B" is similarly for B
            # The "C" is a 2x2 matrix whose entries contribute to C's blocks
            
            # For each of the 7 Strassen 2x2x2 rank-1 tensors:
            for (u_dict, v_dict, w_dict) in strassen_222:
                # u_dict maps (r,c) in 2x2 to coefficient
                # This corresponds to linear combo of the "A" 2x2 matrix entries
                # The "A" 2x2 matrix entry (r,c) is a linear combo of A's blocks
                
                # Build full U_dict for 4x4 A
                full_u = {}
                for (r, c), coeff in u_dict.items():
                    # The (r,c) entry of the "A" 2x2 matrix
                    # This is a linear combo of A's blocks
                    # A_block (bi, bj) where bi = r, bj = c
                    # Each A_block is a 2x2 matrix
                    for (bi, bj), block_coeff in a_blocks.items():
                        if bi == r and bj == c:
                            # Add contribution from this block
                            for (dr, dc) in [(0,0), (0,1), (1,0), (1,1)]:
                                a_idx = (2*bi + dr, 2*bj + dc)
                                full_u[a_idx] = full_u.get(a_idx, 0) + coeff * block_coeff
                
                # Similarly for V
                full_v = {}
                for (r, c), coeff in v_dict.items():
                    for (bi, bj), block_coeff in b_blocks.items():
                        if bi == r and bj == c:
                            for (dr, dc) in [(0,0), (0,1), (1,0), (1,1)]:
                                b_idx = (2*bi + dr, 2*bj + dc)
                                full_v[b_idx] = full_v.get(b_idx, 0) + coeff * block_coeff
                
                # Similarly for W
                full_w = {}
                for (r, c), coeff in w_dict.items():
                    for (bi, bj), block_coeff in c_blocks.items():
                        if bi == r and bj == c:
                            for (dr, dc) in [(0,0), (0,1), (1,0), (1,1)]:
                                c_idx = (2*bi + dr, 2*bj + dc)
                                full_w[c_idx] = full_w.get(c_idx, 0) + coeff * block_coeff
                
                # Remove zero entries
                full_u = {k: v for k, v in full_u.items() if v != 0}
                full_v = {k: v for k, v in full_v.items() if v != 0}
                full_w = {k: v for k, v in full_w.items() if v != 0}
                
                if full_u and full_v and full_w:
                    tensors.append((full_u, full_v, full_w))
        
        return tensors
    
    # Build 4x4x1 schoolbook
    def build_441():
        """Build 4x4x1 decomposition. Returns list of (U_dict, V_dict, W_dict)"""
        tensors = []
        for i in range(4):
            for j in range(4):
                u = {(i, j): 1}
                v = {(j, 0): 1}  # B is 4x1, so column 0
                w = {(i, 0): 1}  # C is 4x1, so column 0
                tensors.append((u, v, w))
        return tensors
    
    # Now combine for 4x4x9
    # Split B's 9 columns as 4+4+1
    # Part 1: columns 0-3 (4x4x4)
    # Part 2: columns 4-7 (4x4x4)
    # Part 3: column 8 (4x4x1)
    
    all_tensors = []
    
    # Part 1: 4x4x4, B columns 0-3
    for (u, v, w) in build_444():
        # u is for A (4x4), v is for B (4x4, columns 0-3), w is for C (4x4, columns 0-3)
        # Map B indices: (j, k) -> (j, k) for k in 0-3
        # Map C indices: (i, k) -> (i, k) for k in 0-3
        all_tensors.append((u, v, w))
    
    # Part 2: 4x4x4, B columns 4-7
    for (u, v, w) in build_444():
        # Map B indices: (j, k) -> (j, k+4)
        v_shifted = {}
        for (j, k), coeff in v.items():
            v_shifted[(j, k+4)] = coeff
        # Map C indices: (i, k) -> (i, k+4)
        w_shifted = {}
        for (i, k), coeff in w.items():
            w_shifted[(i, k+4)] = coeff
        all_tensors.append((u, v_shifted, w_shifted))
    
    # Part 3: 4x4x1, B column 8
    for (u, v, w) in build_441():
        # Map B indices: (j, 0) -> (j, 8)
        v_shifted = {}
        for (j, k), coeff in v.items():
            v_shifted[(j, 8)] = coeff
        # Map C indices: (i, 0) -> (i, 8)
        w_shifted = {}
        for (i, k), coeff in w.items():
            w_shifted[(i, 8)] = coeff
        all_tensors.append((u, v_shifted, w_shifted))
    
    # Now convert to U, V, W matrices
    rank = len(all_tensors)
    
    # U: 16 rows (4x4 A), rank columns
    U = [[0] * rank for _ in range(16)]
    # V: 36 rows (4x9 B), rank columns
    V = [[0] * rank for _ in range(36)]
    # W: 36 rows (4x9 C, using W[k*n+i] convention), rank columns
    W = [[0] * rank for _ in range(36)]
    
    for col, (u_dict, v_dict, w_dict) in enumerate(all_tensors):
        for (i, j), coeff in u_dict.items():
            U[i*4 + j][col] = coeff
        for (j, k), coeff in v_dict.items():
            V[j*9 + k][col] = coeff
        for (i, k), coeff in w_dict.items():
            # W[k*n+i] = W[k*4+i]
            W[k*4 + i][col] = coeff
    
    # Verify
    # C[i,k] = sum_j A[i,j] * B[j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * ... 
    # Actually, the decomposition is: C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_k V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
    # Wait, that's not right. The tensor decomposition is:
    # T[i,j,k] = sum_r U[i,j, r] * V[j,k, r] * W[k,i, r]
    # where T is the matrix multiplication tensor.
    # For matrix multiplication, T[i,j,k] = 1 if the (i,j) of A and (j,k) of B contribute to (i,k) of C.
    # Actually, T[i,j,k] is 1 for all valid (i,j,k), and the product is:
    # C[i,k] = sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k]
    # = sum_r (sum_{i,j} U[i,j,r] * A[i,j]) * (sum_{j,k} V[j,k,r] * B[j,k]) * ... 
    # No, that's not right either.
    #
    # The correct form: C[i,k] = sum_r U[i*m+?, r] * ... 
    # Let me think again.
    #
    # The tensor is T with indices (i, j, k) where i in 0..n-1, j in 0..m-1, k in 0..p-1.
    # T[i,j,k] = 1 for all valid indices.
    # The decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # So: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    #
    # And the matrix multiplication is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_{i,j,k} delta(i,i) * A[i,j] * B[j,k] * delta(k,k)
    # Hmm, let me just verify numerically.
    
    # Convert to numpy for verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
    # And for invalid combinations, it should be 0? No, the tensor only has non-zero entries for valid (i,j,k).
    # Actually, the tensor T[i,j,k] is defined for all i in 0..n-1, j in 0..m-1, k in 0..p-1, and T[i,j,k] = 1.
    # The decomposition should satisfy: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid (i,j,k).
    
    # But wait, that's not quite right. The tensor for matrix multiplication is:
    # T[i,j,k] = 1 if i in 0..n-1, j in 0..m-1, k in 0..p-1
    # The matrix multiplication is: C[i,k] = sum_j A[i,j] * B[j,k]
    # This can be written as: C[i,k] = sum_{i',j,k'} T[i',j,k'] * A[i',j] * B[j,k'] * delta(i',i) * delta(k',k)
    # Hmm, I think the standard form is:
    # The tensor has entries T[a,b,c] where a indexes A, b indexes B, c indexes C.
    # T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(i,k) for some i,j,k.
    # So T is non-zero only when the indices are consistent.
    #
    # In our indexing:
    # a = i*m+j (0..15)
    # b = j*p+k (0..35)
    # c = k*n+i (0..35)  [using W[k*n+i] convention]
    #
    # T[a,b,c] = 1 if there exist i,j,k such that a=i*m+j, b=j*p+k, c=k*n+i.
    # This requires: i = a//m, j = a%m, j = b//p, k = b%p, k = c//n, i = c%n
    # So: a%m == b//p and b%p == c//n and a//m == c%n
    
    # Let me verify our decomposition satisfies this.
    
    verified = True
    for i in range(4):
        for j in range(4):
            for k in range(9):
                a = i*4 + j
                b = j*9 + k
                c = k*4 + i
                val = 0
                for r in range(rank):
                    val += U_np[a, r] * V_np[b, r] * W_np[c, r]
                if abs(val - 1) > 1e-9:
                    verified = False
                    break
            if not verified:
                break
        if not verified:
            break
    
    if not verified:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
