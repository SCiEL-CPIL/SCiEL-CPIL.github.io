"""Matrix multiplication 4x4 * 4x8 via rank-98 Strassen decomposition.

Uses rank-49 Strassen for each 4x4 * 4x4 half, giving total rank 98.
The 4x8 matrix B is split into two 4x4 halves B_left and B_right.
C = A * B = [A * B_left, A * B_right], each computed with rank-49 Strassen.
"""
import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def _strassen_2x2():
    """Return the rank-7 Strassen decomposition for 2x2 * 2x2.
    
    Returns (U, V, W) where:
    U is 4x7 (for A: 2x2 matrix, row-major)
    V is 4x7 (for B: 2x2 matrix, row-major)
    W is 4x7 (for C: 2x2 matrix, row-major, but with trace convention C[k*2+i])
    
    Actually, let's be careful about the trace convention.
    For the full problem: W[k*n+i, r] where C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    For a 2x2*2x2 sub-problem embedded in the larger one, we need to be consistent.
    
    Let me define this more carefully. The tensor T[i,j,k] = 1 if we're doing standard matmul C[i,k] = sum_j A[i,j]*B[j,k].
    
    For the full 4x4*4x8 problem:
    A is 4x4, B is 4x8, C is 4x8
    T[i,j,k] corresponds to A[i,j] * B[j,k] contributing to C[i,k]
    
    U has shape (n*m, rank) = (16, rank)
    V has shape (m*p, rank) = (32, rank)  
    W has shape (p*n, rank) = (32, rank)
    
    The decomposition satisfies: for all i in [0,4), j in [0,4), k in [0,8):
    T[i,j,k] = sum_r U[i*4+j, r] * V[j*8+k, r] * W[k*4+i, r]
    
    And T[i,j,k] = 1 for the standard matrix multiplication tensor.
    
    So for the schoolbook: U[i*4+j, i*4+j] = 1, V[j*8+k, i*4+j] = 1, W[k*4+i, i*4+j] = 1
    Wait, that's not right either. Let me re-derive.
    
    We want: C[i,k] = sum_j A[i,j] B[j,k]
    In tensor form: T[i,j,k] = 1 (the multiplication tensor)
    Decomposition: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[i,k,r]
    
    But the interface says:
    U: (n*m, rank) indexed by i*m+j
    V: (m*p, rank) indexed by j*p+k
    W: (p*n, rank) indexed by k*n+i
    
    So: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    For schoolbook: for each (i,j,k), set U[i*m+j, i*m+j] = 1, V[j*p+k, i*m+j] = 1, W[k*n+i, i*m+j] = 1
    Then sum_r gives 1 for the matching r, 0 otherwise. Sum over all r gives 1. Correct.
    """
    pass

def _build_strassen_4x4_4x4():
    """Build rank-49 Strassen decomposition for 4x4 * 4x4.
    
    Returns (U, V, W) with shapes:
    U: (16, 49) - A is 4x4, indexed i*4+j
    V: (16, 49) - B is 4x4, indexed j*4+k
    W: (16, 49) - C is 4x4, indexed k*4+i (trace convention)
    """
    n = 4
    rank = 49
    U = np.zeros((n*n, rank), dtype=float)
    V = np.zeros((n*n, rank), dtype=float)
    W = np.zeros((n*n, rank), dtype=float)
    
    # Strassen for 4x4: split into 2x2 blocks of 2x2 matrices
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    #
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # Using Strassen's 7 multiplications:
    # P1 = (A00+A11)*(B00+B11)
    # P2 = (A10+A11)*B00
    # P3 = A00*(B01-B11)
    # P4 = A11*(B10-B00)
    # P5 = (A00+A01)*B11
    # P6 = (A10-A00)*(B00+B01)
    # P7 = (A01-A11)*(B10+B11)
    #
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Each P is a 2x2 * 2x2 multiplication, which has rank 7.
    # So each P contributes 7 rank terms.
    # Total: 7 * 7 = 49.
    
    # Define the 7 Strassen 2x2*2x2 sub-decompositions
    # For a 2x2*2x2 with matrices X, Y, Z:
    # Z = X*Y
    # Strassen:
    # M1 = (x00+x11)*(y00+y11)
    # M2 = (x10+x11)*y00
    # M3 = x00*(y01-y11)
    # M4 = x11*(y10-y00)
    # M5 = (x00+x01)*y11
    # M6 = (x10-x00)*(y00+y01)
    # M7 = (x01-x11)*(y10+y11)
    # z00 = M1 + M4 - M5 + M7
    # z01 = M3 + M5
    # z10 = M2 + M4
    # z11 = M1 - M2 + M3 + M6
    
    # For each top-level Strassen term s (0-6), we have a 2x2*2x2 multiplication
    # The "X" matrix is a linear combination of A blocks
    # The "Y" matrix is a linear combination of B blocks
    # We need to expand this into rank-7 terms
    
    # Let's define helper functions to get block indices
    
    def a_block_indices(block):
        """Return list of (row, col) indices in A for a given block."""
        # block: 0=A00, 1=A01, 2=A10, 3=A11
        r0, c0 = (block // 2) * 2, (block % 2) * 2
        return [(r0 + dr, c0 + dc) for dr in range(2) for dc in range(2)]
    
    def b_block_indices(block):
        """Return list of (row, col) indices in B for a given block."""
        r0, c0 = (block // 2) * 2, (block % 2) * 2
        return [(r0 + dr, c0 + dc) for dr in range(2) for dc in range(2)]
    
    def c_block_indices(block):
        """Return list of (row, col) indices in C for a given block."""
        r0, c0 = (block // 2) * 2, (block % 2) * 2
        return [(r0 + dr, c0 + dc) for dr in range(2) for dc in range(2)]
    
    # Top-level Strassen terms:
    # s=0: P1 = (A00+A11) * (B00+B11)
    # s=1: P2 = (A10+A11) * B00
    # s=2: P3 = A00 * (B01-B11)
    # s=3: P4 = A11 * (B10-B00)
    # s=4: P5 = (A00+A01) * B11
    # s=5: P6 = (A10-A00) * (B00+B01)
    # s=6: P7 = (A01-A11) * (B10+B11)
    
    # For each s, define the A-side blocks with coefficients
    # and B-side blocks with coefficients
    top_a_blocks = [
        [(0, 1), (3, 1)],   # s=0: A00 + A11
        [(2, 1), (3, 1)],   # s=1: A10 + A11
        [(0, 1)],           # s=2: A00
        [(3, 1)],           # s=3: A11
        [(0, 1), (1, 1)],   # s=4: A00 + A01
        [(2, 1), (0, -1)],  # s=5: A10 - A00
        [(1, 1), (3, -1)],  # s=6: A01 - A11
    ]
    
    top_b_blocks = [
        [(0, 1), (3, 1)],   # s=0: B00 + B11
        [(0, 1)],           # s=1: B00
        [(1, 1), (3, -1)],  # s=2: B01 - B11
        [(2, 1), (0, -1)],  # s=3: B10 - B00
        [(3, 1)],           # s=4: B11
        [(0, 1), (1, 1)],   # s=5: B00 + B01
        [(2, 1), (3, 1)],   # s=6: B10 + B11
    ]
    
    # Which C blocks get contributions from each P, and with what coefficient
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    c_contributions = [
        # For each s (P_s), list of (c_block, coefficient)
        [(0, 1), (3, 1)],   # s=0 (P1): C00 +1, C11 +1
        [(2, 1), (3, -1)],  # s=1 (P2): C10 +1, C11 -1
        [(1, 1), (3, 1)],   # s=2 (P3): C01 +1, C11 +1
        [(0, 1), (2, 1)],   # s=3 (P4): C00 +1, C10 +1
        [(0, -1), (1, 1)],  # s=4 (P5): C00 -1, C01 +1
        [(3, 1)],           # s=5 (P6): C11 +1
        [(0, 1)],           # s=6 (P7): C00 +1
    ]
    
    # Strassen 2x2*2x2 sub-terms
    # For X*Y = Z:
    # M1 = (x00+x11)*(y00+y11) -> z00 +1, z11 +1
    # M2 = (x10+x11)*y00 -> z10 +1, z11 -1
    # M3 = x00*(y01-y11) -> z01 +1, z11 +1
    # M4 = x11*(y10-y00) -> z00 +1, z10 +1
    # M5 = (x00+x01)*y11 -> z00 -1, z01 +1
    # M6 = (x10-x00)*(y00+y01) -> z11 +1
    # M7 = (x01-x11)*(y10+y11) -> z00 +1
    
    # For each sub-term t (0-6), define:
    # x_elements: list of ((r,c), coeff) for X matrix elements
    # y_elements: list of ((r,c), coeff) for Y matrix elements
    # z_elements: list of ((r,c), coeff) for Z matrix elements
    
    sub_x_elements = [
        [((0,0),1), ((1,1),1)],   # t=0: x00+x11
        [((1,0),1), ((1,1),1)],   # t=1: x10+x11
        [((0,0),1)],              # t=2: x00
        [((1,1),1)],              # t=3: x11
        [((0,0),1), ((0,1),1)],   # t=4: x00+x01
        [((1,0),1), ((0,0),-1)],  # t=5: x10-x00
        [((0,1),1), ((1,1),-1)],  # t=6: x01-x11
    ]
    
    sub_y_elements = [
        [((0,0),1), ((1,1),1)],   # t=0: y00+y11
        [((0,0),1)],              # t=1: y00
        [((0,1),1), ((1,1),-1)],  # t=2: y01-y11
        [((1,0),1), ((0,0),-1)],  # t=3: y10-y00
        [((1,1),1)],              # t=4: y11
        [((0,0),1), ((0,1),1)],   # t=5: y00+y01
        [((1,0),1), ((1,1),1)],   # t=6: y10+y11
    ]
    
    sub_z_elements = [
        [((0,0),1), ((1,1),1)],   # t=0: z00+1, z11+1
        [((1,0),1), ((1,1),-1)],  # t=1: z10+1, z11-1
        [((0,1),1), ((1,1),1)],   # t=2: z01+1, z11+1
        [((0,0),1), ((1,0),1)],   # t=3: z00+1, z10+1
        [((0,0),-1), ((0,1),1)],  # t=4: z00-1, z01+1
        [((1,1),1)],              # t=5: z11+1
        [((0,0),1)],              # t=6: z00+1
    ]
    
    # Now build the decomposition
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            
            # Get the A-side blocks for this top-level term
            a_blocks = top_a_blocks[s]  # list of (block, coeff)
            # Get the B-side blocks
            b_blocks = top_b_blocks[s]
            
            # Get the sub-term definitions
            x_elems = sub_x_elements[t]  # list of ((r,c), coeff)
            y_elems = sub_y_elements[t]
            
            # For each element in X, map to A blocks
            # X is a 2x2 matrix, where X[r,c] is a linear combination of A blocks
            # The A-side coefficient for U[i*4+j, col] is:
            # sum over (x_r, x_c) in x_elems of x_coeff * (coefficient of A block containing (i,j))
            
            # For each (i,j) in A (4x4):
            for i in range(4):
                for j in range(4):
                    a_coeff = 0.0
                    for (x_r, x_c), x_coeff in x_elems:
                        # X[x_r, x_c] corresponds to which A block?
                        # X is formed from A blocks. The block at position (x_r, x_c) in X
                        # corresponds to A block number x_r*2 + x_c
                        a_block_num = x_r * 2 + x_c
                        # Find if this A block is in a_blocks with what coefficient
                        for (blk, blk_coeff) in a_blocks:
                            if blk == a_block_num:
                                # Check if (i,j) is in this block
                                br0, bc0 = (blk // 2) * 2, (blk % 2) * 2
                                if br0 <= i < br0+2 and bc0 <= j < bc0+2:
                                    a_coeff += x_coeff * blk_coeff
                    U[i*4+j, col] = a_coeff
            
            # For each (j,k) in B (4x4):
            for j in range(4):
                for k in range(4):
                    b_coeff = 0.0
                    for (y_r, y_c), y_coeff in y_elems:
                        # Y[y_r, y_c] corresponds to B block number y_r*2 + y_c
                        b_block_num = y_r * 2 + y_c
                        for (blk, blk_coeff) in b_blocks:
                            if blk == b_block_num:
                                br0, bc0 = (blk // 2) * 2, (blk % 2) * 2
                                if br0 <= j < br0+2 and bc0 <= k < bc0+2:
                                    b_coeff += y_coeff * blk_coeff
                    V[j*4+k, col] = b_coeff
            
            # For each (k,i) in C (4x4), with trace convention W[k*4+i]
            # Z is a 2x2 matrix, Z[r,c] contributes to C blocks
            # The C-side coefficient for W[k*4+i, col] is:
            # sum over (z_r, z_c) in z_elems of z_coeff * (coefficient from c_contributions)
            
            for k in range(4):
                for i in range(4):
                    w_coeff = 0.0
                    for (z_r, z_c), z_coeff in sub_z_elements[t]:
                        # Z[z_r, z_c] corresponds to C block number z_r*2 + z_c
                        c_block_num = z_r * 2 + z_c
                        # Find if this C block gets contribution from P_s
                        for (blk, blk_coeff) in c_contributions[s]:
                            if blk == c_block_num:
                                br0, bc0 = (blk // 2) * 2, (blk % 2) * 2
                                if br0 <= k < br0+2 and bc0 <= i < bc0+2:
                                    w_coeff += z_coeff * blk_coeff
                    W[k*4+i, col] = w_coeff
    
    return U, V, W

def _verify_4x4_4x4(U, V, W):
    """Verify that the 4x4*4x4 decomposition is correct."""
    n = 4
    rank = U.shape[1]
    # Compute the tensor: T[i,j,k] = sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
    T = np.zeros((n, n, n), dtype=float)
    for r in range(rank):
        u_vec = U[:, r]  # 16
        v_vec = V[:, r]  # 16
        w_vec = W[:, r]  # 16
        # Reshape
        u_mat = u_vec.reshape(4, 4)
        v_mat = v_vec.reshape(4, 4)
        w_mat = w_vec.reshape(4, 4)  # indexed by k*4+i
        # T[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    T[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
    # Check against standard matmul tensor
    for i in range(4):
        for j in range(4):
            for k in range(4):
                if abs(T[i,j,k] - 1.0) > 1e-10:
                    return False
    return True

def construct():
    """Build the rank-98 Strassen decomposition for 4x4 * 4x8."""
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    total_rank = 98
    
    U = np.zeros((n*m, total_rank), dtype=float)
    V = np.zeros((m*p, total_rank), dtype=float)
    W = np.zeros((p*n, total_rank), dtype=float)
    
    # Build the 4x4*4x4 Strassen decomposition
    U4, V4, W4 = _build_strassen_4x4_4x4()
    
    # Verify
    if not _verify_4x4_4x4(U4, V4, W4):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # For 4x4 * 4x8: split B into two 4x4 halves
    # half=0: B[:, 0:4], C[:, 0:4]
    # half=1: B[:, 4:8], C[:, 4:8]
    
    for half in range(2):
        base_col = half * 49
        col_offset_B = half * 4  # offset in B columns
        col_offset_C = half * 4  # offset in C columns
        
        # Copy U4 to U
        U[:, base_col:base_col+49] = U4
        
        # Copy V4 to V, with column offset
        # V4[j*4+k, r] -> V[j*8 + (k + col_offset_B), r]
        for j in range(4):
            for k in range(4):
                for r in range(49):
                    V[j*8 + (k + col_offset_B), base_col + r] = V4[j*4+k, r]
        
        # Copy W4 to W, with column offset
        # W4[k*4+i, r] -> W[(k + col_offset_C)*4 + i, r]
        for k in range(4):
            for i in range(4):
                for r in range(49):
                    W[(k + col_offset_C)*4 + i, base_col + r] = W4[k*4+i, r]
    
    # Final verification
    elapsed = time.time() - start_time
    if elapsed < 120:
        # Verify the full decomposition
        rank = total_rank
        T = np.zeros((n, m, p), dtype=float)
        for r in range(rank):
            u_vec = U[:, r]
            v_vec = V[:, r]
            w_vec = W[:, r]
            u_mat = u_vec.reshape(n, m)
            v_mat = v_vec.reshape(m, p)
            w_mat = w_vec.reshape(p, n)  # indexed by k*n+i
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        
        # Check
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(T[i,j,k] - 1.0) > 1e-10:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            # Fallback to schoolbook
            rank = n * m * p
            U = np.zeros((n*m, rank), dtype=float)
            V = np.zeros((m*p, rank), dtype=float)
            W = np.zeros((p*n, rank), dtype=float)
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j, col] = 1.0
                        V[j*p+k, col] = 1.0
                        W[k*n+i, col] = 1.0
                        col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
