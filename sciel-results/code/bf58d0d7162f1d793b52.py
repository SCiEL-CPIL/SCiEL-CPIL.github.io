"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Target rank is 28
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # Products P1..P7
    # Z00 = P1 + P4 - P5 + P7
    # Z01 = P3 + P5
    # Z10 = P2 + P4
    # Z11 = P1 - P2 + P3 + P6
    
    # We define the linear forms for X and Y for each product
    # Each product r has a vector u_r (coeffs for X) and v_r (coeffs for Y)
    # and w_r (coeffs for Z)
    
    # X indices: 00, 01, 10, 11
    # Y indices: 00, 01, 10, 11
    # Z indices: 00, 01, 10, 11
    
    # P1: (x00+x11)(y00+y11)
    # u: [1, 0, 0, 1]
    # v: [1, 0, 0, 1]
    # P2: (x10+x11)y00
    # u: [0, 0, 1, 1]
    # v: [1, 0, 0, 0]
    # P3: x00(y01-y11)
    # u: [1, 0, 0, 0]
    # v: [0, 1, 0, -1]
    # P4: x11(y10-y00)
    # u: [0, 0, 0, 1]
    # v: [0, 0, 1, -1] # Wait, y10 is index 2, y00 is index 0. So [ -1, 0, 1, 0 ]
    # Correction: v: [-1, 0, 1, 0]
    # P5: (x00+x01)y11
    # u: [1, 1, 0, 0]
    # v: [0, 0, 0, 1]
    # P6: (x10-x00)(y00+y01)
    # u: [-1, 0, 1, 0]
    # v: [1, 1, 0, 0]
    # P7: (x01-x11)(y10+y11)
    # u: [0, 1, 0, -1]
    # v: [0, 0, 1, 1]
    
    strassen_u = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    strassen_v = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    # W coefficients for Z
    # Z00 = P1 + P4 - P5 + P7
    # Z01 = P3 + P5
    # Z10 = P2 + P4
    # Z11 = P1 - P2 + P3 + P6
    
    # w[r][z_idx]
    strassen_w = [
        [1, 0, 0, 0], # P1
        [0, 0, 1, 0], # P2
        [0, 1, 0, 0], # P3
        [1, 0, 1, 0], # P4
        [-1, 1, 0, 0],# P5
        [0, 0, 0, 1], # P6
        [1, 0, 0, 0]  # P7
    ]
    # Wait, Z11 = P1 - P2 + P3 + P6
    # P1: +1 to Z11
    # P2: -1 to Z11
    # P3: +1 to Z11
    # P6: +1 to Z11
    
    # Let's re-verify W matrix structure.
    # W is (p*n, rank).
    # Index is k*n + i.
    # For a 2x2 block, local i, k are 0,1.
    # Global i, k depend on the block.
    
    # We will define a function to add a Strassen block
    # block_type: 'A1Ba1', 'A2Ba2', 'A1Bb1', 'A2Bb2'
    
    def add_strassen_block(base_col, i_offset, j_offset, k_offset):
        """
        Adds a Strassen decomposition for a 2x2 multiplication.
        i_offset: global row offset for A (0 or 2? No, i is 0..1, j is 0..3)
        Actually, A is 2x4. i is 0,1. j is 0,1,2,3.
        U index: i*m + j = i*4 + j.
        
        B is 4x6. j is 0..3, k is 0..5.
        V index: j*p + k = j*6 + k.
        
        W index: k*n + i = k*2 + i.
        
        For a block:
        A_part: i in {0,1}, j in {j_offset, j_offset+1}
        B_part: j in {j_offset, j_offset+1}, k in {k_offset, k_offset+1}
        Z_part: i in {0,1}, k in {k_offset, k_offset+1}
        """
        for r in range(7):
            col = base_col + r
            
            # Update U
            # u_r = [x00, x01, x10, x11]
            # x00 corresponds to i=0, j=j_offset
            # x01 corresponds to i=0, j=j_offset+1
            # x10 corresponds to i=1, j=j_offset
            # x11 corresponds to i=1, j=j_offset+1
            
            u_coeffs = strassen_u[r]
            U[i_offset*0 + 0*4 + j_offset][col] += u_coeffs[0] # i=0, j=j_offset
            U[i_offset*0 + 0*4 + j_offset+1][col] += u_coeffs[1] # i=0, j=j_offset+1
            U[i_offset*0 + 1*4 + j_offset][col] += u_coeffs[2] # i=1, j=j_offset
            U[i_offset*0 + 1*4 + j_offset+1][col] += u_coeffs[3] # i=1, j=j_offset+1
            
            # Update V
            # v_r = [y00, y01, y10, y11]
            # y00 corresponds to j=j_offset, k=k_offset
            # y01 corresponds to j=j_offset, k=k_offset+1
            # y10 corresponds to j=j_offset+1, k=k_offset
            # y11 corresponds to j=j_offset+1, k=k_offset+1
            
            v_coeffs = strassen_v[r]
            V[j_offset*6 + k_offset][col] += v_coeffs[0]
            V[j_offset*6 + k_offset+1][col] += v_coeffs[1]
            V[j_offset+1*6 + k_offset][col] += v_coeffs[2]
            V[j_offset+1*6 + k_offset+1][col] += v_coeffs[3]
            
            # Update W
            # w_r = [z00, z01, z10, z11]
            # z00 corresponds to i=0, k=k_offset
            # z01 corresponds to i=0, k=k_offset+1
            # z10 corresponds to i=1, k=k_offset
            # z11 corresponds to i=1, k=k_offset+1
            
            w_coeffs = strassen_w[r]
            # W index: k*n + i
            W[k_offset*2 + 0][col] += w_coeffs[0]
            W[k_offset*2 + 1][col] += w_coeffs[1] # Wait, z01 is i=0, k=k_offset+1? No.
            # Z is 2x2.
            # Z[i_local, k_local]
            # Z00: i=0, k=0
            # Z01: i=0, k=1
            # Z10: i=1, k=0
            # Z11: i=1, k=1
            
            # My strassen_w definition:
            # index 0: Z00
            # index 1: Z01
            # index 2: Z10
            # index 3: Z11
            
            # So:
            # Z00 -> i=0, k=k_offset -> W[k_offset*2 + 0]
            # Z01 -> i=0, k=k_offset+1 -> W[(k_offset+1)*2 + 0]
            # Z10 -> i=1, k=k_offset -> W[k_offset*2 + 1]
            # Z11 -> i=1, k=k_offset+1 -> W[(k_offset+1)*2 + 1]
            
            # Let's re-map carefully.
            # w_coeffs[0] is for Z00 (i=0, k=0)
            # w_coeffs[1] is for Z01 (i=0, k=1)
            # w_coeffs[2] is for Z10 (i=1, k=0)
            # w_coeffs[3] is for Z11 (i=1, k=1)
            
            W[k_offset*2 + 0][col] += w_coeffs[0]
            W[(k_offset+1)*2 + 0][col] += w_coeffs[1]
            W[k_offset*2 + 1][col] += w_coeffs[2]
            W[(k_offset+1)*2 + 1][col] += w_coeffs[3]
            
    # Block 1: A1 Ba1
    # A1: j in 0,1. i in 0,1.
    # Ba1: j in 0,1. k in 0,1.
    # Z: i in 0,1. k in 0,1.
    add_strassen_block(0, 0, 0, 0)
    
    # Block 2: A2 Ba2
    # A2: j in 2,3. i in 0,1.
    # Ba2: j in 2,3. k in 2,3.
    # Z: i in 0,1. k in 2,3.
    add_strassen_block(7, 0, 2, 2)
    
    # Block 3: A1 Bb1
    # A1: j in 0,1. i in 0,1.
    # Bb1: j in 0,1. k in 4,5.
    # Z: i in 0,1. k in 4,5.
    add_strassen_block(14, 0, 0, 4)
    
    # Block 4: A2 Bb2
    # A2: j in 2,3. i in 0,1.
    # Bb2: j in 2,3. k in 4,5.
    # Z: i in 0,1. k in 4,5.
    add_strassen_block(21, 0, 2, 4)
    
    # Verify time
    if time.time() > deadline:
        # If we ran out of time, we still return the result as it's static
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
