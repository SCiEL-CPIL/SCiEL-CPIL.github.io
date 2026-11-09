import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    n, m, p = 4, 4, 4
    rank = 49
    
    # Strassen 2x2 element patterns
    # Indexing: [elem00, elem01, elem10, elem11]
    # Corresponds to 2x2 matrix indices (0,0), (0,1), (1,0), (1,1)
    U_elem = [
        [1, 0, 0, 1],   # t=0: a00+a11
        [0, 0, 1, 1],   # t=1: a10+a11
        [1, 0, 0, 0],   # t=2: a00
        [0, 0, 0, 1],   # t=3: a11
        [1, 1, 0, 0],   # t=4: a00+a01
        [-1, 0, 1, 0],  # t=5: a10-a00
        [0, 1, 0, -1],  # t=6: a01-a11
    ]
    V_elem = [
        [1, 0, 0, 1],   # t=0: b00+b11
        [1, 0, 0, 0],   # t=1: b00
        [0, 1, 0, -1],  # t=2: b01-b11
        [-1, 0, 1, 0],  # t=3: b10-b00
        [0, 0, 0, 1],   # t=4: b11
        [1, 1, 0, 0],   # t=5: b00+b01
        [0, 0, 1, 1],   # t=6: b10+b11
    ]
    W_elem = [
        [[1, 0], [0, 1]],   # t=0: C00+=1, C11+=1
        [[0, 0], [1, -1]],  # t=1: C10+=1, C11-=1
        [[0, 1], [0, 1]],   # t=2: C01+=1, C11+=1
        [[1, 0], [1, 0]],   # t=3: C00+=1, C10+=1
        [[-1, 1], [0, 0]],  # t=4: C00-=1, C01+=1
        [[0, 0], [0, 1]],   # t=5: C11+=1
        [[1, 0], [0, 0]],   # t=6: C00+=1
    ]
    
    # Block-level Strassen: which 2x2 blocks and coefficients
    # (block_row, block_col, coeff)
    A_blocks = [
        [(0,0,1), (1,1,1)],      # s=0: A00+A11
        [(1,0,1), (1,1,1)],      # s=1: A10+A11
        [(0,0,1)],               # s=2: A00
        [(1,1,1)],               # s=3: A11
        [(0,0,1), (0,1,1)],      # s=4: A00+A01
        [(1,0,1), (0,0,-1)],     # s=5: A10-A00
        [(0,1,1), (1,1,-1)],     # s=6: A01-A11
    ]
    B_blocks = [
        [(0,0,1), (1,1,1)],      # s=0: B00+B11
        [(0,0,1)],               # s=1: B00
        [(0,1,1), (1,1,-1)],     # s=2: B01-B11
        [(1,0,1), (0,0,-1)],     # s=3: B10-B00
        [(1,1,1)],               # s=4: B11
        [(0,0,1), (0,1,1)],      # s=5: B00+B01
        [(1,0,1), (1,1,1)],      # s=6: B10+B11
    ]
    C_blocks = [
        [(0,0,1), (1,1,1)],      # s=0: C00+=M, C11+=M
        [(1,0,1), (1,1,-1)],     # s=1: C10+=M, C11-=M
        [(0,1,1), (1,1,1)],      # s=2: C01+=M, C11+=M
        [(0,0,1), (1,0,1)],      # s=3: C00+=M, C10+=M
        [(0,0,-1), (0,1,1)],     # s=4: C00-=M, C01+=M
        [(1,1,1)],               # s=5: C11+=M
        [(0,0,1)],               # s=6: C00+=M
    ]
    
    # Initialize matrices with integers
    # U shape: (n*m, rank) = (16, 49)
    # V shape: (m*p, rank) = (16, 49)
    # W shape: (p*n, rank) = (16, 49)
    # W index convention: W[k*n+i, r] where k is col, i is row.
    
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    # Construct U, V, W
    for s in range(7):
        for t in range(7):
            col = s*7 + t
            
            # Build U for this term
            for (bi, bj, c_a) in A_blocks[s]:
                for ei in range(2):
                    for ej in range(2):
                        row_a = (2*bi + ei)*4 + (2*bj + ej)
                        val = U_elem[t][ei*2 + ej]
                        U[row_a][col] += c_a * val
            
            # Build V for this term
            for (bj, bk, c_b) in B_blocks[s]:
                for ej in range(2):
                    for ek in range(2):
                        row_b = (2*bj + ej)*4 + (2*bk + ek)
                        val = V_elem[t][ej*2 + ek]
                        V[row_b][col] += c_b * val
            
            # Build W for this term
            for (bi, bk, c_c) in C_blocks[s]:
                for ei in range(2):
                    for ek in range(2):
                        row_c = (2*bk + ek)*4 + (2*bi + ei)
                        val = W_elem[t][ei][ek]
                        W[row_c][col] += c_c * val
                        
    # Exact integer verification
    try:
        # Use a fixed set of test cases to ensure determinism
        # Test 1: Identity matrices
        A1 = [[1 if i==j else 0 for j in range(4)] for i in range(4)]
        B1 = [[1 if i==j else 0 for j in range(4)] for i in range(4)]
        
        # Test 2: Random small integers
        import random
        random.seed(123)
        A2 = [[random.randint(-3, 3) for _ in range(4)] for _ in range(4)]
        B2 = [[random.randint(-3, 3) for _ in range(4)] for _ in range(4)]
        
        def verify(A, B):
            # Schoolbook
            AB = [[0]*4 for _ in range(4)]
            for i in range(4):
                for j in range(4):
                    s = 0
                    for k in range(4):
                        s += A[i][k] * B[k][j]
                    AB[i][j] = s
            
            # Decomposition
            # u_vals = U^T A_flat
            # v_vals = V^T B_flat
            # scalars = u * v
            # C_flat = W^T scalars
            # C[i][k] = C_flat[k*4+i]
            
            # Compute u_vals
            u_vals = [0]*rank
            for r in range(rank):
                u = 0
                for i in range(4):
                    for j in range(4):
                        u += U[i*4+j][r] * A[i][j]
                u_vals[r] = u
                
            # Compute v_vals
            v_vals = [0]*rank
            for r in range(rank):
                v = 0
                for j in range(4):
                    for k in range(4):
                        v += V[j*4+k][r] * B[j][k]
                v_vals[r] = v
                
            # Compute scalars
            scalars = [u_vals[r] * v_vals[r] for r in range(rank)]
            
            # Compute C_flat
            C_flat = [0]*16
            for idx in range(16):
                c = 0
                for r in range(rank):
                    c += W[idx][r] * scalars[r]
                C_flat[idx] = c
                
            # Reconstruct C
            C_decomp = [[0]*4 for _ in range(4)]
            for i in range(4):
                for k in range(4):
                    C_decomp[i][k] = C_flat[k*4+i]
                    
            # Compare
            for i in range(4):
                for j in range(4):
                    if AB[i][j] != C_decomp[i][j]:
                        return False
            return True
            
        if not verify(A1, B1) or not verify(A2, B2):
            # Fallback to schoolbook rank 64
            rank_fb = 64
            U_fb = [[0]*rank_fb for _ in range(16)]
            V_fb = [[0]*rank_fb for _ in range(16)]
            W_fb = [[0]*rank_fb for _ in range(16)]
            col_fb = 0
            for i2 in range(4):
                for j2 in range(4):
                    for k2 in range(4):
                        U_fb[i2*4+j2][col_fb] = 1
                        V_fb[j2*4+k2][col_fb] = 1
                        W_fb[k2*4+i2][col_fb] = 1
                        col_fb += 1
            return {"U": U_fb, "V": V_fb, "W": W_fb}
            
    except Exception:
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
