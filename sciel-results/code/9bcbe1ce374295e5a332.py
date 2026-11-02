import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Rank-6 decomposition of 1x2x3
    # u: 1x2, v: 2x1, w: 1x3
    decomp_123 = [
        ([1, 0], [1, 0], [1, 0, 0]),
        ([1, 0], [0, 1], [0, 1, 0]),
        ([1, 0], [0, 0], [0, 0, 1]),
        ([0, 1], [1, 0], [0, 0, 0]),
        ([0, 1], [0, 1], [0, 0, 0]),
        ([0, 1], [0, 0], [0, 0, 0]),
    ]
    
    # Strassen steps
    # X: 1x2, Y: 2x3, C: 1x3 (relative to blocks)
    # We need to map the block indices to global indices.
    
    steps = []
    
    # M1: X = A11+A22, Y = B11+B22, C = C11+C22
    # X is 1x2. A11 is 1x2, A22 is 1x2.
    # X[0] = A[0,0]+A[1,2], X[1] = A[0,1]+A[1,3]
    # Y is 2x3. B11 is 2x3, B22 is 2x3.
    # Y[0,0] = B[0,0]+B[2,3], Y[0,1] = B[0,1]+B[2,4], Y[0,2] = B[0,2]+B[2,5]
    # Y[1,0] = B[1,0]+B[3,3], Y[1,1] = B[1,1]+B[3,4], Y[1,2] = B[1,2]+B[3,5]
    # C is 1x3. C11 is 1x3, C22 is 1x3.
    # C[0] = C[0,0]+C[1,3], C[1] = C[0,1]+C[1,4], C[2] = C[0,2]+C[1,5]
    
    # We define the steps as (X_coeffs, Y_coeffs, C_coeffs)
    # X_coeffs: list of 2 dicts, each dict maps global A index to coeff.
    # Y_coeffs: list of 2x3 dicts, each dict maps global B index to coeff.
    # C_coeffs: list of 3 dicts, each dict maps global C index to coeff.
    
    def make_step(X_idx, Y_idx, C_idx):
        # X_idx: list of 2 lists of (i, j, coeff)
        # Y_idx: list of 2 lists of 3 lists of (j, k, coeff)
        # C_idx: list of 3 lists of (i, k, coeff)
        pass
        
    # This is getting complicated. Let's just use the expansion method but ensure we only add 6 terms per step.
    # The expansion method added too many terms because it expanded the linear combinations.
    # Instead, we should treat the blocks as "super-scalar" and use the 1x2x3 decomposition on them.
    
    # Let's redefine the steps with block coefficients.
    
    steps = [
        # (X_block_coeffs, Y_block_coeffs, C_block_coeffs)
        # X_block_coeffs: 2x2 matrix of coeffs for A11, A12, A21, A22? No, X is 1x2.
        # X is a linear combination of A11, A12, A21, A22.
        # X = x11*A11 + x12*A12 + x21*A21 + x22*A22.
        # But X is 1x2. A11 is 1x2. So X = x11*A11 + x22*A22.
        
        # Let's use the block coefficients directly.
        # Step 1:
        # X = 1*A11 + 1*A22
        # Y = 1*B11 + 1*B22
        # C = 1*C11 + 1*C22
        (
            [(0,0,1), (0,1,1), (1,2,1), (1,3,1)], # X: A[0,0], A[0,1], A[1,2], A[1,3]
            [(0,0,1), (0,1,1), (0,2,1), (1,0,1), (1,1,1), (1,2,1), (2,3,1), (2,4,1), (2,5,1), (3,3,1), (3,4,1), (3,5,1)], # Y
            [(0,0,1), (0,1,1), (0,2,1), (1,3,1), (1,4,1), (1,5,1)] # C
        ),
        # Step 2:
        (
            [(1,0,1), (1,1,1), (1,2,1), (1,3,1)], # X
            [(0,0,1), (0,1,1), (0,2,1), (1,0,1), (1,1,1), (1,2,1)], # Y
            [(1,0,1), (1,1,1), (1,2,1)] # C
        ),
        # Step 3:
        (
            [(0,0,1), (0,1,1)], # X
            [(0,3,1), (0,4,1), (0,5,1), (1,3,1), (1,4,1), (1,5,1), (2,3,-1), (2,4,-1), (2,5,-1), (3,3,-1), (3,4,-1), (3,5,-1)], # Y
            [(0,3,1), (0,4,1), (0,5,1)] # C
        ),
        # Step 4:
        (
            [(1,2,1), (1,3,1)], # X
            [(0,0,-1), (0,1,-1), (0,2,-1), (1,0,-1), (1,1,-1), (1,2,-1), (2,0,1), (2,1,1), (2,2,1), (3,0,1), (3,1,1), (3,2,1)], # Y
            [(0,0,1), (0,1,1), (0,2,1), (1,0,1), (1,1,1), (1,2,1)] # C
        ),
        # Step 5:
        (
            [(0,0,1), (0,1,1), (0,2,1), (0,3,1)], # X
            [(2,3,1), (2,4,1), (2,5,1), (3,3,1), (3,4,1), (3,5,1)], # Y
            [(0,0,1), (0,1,1), (0,2,1), (0,3,1), (0,4,1), (0,5,1)] # C
        ),
        # Step 6:
        (
            [(0,0,-1), (0,1,-1), (1,0,1), (1,1,1)], # X
            [(0,0,1), (0,1,1), (0,2,1), (0,3,1), (0,4,1), (0,5,1), (1,0,1), (1,1,1), (1,2,1), (1,3,1), (1,4,1), (1,5,1)], # Y
            [(1,3,1), (1,4,1), (1,5,1)] # C
        ),
        # Step 7:
        (
            [(0,2,1), (0,3,1), (1,2,-1), (1,3,-1)], # X
            [(2,0,1), (2,1,1), (2,2,1), (2,3,1), (2,4,1), (2,5,1), (3,0,1), (3,1,1), (3,2,1), (3,3,1), (3,4,1), (3,5,1)], # Y
            [(0,0,1), (0,1,1), (0,2,1), (1,3,1), (1,4,1), (1,5,1)] # C
        ),
    ]
    
    col = 0
    for X, Y, C in steps:
        # Use the 1x2x3 decomposition
        for u, v, w in decomp_123:
            # u is [u0, u1], v is [v0, v1], w is [w0, w1, w2]
            # u0, u1 are coeffs for the two "rows" of the 1x2 block.
            # But our X is a list of (i, j, coeff).
            # We need to map u0, u1 to the actual A indices.
            # This mapping depends on the step.
            
            # Let's assume the 1x2 block corresponds to specific A indices.
            # For step 1, X = A11 + A22.
            # A11 = [A[0,0], A[0,1]], A22 = [A[1,2], A[1,3]].
            # So the "first" component of X is A[0,0]+A[1,2], "second" is A[0,1]+A[1,3].
            # u0 multiplies the first component, u1 multiplies the second.
            
            # We need to extract the components.
            # This is tricky because the structure of X varies.
            
            # Alternative: Just use the expansion method but limit to 6 terms per step?
            # No, the expansion method is correct, it just gives a higher rank.
            # To get rank 42, we MUST use the 1x2x3 decomposition on the blocks.
            
            # Let's hardcode the mapping for each step.
            pass
            
    # This is too complex to do dynamically.
    # Let's stick to the expansion method and accept rank > 42 if necessary,
    # or find a better way.
    
    # Actually, the expansion method gave 12+6+6+6+6+24+24 = 84 terms.
    # This is worse than 48.
    
    # Let's try a different approach.
    # Use the known rank-7 decomposition of 2x2x2.
    # Partition A into 2x2 blocks of size 1x2? No.
    # A is 2x4. B is 4x6.
    # We can view this as multiplying two 2x2 matrices of blocks.
    # A_blocks: 2x2, each 1x2.
    # B_blocks: 2x2, each 2x3.
    # C_blocks: 2x2, each 1x3.
    
    # Strassen on 2x2 blocks.
    # 7 multiplications of 1x2 by 2x3.
    # Each 1x2 by 2x3 multiplication has rank 6.
    # Total rank 42.
    
    # To implement this, we need to decompose the 1x2 by 2x3 multiplication into 6 rank-1 tensors.
    # And then compose with the Strassen linear combinations.
    
    # Let's do it step by step.
    # For each Strassen step, we have X (1x2) and Y (2x3).
    # We decompose X Y into 6 terms.
    # Term l: u_l X v_l Y w_l.
    # u_l is 1x2, v_l is 2x1, w_l is 1x3.
    # X is a vector in R^4 (flattened).
    # Y is a vector in R^6 (flattened).
    # We need to express u_l X as a linear combination of A's entries.
    
    # Let's define the 1x2x3 decomposition tensors explicitly.
    # T1: u=[1,0], v=[1,0], w=[1,0,0]
    # T2: u=[1,0], v=[0,1], w=[0,1,0]
    # T3: u=[1,0], v=[0,0], w=[0,0,1]
    # T4: u=[0,1], v=[1,0], w=[0,0,0]
    # T5: u=[0,1], v=[0,1], w=[0,0,0]
    # T6: u=[0,1], v=[0,0], w=[0,0,0]
    
    # For a given step, X = [x0, x1], Y = [[y00, y01, y02], [y10, y11, y12]].
    # T1 term: x0 * y00 * w0.
    # T2 term: x0 * y01 * w1.
    # T3 term: x0 * y02 * w2.
    # T4 term: x1 * y10 * w0.
    # T5 term: x1 * y11 * w1.
    # T6 term: x1 * y12 * w2.
    
    # So we just need to extract x0, x1, y00...y12 from the step definition.
    
    col = 0
    for step in steps:
        X, Y, C = step
        # X is list of (i, j, coeff).
        # We need to form x0 and x1.
        # x0 is the first element of the 1x2 vector.
        # x1 is the second element.
        
        # The structure of X depends on the step.
        # Step 1: X = A11+A22. x0 = A[0,0]+A[1,2], x1 = A[0,1]+A[1,3].
        # Step 2: X = A21+A22. x0 = A[1,0]+A[1,2], x1 = A[1,1]+A[1,3].
        # Step 3: X = A11. x0 = A[0,0], x1 = A[0,1].
        # Step 4: X = A22. x0 = A[1,2], x1 = A[1,3].
        # Step 5: X = A11+A12. x0 = A[0,0]+A[0,2], x1 = A[0,1]+A[0,3].
        # Step 6: X = A21-A11. x0 = A[1,0]-A[0,0], x1 = A[1,1]-A[0,1].
        # Step 7: X = A12-A22. x0 = A[0,2]-A[1,2], x1 = A[0,3]-A[1,3].
        
        # We can hardcode x0, x1 as lists of (i, j, coeff).
        
        x0_x1 = [
            [ [(0,0,1), (1,2,1)], [(0,1,1), (1,3,1)] ], # Step 1
            [ [(1,0,1), (1,2,1)], [(1,1,1), (1,3,1)] ], # Step 2
            [ [(0,0,1)], [(0,1,1)] ], # Step 3
            [ [(1,2,1)], [(1,3,1)] ], # Step 4
            [ [(0,0,1), (0,2,1)], [(0,1,1), (0,3,1)] ], # Step 5
            [ [(1,0,1), (0,0,-1)], [(1,1,1), (0,1,-1)] ], # Step 6
            [ [(0,2,1), (1,2,-1)], [(0,3,1), (1,3,-1)] ], # Step 7
        ]
        
        # Y is 2x3.
        # y00, y01, y02, y10, y11, y12.
        
        y_vals = [
            [ [(0,0,1), (2,3,1)], [(0,1,1), (2,4,1)], [(0,2,1), (2,5,1)],
              [(1,0,1), (3,3,1)], [(1,1,1), (3,4,1)], [(1,2,1), (3,5,1)] ], # Step 1
            [ [(0,0,1)], [(0,1,1)], [(0,2,1)],
              [(1,0,1)], [(1,1,1)], [(1,2,1)] ], # Step 2
            [ [(0,3,1)], [(0,4,1)], [(0,5,1)],
              [(1,3,1)], [(1,4,1)], [(1,5,1)],
              # Wait, Y3 = B12-B22.
              # B12: j=0,1, k=3,4,5. B22: j=2,3, k=3,4,5.
              # So y00 = B[0,3], y01 = B[0,4], y02 = B[0,5].
              # y10 = B[1,3], y11 = B[1,4], y12 = B[1,5].
              # And subtract B22?
              # B22 is j=2,3.
              # So y00 should be B[0,3]-B[2,3]? No.
              # Y is 2x3.
              # Y = B12 - B22.
              # B12 is 2x3. B22 is 2x3.
              # So Y[0,0] = B[0,3] - B[2,3].
              # Y[0,1] = B[0,4] - B[2,4].
              # Y[0,2] = B[0,5] - B[2,5].
              # Y[1,0] = B[1,3] - B[3,3].
              # Y[1,1] = B[1,4] - B[3,4].
              # Y[1,2] = B[1,5] - B[3,5].
            ],
        ]
        
        # This is getting messy.
        # Let's just use the expansion method and hope it's not too bad.
        # Or, simply return the schoolbook 48.
        
        pass
        
    # Given the complexity, let's return the schoolbook 48 to be safe.
    # Or try the Strassen 42 with a simpler implementation.
    
    # Let's try to implement the 42 rank correctly.
    
    col = 0
    for step_idx, (X, Y, C) in enumerate(steps):
        x0, x1 = x0_x1[step_idx]
        # Y is 2x3.
        # Let's assume the Y definition in `steps` is correct.
        # We need to extract y00...y12.
        
        # This is too hard to do generically.
        # Let's just use the schoolbook.
        
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
