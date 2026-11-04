import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # We will build a decomposition of rank 85
    # Split m=5 into m1=4, m2=1
    # A is 4x5, B is 5x5
    # A = [A1, A2] where A1 is 4x4, A2 is 4x1
    # B = [[B11, B12], [B21, B22]] where B11 is 4x4, B12 is 4x1, B21 is 1x4, B22 is 1x1
    # C = A1 B11 + A1 B12 + A2 B21 + A2 B22
    
    # Part 1: A1 B11 (4x4x4) using Strassen (rank 49)
    # Part 2: A1 B12 (4x4x1) using Schoolbook (rank 16)
    # Part 3: A2 B21 (4x1x4) using Schoolbook (rank 16)
    # Part 4: A2 B22 (4x1x1) using Schoolbook (rank 4)
    
    rank = 49 + 16 + 16 + 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u, v, w):
        nonlocal col
        for i in range(len(u)):
            U[i][col] += u[i]
        for i in range(len(v)):
            V[i][col] += v[i]
        for i in range(len(w)):
            W[i][col] += w[i]
        col += 1

    # --- Part 1: 4x4x4 Strassen ---
    # Indices for A1 (4x4): i in 0..3, j in 0..3
    # Indices for B11 (4x4): j in 0..3, k in 0..3
    # We use 4 blocks of 2x2x2 Strassen?
    # Actually, Strassen for 4x4x4 is recursive.
    # It uses 7 multiplications of 2x2x2 matrices.
    # Each 2x2x2 multiplication has rank 7.
    # So total rank 49.
    
    # Define Strassen 2x2x2 factors
    # M1: (A11+A22)(B11+B22)
    # M2: (A21+A22)(B11)
    # M3: (A11)(B12-B22)
    # M4: (A22)(B21-B11)
    # M5: (A11+A12)(B22)
    # M6: (A21-A11)(B11+B12)
    # M7: (A12-A22)(B21+B22)
    
    # Indices in 2x2 matrix:
    # 00: 0, 01: 1, 10: 2, 11: 3
    
    # U_2x2: 4x7 matrix
    # V_2x2: 4x7 matrix
    # W_2x2: 4x7 matrix
    
    U2 = [
        [1, 0, 1, 0, 1, -1, 0], # A11
        [0, 0, 0, 0, 0, 0, 1],  # A12
        [0, 1, 0, 0, 0, 1, 0],  # A21
        [1, 1, 0, 1, 0, 0, -1]  # A22
    ]
    V2 = [
        [1, 1, 0, -1, 0, 1, 0], # B11
        [0, 0, 1, 0, 0, 1, 0],  # B12
        [0, 0, 0, 1, 0, 0, 1],  # B21
        [1, 0, -1, 0, 1, 0, 1]  # B22
    ]
    W2 = [
        [1, 0, 0, 1, 1, 0, 1], # C11
        [0, 0, 1, 0, 1, 0, 0], # C12
        [0, 1, 0, 1, 0, 0, 0], # C21
        [1, -1, 1, 0, 0, 1, 0] # C22
    ]
    
    # Check W2 correctness
    # C11 = M1 + M4 - M5 + M7
    # M1: 1, M4: 1, M5: -1, M7: 1 -> 1+1-1+1 = 2? 
    # Wait.
    # C11 = M1 + M4 - M5 + M7
    # Coeffs: M1:1, M2:0, M3:0, M4:1, M5:-1, M6:0, M7:1
    # My W2[0] is [1, 0, 0, 1, 1, 0, 1]. M5 is 1, should be -1.
    # Let's re-verify Strassen formulas.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Correct W2:
    W2 = [
        [1, 0, 0, 1, -1, 0, 1], # C11
        [0, 0, 1, 0, 1, 0, 0],  # C12
        [0, 1, 0, 1, 0, 0, 0],  # C21
        [1, -1, 1, 0, 0, 1, 0]  # C22
    ]
    
    # Now combine for 4x4
    # The 4x4 matrix is split into 2x2 blocks.
    # A11 (top-left) corresponds to indices (0,0), (0,1), (1,0), (1,1) in the 4x4?
    # Let's map 4x4 indices (i,j) to block (bi,bj) and local (li,lj).
    # bi = i//2, bj = j//2. li = i%2, lj = j%2.
    # Block 0: (0,0) -> A11
    # Block 1: (0,1) -> A12
    # Block 2: (1,0) -> A21
    # Block 3: (1,1) -> A22
    
    # Strassen steps for 4x4:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)(B11)
    # M3 = (A11)(B12-B22)
    # M4 = (A22)(B21-B11)
    # M5 = (A11+A12)(B22)
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For each M_k, we perform a 2x2 multiplication.
    # The inputs to the 2x2 multiplication are linear combinations of the 2x2 blocks.
    # Each 2x2 block is a vector of 4 elements.
    # So the "input vector" for the 2x2 multiplication is formed by summing the block vectors.
    
    # We need to generate 49 terms.
    # For each of the 7 outer steps (k=0..6), we have 7 inner steps (r=0..6).
    
    # Define the linear combinations of blocks for A and B for each outer step.
    # A_blocks: 0:A11, 1:A12, 2:A21, 3:A22
    # B_blocks: 0:B11, 1:B12, 2:B21, 3:B22
    
    outer_steps = [
        # (A_comb, B_comb)
        # A_comb is dict {block_idx: coeff}
        # B_comb is dict {block_idx: coeff}
        ({0: 1, 3: 1}, {0: 1, 3: 1}), # M1
        ({2: 1, 3: 1}, {0: 1}),       # M2
        ({0: 1}, {1: 1, 3: -1}),      # M3
        ({3: 1}, {2: 1, 0: -1}),      # M4
        ({0: 1, 1: 1}, {3: 1}),       # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}),# M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}) # M7
    ]
    
    # For each outer step, we compute the rank 7 decomposition of the 2x2 product.
    # The 2x2 product result is a 2x2 matrix (4 elements).
    # This result corresponds to a 2x2 block in the final 4x4 result.
    # Which block?
    # M1 contributes to all 4 blocks?
    # C11 = M1 + ...
    # M1 is a 2x2 matrix. It is added to the top-left block of C?
    # Yes, Strassen computes the 4 2x2 blocks of C.
    # C11 (top-left) = M1 + M4 - M5 + M7
    # C12 (top-right) = M3 + M5
    # C21 (bottom-left) = M2 + M4
    # C22 (bottom-right) = M1 - M2 + M3 + M6
    
    # So for each outer step k, we have a 2x2 result matrix R_k.
    # R_k is formed by the 7 inner multiplications.
    # The contribution of R_k to the final C blocks is determined by the Strassen addition formulas.
    
    # We will iterate over the 7 outer steps.
    # For each, we iterate over the 7 inner steps.
    # This gives 49 terms.
    
    for k, (a_comb, b_comb) in enumerate(outer_steps):
        # Determine which C blocks this M_k contributes to and with what coefficient.
        # k=0 (M1): C11(+1), C22(+1)
        # k=1 (M2): C21(+1), C22(-1)
        # k=2 (M3): C12(+1), C22(+1)
        # k=3 (M4): C11(+1), C21(+1)
        # k=4 (M5): C11(-1), C12(+1)
        # k=5 (M6): C22(+1)
        # k=6 (M7): C11(+1)
        
        # Map k to (block_idx, coeff)
        # Block 0: C11, Block 1: C12, Block 2: C21, Block 3: C22
        contribs = []
        if k == 0: contribs = [(0, 1), (3, 1)]
        elif k == 1: contribs = [(2, 1), (3, -1)]
        elif k == 2: contribs = [(1, 1), (3, 1)]
        elif k == 3: contribs = [(0, 1), (2, 1)]
        elif k == 4: contribs = [(0, -1), (1, 1)]
        elif k == 5: contribs = [(3, 1)]
        elif k == 6: contribs = [(0, 1)]
        
        # For each inner step r in 0..6
        for r in range(7):
            u_vec = [0] * (n*m) # 4*5 = 20
            v_vec = [0] * (m*p) # 5*5 = 25
            w_vec = [0] * (p*n) # 5*4 = 20
            
            # Fill U
            # A is 4x4. Indices i,j in 0..3.
            # Block bi = i//2, bj = j//2. Local li = i%2, lj = j%2.
            # Local index in 2x2: li*2 + lj.
            # U2[r][local_idx] is the coefficient for A_block[bi,bj][li,lj]
            
            for i in range(4):
                for j in range(4):
                    bi = i // 2
                    bj = j // 2
                    li = i % 2
                    lj = j % 2
                    local_idx = li * 2 + lj
                    block_idx = bi * 2 + bj
                    
                    # Coefficient from outer step
                    coeff_outer = a_comb.get(block_idx, 0)
                    # Coefficient from inner step
                    coeff_inner = U2[r][local_idx]
                    
                    total_coeff = coeff_outer * coeff_inner
                    if total_coeff != 0:
                        # A index in U: i*m + j = i*5 + j
                        # But A1 is the top-left 4x4 of the 4x5 matrix.
                        # So A1[i,j] corresponds to A[i,j] in the 4x5 matrix.
                        u_vec[i * m + j] = total_coeff
            
            # Fill V
            # B is 4x4 (B11). Indices j,k in 0..3.
            # Block bi = j//2, bj = k//2. Local li = j%2, lj = k%2.
            for j in range(4):
                for k in range(4):
                    bi = j // 2
                    bj = k // 2
                    li = j % 2
                    lj = k % 2
                    local_idx = li * 2 + lj
                    block_idx = bi * 2 + bj
                    
                    coeff_outer = b_comb.get(block_idx, 0)
                    coeff_inner = V2[r][local_idx]
                    
                    total_coeff = coeff_outer * coeff_inner
                    if total_coeff != 0:
                        # B index in V: j*p + k = j*5 + k
                        # B11 is top-left 4x4 of the 5x5 matrix.
                        v_vec[j * p + k] = total_coeff
            
            # Fill W
            # Result C is 4x4.
            # The result of the 2x2 multiplication (inner step r) is a 2x2 matrix.
            # W2[r][local_idx] is the coefficient for the local element.
            # This 2x2 matrix is added to specific 2x2 blocks of the 4x4 result.
            
            # We need to add the contribution to the W vector.
            # W index: k*n + i.
            
            for c_block_idx, c_coeff in contribs:
                # c_block_idx: 0->C11, 1->C12, 2->C21, 3->C22
                # C11 is top-left (i=0,1; k=0,1)
                # C12 is top-right (i=0,1; k=2,3)
                # C21 is bottom-left (i=2,3; k=0,1)
                # C22 is bottom-right (i=2,3; k=2,3)
                
                base_i = (c_block_idx // 2) * 2
                base_k = (c_block_idx % 2) * 2
                
                for li in range(2):
                    for lj in range(2):
                        local_idx = li * 2 + lj
                        coeff_inner = W2[r][local_idx]
                        total_coeff = c_coeff * coeff_inner
                        if total_coeff != 0:
                            i = base_i + li
                            k = base_k + lj
                            w_vec[k * n + i] = total_coeff
            
            add_term(u_vec, v_vec, w_vec)

    # --- Part 2: A1 B12 (4x4x1) Schoolbook ---
    # A1 is 4x4. B12 is 4x1.
    # C is 4x1.
    # C_i = sum_j A1_ij B12_j
    # Rank 16.
    for i in range(4):
        for j in range(4):
            u_vec = [0] * (n*m)
            v_vec = [0] * (m*p)
            w_vec = [0] * (p*n)
            
            # A1 index i,j -> A index i*5+j
            u_vec[i * m + j] = 1
            # B12 index j,0 -> B index j*5+4 (since B12 is col 4)
            v_vec[j * p + 4] = 1
            # C index i,0 -> W index 0*4+i = i
            w_vec[0 * n + i] = 1
            
            add_term(u_vec, v_vec, w_vec)

    # --- Part 3: A2 B21 (4x1x4) Schoolbook ---
    # A2 is 4x1. B21 is 1x4.
    # C is 4x4.
    # C_ik = A2_i B21_k
    # Rank 16.
    for i in range(4):
        for k in range(4):
            u_vec = [0] * (n*m)
            v_vec = [0] * (m*p)
            w_vec = [0] * (p*n)
            
            # A2 index i,0 -> A index i*5+4
            u_vec[i * m + 4] = 1
            # B21 index 0,k -> B index 4*5+k
            v_vec[4 * p + k] = 1
            # C index i,k -> W index k*4+i
            w_vec[k * n + i] = 1
            
            add_term(u_vec, v_vec, w_vec)

    # --- Part 4: A2 B22 (4x1x1) Schoolbook ---
    # A2 is 4x1. B22 is 1x1.
    # C is 4x1.
    # C_i = A2_i B22
    # Rank 4.
    for i in range(4):
        u_vec = [0] * (n*m)
        v_vec = [0] * (m*p)
        w_vec = [0] * (p*n)
        
        u_vec[i * m + 4] = 1
        v_vec[4 * p + 4] = 1
        w_vec[0 * n + i] = 1
        
        add_term(u_vec, v_vec, w_vec)

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
