import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    DEADLINE = 140  # seconds, safety margin before 150s limit
    
    # Rank 20: 16 (Winograd F(4) for 4x4x4) + 4 (Winograd F(2) for 4x4x1)
    rank = 20
    U = [[0] * rank for _ in range(n * m)]  # 16 x 20
    V = [[0] * rank for _ in range(m * p)]  # 20 x 20
    W = [[0] * rank for _ in range(p * n)]  # 20 x 20
    
    col = 0
    
    # Winograd F(4) transform for 4x4x4 matrix multiplication
    # F(4) = [[1, 0, 0, 0],
    #          [0, 1, 1, 0],
    #          [0, 1, -1, 0],
    #          [0, 0, 0, 1]]
    # 
    # For multiplying A(4x4) * B(4x4) = C(4x4):
    # M = F(4) @ A @ F(4)^T  (16 multiplications)
    # C = F(4)^{-1} @ M @ F(4)
    #
    # F(4) inverse = F(4)^T / 4 for the off-diagonal elements
    # Actually F(4) is symmetric, and F(4)^{-1} = (1/4)*F(4) for the 2x2 block
    # Let me use the standard Winograd F(4) approach:
    #
    # The 4x4x4 tensor can be decomposed using the Kronecker product structure.
    # F(4) ⊗ F(4) gives a 16x16 transform with 16 nonzero entries per row/col
    # 
    # Actually, let me use the direct Winograd F(4) construction:
    # The 4x4x4 multiplication tensor T[i,j,k] = delta_{i,k} * delta_{j,i}
    # 
    # Using Winograd: if we can write T = U * V * W^T with rank r
    # For F(4): the 2x2x2 case uses rank 7 (Strassen)
    # For 4x4x4: we can use the Kronecker product of two 2x2x2 Strassen decomp
    # giving rank 7*7 = 49... that's what we had before.
    #
    # Winograd F(4) approach: 
    # F(4) = F(2) ⊗ F(2) where F(2) = [[1,0],[0,1],[0,1],[0,-1]]... no
    #
    # Let me think differently. The Winograd F(n) algorithm for n×n×n:
    # F(2) has rank 7 (Strassen)
    # F(4) = F(2) ⊗ F(2) would give rank 7^2 = 49
    #
    # But there's a better approach: use the fact that 4x4x4 can be done
    # with the "super Strassen" or by noting that the 4x4x4 tensor is
    # the Kronecker product of two 2x2x2 tensors.
    #
    # Actually, the minimum rank of 4x4x4 is known to be at most 49 (via Kronecker)
    # but potentially lower. The best known upper bound is 49.
    #
    # Wait - let me reconsider. The inspiration mentions Winograd F(4) giving rank 16.
    # That would be the schoolbook rank for 4x4x4! That doesn't make sense as an improvement.
    #
    # Let me re-read: "Winograd F(4) transform for the 4×4×4 sub-problem, which gives rank 16"
    # Rank 16 for 4x4x4 is just schoolbook. That's worse than 49.
    #
    # Hmm, but maybe the inspiration is wrong or I'm misunderstanding. Let me stick with
    # the known good decomposition: double Strassen (rank 49) + schoolbook (rank 16) = 65.
    #
    # Actually wait - let me reconsider the problem. We're multiplying 4x4 by 4x5.
    # The output is 4x5. So we need C[i,k] for i in 0..3, k in 0..4.
    #
    # C = A * B where A is 4x4, B is 4x5, C is 4x5.
    #
    # We can split B into B_left (4x4) and B_right (4x1).
    # C_left = A * B_left (4x4x4, rank 49 via double Strassen)
    # C_right = A * B_right (4x4x1, rank 16 via schoolbook)
    # Total: 65.
    #
    # Can we do better? The 4x4x1 part: C[i] = sum_j A[i,j] * B[j]
    # This is just a linear combination, rank 16 (each output is a sum of 4 products).
    # Actually, this is rank 16 because we have 4 outputs, each requiring 4 multiplications.
    # But we can potentially share computations... no, each A[i,j]*B[j] is a distinct product.
    #
    # So the minimum is at least 49 + 16 = 65? Not necessarily - maybe there's a smarter
    # decomposition that mixes the two parts.
    #
    # For now, let's keep the rank 65 solution which is verified working.
    
    # Let me just keep the current working solution with rank 65
    rank = 65
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank for _ in range(p * n)]  # 20 x 65
    
    def add_term(col, u_list, v_list, w_list):
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c
    
    col = 0
    
    # Strassen 2x2x2 decomposition
    inner_defs = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,0,1),(1,1,1)]),       # M1
        ([(2,1),(3,1)], [(0,1)], [(1,0,1),(1,1,-1)]),             # M2
        ([(0,1)], [(1,1),(3,-1)], [(0,1,1),(1,1,1)]),             # M3
        ([(3,1)], [(2,1),(0,-1)], [(0,0,1),(1,0,1)]),             # M4
        ([(0,1),(1,1)], [(3,1)], [(0,0,-1),(0,1,1)]),             # M5
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(1,1,1)]),               # M6
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,0,1)]),               # M7
    ]
    
    # Super Strassen for 4x4x4
    super_defs = [
        ([1,0,0,1], [1,0,0,1], [(0,0,1),(1,1,1)]),       # M1 -> C00, C11
        ([0,0,1,1], [1,0,0,0], [(1,0,1),(1,1,-1)]),      # M2 -> C10, -C11
        ([1,0,0,0], [0,1,0,-1], [(0,1,1),(1,1,1)]),      # M3 -> C01, C11
        ([0,0,0,1], [-1,0,1,0], [(0,0,1),(1,0,1)]),      # M4 -> C00, C10
        ([1,1,0,0], [0,0,0,1], [(0,0,-1),(0,1,1)]),      # M5 -> -C00, C01
        ([-1,0,1,0], [1,1,0,0], [(1,1,1)]),              # M6 -> C11
        ([0,1,0,-1], [0,0,1,1], [(0,0,1)]),              # M7 -> C00
    ]
    
    def get_p_element(p_elem_idx, p_block_coeffs):
        r = p_elem_idx // 2
        c = p_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(p_block_coeffs):
            if coeff == 0:
                continue
            if b_idx == 0:    # A00
                gi, gj = r, c
            elif b_idx == 1:  # A01
                gi, gj = r, 2 + c
            elif b_idx == 2:  # A10
                gi, gj = 2 + r, c
            elif b_idx == 3:  # A11
                gi, gj = 2 + r, 2 + c
            res.append((gi, gj, coeff))
        return res
    
    def get_q_element(q_elem_idx, q_block_coeffs):
        r = q_elem_idx // 2
        c = q_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(q_block_coeffs):
            if coeff == 0:
                continue
            if b_idx == 0:    # B00
                gj, gk = r, c
            elif b_idx == 1:  # B01
                gj, gk = r, 2 + c
            elif b_idx == 2:  # B10
                gj, gk = 2 + r, c
            elif b_idx == 3:  # B11
                gj, gk = 2 + r, 2 + c
            res.append((gj, gk, coeff))
        return res
    
    # Build all 49 terms for 4x4x4
    for s_idx, (p_bc, q_bc, c_contribs) in enumerate(super_defs):
        for i_idx, (p_elems, q_elems, r_contribs) in enumerate(inner_defs):
            u_list = []
            v_list = []
            w_list = []
            
            for p_elem, coeff in p_elems:
                for gi, gj, c in get_p_element(p_elem, p_bc):
                    u_list.append((gi, gj, coeff * c))
            
            for q_elem, coeff in q_elems:
                for gj, gk, c in get_q_element(q_elem, q_bc):
                    v_list.append((gj, gk, coeff * c))
            
            for br, bc, sign in c_contribs:
                for lr, lc, local_coeff in r_contribs:
                    gi = br * 2 + lr
                    gk = bc * 2 + lc
                    w_list.append((gk, gi, sign * local_coeff))
            
            if u_list and v_list and w_list:
                add_term(col, u_list, v_list, w_list)
                col += 1
    
    # Part 2: 4x4x1 (rank 16) - schoolbook
    # C[i,4] = sum_j A[i,j] * B[j,4]
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1  # W[k*n+i] with k=4, i=i
            col += 1
    
    if time.time() - start_time > DEADLINE - 5:
        pass
    
    return {"U": U, "V": V, "W": W}
