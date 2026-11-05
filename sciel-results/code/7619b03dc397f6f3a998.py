import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    DEADLINE = 130  # seconds safety margin

    # Strategy: Winograd F(4) for 4x4x4 (rank 16) + schoolbook for 4x4x1 (rank 16) = rank 32
    rank = 32
    U = [[0] * rank for _ in range(n * m)]  # 16 x 32
    V = [[0] * rank for _ in range(m * p)]  # 20 x 32
    W = [[0] * rank for _ in range(p * n)]  # 20 x 32

    # Winograd F(4) transform matrix
    T = [[1, 0, 1, 0],
         [0, 1, 0, -1],
         [1, 0, -1, 0],
         [0, 1, 0, 1]]
    
    col = 0
    for r in range(4):
        for s in range(4):
            for i in range(4):
                U[i * 4 + s][col] += T[r][i]
            for j in range(4):
                V[j * 5 + r][col] += T[s][j]
            for i in range(4):
                for j in range(4):
                    W[j * 4 + i][col] += T[r][i] * T[s][j]
            col += 1
    
    # Part 2: 4x4x1 schoolbook (rank 16)
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
            col += 1
    
    # Verify the decomposition is correct
    if time.time() - start_time < DEADLINE - 10:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check: for the 4x4x4 part, the decomposition should give correct matrix mult
        # C[i,j] = sum_k A[i,k]*B[k,j]
        # Decomposition: sum_alpha (sum_{i,k} U[i*4+k, alpha]*A[i,k]) * (sum_{k,j} V[k*5+j, alpha]*B[k,j]) * W[j*4+i, alpha]
        
        # Test with random matrices
        rng = np.random.RandomState(42)
        A_test = rng.rand(4, 4)
        B_test = rng.rand(4, 5)
        
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((4, 5))
        for alpha in range(rank):
            a_part = sum(A_test[i, k] * U_arr[i * 4 + k, alpha] for i in range(4) for k in range(4))
            b_part = sum(B_test[k, j] * V_arr[k * 5 + j, alpha] for k in range(4) for j in range(5))
            for i in range(4):
                for j in range(5):
                    C_computed[i, j] += a_part * b_part * W_arr[j * 4 + i, alpha]
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            # F(4) transform is incorrect, fall back to Strassen + schoolbook (rank 65)
            rank = 65
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
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
            inner_defs = [
                ([(0,1),(3,1)], [(0,1),(3,1)], [(0,0,1),(1,1,1)]),
                ([(2,1),(3,1)], [(0,1)], [(1,0,1),(1,1,-1)]),
                ([(0,1)], [(1,1),(3,-1)], [(0,1,1),(1,1,1)]),
                ([(3,1)], [(2,1),(0,-1)], [(0,0,1),(1,0,1)]),
                ([(0,1),(1,1)], [(3,1)], [(0,0,-1),(0,1,1)]),
                ([(2,1),(0,-1)], [(0,1),(1,1)], [(1,1,1)]),
                ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,0,1)]),
            ]
            super_defs = [
                ([1,0,0,1], [1,0,0,1], [(0,0,1),(1,1,1)]),
                ([0,0,1,1], [1,0,0,0], [(1,0,1),(1,1,-1)]),
                ([1,0,0,0], [0,1,0,-1], [(0,1,1),(1,1,1)]),
                ([0,0,0,1], [-1,0,1,0], [(0,0,1),(1,0,1)]),
                ([1,1,0,0], [0,0,0,1], [(0,0,-1),(0,1,1)]),
                ([-1,0,1,0], [1,1,0,0], [(1,1,1)]),
                ([0,1,0,-1], [0,0,1,1], [(0,0,1)]),
            ]
            
            def get_p_element(p_elem_idx, p_block_coeffs):
                r = p_elem_idx // 2
                c = p_elem_idx % 2
                res = []
                for b_idx, coeff in enumerate(p_block_coeffs):
                    if coeff == 0:
                        continue
                    if b_idx == 0:
                        gi, gj = r, c
                    elif b_idx == 1:
                        gi, gj = r, 2 + c
                    elif b_idx == 2:
                        gi, gj = 2 + r, c
                    elif b_idx == 3:
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
                    if b_idx == 0:
                        gj, gk = r, c
                    elif b_idx == 1:
                        gj, gk = r, 2 + c
                    elif b_idx == 2:
                        gj, gk = 2 + r, c
                    elif b_idx == 3:
                        gj, gk = 2 + r, 2 + c
                    res.append((gj, gk, coeff))
                return res
            
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
            
            for i in range(4):
                for j in range(4):
                    U[i * 4 + j][col] = 1
                    V[j * 5 + 4][col] = 1
                    W[4 * 4 + i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
