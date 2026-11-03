import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = 4, 4, 8
    
    # Strassen 2x2x2: 7 terms
    # For 4x4x8: split into two 4x4x4 problems, each using 8 Strassen 2x2x2 block mults.
    # Total: 2 * 8 * 7 = 112 terms.
    
    def strassen_2x2x2_terms():
        """Return the 7 Strassen terms for 2x2x2 matrix multiplication.
        Each term is (u_list, v_list, w_list) with flat indices for 2x2 matrices.
        U index: i*2+j, V index: j*2+k, W index: k*2+i
        """
        terms = []
        # M1 = (A11+A22)(B11+B22)
        terms.append(([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]))
        # M2 = (A21+A22)B11
        terms.append(([(2,1),(3,1)], [(0,1)], [(2,1)]))
        # M3 = A11(B21-B22)
        terms.append(([(0,1)], [(2,1),(3,-1)], [(1,1)]))
        # M4 = A22(B12-B11)
        terms.append(([(3,1)], [(1,1),(0,-1)], [(1,1)]))
        # M5 = (A11+A12)B22
        terms.append(([(0,1),(1,1)], [(3,1)], [(1,1)]))
        # M6 = (A21-A11)(B11+B12)
        terms.append(([(2,1),(0,-1)], [(0,1),(1,1)], [(2,1)]))
        # M7 = (A12-A22)(B21+B22)
        terms.append(([(1,1),(3,-1)], [(2,1),(3,1)], [(3,1)]))
        return terms
    
    strassen_terms = strassen_2x2x2_terms()
    
    U_list = []
    V_list = []
    W_list = []
    
    # For each of the two 4x4x4 blocks (left and right):
    for block_offset in [0, 4]:
        block_mults = [
            (0, 0, 0, 0, 0, 0),
            (0, 1, 1, 0, 0, 0),
            (0, 0, 0, 1, 0, 1),
            (0, 1, 1, 1, 0, 1),
            (1, 0, 0, 0, 1, 0),
            (1, 1, 1, 0, 1, 0),
            (1, 0, 0, 1, 1, 1),
            (1, 1, 1, 1, 1, 1),
        ]
        
        for (ar, ac, br, bc, cr, cc) in block_mults:
            for (u_l, v_l, w_l) in strassen_terms:
                u_entries = []
                v_entries = []
                w_entries = []
                
                for (u_idx, u_coeff) in u_l:
                    local_i = u_idx // 2
                    local_j = u_idx % 2
                    global_u = (ar*2 + local_i) * 4 + (ac*2 + local_j)
                    u_entries.append((global_u, u_coeff))
                
                for (v_idx, v_coeff) in v_l:
                    local_j = v_idx // 2
                    local_k = v_idx % 2
                    global_v = (br*2 + local_j) * 8 + (block_offset + bc*2 + local_k)
                    v_entries.append((global_v, v_coeff))
                
                for (w_idx, w_coeff) in w_l:
                    # W convention: W[k*n+i, r] where k is output col, i is output row
                    local_k = w_idx // 2
                    local_i = w_idx % 2
                    # Output row = cr*2 + local_i, output col = block_offset + cc*2 + local_k
                    global_w = (block_offset + cc*2 + local_k) * 4 + (cr*2 + local_i)
                    w_entries.append((global_w, w_coeff))
                
                U_list.append(u_entries)
                V_list.append(v_entries)
                W_list.append(w_entries)
    
    total_rank = len(U_list)
    
    # Verify the decomposition
    def verify_decomposition(U_terms, V_terms, W_terms, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for u_l, v_l, w_l in zip(U_terms, V_terms, W_terms):
                        u_val = sum(c for idx, c in u_l if idx == i*m+j)
                        v_val = sum(c for idx, c in v_l if idx == j*p+k)
                        w_val = sum(c for idx, c in w_l if idx == k*n+i)
                        total += u_val * v_val * w_val
                    if abs(total - 1) > 1e-9:
                        return False
        return True
    
    if not verify_decomposition(U_list, V_list, W_list, n, m, p):
        # Fallback to naive
        U_list = []
        V_list = []
        W_list = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_list.append([(i*m+j, 1)])
                    V_list.append([(j*p+k, 1)])
                    W_list.append([(k*n+i, 1)])
        total_rank = len(U_list)
    
    # Convert to matrices
    rank = total_rank
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for r in range(rank):
        for (idx, coeff) in U_list[r]:
            U[idx, r] = coeff
        for (idx, coeff) in V_list[r]:
            V[idx, r] = coeff
        for (idx, coeff) in W_list[r]:
            W[idx, r] = coeff
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
