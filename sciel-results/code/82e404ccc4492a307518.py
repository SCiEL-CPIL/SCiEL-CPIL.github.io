import time

DIMENSIONS = [5, 5, 5]
_START_TIME = time.time()
TIME_LIMIT = 150.0

def _deadline_exceeded():
    return (time.time() - _START_TIME) > (TIME_LIMIT - 5.0)

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # 4+1 partition
    sizes = [4, 1]
    starts = [0, 4]
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        if _deadline_exceeded():
            return False
        U_col = [0.0] * (n * m)
        V_col = [0.0] * (m * p)
        W_col = [0.0] * (p * n)
        
        for idx, coeff in u_coeffs:
            U_col[idx] = coeff
        for idx, coeff in v_coeffs:
            V_col[idx] = coeff
        for idx, coeff in w_coeffs:
            W_col[idx] = coeff
            
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)
        return True

    def map_u(i, k, p_local, q_local):
        return (starts[i] + p_local) * m + (starts[k] + q_local)

    def map_v(k, j, q_local, t_local):
        return (starts[k] + q_local) * p + (starts[j] + t_local)

    def map_w(i, j, p_local, t_local):
        return (starts[j] + t_local) * n + (starts[i] + p_local)

    def add_strassen_4x4(i, k, j):
        """
        Add 49 terms for a 4x4x4 Strassen decomposition using recursive 2x2 Strassen.
        """
        def a_idx(bi, bk, p, q):
            return (starts[i] + bi*2 + p) * m + (starts[k] + bk*2 + q)
        
        def b_idx(bk, bj, q, t):
            return (starts[k] + bk*2 + q) * p + (starts[j] + bj*2 + t)
        
        def c_idx(bi, bj, p, t):
            return (starts[j] + bj*2 + t) * n + (starts[i] + bi*2 + p)
        
        outer_terms = [
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
        ]
        
        inner_terms = [
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
            ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
            ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
            ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
            ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
            ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
            ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]),
        ]
        
        for A_blocks, B_blocks, C_blocks in outer_terms:
            for X_entries, Y_entries, P_entries in inner_terms:
                if _deadline_exceeded():
                    return False
                u_coeffs = []
                v_coeffs = []
                w_coeffs = []
                
                for (x_row, x_col, x_coeff) in X_entries:
                    block_coeff = A_blocks.get((x_row, x_col), 0)
                    if block_coeff != 0:
                        for pr in range(2):
                            for pc in range(2):
                                gi = (starts[i] + x_row*2 + pr) * m + (starts[k] + x_col*2 + pc)
                                u_coeffs.append((gi, x_coeff * block_coeff))
                
                for (y_row, y_col, y_coeff) in Y_entries:
                    block_coeff = B_blocks.get((y_row, y_col), 0)
                    if block_coeff != 0:
                        for pr in range(2):
                            for pc in range(2):
                                gi = (starts[k] + y_row*2 + pr) * p + (starts[j] + y_col*2 + pc)
                                v_coeffs.append((gi, y_coeff * block_coeff))
                
                for (p_row, p_col, p_coeff) in P_entries:
                    block_coeff = C_blocks.get((p_row, p_col), 0)
                    if block_coeff != 0:
                        for pr in range(2):
                            for pc in range(2):
                                gi = (starts[j] + p_col*2 + pc) * n + (starts[i] + p_row*2 + pr)
                                w_coeffs.append((gi, p_coeff * block_coeff))
                
                u_dict = {}
                for idx, coeff in u_coeffs:
                    u_dict[idx] = u_dict.get(idx, 0) + coeff
                v_dict = {}
                for idx, coeff in v_coeffs:
                    v_dict[idx] = v_dict.get(idx, 0) + coeff
                w_dict = {}
                for idx, coeff in w_coeffs:
                    w_dict[idx] = w_dict.get(idx, 0) + coeff
                
                u_clean = [(idx, coeff) for idx, coeff in u_dict.items() if coeff != 0]
                v_clean = [(idx, coeff) for idx, coeff in v_dict.items() if coeff != 0]
                w_clean = [(idx, coeff) for idx, coeff in w_dict.items() if coeff != 0]
                
                if u_clean and v_clean and w_clean:
                    if not add_term(u_clean, v_clean, w_clean):
                        return False
        
        return True
    
    for i in range(2):
        for k in range(2):
            for j in range(2):
                if _deadline_exceeded():
                    break
                r = sizes[i]
                s = sizes[k]
                c = sizes[j]
                
                if r == 4 and s == 4 and c == 4:
                    if not add_strassen_4x4(i, k, j):
                        break
                else:
                    for p_local in range(r):
                        for q_local in range(s):
                            for t_local in range(c):
                                if _deadline_exceeded():
                                    break
                                if not add_term(
                                    [(map_u(i, k, p_local, q_local), 1)],
                                    [(map_v(k, j, q_local, t_local), 1)],
                                    [(map_w(i, j, p_local, t_local), 1)]
                                ):
                                    break
                    else:
                        continue
                    break
            else:
                continue
            break
    
    rank = len(U_cols)
    
    if rank == 0:
        # Fallback to schoolbook
        U_cols = []
        V_cols = []
        W_cols = []
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    add_term(
                        [(i * m + k, 1)],
                        [(k * p + j, 1)],
                        [(j * n + i, 1)]
                    )
        rank = len(U_cols)
    
    U = [[U_cols[c][r] for c in range(rank)] for r in range(n*m)]
    V = [[V_cols[c][r] for c in range(rank)] for r in range(m*p)]
    W = [[W_cols[c][r] for c in range(rank)] for r in range(p*n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
