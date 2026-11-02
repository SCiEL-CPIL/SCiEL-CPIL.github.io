import numpy as np

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Global dimensions
    dim_U = n * m  # 9
    dim_V = m * p  # 24
    dim_W = p * n  # 24
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)
        
    def make_zero_vec(dim):
        return [0] * dim
        
    # Strassen decomposition for 2x2x2
    # Local indices: i,j,k in {0,1}
    # u: index i*2+j
    # v: index j*2+k
    # w: index k*2+i
    strassen_terms = [
        # u, v, w
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]), # M2
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 0]), # M3
        ([0, 0, 0, 1], [-1, 1, 0, 0], [1, 0, 1, 0]), # M4
        ([1, 0, 1, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1])  # M7
    ]
    
    # Sub-problem 1: 2x2x8
    # i in {0,1}, j in {0,1}, k in {0..7}
    for b in range(4):
        k_offset = b * 2
        for u_loc, v_loc, w_loc in strassen_terms:
            u = make_zero_vec(dim_U)
            v = make_zero_vec(dim_V)
            w = make_zero_vec(dim_W)
            
            # Map local u indices to global
            # local index l = i*2 + j
            # global index g = i*3 + j
            for l, val in enumerate(u_loc):
                if val != 0:
                    i = l // 2
                    j = l % 2
                    g = i * 3 + j
                    u[g] = val
                    
            # Map local v indices to global
            # local index l = j*2 + k_local
            # k_local is 0 or 1
            # global k = k_offset + k_local
            # global index g = j*8 + k
            for l, val in enumerate(v_loc):
                if val != 0:
                    j = l // 2
                    k_local = l % 2
                    k = k_offset + k_local
                    g = j * 8 + k
                    v[g] = val
                    
            # Map local w indices to global
            # local index l = k_local*2 + i
            # global index g = k*3 + i
            for l, val in enumerate(w_loc):
                if val != 0:
                    k_local = l // 2
                    i = l % 2
                    k = k_offset + k_local
                    g = k * 3 + i
                    w[g] = val
                    
            add_term(u, v, w)
            
    # Sub-problem 2: 2x1x8
    # i in {0,1}, j=2, k in {0..7}
    for i in range(2):
        for k in range(8):
            u = make_zero_vec(dim_U)
            v = make_zero_vec(dim_V)
            w = make_zero_vec(dim_W)
            
            u[i * 3 + 2] = 1
            v[2 * 8 + k] = 1
            w[k * 3 + i] = 1
            
            add_term(u, v, w)
            
    # Sub-problem 3: 1x2x8
    # i=2, j in {0,1}, k in {0..7}
    for j in range(2):
        for k in range(8):
            u = make_zero_vec(dim_U)
            v = make_zero_vec(dim_V)
            w = make_zero_vec(dim_W)
            
            u[2 * 3 + j] = 1
            v[j * 8 + k] = 1
            w[k * 3 + 2] = 1
            
            add_term(u, v, w)
            
    # Sub-problem 4: 1x1x8
    # i=2, j=2, k in {0..7}
    for k in range(8):
        u = make_zero_vec(dim_U)
        v = make_zero_vec(dim_V)
        w = make_zero_vec(dim_W)
        
        u[2 * 3 + 2] = 1
        v[2 * 8 + k] = 1
        w[k * 3 + 2] = 1
        
        add_term(u, v, w)
        
    rank = len(U_list)
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'])}")
