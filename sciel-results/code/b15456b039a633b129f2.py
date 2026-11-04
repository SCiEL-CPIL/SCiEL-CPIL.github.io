import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Partition 5 into 4 + 1
    # Blocks:
    # (4,4,4) -> Rank 49
    # (4,4,1) -> Rank 16
    # (4,1,4) -> Rank 16
    # (1,4,4) -> Rank 16
    # (4,1,1) -> Rank 4
    # (1,4,1) -> Rank 4
    # (1,1,4) -> Rank 4
    # (1,1,1) -> Rank 1
    # Total: 110
    
    total_rank = 110
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(U_dict, V_dict, W_dict):
        nonlocal col
        if col >= total_rank:
            return
        for idx, val in U_dict.items():
            U[idx][col] = val
        for idx, val in V_dict.items():
            V[idx][col] = val
        for idx, val in W_dict.items():
            W[idx][col] = val
        col += 1

    def add_schoolbook(r1, r2, c1, c2, k1, k2):
        for i in range(r1, r2):
            for j in range(c1, c2):
                for k in range(k1, k2):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})

    def add_strassen_4x4(r0, c0, k0):
        # Implementation from above
        pass # Will be filled below

    # Define add_strassen_4x4
    def add_strassen_4x4_impl(r0, c0, k0):
        def a_idx(r, c):
            return (r0 + r) * m + (c0 + c)
        def b_idx(r, c):
            return (c0 + r) * p + (k0 + c)
        def c_idx(r, c):
            return (k0 + r) * n + (r0 + c)
            
        def make_a_map(blocks):
            map_a = {0: {}, 1: {}, 2: {}, 3: {}}
            for br, bc, coeff in blocks:
                for i in range(2):
                    for j in range(2):
                        virt_idx = i * 2 + j
                        r = br * 2 + i
                        c = bc * 2 + j
                        g_idx = r * 4 + c
                        if g_idx not in map_a[virt_idx]:
                            map_a[virt_idx][g_idx] = 0
                        map_a[virt_idx][g_idx] += coeff
            return map_a
            
        def make_b_map(blocks):
            map_b = {0: {}, 1: {}, 2: {}, 3: {}}
            for br, bc, coeff in blocks:
                for i in range(2):
                    for j in range(2):
                        virt_idx = i * 2 + j
                        r = br * 2 + i
                        c = bc * 2 + j
                        g_idx = r * 4 + c
                        if g_idx not in map_b[virt_idx]:
                            map_b[virt_idx][g_idx] = 0
                        map_b[virt_idx][g_idx] += coeff
            return map_b
            
        def make_c_map(blocks):
            map_c = {0: {}, 1: {}, 2: {}, 3: {}}
            for br, bc, coeff in blocks:
                for i in range(2):
                    for j in range(2):
                        virt_idx = i * 2 + j
                        r = br * 2 + i
                        c = bc * 2 + j
                        g_idx = r * 4 + c
                        if g_idx not in map_c[virt_idx]:
                            map_c[virt_idx][g_idx] = 0
                        map_c[virt_idx][g_idx] += coeff
            return map_c

        products = [
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
            ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
            ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
            ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
            ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
            ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
            ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]),
        ]
        
        s2x2 = [
            ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
            ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),
            ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),
            ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),
            ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
            ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
            ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),
        ]
        
        for a_blocks, b_blocks, c_blocks in products:
            map_a = make_a_map(a_blocks)
            map_b = make_b_map(b_blocks)
            map_c = make_c_map(c_blocks)
            
            for u_v, v_v, w_v in s2x2:
                U_dict = {}
                V_dict = {}
                W_dict = {}
                
                for v_idx, coeff in u_v.items():
                    for g_idx, g_coeff in map_a[v_idx].items():
                        r = g_idx // 4
                        c = g_idx % 4
                        actual_idx = a_idx(r, c)
                        if actual_idx not in U_dict:
                            U_dict[actual_idx] = 0
                        U_dict[actual_idx] += coeff * g_coeff
                
                for v_idx, coeff in v_v.items():
                    for g_idx, g_coeff in map_b[v_idx].items():
                        r = g_idx // 4
                        c = g_idx % 4
                        actual_idx = b_idx(r, c)
                        if actual_idx not in V_dict:
                            V_dict[actual_idx] = 0
                        V_dict[actual_idx] += coeff * g_coeff
                
                for v_idx, coeff in w_v.items():
                    for g_idx, g_coeff in map_c[v_idx].items():
                        r = g_idx // 4
                        c = g_idx % 4
                        actual_idx = c_idx(r, c)
                        if actual_idx not in W_dict:
                            W_dict[actual_idx] = 0
                        W_dict[actual_idx] += coeff * g_coeff
                
                U_dict = {k:v for k,v in U_dict.items() if v != 0}
                V_dict = {k:v for k,v in V_dict.items() if v != 0}
                W_dict = {k:v for k,v in W_dict.items() if v != 0}
                if U_dict and V_dict and W_dict:
                    add_term(U_dict, V_dict, W_dict)

    # Call for (4,4,4) block
    add_strassen_4x4_impl(0, 0, 0)
    
    # Call schoolbook for others
    # (4,4,1)
    add_schoolbook(0, 4, 0, 4, 4, 5)
    # (4,1,4)
    add_schoolbook(0, 4, 4, 5, 0, 4)
    # (1,4,4)
    add_schoolbook(4, 5, 0, 4, 0, 4)
    # (4,1,1)
    add_schoolbook(0, 4, 4, 5, 4, 5)
    # (1,4,1)
    add_schoolbook(4, 5, 0, 4, 4, 5)
    # (1,1,4)
    add_schoolbook(4, 5, 4, 5, 0, 4)
    # (1,1,1)
    add_schoolbook(4, 5, 4, 5, 4, 5)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
