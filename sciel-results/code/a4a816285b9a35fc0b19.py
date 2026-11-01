"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    # Target rank is 58
    # 7 (Strassen 2x2x2 for A1B1) + 7 (Strassen 2x2x2 for A2aB2a) + 4 (Schoolbook 2x1x2 for A2bB2b) + 40 (Schoolbook 2x5x4 for rest)
    rank = 58
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
        col += 1

    # Part 1: A1 (2x2) * B1 (2x2) -> C[0:2, 0:2]
    # A1 indices: (0,0)->0, (0,1)->1, (1,0)->5, (1,1)->6
    # B1 indices: (0,0)->0, (0,1)->1, (1,0)->6, (1,1)->7
    # C indices: (0,0)->0, (1,0)->1, (0,1)->2, (1,1)->3
    
    # m1 = (a+d)(e+h)
    add_term(0, 1, 0, 1, 0, 1)
    add_term(6, 1, 0, 0, 0, 0) # Wait, add_term handles one entry per call? 
    # My add_term above is flawed for multiple non-zero entries in one rank step.
    # I need to set multiple entries in U, V, W for the SAME column index.
    
    # Let's rewrite the logic to handle vectors of indices/values for one rank step.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_rank_step(u_pairs, v_pairs, w_pairs):
        nonlocal col
        if col >= rank:
            return
        for idx, val in u_pairs:
            U[idx][col] = val
        for idx, val in v_pairs:
            V[idx][col] = val
        for idx, val in w_pairs:
            W[idx][col] = val
        col += 1

    # Part 1: Strassen for A1B1
    # m1 = (a+d)(e+h) -> C00 += 1, C11 += 1
    add_rank_step([(0, 1), (6, 1)], [(0, 1), (7, 1)], [(0, 1), (3, 1)])
    
    # m2 = (c+d)e -> C10 += 1, C11 -= 1
    add_rank_step([(5, 1), (6, 1)], [(0, 1)], [(1, 1), (3, -1)])
    
    # m3 = a(f-h) -> C01 += 1, C11 += 1
    add_rank_step([(0, 1)], [(1, 1), (7, -1)], [(2, 1), (3, 1)])
    
    # m4 = d(g-e) -> C00 += 1, C10 += 1
    add_rank_step([(6, 1)], [(6, 1), (0, -1)], [(0, 1), (1, 1)])
    
    # m5 = (a+b)h -> C00 -= 1, C01 += 1
    add_rank_step([(0, 1), (1, 1)], [(7, 1)], [(0, -1), (2, 1)])
    
    # m6 = (c-a)(e+f) -> C11 += 1
    add_rank_step([(5, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)])
    
    # m7 = (b-d)(g+h) -> C00 += 1
    add_rank_step([(1, 1), (6, -1)], [(6, 1), (7, 1)], [(0, 1)])

    # Part 2: Strassen for A2a B2a (cols 2,3 of A; rows 2,3 of B) -> C[0:2, 0:2]
    # A2a indices: (0,2)->2, (0,3)->3, (1,2)->7, (1,3)->8
    # B2a indices: (2,0)->12, (2,1)->13, (3,0)->18, (3,1)->19
    # C indices: same as Part 1
    
    # m1 = (a+d)(e+h)
    add_rank_step([(2, 1), (8, 1)], [(12, 1), (19, 1)], [(0, 1), (3, 1)])
    
    # m2 = (c+d)e
    add_rank_step([(7, 1), (8, 1)], [(12, 1)], [(1, 1), (3, -1)])
    
    # m3 = a(f-h)
    add_rank_step([(2, 1)], [(13, 1), (19, -1)], [(2, 1), (3, 1)])
    
    # m4 = d(g-e)
    add_rank_step([(8, 1)], [(18, 1), (12, -1)], [(0, 1), (1, 1)])
    
    # m5 = (a+b)h
    add_rank_step([(2, 1), (3, 1)], [(19, 1)], [(0, -1), (2, 1)])
    
    # m6 = (c-a)(e+f)
    add_rank_step([(7, 1), (2, -1)], [(12, 1), (13, 1)], [(3, 1)])
    
    # m7 = (b-d)(g+h)
    add_rank_step([(3, 1), (8, -1)], [(18, 1), (19, 1)], [(0, 1)])

    # Part 3: Schoolbook for A2b B2b (col 4 of A; row 4 of B) -> C[0:2, 0:2]
    # A2b indices: (0,4)->4, (1,4)->9
    # B2b indices: (4,0)->24, (4,1)->25
    
    # i=0, j=4, k=0
    add_rank_step([(4, 1)], [(24, 1)], [(0, 1)])
    # i=0, j=4, k=1
    add_rank_step([(4, 1)], [(25, 1)], [(2, 1)])
    # i=1, j=4, k=0
    add_rank_step([(9, 1)], [(24, 1)], [(1, 1)])
    # i=1, j=4, k=1
    add_rank_step([(9, 1)], [(25, 1)], [(3, 1)])

    # Part 4: Schoolbook for A B_rest (all A, cols 2-5 of B) -> C[0:2, 2:6]
    # A indices: all 10
    # B indices: j in 0..4, k in 2..5
    # C indices: k in 2..5, i in 0..1 -> k*2+i
    
    for i in range(2):
        for j in range(5):
            for k in range(2, 6):
                u_idx = i * 5 + j
                v_idx = j * 6 + k
                w_idx = k * 2 + i
                add_rank_step([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])

    if col != rank:
        # If we didn't use all ranks, it's still valid, but rank should be minimal.
        # If we used more, it's an error in count.
        pass

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
