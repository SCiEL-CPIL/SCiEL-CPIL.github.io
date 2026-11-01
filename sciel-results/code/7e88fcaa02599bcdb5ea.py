"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Strategy: Use Strassen for two 2x2x2 blocks (cols 0-1, rows 0-1 and cols 2-3, rows 2-3)
    # and schoolbook for the remaining terms.
    # Rank breakdown:
    # - Strassen 2x2x2 for A[0:2,0:2]*B[0:2,0:2] -> C[0:2,0:2]: 7 terms
    # - Strassen 2x2x2 for A[0:2,2:4]*B[2:4,0:2] -> C[0:2,0:2]: 7 terms  
    # - Schoolbook for A[0:2,4]*B[4,0:2] -> C[0:2,0:2]: 4 terms
    # - Schoolbook for A[0:2,0:5]*B[0:5,2:6] -> C[0:2,2:6]: 2*5*4 = 40 terms
    # Total: 7 + 7 + 4 + 40 = 58
    
    rank = 58
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_rank_step(u_pairs, v_pairs, w_pairs):
        nonlocal col
        if col >= rank:
            return False
        for idx, val in u_pairs:
            U[idx][col] = val
        for idx, val in v_pairs:
            V[idx][col] = val
        for idx, val in w_pairs:
            W[idx][col] = val
        col += 1
        return True
    
    def check_time():
        return time.time() - start_time < deadline
    
    # Part 1: Strassen for A1(2x2) * B1(2x2) -> C[0:2, 0:2]
    # A1: (0,0)=0, (0,1)=1, (1,0)=5, (1,1)=6
    # B1: (0,0)=0, (0,1)=1, (1,0)=6, (1,1)=7
    # C: (0,0)=0, (1,0)=1, (0,1)=2, (1,1)=3
    
    if not check_time():
        return {"U": U, "V": V, "W": W}
    
    # m1 = (a00+a11)(b00+b11) -> C00 += 1, C11 += 1
    add_rank_step([(0, 1), (6, 1)], [(0, 1), (7, 1)], [(0, 1), (3, 1)])
    
    # m2 = (a10+a11)*b00 -> C10 += 1, C11 -= 1
    add_rank_step([(5, 1), (6, 1)], [(0, 1)], [(1, 1), (3, -1)])
    
    # m3 = a00*(b01-b11) -> C01 += 1, C11 += 1
    add_rank_step([(0, 1)], [(1, 1), (7, -1)], [(2, 1), (3, 1)])
    
    # m4 = a11*(b10-b00) -> C00 += 1, C10 += 1
    add_rank_step([(6, 1)], [(6, 1), (0, -1)], [(0, 1), (1, 1)])
    
    # m5 = (a00+a01)*b11 -> C00 -= 1, C01 += 1
    add_rank_step([(0, 1), (1, 1)], [(7, 1)], [(0, -1), (2, 1)])
    
    # m6 = (a10-a00)*(b00+b01) -> C11 += 1
    add_rank_step([(5, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)])
    
    # m7 = (a01-a11)*(b10+b11) -> C00 += 1
    add_rank_step([(1, 1), (6, -1)], [(6, 1), (7, 1)], [(0, 1)])

    if not check_time():
        return {"U": U, "V": V, "W": W}
    
    # Part 2: Strassen for A2a(2x2) * B2a(2x2) -> C[0:2, 0:2]
    # A2a: cols 2,3 -> (0,2)=2, (0,3)=3, (1,2)=7, (1,3)=8
    # B2a: rows 2,3 -> (2,0)=12, (2,1)=13, (3,0)=18, (3,1)=19
    # C: same as Part 1
    
    # m1 = (a02+a13)(b20+b31) -> C00 += 1, C11 += 1
    add_rank_step([(2, 1), (8, 1)], [(12, 1), (19, 1)], [(0, 1), (3, 1)])
    
    # m2 = (a12+a13)*b20 -> C10 += 1, C11 -= 1
    add_rank_step([(7, 1), (8, 1)], [(12, 1)], [(1, 1), (3, -1)])
    
    # m3 = a02*(b21-b31) -> C01 += 1, C11 += 1
    add_rank_step([(2, 1)], [(13, 1), (19, -1)], [(2, 1), (3, 1)])
    
    # m4 = a13*(b30-b20) -> C00 += 1, C10 += 1
    add_rank_step([(8, 1)], [(18, 1), (12, -1)], [(0, 1), (1, 1)])
    
    # m5 = (a02+a03)*b31 -> C00 -= 1, C01 += 1
    add_rank_step([(2, 1), (3, 1)], [(19, 1)], [(0, -1), (2, 1)])
    
    # m6 = (a12-a02)*(b20+b21) -> C11 += 1
    add_rank_step([(7, 1), (2, -1)], [(12, 1), (13, 1)], [(3, 1)])
    
    # m7 = (a03-a13)*(b30+b31) -> C00 += 1
    add_rank_step([(3, 1), (8, -1)], [(18, 1), (19, 1)], [(0, 1)])

    if not check_time():
        return {"U": U, "V": V, "W": W}
    
    # Part 3: Schoolbook for A2b(2x1) * B2b(1x2) -> C[0:2, 0:2]
    # A2b: col 4 -> (0,4)=4, (1,4)=9
    # B2b: row 4 -> (4,0)=24, (4,1)=25
    
    # i=0, j=4, k=0
    add_rank_step([(4, 1)], [(24, 1)], [(0, 1)])
    # i=0, j=4, k=1
    add_rank_step([(4, 1)], [(25, 1)], [(2, 1)])
    # i=1, j=4, k=0
    add_rank_step([(9, 1)], [(24, 1)], [(1, 1)])
    # i=1, j=4, k=1
    add_rank_step([(9, 1)], [(25, 1)], [(3, 1)])

    if not check_time():
        return {"U": U, "V": V, "W": W}
    
    # Part 4: Schoolbook for A(2x5) * B_rest(5x4) -> C[0:2, 2:6]
    # All A entries, B entries with k in 2..5
    # C indices: k in 2..5, i in 0..1 -> k*2+i
    
    for i in range(2):
        for j in range(5):
            for k in range(2, 6):
                if not check_time():
                    return {"U": U, "V": V, "W": W}
                u_idx = i * 5 + j
                v_idx = j * 6 + k
                w_idx = k * 2 + i
                add_rank_step([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])

    # If we didn't use all ranks, the decomposition is still valid
    # The rank is determined by the number of non-zero columns
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
