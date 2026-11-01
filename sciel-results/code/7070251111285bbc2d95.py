"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit

    n, m, p = DIMENSIONS

    # Systematic decomposition:
    # j-groups: {0,1}, {2,3}, {4}
    # k-groups: {0,1}, {2,3}, {4,5}
    # For 2x2 j-groups with 2x2 k-groups: Strassen (rank 7)
    # For 2x1 j-groups with 2x2 k-groups: schoolbook (rank 4)
    # Total: 4*7 + 5*4 = 28 + 20 = 48... 
    # Wait, let me recount: 
    # (j={0,1},k={0,1}), (j={0,1},k={2,3}), (j={0,1},k={4,5}): 3 Strassen blocks = 21
    # (j={2,3},k={0,1}), (j={2,3},k={2,3}), (j={2,3},k={4,5}): 3 Strassen blocks = 21
    # (j={4},k={0,1}), (j={4},k={2,3}), (j={4},k={4,5}): 3 schoolbook blocks = 12
    # Total = 21 + 21 + 12 = 54

    rank = 54

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_rank_step(u_pairs, v_pairs, w_pairs):
        nonlocal col
        if col >= rank:
            return False
        if time.time() - start_time > deadline:
            return False
        for idx, val in u_pairs:
            U[idx][col] = val
        for idx, val in v_pairs:
            V[idx][col] = val
        for idx, val in w_pairs:
            W[idx][col] = val
        col += 1
        return True

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    def check_time():
        return time.time() - start_time < deadline

    def add_strassen_block(j1, j2, k1, k2):
        """Add Strassen rank-7 terms for A[:, j1:j2+1] * B[j1:j2+1, k1:k2+1]"""
        if not check_time():
            return

        # m1 = (a00+a11)(b00+b11) -> C00 += 1, C11 += 1
        add_rank_step(
            [(u_idx(0, j1), 1), (u_idx(1, j2), 1)],
            [(v_idx(j1, k1), 1), (v_idx(j2, k2), 1)],
            [(w_idx(k1, 0), 1), (w_idx(k2, 1), 1)]
        )

        # m2 = (a10+a11)*b00 -> C10 += 1, C11 -= 1
        add_rank_step(
            [(u_idx(1, j1), 1), (u_idx(1, j2), 1)],
            [(v_idx(j1, k1), 1)],
            [(w_idx(k1, 1), 1), (w_idx(k2, 1), -1)]
        )

        # m3 = a00*(b01-b11) -> C01 += 1, C11 += 1
        add_rank_step(
            [(u_idx(0, j1), 1)],
            [(v_idx(j1, k2), 1), (v_idx(j2, k2), -1)],
            [(w_idx(k2, 0), 1), (w_idx(k2, 1), 1)]
        )

        # m4 = a11*(b10-b00) -> C00 += 1, C10 += 1
        add_rank_step(
            [(u_idx(1, j2), 1)],
            [(v_idx(j2, k1), 1), (v_idx(j1, k1), -1)],
            [(w_idx(k1, 0), 1), (w_idx(k1, 1), 1)]
        )

        # m5 = (a00+a01)*b11 -> C00 -= 1, C01 += 1
        add_rank_step(
            [(u_idx(0, j1), 1), (u_idx(0, j2), 1)],
            [(v_idx(j2, k2), 1)],
            [(w_idx(k1, 0), -1), (w_idx(k2, 0), 1)]
        )

        # m6 = (a10-a00)*(b00+b01) -> C11 += 1
        add_rank_step(
            [(u_idx(1, j1), 1), (u_idx(0, j1), -1)],
            [(v_idx(j1, k1), 1), (v_idx(j1, k2), 1)],
            [(w_idx(k2, 1), 1)]
        )

        # m7 = (a01-a11)*(b10+b11) -> C00 += 1
        add_rank_step(
            [(u_idx(0, j2), 1), (u_idx(1, j2), -1)],
            [(v_idx(j2, k1), 1), (v_idx(j2, k2), 1)],
            [(w_idx(k1, 0), 1)]
        )

    def add_schoolbook_2x1x2(j, k1, k2):
        """Add schoolbook rank-4 terms for A[:, j] * B[j, k1:k2+1]"""
        if not check_time():
            return

        # i=0, j, k1
        add_rank_step([(u_idx(0, j), 1)], [(v_idx(j, k1), 1)], [(w_idx(k1, 0), 1)])
        # i=0, j, k2
        add_rank_step([(u_idx(0, j), 1)], [(v_idx(j, k2), 1)], [(w_idx(k2, 0), 1)])
        # i=1, j, k1
        add_rank_step([(u_idx(1, j), 1)], [(v_idx(j, k1), 1)], [(w_idx(k1, 1), 1)])
        # i=1, j, k2
        add_rank_step([(u_idx(1, j), 1)], [(v_idx(j, k2), 1)], [(w_idx(k2, 1), 1)])

    # Strassen blocks for 2x2 j-groups and 2x2 k-groups
    add_strassen_block(0, 1, 0, 1)
    if not check_time():
        return {"U": U, "V": V, "W": W}
    
    add_strassen_block(0, 1, 2, 3)
    if not check_time():
        return {"U": U, "V": V, "W": W}
    
    add_strassen_block(0, 1, 4, 5)
    if not check_time():
        return {"U": U, "V": V, "W": W}
    
    add_strassen_block(2, 3, 0, 1)
    if not check_time():
        return {"U": U, "V": V, "W": W}
    
    add_strassen_block(2, 3, 2, 3)
    if not check_time():
        return {"U": U, "V": V, "W": W}
    
    add_strassen_block(2, 3, 4, 5)
    if not check_time():
        return {"U": U, "V": V, "W": W}
    
    # Schoolbook blocks for 2x1 j-group (j=4) and 2x2 k-groups
    add_schoolbook_2x1x2(4, 0, 1)
    if not check_time():
        return {"U": U, "V": V, "W": W}
    
    add_schoolbook_2x1x2(4, 2, 3)
    if not check_time():
        return {"U": U, "V": V, "W": W}
    
    add_schoolbook_2x1x2(4, 4, 5)

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
