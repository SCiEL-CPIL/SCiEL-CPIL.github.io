"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()

def construct():
    n, m, p = DIMENSIONS
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_dict, v_dict, w_dict):
        nonlocal col
        if col >= rank:
            raise ValueError("Rank exceeded")
        for idx, val in u_dict.items():
            U[idx][col] = val
        for idx, val in v_dict.items():
            V[idx][col] = val
        for idx, val in w_dict.items():
            W[idx][col] = val
        col += 1

    # Strassen for 2x2 * 2x2
    # Local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    def add_strassen(a_i, a_j, b_i, b_j, c_i, c_j):
        def ma(l): return (a_i + l//2) * m + (a_j + l%2)
        def mb(l): return (b_i + l//2) * p + (b_j + l%2)
        def mc(l):
            i = c_i + l//2
            k = c_j + l%2
            return k * n + i

        # M1
        add_term({ma(0): 1, ma(3): 1}, {mb(0): 1, mb(3): 1}, {mc(0): 1, mc(3): 1})
        # M2
        add_term({ma(2): 1, ma(3): 1}, {mb(0): 1}, {mc(2): 1, mc(3): -1})
        # M3
        add_term({ma(0): 1}, {mb(1): 1, mb(3): -1}, {mc(1): 1, mc(3): 1})
        # M4
        add_term({ma(3): 1}, {mb(2): 1, mb(0): -1}, {mc(0): 1, mc(2): 1})
        # M5
        add_term({ma(0): 1, ma(1): 1}, {mb(3): 1}, {mc(0): -1, mc(1): 1})
        # M6
        add_term({ma(2): 1, ma(0): -1}, {mb(0): 1, mb(1): 1}, {mc(3): 1})
        # M7
        add_term({ma(1): 1, ma(3): -1}, {mb(2): 1, mb(3): 1}, {mc(0): 1})

    # Schoolbook for 2x2 * 2x1
    def add_schoolbook_2x1(a_i, a_j, b_i, b_j, c_i, c_j):
        def ma(l): return (a_i + l//2) * m + (a_j + l%2)
        def mb(l): return (b_i + l) * p + b_j
        def mc(l):
            i = c_i + l
            k = c_j
            return k * n + i

        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        add_term({ma(0): 1}, {mb(0): 1}, {mc(0): 1})
        add_term({ma(1): 1}, {mb(1): 1}, {mc(0): 1})
        add_term({ma(2): 1}, {mb(0): 1}, {mc(1): 1})
        add_term({ma(3): 1}, {mb(1): 1}, {mc(1): 1})

    # Blocks:
    # A1 (0,0) * B1 (0,0) -> C (0,0)
    add_strassen(0, 0, 0, 0, 0, 0)
    # A1 (0,0) * B1 (0,2) -> C (0,2)
    add_strassen(0, 0, 0, 2, 0, 2)
    # A1 (0,0) * B1 (0,4) -> C (0,4)
    add_schoolbook_2x1(0, 0, 0, 4, 0, 4)
    
    # A2 (0,2) * B2 (2,0) -> C (0,0)
    add_strassen(0, 2, 2, 0, 0, 0)
    # A2 (0,2) * B2 (2,2) -> C (0,2)
    add_strassen(0, 2, 2, 2, 0, 2)
    # A2 (0,2) * B2 (2,4) -> C (0,4)
    add_schoolbook_2x1(0, 2, 2, 4, 0, 4)
    
    # Safety: ensure we're within time budget
    elapsed = time.time() - _start_time
    if elapsed > 140:
        raise TimeoutError("Time limit exceeded")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
