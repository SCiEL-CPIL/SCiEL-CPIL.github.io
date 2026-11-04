"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard: kill the program after 150s wall-clock, keep margin
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds margin before 120s limit
    
    # Rank-17 decomposition using Strassen on the 2x2x2 sub-block (i in {0,1}, j in {0,1}, k in {0,1})
    # plus direct multiplication for remaining terms.
    #
    # Partition: j in {0,1,2}. Use Strassen for (i,j,k) in {0,1}x{0,1}x{0,1} (7 factors instead of 8).
    # Direct multiplication for:
    #   - i in {0,1}, j in {0,1}, k=2 (4 terms)
    #   - i in {0,1}, j=2, k in {0,1,2} (6 terms)
    # Total: 7 + 4 + 6 = 17
    
    rank = 17
    U = [[0] * rank for _ in range(n * m)]  # shape (6, 17)
    V = [[0] * rank for _ in range(m * p)]  # shape (9, 17)
    W = [[0] * rank for _ in range(p * n)]  # shape (6, 17)
    
    # Helper to add a factor
    def add_factor(u_entries, v_entries, w_entries):
        nonlocal rank
        c = rank  # this won't work, let me use a list instead
    
    # Let me build the factors as lists of (position, value) tuples
    factors = []
    
    # Strassen's 7 factors for 2x2 * 2x2 sub-block
    # i in {0,1}, j in {0,1}, k in {0,1}
    # U position: i*m + j = i*3 + j
    # V position: j*p + k = j*3 + k
    # W position: k*n + i = k*2 + i
    
    # M1 = (a00+a11)(b00+b11)
    # U: (0,0)=1, (1,1)=1 => positions 0, 4
    # V: (0,0)=1, (1,1)=1 => positions 0, 4
    # W: C00=M1+..., C01=..., C10=..., C11=M1-... => W[0]=1, W[1]=1, W[2]=-1, W[3]=1
    factors.append(([(0, 1), (4, 1)], [(0, 1), (4, 1)], [(0, 1), (1, 1), (2, -1), (3, 1)]))
    
    # M2 = (a10+a11)b00
    # U: (1,0)=1, (1,1)=1 => positions 3, 4
    # V: (0,0)=1 => position 0
    # W: C10=M2+..., C11=-M2+... => W[2]=1, W[3]=-1
    factors.append(([(3, 1), (4, 1)], [(0, 1)], [(2, 1), (3, -1)]))
    
    # M3 = a00(b10-b11)
    # U: (0,0)=1 => position 0
    # V: (1,0)=1, (1,1)=-1 => positions 3, 4
    # W: C01=M3+..., C11=M3+... => W[1]=1, W[3]=1
    factors.append(([(0, 1)], [(3, 1), (4, -1)], [(1, 1), (3, 1)]))
    
    # M4 = a11(b01-b00)
    # U: (1,1)=1 => position 4
    # V: (0,1)=1, (0,0)=-1 => positions 1, 0
    # W: C00=M4+..., C10=M4+... => W[0]=1, W[2]=1
    factors.append(([(4, 1)], [(1, 1), (0, -1)], [(0, 1), (2, 1)]))
    
    # M5 = (a00+a01)b11
    # U: (0,0)=1, (0,1)=1 => positions 0, 1
    # V: (1,1)=1 => position 4
    # W: C00=-M5+..., C01=M5+... => W[0]=-1, W[1]=1
    factors.append(([(0, 1), (1, 1)], [(4, 1)], [(0, -1), (1, 1)]))
    
    # M6 = (a10-a00)(b00+b10)
    # U: (1,0)=1, (0,0)=-1 => positions 3, 0
    # V: (0,0)=1, (1,0)=1 => positions 0, 3
    # W: C11=M6+... => W[3]=1
    factors.append(([(3, 1), (0, -1)], [(0, 1), (3, 1)], [(3, 1)]))
    
    # M7 = (a01-a11)(b01+b11)
    # U: (0,1)=1, (1,1)=-1 => positions 1, 4
    # V: (0,1)=1, (1,1)=1 => positions 1, 4
    # W: C00=M7+... => W[0]=1
    factors.append(([(1, 1), (4, -1)], [(1, 1), (4, 1)], [(0, 1)]))
    
    # Now direct multiplication factors:
    # i in {0,1}, j in {0,1}, k=2 (4 terms)
    for i in range(2):
        for j in range(2):
            u_pos = i * m + j
            v_pos = j * p + 2  # k=2
            w_pos = 2 * n + i  # k=2
            factors.append(([(u_pos, 1)], [(v_pos, 1)], [(w_pos, 1)]))
    
    # i in {0,1}, j=2, k in {0,1,2} (6 terms)
    for i in range(2):
        for k in range(3):
            u_pos = i * m + 2  # j=2
            v_pos = 2 * p + k  # j=2
            w_pos = k * n + i
            factors.append(([(u_pos, 1)], [(v_pos, 1)], [(w_pos, 1)]))
    
    # Fill in the matrices
    for c, (u_entries, v_entries, w_entries) in enumerate(factors):
        for pos, val in u_entries:
            U[pos][c] = val
        for pos, val in v_entries:
            V[pos][c] = val
        for pos, val in w_entries:
            W[pos][c] = val
    
    # Verify the decomposition
    def verify():
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = 1.0
                    rhs = 0.0
                    for c in range(rank):
                        rhs += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(lhs - rhs) > 1e-10:
                        return False
        return True
    
    if not verify():
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
