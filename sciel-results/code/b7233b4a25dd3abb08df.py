"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 100  # safety margin
    
    n, m, p = DIMENSIONS  # 2, 2, 15
    
    # Decompose into 7 Strassen blocks (2x2x2, rank 7) + 1 schoolbook block (2x2x1, rank 4)
    total_rank = 7 * 7 + 4  # 53
    
    U = [[0] * total_rank for _ in range(n * m)]   # 4 x 53
    V = [[0] * total_rank for _ in range(m * p)]   # 30 x 53
    W = [[0] * total_rank for _ in range(p * n)]   # 30 x 53
    
    # Strassen 2x2x2: 7 rank-1 tensors
    # Each tensor: U_vec (len 4), V_vec (len 4), W_vec (len 4)
    # U index: i*2+j, V index: j*2+k, W index: k*2+i
    strassen_tensors = [
        # r=0: m1 = (a00+a11)(b00+b11) -> c00+=1, c11+=1
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        # r=1: m2 = (a10+a11)b00 -> c10+=1
        ([0,0,1,1], [1,0,0,0], [0,1,0,0]),
        # r=2: m3 = a00(b01-b11) -> c01+=1
        ([1,0,0,0], [0,1,0,-1], [0,0,1,0]),
        # r=3: m4 = a11(b10-b00) -> c10+=1, c11+=1
        ([0,0,0,1], [0,0,1,-1], [0,1,0,1]),
        # r=4: m5 = (a00+a01)b11 -> c01+=1, c11+=1
        ([1,1,0,0], [0,0,0,1], [0,0,1,1]),
        # r=5: m6 = (a10-a00)(b00+b01) -> c10+=1
        ([0,1,1,0], [1,1,0,0], [0,1,0,0]),
        # r=6: m7 = (a01-a11)(b10+b11) -> c01+=1, c11+=1
        ([0,1,0,-1], [0,0,1,1], [0,0,1,1]),
    ]
    
    col = 0
    for block in range(7):
        k_off = block * 2
        for r in range(7):
            c = col + r
            u_vec, v_vec, w_vec = strassen_tensors[r]
            for ui in range(4):
                U[ui][c] = u_vec[ui]
            for j in range(2):
                for kk in range(2):
                    V[j*p + (k_off+kk)][c] = v_vec[j*2+kk]
            for kk in range(2):
                for i in range(2):
                    W[(k_off+kk)*n + i][c] = w_vec[kk*2+i]
        col += 7
    
    # Schoolbook 2x2x1 for last column (k=14)
    for i in range(2):
        for j in range(2):
            U[i*m+j][col] = 1
            V[j*p + 14][col] = 1
            W[14*n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
