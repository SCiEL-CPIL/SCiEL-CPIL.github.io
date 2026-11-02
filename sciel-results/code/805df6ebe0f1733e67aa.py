import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill

    # Winograd 2x2x2 factors (verified)
    U_w = np.array([
        [1, 0, 1, 1, 0, 1, 0],
        [0, 1, 1, 0, 1, -1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [0, 1, 0, 0, 1, 0, -1]
    ], dtype=float)

    V_w = np.array([
        [1, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 1],
        [0, 1, 0, 0, 1, 0, 0]
    ], dtype=float)

    W_w = np.array([
        [1, 0, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 1, 1, 0],
        [0, 0, 0, 1, 1, -1, 1],
        [1, 0, 1, 0, 0, 0, 1]
    ], dtype=float)

    # Schoolbook 2x2x1 factors
    U_sb = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=float)
    V_sb = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ], dtype=float)
    W_sb = np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=float)

    # Verify Winograd 2x2x2
    def verify_222(U, V, W):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    a = i * 2 + j
                    b = j * 2 + k
                    c = k * 2 + i
                    s = np.dot(U[a], np.multiply(V[b], W[c]))
                    if abs(s - 1.0) > 1e-9:
                        return False
        for a in range(4):
            ia, ja = divmod(a, 2)
            for b in range(4):
                jb, kb = divmod(b, 2)
                for c in range(4):
                    kc, ic = divmod(c, 2)
                    if ia == ic and ja == jb and kb == kc:
                        continue
                    s = np.dot(U[a], np.multiply(V[b], W[c]))
                    if abs(s) > 1e-9:
                        return False
        return True

    if not verify_222(U_w, V_w, W_w):
        return _schoolbook()

    # Build 2x2x7 block: split k into {0,1},{2,3},{4,5},{6}
    # 3 Winograd blocks (7 each) + 1 schoolbook block (4) = 25 rank
    r227 = 7 + 7 + 7 + 4  # 25

    U_227 = np.zeros((4, r227))
    V_227 = np.zeros((14, r227))  # rows: (j_local, k) for j_local in 0,1
    W_227 = np.zeros((14, r227))  # rows: (k, i) for k in 0..6

    col = 0
    for blk in range(3):
        U_227[:, col:col+7] = U_w
        # V: rows (j_local, k) where k = blk*2, blk*2+1
        # V_w rows: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        # V_227 row for (j_local, k): index = j_local*7 + k
        V_227[blk*2, col:col+7] = V_w[0]      # j_local=0, k=blk*2
        V_227[blk*2+1, col:col+7] = V_w[1]    # j_local=0, k=blk*2+1
        V_227[7+blk*2, col:col+7] = V_w[2]    # j_local=1, k=blk*2
        V_227[7+blk*2+1, col:col+7] = V_w[3]  # j_local=1, k=blk*2+1
        # W: rows (k, i)
        W_227[blk*2, col:col+7] = W_w[0]      # k=blk*2, i=0
        W_227[blk*2+1, col:col+7] = W_w[1]    # k=blk*2+1, i=0
        W_227[blk*2+2, col:col+7] = W_w[2]    # k=blk*2, i=1 -- WRONG!
        # Fix: W_w row 2 is (k=1, i=0)? No.
        # W_w rows: 0=(k=0,i=0), 1=(k=0,i=1), 2=(k=1,i=0), 3=(k=1,i=1)
        # So for k=blk*2: W_w[0] for i=0, W_w[2] for i=1
        # For k=blk*2+1: W_w[1] for i=0, W_w[3] for i=1
        W_227[blk*2, col:col+7] = W_w[0]
        W_227[blk*2+1, col:col+7] = W_w[2]
        W_227[blk*2+2, col:col+7] = W_w[1]
        W_227[blk*2+3, col:col+7] = W_w[3]
        # Wait, that's 4 rows but we only have 2 k values per block.
        # W_227 has 14 rows total: (k=0,i=0),(k=0,i=1),(k=1,i=0),(k=1,i=1),...,(k=6,i=0),(k=6,i=1)
        # Row index = k*2 + i
        W_227[blk*2*2, col:col+7] = W_w[0]       # k=blk*2, i=0
        W_227[blk*2*2+1, col:col+7] = W_w[1]     # k=blk*2, i=1
        W_227[blk*2*2+2, col:col+7] = W_w[2]     # k=blk*2+1, i=0
        W_227[blk*2*2+3, col:col+7] = W_w[3]     # k=blk*2+1, i=1
        col += 7

    # Schoolbook block for k=6
    U_227[:, col:col+4] = U_sb
    V_227[6, col:col+4] = V_sb[0]   # j_local=0, k=6
    V_227[13, col:col+4] = V_sb[1]  # j_local=1, k=6
    W_227[12, col:col+4] = W_sb[0]  # k=6, i=0
    W_227[13, col:col+4] = W_sb[1]  # k=6, i=1
    col += 4

    # Now combine two blocks: j_global in {0,1} and {2,3}
    total_rank = 2 * r227  # 50
    U = np.zeros((8, total_rank))
    V = np.zeros((28, total_rank))
    W = np.zeros((14, total_rank))

    # Block 1: j_global=0,1 -> j_local=0,1
    # U rows: i*m+j_global. For block1: (0,0)=0, (0,1)=1, (1,0)=4, (1,1)=5
    U[0, 0:25] = U_227[0]  # i=0, j_local=0
    U[1, 0:25] = U_227[1]  # i=0, j_local=1
    U[4, 0:25] = U_227[2]  # i=1, j_local=0
    U[5, 0:25] = U_227[3]  # i=1, j_local=1

    # V rows: j_global*p + k. For block1 j_global=0,1:
    # V_227 row (j_local, k) -> V row j_global*p + k
    # j_local=0 -> j_global=0, j_local=1 -> j_global=1
    for j_loc in range(2):
        for k in range(7):
            V[j_loc * 7 + k, 0:25] = V_227[j_loc * 7 + k, 0:25]

    # W rows: k*n + i. Same for both blocks, so add.
    W[0:14, 0:25] = W_227

    # Block 2: j_global=2,3 -> j_local=0,1
    U[2, 25:50] = U_227[0]  # i=0, j_global=2
    U[3, 25:50] = U_227[1]  # i=0, j_global=3
    U[6, 25:50] = U_227[2]  # i=1, j_global=2
    U[7, 25:50] = U_227[3]  # i=1, j_global=3

    for j_loc in range(2):
        j_glob = 2 + j_loc
        for k in range(7):
            V[j_glob * 7 + k, 25:50] = V_227[j_loc * 7 + k, 0:25]

    W[0:14, 25:50] = W_227

    # Verify
    def check_valid(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = np.dot(U[a], np.multiply(V[b], W[c]))
                    if abs(s - 1.0) > 1e-6:
                        return False
        for a in range(n * m):
            ia = a // m
            ja = a % m
            for b in range(m * p):
                jb = b // p
                kb = b % p
                for c in range(p * n):
                    kc = c // n
                    ic = c % n
                    if ia == ic and ja == jb and kb == kc:
                        continue
                    s = np.dot(U[a], np.multiply(V[b], W[c]))
                    if abs(s) > 1e-6:
                        return False
        return True

    if time.time() < deadline - 5:
        if check_valid(U, V, W, total_rank):
            def clean(x):
                if abs(x - round(x)) < 1e-10:
                    return int(round(x))
                if abs(2*x - round(2*x)) < 1e-10:
                    return round(2*x) / 2
                return float(x)
            return {
                "U": [[clean(U[i, j]) for j in range(total_rank)] for i in range(n*m)],
                "V": [[clean(V[i, j]) for j in range(total_rank)] for i in range(m*p)],
                "W": [[clean(W[i, j]) for j in range(total_rank)] for i in range(p*n)]
            }

    return _schoolbook()

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
