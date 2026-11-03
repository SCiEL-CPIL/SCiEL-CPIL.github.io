"""Direct construction of rank-minimal (or near-minimal) tensor for 4x4 * 4x8."""
import time

DIMENSIONS = [4, 4, 8]

def _make_strassen_2x2():
    """Rank-7 decomposition for 2x2 * 2x2.
    U: 4x7 (rows indexed i*2+j), V: 4x7 (rows j*2+k), W: 4x7 (rows k*2+i)
    W[k*2+i] corresponds to C_{i,k} (row i, col k).
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    # U[i*2+j]: a_{i,j}
    # V[j*2+k]: b_{j,k}
    # W[k*2+i]: c_{i,k}
    # Strassen:
    # M1=(a00+a11)(b00+b11)
    # M2=(a10+a11)b00
    # M3=a00(b01-b11)
    # M4=a11(b10-b00)
    # M5=(a00+a10)b11
    # M6=(a10-a00)(b00+b10)
    # M7=(a01-a11)(b01+b11)
    # C00=M1-M4+M5+M6, C01=M3+M1-M5+M7, C10=M2+M4-M1+M3, C11=M1-M2+M6+M7
    # W indices: W[0]=C00, W[1]=C10, W[2]=C01, W[3]=C11
    terms = [
        # M1: U:{0:1,3:1}, V:{0:1,3:1}, W: C00+=1,C01+=1,C10-=1,C11+=1 -> {0:1,1:-1,2:1,3:1}
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 1:-1, 2:1, 3:1}),
        # M2: U:{2:1,3:1}, V:{0:1}, W: C10+=1,C11-=1 -> {1:1,3:-1}
        ({2:1, 3:1}, {0:1}, {1:1, 3:-1}),
        # M3: U:{0:1}, V:{1:1,3:-1}, W: C01+=1,C10+=1 -> {1:1,2:1}
        ({0:1}, {1:1, 3:-1}, {1:1, 2:1}),
        # M4: U:{3:1}, V:{2:1,0:-1}, W: C00-=1,C10+=1 -> {0:-1,1:1}
        ({3:1}, {2:1, 0:-1}, {0:-1, 1:1}),
        # M5: U:{0:1,2:1}, V:{3:1}, W: C00+=1,C01-=1 -> {0:1,2:-1}
        ({0:1, 2:1}, {3:1}, {0:1, 2:-1}),
        # M6: U:{2:1,0:-1}, V:{0:1,2:1}, W: C00+=1,C11+=1 -> {0:1,3:1}
        ({2:1, 0:-1}, {0:1, 2:1}, {0:1, 3:1}),
        # M7: U:{1:1,3:-1}, V:{1:1,3:1}, W: C01+=1,C11+=1 -> {2:1,3:1}
        ({1:1, 3:-1}, {1:1, 3:1}, {2:1, 3:1}),
    ]
    for t, (u_c, v_c, w_c) in enumerate(terms):
        for idx, val in u_c.items():
            U[idx][t] = val
        for idx, val in v_c.items():
            V[idx][t] = val
        for idx, val in w_c.items():
            W[idx][t] = val
    return U, V, W

def _verify_2x2():
    """Verify the 2x2 Strassen decomposition is correct."""
    import numpy as np
    U, V, W = _make_strassen_2x2()
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = 7
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = 0.0
                for r in range(rank):
                    val += U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r]
                if abs(val - 1.0) > 1e-10:
                    raise ValueError(f"2x2 mismatch at ({i},{j},{k}): got {val}, expected 1.0")

def _strassen_4x4_4x4():
    """Rank-49 decomposition for 4x4 * 4x4.
    U: (16, 49) indexed by i*4+j
    V: (16, 49) indexed by j*4+k
    W: (16, 49) indexed by k*4+i
    """
    import numpy as np
    u2, v2, w2 = _make_strassen_2x2()
    u2 = np.array(u2, dtype=float)
    v2 = np.array(v2, dtype=float)
    w2 = np.array(w2, dtype=float)
    
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    def u_block_rows(br, bc):
        return [br*8 + bc*2, br*8 + bc*2+1, br*8+4 + bc*2, br*8+4 + bc*2+1]
    
    def v_block_rows(br, bc):
        return [br*8 + bc*2, br*8 + bc*2+1, br*8+4 + bc*2, br*8+4 + bc*2+1]
    
    def w_block_rows(br, bc):
        return [bc*8 + br*2, bc*8 + br*2+1, bc*8+4 + br*2, bc*8+4 + br*2+1]
    
    # Block index: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen_defs = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(1,1),(2,-1),(3,1)]),
        ([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]),
        ([(0,1)], [(1,1),(3,-1)], [(1,1),(2,1)]),
        ([(3,1)], [(2,1),(0,-1)], [(0,-1),(2,1)]),
        ([(0,1),(2,1)], [(3,1)], [(0,1),(1,-1)]),
        ([(2,1),(0,-1)], [(0,1),(2,1)], [(0,1),(3,1)]),
        ([(1,1),(3,-1)], [(1,1),(3,1)], [(1,1),(3,1)]),
    ]
    
    def block_bc(idx):
        return (idx // 2, idx % 2)
    
    for s, (a_blocks, b_blocks, c_blocks) in enumerate(strassen_defs):
        for t in range(7):
            col = s * 7 + t
            for (a_idx, a_sign) in a_blocks:
                abr, abc = block_bc(a_idx)
                rows = u_block_rows(abr, abc)
                for lr, row in enumerate(rows):
                    U[row, col] += a_sign * u2[lr, t]
            for (b_idx, b_sign) in b_blocks:
                bbr, bbc = block_bc(b_idx)
                rows = v_block_rows(bbr, bbc)
                for lr, row in enumerate(rows):
                    V[row, col] += b_sign * v2[lr, t]
            for (c_idx, c_sign) in c_blocks:
                cbr, cbc = block_bc(c_idx)
                rows = w_block_rows(cbr, cbc)
                for lr, row in enumerate(rows):
                    W[row, col] += c_sign * w2[lr, t]
    
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    deadline = time.time() + 100
    
    U1, V1, W1 = _strassen_4x4_4x4()
    U2, V2, W2 = _strassen_4x4_4x4()
    
    rank = 98
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for i in range(n*m):
        for r in range(49):
            U[i][r] = U1[i][r]
    
    for j in range(4):
        for k in range(4):
            v_row = j*8 + k
            v1_row = j*4 + k
            for r in range(49):
                V[v_row][r] = V1[v1_row][r]
    
    for k in range(4):
        for i in range(4):
            w_row = k*4 + i
            w1_row = k*4 + i
            for r in range(49):
                W[w_row][r] = W1[w1_row][r]
    
    for i in range(n*m):
        for r in range(49):
            U[i][49+r] = U2[i][r]
    
    for j in range(4):
        for k in range(4):
            v_row = j*8 + (k+4)
            v2_row = j*4 + k
            for r in range(49):
                V[v_row][49+r] = V2[v2_row][r]
    
    for k in range(4):
        for i in range(4):
            w_row = (k+4)*4 + i
            w2_row = k*4 + i
            for r in range(49):
                W[w_row][49+r] = W2[w2_row][r]
    
    return {"U": U, "V": V, "W": W}
