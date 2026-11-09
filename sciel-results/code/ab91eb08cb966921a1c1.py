import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)
    
    # Build 4x4 multiplication via 2-level Strassen
    # Level 1: 7 outer products of 2x2 blocks
    # Level 2: each 2x2 product uses 7 inner Strassen terms
    # Total: 49 terms
    
    # Inner 2x2 Strassen: for C = A*B where A,B,C are 2x2
    # Flattened indices: A=[a00,a01,a10,a11], B=[b00,b01,b10,b11], C=[c00,c01,c10,c11]
    # 7 products:
    # P0 = (a00+a11)(b00+b11) -> c00+=P0, c11+=P0
    # P1 = (a10+a11)b00 -> c10+=P1, c11-=P1
    # P2 = a00(b01-b11) -> c01+=P2, c11+=P2
    # P3 = a11(b10-b00) -> c00+=P3, c10+=P3
    # P4 = (a00+a01)b11 -> c00-=P4, c01+=P4
    # P5 = (a10-a00)(b00+b01) -> c11+=P5
    # P6 = (a01-a11)(b10+b11) -> c00+=P6
    
    inner_U = [
        [1,0,0,1],  # P0: a00+a11
        [0,0,1,1],  # P1: a10+a11
        [1,0,0,0],  # P2: a00
        [0,0,0,1],  # P3: a11
        [1,1,0,0],  # P4: a00+a01
        [-1,0,1,0], # P5: a10-a00
        [0,1,0,-1]  # P6: a01-a11
    ]
    inner_V = [
        [1,0,0,1],  # P0: b00+b11
        [1,0,0,0],  # P1: b00
        [0,1,-1,0], # P2: b01-b11
        [0,0,1,-1], # P3: b10-b00
        [0,0,0,1],  # P4: b11
        [1,1,0,0],  # P5: b00+b01
        [0,0,1,1]   # P6: b10+b11
    ]
    inner_W = [
        [1,0,0,1],  # P0: c00,c11
        [0,0,1,-1], # P1: c10,c11
        [0,1,0,1],  # P2: c01,c11
        [1,0,1,0],  # P3: c00,c10
        [-1,1,0,0], # P4: c00,c01
        [0,0,0,1],  # P5: c11
        [1,0,0,0]   # P6: c00
    ]
    
    # Outer 4x4: 7 terms combining 2x2 blocks
    # A blocks: A00(top-left), A01(top-right), A10(bottom-left), A11(bottom-right)
    # B blocks: B00, B01, B10, B11
    # C blocks: C00, C01, C10, C11
    
    # O0 = (A00+A11)(B00+B11) -> C00+=O0, C11+=O0
    # O1 = (A10+A11)B00 -> C10+=O1, C11-=O1
    # O2 = A00(B01-B11) -> C01+=O2, C11+=O2
    # O3 = A11(B10-B00) -> C00+=O3, C10+=O3
    # O4 = (A00+A01)B11 -> C00-=O4, C01+=O4
    # O5 = (A10-A00)(B00+B01) -> C11+=O5
    # O6 = (A01-A11)(B10+B11) -> C00+=O6
    
    outer_A = [
        [(0,1),(3,1)],  # O0: A00+A11
        [(2,1),(3,1)],  # O1: A10+A11
        [(0,1)],        # O2: A00
        [(3,1)],        # O3: A11
        [(0,1),(1,1)],  # O4: A00+A01
        [(2,1),(0,-1)], # O5: A10-A00
        [(1,1),(3,-1)]  # O6: A01-A11
    ]
    outer_B = [
        [(0,1),(3,1)],  # O0: B00+B11
        [(0,1)],        # O1: B00
        [(1,1),(3,-1)], # O2: B01-B11
        [(2,1),(0,-1)], # O3: B10-B00
        [(3,1)],        # O4: B11
        [(0,1),(1,1)],  # O5: B00+B01
        [(2,1),(3,1)]   # O6: B10+B11
    ]
    outer_C = [
        [(0,1),(3,1)],  # O0: C00,C11
        [(2,1),(3,-1)], # O1: C10,C11
        [(1,1),(3,1)],  # O2: C01,C11
        [(0,1),(2,1)],  # O3: C00,C10
        [(0,-1),(1,1)], # O4: C00,C01
        [(3,1)],        # O5: C11
        [(0,1)]         # O6: C00
    ]
    
    def map_A(block, li):
        r = li // 2
        c = li % 2
        gr = (block // 2) * 2 + r
        gc = (block % 2) * 2 + c
        return gr * 4 + gc
    
    def map_B(block, li):
        r = li // 2
        c = li % 2
        gr = (block // 2) * 2 + r
        gc = (block % 2) * 2 + c
        return gr * 4 + gc
    
    def map_C(block, li):
        r = li // 2
        c = li % 2
        gr = (block // 2) * 2 + r
        gc = (block % 2) * 2 + c
        return gc * 4 + gr
    
    for t in range(7):
        for s in range(7):
            col = t * 7 + s
            for li in range(4):
                for bi, cb in outer_A[t]:
                    if inner_U[s][li] != 0:
                        U[map_A(bi, li), col] += inner_U[s][li] * cb
            for li in range(4):
                for bi, cb in outer_B[t]:
                    if inner_V[s][li] != 0:
                        V[map_B(bi, li), col] += inner_V[s][li] * cb
            for li in range(4):
                for bi, cb in outer_C[t]:
                    if inner_W[s][li] != 0:
                        W[map_C(bi, li), col] += inner_W[s][li] * cb
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
