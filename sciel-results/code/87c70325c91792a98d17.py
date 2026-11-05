"""Matrix multiplication 2x5, 5x5 tensor decomposition.
Uses a 2+2+1 partition of the 5x5 B matrix with Strassen (7-mult) on the
2x2 x 2x2 sub-blocks and schoolbook for 1-wide blocks, reducing rank to 42."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 5

    # Partition indices 0..4 into blocks of sizes [2,2,1]
    sizes = [2, 2, 1]
    blocks = []
    off = 0
    for s in sizes:
        blocks.append(list(range(off, off + s)))
        off += s
    # blocks = [[0,1],[2,3],[4]]

    terms = []  # each: (U dict, V dict, W dict)

    def add_term(U, V, W):
        terms.append((U, V, W))

    # Strassen 2x2 x 2x2:
    #   M1=(A00+A11)(B00+B11)  M2=(A10+A11)B00  M3=A00(B01-B11)
    #   M4=A11(B10-B00)        M5=(A00+A01)B11  M6=(A10-A00)(B00+B01)
    #   M7=(A01-A11)(B10+B11)
    #   C00=M1+M4-M5+M7  C01=M3+M5  C10=M2+M4  C11=M1-M2+M3+M6
    def strassen_block(i0, i1, a0, a1, b0, b1):
        def A(i, a): return i * m + a
        def B(a, b): return a * p + b
        def C(b, i): return b * n + i
        add_term({A(i0,a0):1, A(i1,a1):1}, {B(a0,b0):1, B(a1,b1):1},
                 {C(b0,i0):1, C(b1,i1):1})            # M1
        add_term({A(i1,a0):1, A(i1,a1):1}, {B(a0,b0):1},
                 {C(b0,i1):1, C(b1,i1):-1})           # M2
        add_term({A(i0,a0):1}, {B(a0,b1):1, B(a1,b1):-1},
                 {C(b1,i0):1, C(b1,i1):1})            # M3
        add_term({A(i1,a1):1}, {B(a1,b0):1, B(a0,b0):-1},
                 {C(b0,i0):1, C(b0,i1):1})            # M4
        add_term({A(i0,a0):1, A(i0,a1):1}, {B(a1,b1):1},
                 {C(b0,i0):-1, C(b1,i0):1})           # M5
        add_term({A(i1,a0):1, A(i0,a0):-1}, {B(a0,b0):1, B(a0,b1):1},
                 {C(b1,i1):1})                         # M6
        add_term({A(i0,a1):1, A(i1,a1):-1}, {B(a1,b0):1, B(a1,b1):1},
                 {C(b0,i0):1})                         # M7

    def schoolbook_block(i0, i1, a0, a1, b0, b1):
        def A(i, a): return i * m + a
        def B(a, b): return a * p + b
        def C(b, i): return b * n + i
        for i in (i0, i1):
            for a in range(a0, a1 + 1):
                for b in range(b0, b1 + 1):
                    add_term({A(i, a): 1}, {B(a, b): 1}, {C(b, i): 1})

    i0, i1 = 0, 1

    for rb, a_blk in enumerate(blocks):
        for cb, b_blk in enumerate(blocks):
            a0, a1 = a_blk[0], a_blk[-1]
            b0, b1 = b_blk[0], b_blk[-1]
            if len(a_blk) == 2 and len(b_blk) == 2:
                strassen_block(i0, i1, a0, a1, b0, b1)
            else:
                schoolbook_block(i0, i1, a0, a1, b0, b1)

    rank = len(terms)

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c, (Ud, Vd, Wd) in enumerate(terms):
        for row, coeff in Ud.items():
            U[row][c] = coeff
        for row, coeff in Vd.items():
            V[row][c] = coeff
        for row, coeff in Wd.items():
            W[row][c] = coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
