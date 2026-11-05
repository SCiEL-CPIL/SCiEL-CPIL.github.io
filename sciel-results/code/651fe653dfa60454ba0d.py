"""Matrix multiplication 6x6x6 with rank-189 decomposition via recursive Strassen."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n = m = p = 6
    rank = 189
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    # Strassen on 2x2 blocks of 3x3
    # Blocks: (0,0)=rows0-2,cols0-2; (0,1)=rows0-2,cols3-5; (1,0)=rows3-5,cols0-2; (1,1)=rows3-5,cols3-5
    # T1=(A00+A11)(B00+B11), T2=(A10+A11)B00, T3=A00(B01-B11), T4=A11(B10-B00)
    # T5=(A00+A01)B11, T6=(A10-A00)(B00+B01), T7=(A01-A11)(B10+B11)
    # C00=T1+T4-T5+T7, C01=T3+T5, C10=T2+T4, C11=T1-T2+T3+T6
    
    terms = [
        (([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)])),
        (([(1,0,1),(1,1,1)], [(0,0,1)])),
        (([(0,0,1)], [(0,1,1),(1,1,-1)])),
        (([(1,1,1)], [(1,0,1),(0,0,-1)])),
        (([(0,0,1),(0,1,1)], [(1,1,1)])),
        (([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)])),
        (([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)])),
    ]
    
    contribs = [
        [(0,1),(3,1),(4,-1),(6,1)],
        [(2,1),(4,1)],
        [(1,1),(3,1)],
        [(0,1),(1,-1),(2,1),(5,1)],
    ]
    
    col = 0
    for t in range(7):
        left = terms[t][0]
        right = terms[t][1]
        for i in range(3):
            for a in range(3):
                for k in range(3):
                    for br, bc, cl in left:
                        U[(3*br+i)*6 + (3*bc+a)][col] += cl
                    for br, bc, cr in right:
                        V[(3*br+a)*6 + (3*bc+k)][col] += cr
                    for cb in range(4):
                        for si, sg in contribs[cb]:
                            if si == t:
                                rbr = cb // 2
                                rbc = cb % 2
                                W[(3*rbc+k)*6 + (3*rbr+i)][col] += sg
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
