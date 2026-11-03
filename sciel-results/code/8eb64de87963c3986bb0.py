"""Matrix multiplication 6x6x6: rank-189 via Strassen 2x2 blocks of 3x3 schoolbook."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Strassen 2x2 block decomposition with 3x3 blocks.
    # A, B, C are 6x6, partitioned into 2x2 blocks of 3x3.
    # 
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Each Mi is a 3x3 matrix product (schoolbook, rank 27).
    # Total rank = 7 * 27 = 189.
    
    rank = 7 * 27
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    col = 0
    for s in range(7):
        # Define left block combos, right block combos, and output block combos
        if s == 0:  # M1
            left = [((0,0),1),((1,1),1)]
            right = [((0,0),1),((1,1),1)]
            out = [((0,0),1),((1,1),1)]
        elif s == 1:  # M2
            left = [((1,0),1),((1,1),1)]
            right = [((0,0),1)]
            out = [((1,0),1),((1,1),-1)]
        elif s == 2:  # M3
            left = [((0,0),1)]
            right = [((1,0),1),((1,1),-1)]
            out = [((0,1),1),((1,1),1)]
        elif s == 3:  # M4
            left = [((1,1),1)]
            right = [((0,1),1),((0,0),-1)]
            out = [((0,0),1),((1,0),1)]
        elif s == 4:  # M5
            left = [((0,0),1),((0,1),1)]
            right = [((1,1),1)]
            out = [((0,0),-1),((0,1),1)]
        elif s == 5:  # M6
            left = [((1,0),1),((0,0),-1)]
            right = [((0,0),1),((0,1),1)]
            out = [((1,1),1)]
        elif s == 6:  # M7
            left = [((0,0),1),((1,1),-1)]
            right = [((1,0),1),((1,1),1)]
            out = [((0,0),1)]
        
        for (lb, lc) in left:
            for (rb, rc) in right:
                for (ob, oc) in out:
                    for i in range(3):
                        for j in range(3):
                            for k in range(3):
                                # A index: block lb=(br,cb), local (i,j)
                                a_idx = (lb[0]*3 + i) * 6 + (lb[1]*3 + j)
                                # B index: block rb=(cr,cc), local (j,k)
                                b_idx = (rb[0]*3 + j) * 6 + (rb[1]*3 + k)
                                # W index: output block ob=(or,oc), local (i,k)
                                # W[k*n+i] = W[(oc*3+k)*6 + (or*3+i)]
                                w_idx = (ob[1]*3 + k) * 6 + (ob[0]*3 + i)
                                
                                coeff = lc * rc * oc
                                U[a_idx][col] = coeff
                                V[b_idx][col] = coeff
                                W[w_idx][col] = coeff
                                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
