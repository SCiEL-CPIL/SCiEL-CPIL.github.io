import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 130
    
    n, m, p = 4, 4, 5
    rank = 65
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 4x4x4: 49 terms for columns 0-3
    # Then 16 schoolbook terms for column 4
    
    # Define inner 2x2 Strassen (7 terms)
    # Each term: (u_pattern, v_pattern, w_pattern)
    # u_pattern: which elements of X (2x2) are in the U factor
    # v_pattern: which elements of Y (2x2) are in the V factor  
    # w_pattern: which elements of Z (2x2) get contributions
    
    # For 2x2 Strassen:
    # t1: U=(X00+X11), V=(Y00+Y11), W: Z00+=1, Z11+=1
    # t2: U=(X10+X11), V=Y00, W: Z10+=1, Z11-=1
    # t3: U=X00, V=(Y10-Y11), W: Z01+=1, Z11+=1
    # t4: U=X11, V=(Y00-Y10), W: Z00+=1, Z10+=1
    # t5: U=(X00+X01), V=Y11, W: Z00-=1, Z01+=1
    # t6: U=(X10-X00), V=(Y00+Y01), W: Z11+=1
    # t7: U=(X01-X11), V=(Y10+Y11), W: Z00+=1
    
    # For 4x4, we have 2x2 blocks. Let me define the outer Strassen
    # and compose with inner.
    
    # I'll build the 49 terms explicitly.
    
    # Outer Strassen patterns (which 2x2 blocks of A and B are combined):
    # o1: A_blocks = {00:+1, 11:+1}, B_blocks = {00:+1, 11:+1}
    # o2: A_blocks = {10:+1, 11:+1}, B_blocks = {00:+1}
    # o3: A_blocks = {00:+1}, B_blocks = {10:+1, 11:-1}
    # o4: A_blocks = {11:+1}, B_blocks = {00:+1, 10:-1}
    # o5: A_blocks = {00:+1, 01:+1}, B_blocks = {11:+1}
    # o6: A_blocks = {10:+1, 00:-1}, B_blocks = {00:+1, 01:+1}
    # o7: A_blocks = {01:+1, 11:-1}, B_blocks = {10:+1, 11:+1}
    
    # For each outer term, we get a 2x2x2 multiplication of the block sums.
    # Each such multiplication uses 7 inner Strassen terms.
    
    # The C block contributions from outer Strassen:
    # C00 = o1 + o4 - o5 + o7
    # C01 = o3 + o5
    # C10 = o2 + o4
    # C11 = o1 - o2 + o3 + o6
    
    # So for each outer term o and inner term i, the C contribution is:
    # For C block (br, bc), the coefficient is:
    # c_contrib[br, bc] = sum over outer terms o where C[br,bc] includes o
    
    # Let me define:
    # c_outer_contrib = {
    #   (0,0): {o1:+1, o4:+1, o5:-1, o7:+1},
    #   (0,1): {o3:+1, o5:+1},
    #   (1,0): {o2:+1, o4:+1},
    #   (1,1): {o1:+1, o2:-1, o3:+1, o6:+1},
    # }
    
    # For each outer term o and inner term i:
    # - U: for each element (a_br, a_bc) in A with coefficient from outer o,
    #       and each element (ir, ic) in the 2x2 inner U pattern,
    #       the global A element is at row a_br*2+ir, col a_bc*2+ic
    # - Similarly for V and W
    
    # Let me code this up
    
    outer_a = [
        {0: [(0,0,1), (1,1,1)]},      # o1
        {1: [(1,0,1), (1,1,1)]},      # o2
        {2: [(0,0,1)]},               # o3
        {3: [(1,1,1)]},               # o4
        {4: [(0,0,1), (0,1,1)]},      # o5
        {5: [(1,0,1), (0,0,-1)]},     # o6
        {6: [(0,1,1), (1,1,-1)]},     # o7
    ]
    outer_b = [
        {0: [(0,0,1), (1,1,1)]},      # o1
        {1: [(0,0,1)]},               # o2
        {2: [(1,0,1), (1,1,-1)]},     # o3
        {3: [(0,0,1), (1,0,-1)]},     # o4
        {4: [(1,1,1)]},               # o5
        {5: [(0,0,1), (0,1,1)]},      # o6
        {6: [(1,0,1), (1,1,1)]},      # o7
    ]
    
    # Inner 2x2 Strassen patterns
    # Each: (u_elem, v_elem, w_elem) where elem is dict {(row,col): coeff}
    inner_u = [
        {(0,0):1, (1,1):1},  # t1
        {(1,0):1, (1,1):1},  # t2
        {(0,0):1},            # t3
        {(1,1):1},            # t4
        {(0,0):1, (0,1):1},  # t5
        {(1,0):1, (0,0):-1}, # t6
        {(0,1):1, (1,1):-1}, # t7
    ]
    inner_v = [
        {(0,0):1, (1,1):1},  # t1
        {(0,0):1},            # t2
        {(1,0):1, (1,1):-1}, # t3
        {(0,0):1, (1,0):-1}, # t4
        {(1,1):1},            # t5
        {(0,0):1, (0,1):1},  # t6
        {(1,0):1, (1,1):1},  # t7
    ]
    inner_w = [
        {(0,0):1, (1,1):1},  # t1
        {(1,0):1, (1,1):-1}, # t2
        {(0,1):1, (1,1):1},  # t3
        {(0,0):1, (1,0):1},  # t4
        {(0,0):-1, (0,1):1}, # t5
        {(1,1):1},            # t6
        {(0,0):1},            # t7
    ]
    
    c_outer = {
        (0,0): {0:1, 3:1, 4:-1, 6:1},
        (0,1): {2:1, 4:1},
        (1,0): {1:1, 3:1},
        (1,1): {0:1, 1:-1, 2:1, 5:1},
    }
    
    col = 0
    for o in range(7):
        for i in range(7):
            # Build U, V, W for this term
            for (a_br, a_bc, a_coeff) in outer_a[o]:
                for (ir, ic, iu_coeff) in inner_u[i].items():
                    gr = a_br * 2 + ir
                    gc = a_bc * 2 + ic
                    U[gr * 4 + gc][col] += a_coeff * iu_coeff
            
            for (b_br, b_bc, b_coeff) in outer_b[o]:
                for (ir, ic, iv_coeff) in inner_v[i].items():
                    gr = b_br * 2 + ir
                    gc = b_bc * 2 + ic
                    V[gr * 5 + gc][col] += b_coeff * iv_coeff
            
            for (c_br, c_bc) in c_outer:
                if o not in c_outer[(c_br, c_bc)]:
                    continue
                c_coeff = c_outer[(c_br, c_bc)][o]
                for (ir, ic, iw_coeff) in inner_w[i].items():
                    gr = c_br * 2 + ir
                    gc = c_bc * 2 + ic
                    W[gc * 4 + gr][col] += c_coeff * iw_coeff
            
            col += 1
    
    # Now add 16 schoolbook terms for column 4
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
            col += 1
    
    # Verify
    A_test = np.random.rand(4, 4)
    B_test = np.random.rand(4, 5)
    C_test = A_test @ B_test
    
    C_recon = np.zeros((4, 5))
    for t in range(65):
        for i in range(4):
            for k in range(5):
                val = 0
                for j in range(4):
                    val += U[i*4+j][t] * A_test[i,j] * V[j*5+k][t] * W[k*4+i][t]
                C_recon[i,k] += val
    
    if not np.allclose(C_recon, C_test, atol=1e-10):
        # Fallback to schoolbook
        rank = 80
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(20)]
        W = [[0] * rank for _ in range(20)]
        for i in range(4):
            for j in range(4):
                for k in range(5):
                    c = i * 20 + j * 5 + k
                    U[i*4+j][c] = 1
                    V[j*5+k][c] = 1
                    W[k*4+i][c] = 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
