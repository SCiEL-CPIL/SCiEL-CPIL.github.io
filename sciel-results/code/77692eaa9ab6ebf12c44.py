import time
import sys

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1: (x00+x11)(y00+y11) -> z00+=1, z11+=1
    # M2: (x10+x11)y00      -> z10+=1, z11-=1
    # M3: x00(y01-y11)      -> z00-=1, z01+=1
    # M4: x11(y10-y00)      -> z00+=1, z10+=1
    # M5: (x00+x01)y11      -> z00-=1, z01+=1
    # M6: (x10-x00)(y00+y01)-> z11+=1
    # M7: (x01-x11)(y10+y11)-> z00+=1
    
    strassen_terms = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # u: {00: 1, 11: 1}, v: {00: 1, 11: 1}, w: {00: 1, 11: 1}
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # u: {1: 1, 3: 1}, v: {0: 1}, w: {2: 1, 3: -1}
        ({1: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # u: {0: 1}, v: {1: 1, 3: -1}, w: {0: -1, 1: 1}
        ({0: 1}, {1: 1, 3: -1}, {0: -1, 1: 1}),
        # u: {3: 1}, v: {2: 1, 0: -1}, w: {0: 1, 2: 1}
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # u: {0: 1, 1: 1}, v: {3: 1}, w: {0: -1, 1: 1}
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # u: {1: 1, 0: -1}, v: {0: 1, 1: 1}, w: {3: 1}
        ({1: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # u: {1: 1, 3: -1}, v: {2: 1, 3: 1}, w: {0: 1}
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]
    
    # Mapping local 2x2 indices to global indices
    # Local indices: 0->00, 1->01, 2->10, 3->11
    # Part 0: A_L, B_T
    # Part 1: A_R, B_B
    
    for j_block in range(3):
        for part in range(2):
            base_rank = j_block * 14 + part * 7
            
            for t in range(7):
                r = base_rank + t
                u_coeffs, v_coeffs, w_coeffs = strassen_terms[t]
                
                # Update U
                for local_idx, val in u_coeffs.items():
                    # local_idx 0: 00, 1: 01, 2: 10, 3: 11
                    if part == 0:
                        # A_L: rows 0,1; cols 0,1
                        # 00->0, 01->1, 10->4, 11->5
                        if local_idx == 0: u_row = 0
                        elif local_idx == 1: u_row = 1
                        elif local_idx == 2: u_row = 4
                        elif local_idx == 3: u_row = 5
                    else:
                        # A_R: rows 0,1; cols 2,3
                        # 00->2, 01->3, 10->6, 11->7
                        if local_idx == 0: u_row = 2
                        elif local_idx == 1: u_row = 3
                        elif local_idx == 2: u_row = 6
                        elif local_idx == 3: u_row = 7
                    U[u_row][r] += val
                
                # Update V
                for local_idx, val in v_coeffs.items():
                    if part == 0:
                        # B_T: rows 0,1; cols 2*j_block, 2*j_block+1
                        # 00->2*j_block, 01->2*j_block+1, 10->6+2*j_block, 11->6+2*j_block+1
                        if local_idx == 0: v_row = 2 * j_block
                        elif local_idx == 1: v_row = 2 * j_block + 1
                        elif local_idx == 2: v_row = 6 + 2 * j_block
                        elif local_idx == 3: v_row = 6 + 2 * j_block + 1
                    else:
                        # B_B: rows 2,3; cols 2*j_block, 2*j_block+1
                        # 00->12+2*j_block, 01->12+2*j_block+1, 10->18+2*j_block, 11->18+2*j_block+1
                        if local_idx == 0: v_row = 12 + 2 * j_block
                        elif local_idx == 1: v_row = 12 + 2 * j_block + 1
                        elif local_idx == 2: v_row = 18 + 2 * j_block
                        elif local_idx == 3: v_row = 18 + 2 * j_block + 1
                    V[v_row][r] += val
                
                # Update W
                for local_idx, val in w_coeffs.items():
                    # C: rows 0,1; cols 2*j_block, 2*j_block+1
                    # 00->2*j_block, 01->2*j_block+1, 10->2+2*j_block, 11->2+2*j_block+1
                    if local_idx == 0: w_row = 2 * j_block
                    elif local_idx == 1: w_row = 2 * j_block + 1
                    elif local_idx == 2: w_row = 2 + 2 * j_block
                    elif local_idx == 3: w_row = 2 + 2 * j_block + 1
                    W[w_row][r] += val

    return {"U": U, "V": V, "W": W}
