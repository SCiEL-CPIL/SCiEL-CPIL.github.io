import time
import sys

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # rank = 42
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2x2
    # X = [[x00, x01], [x10, x11]]
    # Y = [[y00, y01], [y10, y11]]
    # Z = XY
    
    # We define the 7 terms as (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict {x_index: coeff}
    # v_coeffs: dict {y_index: coeff}
    # w_coeffs: dict {z_index: coeff}
    
    # x indices: 0:00, 1:01, 2:10, 3:11
    # y indices: 0:00, 1:01, 2:10, 3:11
    # z indices: 0:00, 1:01, 2:10, 3:11
    
    terms = [
        # m1 = (x00 + x11)(y00 + y11) -> z00 += 1, z11 += 1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # m2 = (x10 + x11)y00 -> z10 += 1, z11 -= 1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # m3 = x00(y10 - y11) -> z01 += 1, z11 += 1
        ({0: 1}, {2: 1, 3: -1}, {1: 1, 3: 1}),
        # m4 = x11(y01 - y00) -> z00 += 1, z10 += 1
        ({3: 1}, {1: 1, 0: -1}, {0: 1, 2: 1}),
        # m5 = (x00 + x01)y11 -> z00 -= 1, z01 += 1
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # m6 = (x10 - x00)(y00 + y01) -> z11 += 1
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # m7 = (x01 - x11)(y10 + y11) -> z00 += 1
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]
    
    # Block definitions
    # A blocks: 0 (j=0,1), 1 (j=2,3)
    # B blocks: 0 (k=0,1), 1 (k=2,3), 2 (k=4,5)
    # Note: B is multiplied by A blocks.
    # A_block 0 multiplies B_block 0, 1, 2
    # A_block 1 multiplies B_block 0, 1, 2
    
    col = 0
    for a_block in range(2):
        for b_block in range(3):
            # Determine U indices
            # A_block 0: j=0,1. Rows: 0,1,4,5
            # A_block 1: j=2,3. Rows: 2,3,6,7
            if a_block == 0:
                u_idx = [0, 1, 4, 5] # x00, x01, x10, x11
            else:
                u_idx = [2, 3, 6, 7]
                
            # Determine V indices
            # B_block 0: k=0,1.
            # If a_block=0 (j=0,1): V rows 0,1,6,7
            # If a_block=1 (j=2,3): V rows 12,13,18,19
            # B_block 1: k=2,3.
            # If a_block=0: V rows 2,3,8,9
            # If a_block=1: V rows 14,15,20,21
            # B_block 2: k=4,5.
            # If a_block=0: V rows 4,5,10,11
            # If a_block=1: V rows 16,17,22,23
            
            j_start = a_block * 2
            k_start = b_block * 2
            
            # V indices for y00, y01, y10, y11
            # y00: j=j_start, k=k_start -> j_start*p + k_start
            # y01: j=j_start, k=k_start+1 -> j_start*p + k_start+1
            # y10: j=j_start+1, k=k_start -> (j_start+1)*p + k_start
            # y11: j=j_start+1, k=k_start+1 -> (j_start+1)*p + k_start+1
            
            v_idx = [
                j_start * p + k_start,
                j_start * p + k_start + 1,
                (j_start + 1) * p + k_start,
                (j_start + 1) * p + k_start + 1
            ]
            
            # Determine W indices
            # Output C block corresponds to k_start, k_start+1
            # W index = k*n + i
            # z00: k=k_start, i=0 -> k_start*n + 0
            # z01: k=k_start, i=1 -> k_start*n + 1
            # z10: k=k_start+1, i=0 -> (k_start+1)*n + 0
            # z11: k=k_start+1, i=1 -> (k_start+1)*n + 1
            
            w_idx = [
                k_start * n + 0,
                k_start * n + 1,
                (k_start + 1) * n + 0,
                (k_start + 1) * n + 1
            ]
            
            # Add 7 Strassen terms
            for u_c, v_c, w_c in terms:
                for x_i, coeff in u_c.items():
                    U[u_idx[x_i]][col] += coeff
                for y_i, coeff in v_c.items():
                    V[v_idx[y_i]][col] += coeff
                for z_i, coeff in w_c.items():
                    W[w_idx[z_i]][col] += coeff
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
