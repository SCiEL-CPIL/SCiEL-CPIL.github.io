"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = 6, 6, 6
    rank = 189
    
    # Strassen 2x2 block decomposition with 3x3 schoolbook leaves
    # Block linear index: block(i,j) where i,j in {0,1,2}
    # A11(i,j) = i*6+j, A12(i,j) = i*6+j+3, A21(i,j) = (i+3)*6+j, A22(i,j) = (i+3)*6+j+3
    
    def lin(block, i, j):
        # block: 0=A11, 1=A12, 2=A21, 3=A22
        br, bc = divmod(block, 2)
        return (br*3+i)*6 + (bc*3+j)
    
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    # Strassen products: (left_blocks, right_blocks, output_blocks)
    # Each is a list of (block, sign)
    strassen = [
        # M1
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        # M2
        ([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]),
        # M3
        ([(0,1)], [(1,1),(3,-1)], [(1,1),(3,1)]),
        # M4
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),
        # M5
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(3,1)]),
        # M6
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        # M7
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]
    
    t = 0
    for left, right, output in strassen:
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    for blk, sign in left:
                        U[lin(blk, a, b)][t] += sign
                    for blk, sign in right:
                        V[lin(blk, b, c)][t] += sign
                    for blk, sign in output:
                        W[lin(blk, a, c)][t] += sign
                    t += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
