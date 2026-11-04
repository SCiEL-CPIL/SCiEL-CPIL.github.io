"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    bs = 3  # block size
    nb = 2  # number of blocks per dimension
    
    # Strassen terms: (A_block_coeffs, B_block_coeffs, C_block_coeffs)
    terms = [
        # M1 = (A11+A22)*(B11+B22) -> C11+=1, C22+=1
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        # M2 = (A21+A22)*B11 -> C21+=1, C22-=1
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}),
        # M3 = A11*(B12-B22) -> C12+=1, C22+=1
        ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}),
        # M4 = A22*(B21-B11) -> C11+=1, C21+=1
        ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),
        # M5 = (A11+A12)*B22 -> C11-=1, C12+=1
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),
        # M6 = (A21-A11)*(B11+B12) -> C22+=1
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
        # M7 = (A12-A22)*(B21+B22) -> C11+=1, C22+=1
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
    ]
    
    rank = 7 * bs * bs * bs
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for (ac, bc, cc) in terms:
        for r in range(bs):
            for s in range(bs):
                for t in range(bs):
                    # For this (r,s,t), we need to handle all block combinations
                    # L is formed from A blocks, R from B blocks
                    # L[br*bs+r, bs*bc+s] gets coefficient ac[(br,bc)]
                    # R[br*bs+s, bs*bc+t] gets coefficient bc[(br,bc)] -- wait, need to be careful
                    
                    # Actually:
                    # Left operand L[i,j]: i = br*bs + r, j = bc*bs + s, coeff = ac.get((br,bc),0)
                    # Right operand R[j,k]: j = br*bs + s, k = bc*bs + t, coeff = bc.get((br,bc),0)
                    # Note: the 's' index must match between L's column and R's row
                    # L column block index = bc, R row block index = br (same as L's row block? No!)
                    
                    # Let me use different variables:
                    # L row block = bi, L col block = bj
                    # R row block = bk, R col block = bl
                    # For L*R to be valid, bj must equal bk (L's col = R's row)
                    # So bk = bj
                    
                    # L[bi*bs+r, bj*bs+s] * R[bj*bs+s, bl*bs+t]
                    # -> contributes to C[ci*bs+r, cl*bs+t] with coeff ac[(bi,bj)] * bc[(bj,bl)] * cc[(ci,cl)]
                    
                    for bi in range(nb):
                        for bj in range(nb):
                            for bl in range(nb):
                                a_val = ac.get((bi,bj), 0)
                                b_val = bc.get((bj,bl), 0)
                                if a_val == 0 or b_val == 0:
                                    continue
                                for ci in range(nb):
                                    for cl in range(nb):
                                        c_val = cc.get((ci,cl), 0)
                                        if c_val == 0:
                                            continue
                                        
                                        u_idx = (bi*bs + r) * m + (bj*bs + s)
                                        v_idx = (bj*bs + s) * p + (bl*bs + t)
                                        w_idx = (cl*bs + t) * n + (ci*bs + r)
                                        
                                        U[u_idx][col] += a_val
                                        V[v_idx][col] += b_val
                                        W[w_idx][col] += c_val
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
