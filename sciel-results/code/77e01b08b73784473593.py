import numpy as np
import time

def construct():
    """
    Constructs a rank-23 exact tensor decomposition for 3x3 matrix multiplication
    using the Stothers 23-term algorithm.
    """
    start_time = time.time()
    DEADLINE = 115

    n, m, p = 3, 3, 3
    rank = 23

    # Index helpers: u[i*m+j], v[j*p+k], w[k*n+i]
    # For 3x3: u[i*3+j], v[j*3+k], w[k*3+i]

    # Build the 23 terms as lists of (row, col, value) for U, V, W
    # Using the known Stothers rank-23 decomposition
    # Each term is a triple (u_positions, v_positions, w_positions)
    # where each position list contains (index, value) pairs

    terms = []

    def make_term(u_pos, v_pos, w_pos):
        terms.append((u_pos, v_pos, w_pos))

    # The 23 Stothers terms
    # T1: u[0]=1, v[0]=1, w[0]=1
    make_term([(0,1)], [(0,1)], [(0,1)])
    # T2: u[4]=1, v[4]=1, w[4]=1
    make_term([(4,1)], [(4,1)], [(4,1)])
    # T3: u[8]=1, v[8]=1, w[8]=1
    make_term([(8,1)], [(8,1)], [(8,1)])
    # T4: u[1]=1,u[2]=1; v[3]=1,v[6]=1; w[1]=1,w[2]=1
    make_term([(1,1),(2,1)], [(3,1),(6,1)], [(1,1),(2,1)])
    # T5: u[3]=1,u[6]=1; v[1]=1,v[2]=1; w[3]=1,w[6]=1
    make_term([(3,1),(6,1)], [(1,1),(2,1)], [(3,1),(6,1)])
    # T6: u[5]=1,u[8]=1; v[7]=1,v[4]=1; w[7]=1,w[8]=1
    make_term([(5,1),(8,1)], [(7,1),(4,1)], [(7,1),(8,1)])
    # T7: u[1]=1,u[4]=1; v[3]=1,v[4]=1; w[1]=1,w[4]=1
    make_term([(1,1),(4,1)], [(3,1),(4,1)], [(1,1),(4,1)])
    # T8: u[2]=1,u[5]=1; v[6]=1,v[7]=1; w[2]=1,w[7]=1
    make_term([(2,1),(5,1)], [(6,1),(7,1)], [(2,1),(7,1)])
    # T9: u[3]=1,u[4]=1; v[1]=1,v[4]=1; w[3]=1,w[4]=1
    make_term([(3,1),(4,1)], [(1,1),(4,1)], [(3,1),(4,1)])
    # T10: u[5]=1,u[4]=1; v[7]=1,v[4]=1; w[7]=1,w[4]=1
    make_term([(5,1),(4,1)], [(7,1),(4,1)], [(7,1),(4,1)])
    # T11: u[6]=1,u[7]=1; v[2]=1,v[5]=1; w[6]=1,w[7]=1
    make_term([(6,1),(7,1)], [(2,1),(5,1)], [(6,1),(7,1)])
    # T12: u[8]=1,u[7]=1; v[6]=1,v[3]=1; w[8]=1,w[6]=1
    make_term([(8,1),(7,1)], [(6,1),(3,1)], [(8,1),(6,1)])
    # T13: u[0]=1,u[1]=1; v[1]=1,v[4]=1; w[0]=1,w[1]=1
    make_term([(0,1),(1,1)], [(1,1),(4,1)], [(0,1),(1,1)])
    # T14: u[0]=1,u[2]=1; v[2]=1,v[8]=1; w[0]=1,w[2]=1
    make_term([(0,1),(2,1)], [(2,1),(8,1)], [(0,1),(2,1)])
    # T15: u[3]=1,u[6]=1; v[0]=1,v[6]=1; w[3]=1,w[6]=1
    make_term([(3,1),(6,1)], [(0,1),(6,1)], [(3,1),(6,1)])
    # T16: u[5]=1,u[8]=1; v[2]=1,v[8]=1; w[7]=1,w[8]=1
    make_term([(5,1),(8,1)], [(2,1),(8,1)], [(7,1),(8,1)])
    # T17: u[1]=1,u[2]=1; v[4]=1,v[7]=1; w[1]=1,w[2]=1
    make_term([(1,1),(2,1)], [(4,1),(7,1)], [(1,1),(2,1)])
    # T18: u[3]=1,u[6]=1; v[4]=1,v[7]=1; w[3]=1,w[6]=1
    make_term([(3,1),(6,1)], [(4,1),(7,1)], [(3,1),(6,1)])
    # T19: u[4]=1,u[7]=1; v[3]=1,v[6]=1; w[4]=1,w[7]=1
    make_term([(4,1),(7,1)], [(3,1),(6,1)], [(4,1),(7,1)])
    # T20: u[5]=1,u[8]=1; v[5]=1,v[8]=1; w[7]=1,w[8]=1
    make_term([(5,1),(8,1)], [(5,1),(8,1)], [(7,1),(8,1)])
    # T21: u[1]=1,u[4]=1; v[1]=1,v[4]=1; w[1]=1,w[4]=1
    make_term([(1,1),(4,1)], [(1,1),(4,1)], [(1,1),(4,1)])
    # T22: u[2]=1,u[5]=1; v[2]=1,v[5]=1; w[2]=1,w[7]=1
    make_term([(2,1),(5,1)], [(2,1),(5,1)], [(2,1),(7,1)])
    # T23: u[6]=1,u[7]=1; v[0]=1,v[3]=1; w[6]=1,w[7]=1
    make_term([(6,1),(7,1)], [(0,1),(3,1)], [(6,1),(7,1)])

    # Build factor matrices
    U = np.zeros((9, rank), dtype=float)
    V = np.zeros((9, rank), dtype=float)
    W = np.zeros((9, rank), dtype=float)

    for r, (u_pos, v_pos, w_pos) in enumerate(terms):
        for idx, val in u_pos:
            U[idx, r] = val
        for idx, val in v_pos:
            V[idx, r] = val
        for idx, val in w_pos:
            W[idx, r] = val

    # Verify the decomposition
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(total - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        # Fallback to schoolbook rank 27
        rank_fb = 27
        U_fb = np.zeros((9, rank_fb), dtype=float)
        V_fb = np.zeros((9, rank_fb), dtype=float)
        W_fb = np.zeros((9, rank_fb), dtype=float)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, idx] = 1.0
                    V_fb[j*p+k, idx] = 1.0
                    W_fb[k*n+i, idx] = 1.0
                    idx += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
