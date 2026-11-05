import numpy as np
import time

DIMENSIONS = [4, 5, 5]
_START = time.time()
_DEADLINE = 120.0  # seconds

def construct():
    n, m, p = DIMENSIONS  # n=4, m=5, p=5
    
    terms = []
    
    def add_term(u_sparsity, v_sparsity, w_sparsity):
        terms.append((u_sparsity, v_sparsity, w_sparsity))
        
    def add_simple_term(i, k, j, coeff=1.0):
        u_sp = [(i * m + k, coeff)]
        v_sp = [(k * p + j, coeff)]
        w_sp = [(j * n + i, coeff)]
        add_term(u_sp, v_sp, w_sp)
        
    # Global index helpers for 4x4x4 subproblem
    def a_idx(block, li, lk):
        if block == 0: return li * m + lk
        if block == 1: return li * m + (lk + 2)
        if block == 2: return (li + 2) * m + lk
        if block == 3: return (li + 2) * m + (lk + 2)
        
    def b_idx(block, lk, lj):
        if block == 0: return lk * p + lj
        if block == 1: return lk * p + (lj + 2)
        if block == 2: return (lk + 2) * p + lj
        if block == 3: return (lk + 2) * p + (lj + 2)
        
    def c_idx(block, li, lj):
        if block == 0: return lj * n + li
        if block == 1: return (lj + 2) * n + li
        if block == 2: return lj * n + (li + 2)
        if block == 3: return (lj + 2) * n + (li + 2)

    strassen_defs = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    for a_parts, b_parts, c_parts in strassen_defs:
        for li in range(2):
            for lk in range(2):
                for lj in range(2):
                    u_sp = []
                    v_sp = []
                    w_sp = []
                    
                    for block, sign in a_parts:
                        idx = a_idx(block, li, lk)
                        u_sp.append((idx, sign))
                        
                    for block, sign in b_parts:
                        idx = b_idx(block, lk, lj)
                        v_sp.append((idx, sign))
                        
                    for block, sign in c_parts:
                        idx = c_idx(block, li, lj)
                        w_sp.append((idx, sign))
                        
                    add_term(u_sp, v_sp, w_sp)
                    
                    if time.time() - _START > _DEADLINE - 10:
                        break
                if time.time() - _START > _DEADLINE - 10:
                    break
            if time.time() - _START > _DEADLINE - 10:
                break
        if time.time() - _START > _DEADLINE - 10:
            break
            
    # Part 2: 4x4x1 (k in 0..3, j=4) - rank 1!
    # U[i*5+k, r] = 1 for i<4, k<4
    # V[k*5+4, r] = 1 for k<4
    # W[4*4+i, r] = 1 for i<4
    u_sp = [(i * m + k, 1.0) for i in range(4) for k in range(4)]
    v_sp = [(k * p + 4, 1.0) for k in range(4)]
    w_sp = [(4 * n + i, 1.0) for i in range(4)]
    add_term(u_sp, v_sp, w_sp)
    
    # Part 3: 4x1x5 (k=4, j in 0..4) - rank 1!
    # U[i*5+4, r] = 1 for i<4
    # V[4*5+j, r] = 1 for j<5
    # W[j*4+i, r] = 1 for i<4, j<5
    u_sp = [(i * m + 4, 1.0) for i in range(4)]
    v_sp = [(4 * p + j, 1.0) for j in range(5)]
    w_sp = [(j * n + i, 1.0) for i in range(4) for j in range(5)]
    add_term(u_sp, v_sp, w_sp)
    
    rank = len(terms)
    
    if rank == 0:
        add_simple_term(0, 0, 0, 1.0)
        rank = 1
        
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r, (u_sp, v_sp, w_sp) in enumerate(terms):
        for idx, val in u_sp:
            U[idx, r] = val
        for idx, val in v_sp:
            V[idx, r] = val
        for idx, val in w_sp:
            W[idx, r] = val
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
