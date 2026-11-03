import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    U_new = []
    V_new = []
    W_new = []
    
    def add_term(u, v, w):
        U_new.append(u)
        V_new.append(v)
        W_new.append(w)
    
    def make_vec(length, positions_values):
        vec = [0] * length
        for pos, val in positions_values:
            vec[pos] = val
        return vec
    
    # Strassen decomposition for 2x2x2 block: i in {0,1}, j in {0,1}, k in {0,1}
    # U index: i*4+j, V index: j*4+k, W index: k*3+i
    
    # M1 = (A_{0,0}+A_{1,1})(B_{0,0}+B_{1,1}) -> C_{0,0} + C_{1,1}
    u = make_vec(12, [(0, 1), (5, 1)])
    v = make_vec(16, [(0, 1), (9, 1)])
    w = make_vec(12, [(0, 1), (4, 1)])
    add_term(u, v, w)
    
    # M2 = (A_{1,0}+A_{1,1})B_{0,0} -> C_{1,0} + C_{1,1}
    u = make_vec(12, [(4, 1), (5, 1)])
    v = make_vec(16, [(0, 1)])
    w = make_vec(12, [(1, 1), (4, 1)])
    add_term(u, v, w)
    
    # M3 = A_{0,0}(B_{0,1}-B_{1,1}) -> C_{0,1} + C_{1,1}
    u = make_vec(12, [(0, 1)])
    v = make_vec(16, [(1, 1), (9, -1)])
    w = make_vec(12, [(1, 1), (4, 1)])
    add_term(u, v, w)
    
    # M4 = A_{1,1}(B_{1,0}-B_{0,0}) -> C_{0,0} + C_{1,0}
    u = make_vec(12, [(5, 1)])
    v = make_vec(16, [(4, 1), (0, -1)])
    w = make_vec(12, [(0, 1), (1, 1)])
    add_term(u, v, w)
    
    # M5 = (A_{0,0}+A_{0,1})B_{1,1} -> C_{0,0} + C_{0,1}
    u = make_vec(12, [(0, 1), (1, 1)])
    v = make_vec(16, [(9, 1)])
    w = make_vec(12, [(0, 1), (1, 1)])
    add_term(u, v, w)
    
    # M6 = (A_{1,0}-A_{0,0})(B_{0,0}+B_{0,1}) -> C_{1,1}
    u = make_vec(12, [(4, 1), (0, -1)])
    v = make_vec(16, [(0, 1), (1, 1)])
    w = make_vec(12, [(4, 1)])
    add_term(u, v, w)
    
    # M7 = (A_{0,1}-A_{1,1})(B_{1,0}+B_{1,1}) -> C_{0,0}
    u = make_vec(12, [(1, 1), (5, -1)])
    v = make_vec(16, [(4, 1), (9, 1)])
    w = make_vec(12, [(0, 1)])
    add_term(u, v, w)
    
    # Other blocks with schoolbook decomposition
    # i in {0,1}, j in {2,3}, k in {0,1}
    for i in range(2):
        for k in range(2):
            for j in range(2, 4):
                u = make_vec(12, [(i*4 + j, 1)])
                v = make_vec(16, [(j*4 + k, 1)])
                w = make_vec(12, [(k*3 + i, 1)])
                add_term(u, v, w)
    
    # i in {0,1}, j in {0,1}, k in {2,3}
    for i in range(2):
        for k in range(2, 4):
            for j in range(2):
                u = make_vec(12, [(i*4 + j, 1)])
                v = make_vec(16, [(j*4 + k, 1)])
                w = make_vec(12, [(k*3 + i, 1)])
                add_term(u, v, w)
    
    # i in {0,1}, j in {2,3}, k in {2,3}
    for i in range(2):
        for k in range(2, 4):
            for j in range(2, 4):
                u = make_vec(12, [(i*4 + j, 1)])
                v = make_vec(16, [(j*4 + k, 1)])
                w = make_vec(12, [(k*3 + i, 1)])
                add_term(u, v, w)
    
    # i=2, j in {0,1}, k in {0,1}
    for k in range(2):
        for j in range(2):
            u = make_vec(12, [(2*4 + j, 1)])
            v = make_vec(16, [(j*4 + k, 1)])
            w = make_vec(12, [(k*3 + 2, 1)])
            add_term(u, v, w)
    
    # i=2, j in {2,3}, k in {0,1}
    for k in range(2):
        for j in range(2, 4):
            u = make_vec(12, [(2*4 + j, 1)])
            v = make_vec(16, [(j*4 + k, 1)])
            w = make_vec(12, [(k*3 + 2, 1)])
            add_term(u, v, w)
    
    # i=2, j in {0,1}, k in {2,3}
    for k in range(2, 4):
        for j in range(2):
            u = make_vec(12, [(2*4 + j, 1)])
            v = make_vec(16, [(j*4 + k, 1)])
            w = make_vec(12, [(k*3 + 2, 1)])
            add_term(u, v, w)
    
    # i=2, j in {2,3}, k in {2,3}
    for k in range(2, 4):
        for j in range(2, 4):
            u = make_vec(12, [(2*4 + j, 1)])
            v = make_vec(16, [(j*4 + k, 1)])
            w = make_vec(12, [(k*3 + 2, 1)])
            add_term(u, v, w)
    
    rank = len(U_new)
    
    return {"U": U_new, "V": V_new, "W": W_new}
# EVOLVE-BLOCK-END
