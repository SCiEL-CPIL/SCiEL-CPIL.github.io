import time
import sys

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 5
    rank = 17
    U = [[0] * rank for _ in range(n * m)]  # 6 x 17
    V = [[0] * rank for _ in range(m * p)]  # 15 x 17
    W = [[0] * rank for _ in range(p * n)]  # 10 x 17
    
    r = 0
    
    def add_term(u_entries, v_entries, w_entries):
        nonlocal r
        for idx, val in u_entries:
            U[idx][r] = val
        for idx, val in v_entries:
            V[idx][r] = val
        for idx, val in w_entries:
            W[idx][r] = val
        r += 1
        
    # Block 1: j=2
    # Rank 1 decomposition of all-ones 2x5
    # U[i,2] = 1, V[2,k] = 1, W[k,i] = 1
    u_entries = [(i*m+2, 1) for i in range(n)]
    v_entries = [(2*p+k, 1) for k in range(p)]
    w_entries = [(k*n+i, 1) for k in range(p) for i in range(n)]
    add_term(u_entries, v_entries, w_entries)
    
    # Block 2: j=0,1 for k=4
    # Rank 2 decomposition of I_2
    # r=0: j=0
    add_term([(0*m+0, 1), (1*m+0, 1)], [(0*p+4, 1)], [(4*n+0, 1), (4*n+1, 1)])
    # r=1: j=1
    add_term([(0*m+1, 1), (1*m+1, 1)], [(1*p+4, 1)], [(4*n+0, 1), (4*n+1, 1)])
    
    # Block 3: j=0,1 for k=0,1 (Strassen)
    # M1 = (a00+a11)(b00+b11)
    add_term([(0*m+0, 1), (1*m+1, 1)], [(0*p+0, 1), (1*p+1, 1)], [(0*n+0, 1), (1*n+1, 1)])
    
    # M2 = (a10+a11)(b00)
    add_term([(1*m+0, 1), (1*m+1, 1)], [(0*p+0, 1)], [(0*n+1, 1), (1*n+1, -1)])
    
    # M3 = (a00)(b01-b11)
    add_term([(0*m+0, 1)], [(0*p+1, 1), (1*p+1, -1)], [(1*n+0, 1), (1*n+1, 1)])
    
    # M4 = (a11)(b10-b00)
    add_term([(1*m+1, 1)], [(1*p+0, 1), (0*p+0, -1)], [(0*n+0, 1), (0*n+1, 1)])
    
    # M5 = (a00+a01)(b11)
    add_term([(0*m+0, 1), (0*m+1, 1)], [(1*p+1, 1)], [(0*n+0, -1), (1*n+0, 1)])
    
    # M6 = (a10-a00)(b00+b01)
    add_term([(1*m+0, 1), (0*m+0, -1)], [(0*p+0, 1), (0*p+1, 1)], [(1*n+1, 1)])
    
    # M7 = (a01-a11)(b10+b11)
    add_term([(0*m+1, 1), (1*m+1, -1)], [(1*p+0, 1), (1*p+1, 1)], [(0*n+0, 1)])
    
    # Block 4: j=0,1 for k=2,3 (Strassen)
    # M1 = (a00+a11)(b02+b13)
    add_term([(0*m+0, 1), (1*m+1, 1)], [(0*p+2, 1), (1*p+3, 1)], [(2*n+0, 1), (3*n+1, 1)])
    
    # M2 = (a10+a11)(b02)
    add_term([(1*m+0, 1), (1*m+1, 1)], [(0*p+2, 1)], [(2*n+1, 1), (3*n+1, -1)])
    
    # M3 = (a00)(b03-b13)
    add_term([(0*m+0, 1)], [(0*p+3, 1), (1*p+3, -1)], [(3*n+0, 1), (3*n+1, 1)])
    
    # M4 = (a11)(b12-b02)
    add_term([(1*m+1, 1)], [(1*p+2, 1), (0*p+2, -1)], [(2*n+0, 1), (2*n+1, 1)])
    
    # M5 = (a00+a01)(b13)
    add_term([(0*m+0, 1), (0*m+1, 1)], [(1*p+3, 1)], [(2*n+0, -1), (3*n+0, 1)])
    
    # M6 = (a10-a00)(b02+b03)
    add_term([(1*m+0, 1), (0*m+0, -1)], [(0*p+2, 1), (0*p+3, 1)], [(3*n+1, 1)])
    
    # M7 = (a01-a11)(b12+b13)
    add_term([(0*m+1, 1), (1*m+1, -1)], [(1*p+2, 1), (1*p+3, 1)], [(2*n+0, 1)])
    
    return {"U": U, "V": V, "W": W}
