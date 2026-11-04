"""
Rewritten program for matrix multiplication tensor decomposition.
Task: M_{2,3,10}
Dimensions: n=2, m=3, p=10
Standard rank: 60
Optimized rank: 55 (using Strassen for M_{2,2,10} part)
"""
import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 10]
DEADLINE = 140

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=10
    # U shape: (n*m, rank) = (6, 55)
    # V shape: (m*p, rank) = (30, 55)
    # W shape: (p*n, rank) = (20, 55)
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: M_{2,1,10} (A col 2, B row 2)
    # A index: i*m + 2. i in {0,1}
    # B index: 2*p + k. k in {0..9}
    # W index: k*n + i. k in {0..9}, i in {0,1}
    
    r = 0
    for i in range(n):
        for k in range(p):
            a_idx = i * m + 2
            b_idx = 2 * p + k
            w_idx = k * n + i
            
            U[a_idx][r] = 1
            V[b_idx][r] = 1
            W[w_idx][r] = 1
            r += 1
            
    # Part 2: M_{2,2,10} (A cols 0,1, B rows 0,1)
    # Split B rows 0,1 into 5 blocks of 2 columns each.
    # Block t in {0..4} corresponds to k in {2t, 2t+1}
    
    # Local indices for M_{2,2,2}
    # A local: 0(0,0), 1(0,1), 2(1,0), 3(1,1)
    # B local: 0(0,0), 1(0,1), 2(1,0), 3(1,1)
    # W local: 0(0,0), 1(0,1), 2(1,0), 3(1,1)
    
    # Global A indices for local 0,1,2,3
    # a=0 (i=0,j=0) -> 0
    # a=1 (i=0,j=1) -> 1
    # a=2 (i=1,j=0) -> 3
    # a=3 (i=1,j=1) -> 4
    a_global_map = [0, 1, 3, 4]
    
    # Strassen factors
    # U_local (4x7), V_local (4x7), W_local (4x7)
    U_loc = [
        [1, 0, 1, 0, 1, -1, 0], # a=0
        [0, 0, 0, 0, 1, 0, 1],  # a=1
        [0, 1, 0, 0, 0, 1, 0],  # a=2
        [1, 1, 0, 1, 0, 0, -1]  # a=3
    ]
    V_loc = [
        [1, 1, 0, -1, 0, 1, 0], # b=0
        [0, 0, 0, 1, 0, 1, 0],  # b=1
        [0, 0, 1, 0, 0, 0, 1],  # b=2
        [1, 0, -1, 0, 1, 0, 1]  # b=3
    ]
    W_loc = [
        [1, 0, 0, -1, -1, 0, 1], # c=0
        [0, 0, 1, 0, 1, 0, 0],   # c=1
        [0, 1, 0, 1, 0, 0, 0],   # c=2
        [1, -1, 1, 0, 0, -1, 0]  # c=3
    ]
    
    for t in range(5):
        # B global indices for local 0,1,2,3 in block t
        # b=0 (j=0, k=2t) -> 0*p + 2t = 2t
        # b=1 (j=0, k=2t+1) -> 0*p + 2t+1 = 2t+1
        # b=2 (j=1, k=2t) -> 1*p + 2t = 10+2t
        # b=3 (j=1, k=2t+1) -> 1*p + 2t+1 = 11+2t
        b_global_map = [2*t, 2*t+1, 10+2*t, 11+2*t]
        
        # W global indices for local 0,1,2,3 in block t
        # c=0 (k=2t, i=0) -> 2t*2 + 0 = 4t
        # c=1 (k=2t, i=1) -> 2t*2 + 1 = 4t+1
        # c=2 (k=2t+1, i=0) -> (2t+1)*2 + 0 = 4t+2
        # c=3 (k=2t+1, i=1) -> (2t+1)*2 + 1 = 4t+3
        w_global_map = [4*t, 4*t+1, 4*t+2, 4*t+3]
        
        for s in range(7):
            r = 20 + t * 7 + s
            
            for a_loc in range(4):
                val = U_loc[a_loc][s]
                if val != 0:
                    a_glob = a_global_map[a_loc]
                    U[a_glob][r] += val
                    
            for b_loc in range(4):
                val = V_loc[b_loc][s]
                if val != 0:
                    b_glob = b_global_map[b_loc]
                    V[b_glob][r] += val
                    
            for w_loc in range(4):
                val = W_loc[w_loc][s]
                if val != 0:
                    w_glob = w_global_map[w_loc]
                    W[w_glob][r] += val

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
