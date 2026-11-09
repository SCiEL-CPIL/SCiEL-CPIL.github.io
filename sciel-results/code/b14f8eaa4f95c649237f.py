"""Matrix multiplication 3,3,7 - schoolbook decomposition."""
import numpy as np
import time

def construct():
    n, m, p = 3, 3, 7
    U_cols = []
    V_cols = []
    W_cols = []
    
    for i in range(3):
        for j in range(3):
            for k in range(7):
                u = [0]*9
                v = [0]*21
                w = [0]*21
                u[i*3+j] = 1
                v[j*7+k] = 1
                w[k*3+i] = 1
                U_cols.append(u)
                V_cols.append(v)
                W_cols.append(w)
    
    rank = len(U_cols)
    U = [[U_cols[c][r] for c in range(rank)] for r in range(9)]
    V = [[V_cols[c][r] for c in range(rank)] for r in range(21)]
    W = [[W_cols[c][r] for c in range(rank)] for r in range(21)]
    
    return {"U": U, "V": V, "W": W}
