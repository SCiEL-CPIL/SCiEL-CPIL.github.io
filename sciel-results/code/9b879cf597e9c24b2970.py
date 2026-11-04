import numpy as np

def construct():
       n, m, p = 3, 3, 3
       rank = 27
       U = np.zeros((9, rank), dtype=float)
       V = np.zeros((9, rank), dtype=float)
       W = np.zeros((9, rank), dtype=float)
       
       idx = 0
       for i in range(3):
           for j in range(3):
               for k in range(3):
                   U[i*3 + j, idx] = 1.0
                   V[j*3 + k, idx] = 1.0
                   W[k*3 + i, idx] = 1.0
                   idx += 1
                   
       return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
