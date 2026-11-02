import numpy as np

def construct():
    N, M, P = 2, 4, 6
    NM = N * M
    MP = M * P
    PN = P * N
    
    rank = 1
    
    U = np.ones((NM, rank))
    V = np.ones((MP, rank))
    W = np.ones((PN, rank))
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
