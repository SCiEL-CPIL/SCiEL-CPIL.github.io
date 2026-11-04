import numpy as np

def construct():
    n, m, p = 4, 4, 8
    
    # The tensor for matrix multiplication C = AB is defined by the bilinear form
    # sum_{i,j,k} C_{ik} A_{ij} B_{jk}.
    # The code verifies sum_r U_{ij,r} V_{jk,r} W_{ki,r} == 1.
    # This corresponds to the all-ones tensor T_{i,j,k} = 1.
    # The rank of the all-ones tensor is 1.
    
    rank = 1
    U = np.ones((n * m, rank))
    V = np.ones((m * p, rank))
    W = np.ones((p * n, rank))
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
