import numpy as np
import time
import random

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # The tensor for matrix multiplication C = AB is defined by
    # C_ik = sum_j A_ij B_jk
    # This implies that for every triplet (i, j, k), the coefficient of
    # A_ij * B_jk in the expression for C_ik is 1.
    # The problem asks for a decomposition of the tensor T_ijk such that
    # sum_c U_ij^c V_jk^c W_ki^c = T_ijk
    # Since T_ijk = 1 for all i, j, k, the tensor is the all-ones tensor.
    # The rank of the all-ones tensor is 1.
    # We can use U_ij = 1, V_jk = 1, W_ki = 1.
    
    rank = 1
    
    U = np.ones((n * m, rank), dtype=float)
    V = np.ones((m * p, rank), dtype=float)
    W = np.ones((p * n, rank), dtype=float)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
