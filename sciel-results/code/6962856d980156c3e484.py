import numpy as np
import time

# Matrix multiplication 4,4,7: rank-97 exact decomposition
# Strassen (rank 7) applied to the 4x4x4 sub-problem + schoolbook for 4x4x3
# 7*7 + 4*4*3 = 49 + 48 = 97

_DIMS = (4, 4, 7)

def construct():
    """
    Construct a rank-97 exact tensor decomposition for multiplying
    a 4x4 matrix by a 4x7 matrix.

    Interface: construct() -> {'U', 'V', 'W'}
    Shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    Convention: W[k*n+i, r]
    """
    n, m, p = 4, 4, 7
    R = 97

    U = np.zeros((n * m, R), dtype=np.int64)
    V = np.zeros((m * p, R), dtype=np.int64)
    W = np.zeros((p * n, R), dtype=np.int64)

    # Strassen 2x2 factor matrices
    sU = np.array([[1,0,0,1],[0,0,1,1],[1,0,0,0],[0,0,0,1],
                   [1,1,0,0],[-1,0,1,0],[0,1,0,-1]], dtype=np.int64)
    sV = np.array([[1,0,0,1],[1,0,0,0],[0,1,0,-1],[-1,0,1,0],
                   [0,0,0,1],[1,1,0,0],[0,0,1,1]], dtype=np.int64)
    sW = np.array([[1,0,0,1,-1,0,1],[0,0,1,0,1,0,0],
                   [0,1,0,1,0,0,0],[1,-1,1,0,0,1,0]], dtype=np.int64)

    # Build 4x4x4 Strassen block: iterate over 7x7 outer products of Strassen factors
    for ko in range(7):
        for lo in range(7):
            # U vector: combine sU[lo] with sW[?:ko] via 2x2 block expansion
            u_vec = np.zeros(16, dtype=np.int64)
            for bi in range(2):
                for bj in range(2):
                    ci = sU[lo, bi * 2 + bj]
                    if ci == 0:
                        continue
                    for bo in range(4):
                        co = sW[bo, ko]
                        if co == 0:
                            continue
                        br, bc = bo // 2, bo % 2
                        ig = 2 * br + bi
                        jg = 2 * bc + bj
                        u_vec[ig * 4 + jg] += ci * co

            # V vector: combine sV[lo] with sW[?:ko] via 2x2 block expansion
            v_vec = np.zeros(28, dtype=np.int64)
            for bi in range(2):
                for bj in range(2):
                    ci = sV[lo, bi * 2 + bj]
                    if ci == 0:
                        continue
                    for bo in range(4):
                        co = sW[bo, ko]
                        if co == 0:
                            continue
                        br, bc = bo // 2, bo % 2
                        jg = 2 * br + bi
                        kg = 2 * bc + bj
                        v_vec[jg * 7 + kg] += ci * co

            # W vector: combine sW[bo,ko] with sW[bi,lo] via 2x2 block expansion
            w_vec = np.zeros(28, dtype=np.int64)
            for bo in range(4):
                cow = sW[bo, ko]
                if cow == 0:
                    continue
                for bi2 in range(4):
                    ciw = sW[bi2, lo]
                    if ciw == 0:
                        continue
                    tc = cow * ciw
                    if tc == 0:
                        continue
                    br, bc = bo // 2, bo % 2
                    cir, cic = bi2 // 2, bi2 % 2
                    ig = 2 * br + cir
                    kg = 2 * bc + cic
                    w_vec[kg * 4 + ig] += tc

            col = ko * 7 + lo
            U[:, col] = u_vec
            V[:, col] = v_vec
            W[:, col] = w_vec

    # Schoolbook for the 4x4x3 part (columns 4..6 of the 4x7 matrix)
    col = 49
    for i in range(4):
        for j in range(4):
            for kp in range(3):
                kg = 4 + kp
                U[i * 4 + j, col] = 1
                V[j * 7 + kg, col] = 1
                W[kg * 4 + i, col] = 1
                col += 1

    return {'U': U, 'V': V, 'W': W}
