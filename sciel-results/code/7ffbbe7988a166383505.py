import itertools, time
import numpy as np

N, M, P = 2, 3, 4
NM, MP, PN = 6, 12, 8
NP = 8  # equations per ... no, 24 equations total

def build_system(V, W, R):
    # unknowns U[i,j,r]: index = (i*M + j)*R + r
    # equations: (i,j,k) -> sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
    eqs = []
    for i in range(N):
        for j in range(M):
            for k in range(P):
                row = np.zeros(NM * R)
                for r in range(R):
                    row[(i*M + j)*R + r] = V[j*P + k, r] * W[k*N + i, r]
                eqs.append(row)
    A = np.array(eqs)
    b = np.ones(N*M*P)
    return A, b
