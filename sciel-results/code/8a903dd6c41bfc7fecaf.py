import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    # Laderman rank-23 decomposition for 3x3 matrix multiplication
    U3 = np.zeros((9, 23), dtype=int)
    V3 = np.zeros((9, 23), dtype=int)
    W3 = np.zeros((9, 23), dtype=int)
    
    # Each form: (u_coeffs, v_coeffs, w_coeffs)
    # u: {(i,j): val} for A[i,j], U index = i*3+j
    # v: {(j,k): val} for B[j,k], V index = j*3+k
    # w: {(k,i): val} for C[i,k], W index = k*3+i
    
    laderman = [
        ({(0,0):1}, {(0,0):1}, {(0,0):1}),
        ({(0,1):1}, {(1,0):1}, {(0,0):1}),
        ({(0,2):1}, {(2,0):1}, {(0,0):1}),
        ({(1,0):1}, {(0,1):1}, {(1,1):1}),
        ({(1,1):1}, {(1,1):1}, {(1,1):1}),
        ({(1,2):1}, {(2,1):1}, {(1,1):1}),
        ({(2,0):1}, {(0,2):1}, {(2,2):1}),
        ({(2,1):1}, {(1,2):1}, {(2,2):1}),
        ({(2,2):1}, {(2,2):1}, {(2,2):1}),
        ({(0,0):1,(1,1):1}, {(0,0):1}, {(0,0):-1,(1,1):1}),
        ({(0,1):1,(1,2):1}, {(1,0):1}, {(0,0):-1,(1,1):1}),
        ({(0,2):1,(1,0):1}, {(2,0):1}, {(0,0):-1,(1,1):1}),
        ({(0,0):1}, {(0,1):1,(1,2):1}, {(1,1):-1,(2,2):1}),
        ({(1,1):1}, {(1,2):1,(2,0):1}, {(1,1):-1,(2,2):1}),
        ({(2,2):1}, {(2,0):1,(0,1):1}, {(1,1):-1,(2,2):1}),
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):-1,(1,1):-1}),
        ({(0,1):1,(1,2):1}, {(1,0):1,(2,1):1}, {(0,0):-1,(1,1):-1}),
        ({(0,2):1,(1,0):1}, {(2,0):1,(0,1):1}, {(0,0):-1,(1,1):-1}),
        ({(0,0):1,(1,1):-1}, {(1,1):1,(0,0):-1}, {(0,0):1,(1,1):-1}),
        ({(0,1):1,(1,2):-1}, {(2,1):1,(1,0):-1}, {(0,0):1,(1,1):-1}),
        ({(0,2):1,(1,0):-1}, {(0,1):1,(2,0):-1}, {(0,0):1,(1,1):-1}),
        ({(1,1):1,(2,2):1}, {(1,1):1,(2,2):1}, {(1,1):-1,(2,2):-1}),
        ({(2,2):1,(1,1):-1}, {(2,2):1,(1,1):-1}, {(1,1):1,(2,2):-1}),
    ]
    
    for c, (uc, vc, wc) in enumerate(laderman):
        for (i,j), v in uc.items():
            U3[i*3+j, c] += v
        for (j,k), v in vc.items():
            V3[j*3+k, c] += v
        for (k,i), v in wc.items():
            W3[k*3+i, c] += v

    # Strassen rank-7 for 2x2
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    strassen = [
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}),
        ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}),
        ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
    ]
    
    for c, (uc, vc, wc) in enumerate(strassen):
        for (i,j), v in uc.items():
            U2[i*2+j, c] += v
        for (j,k), v in vc.items():
            V2[j*2+k, c] += v
        for (k,i), v in wc.items():
            W2[k*2+i, c] += v

    # Combine: 6x6 = 2x2 blocks of 3x3
    rank = 7 * 23
    U6 = np.zeros((36, rank), dtype=int)
    V6 = np.zeros((36, rank), dtype=int)
    W6 = np.zeros((36, rank), dtype=int)
    
    for s in range(7):
        for l in range(23):
            r = s * 23 + l
            for u2i in range(4):
                if U2[u2i, s]:
                    i2, j2 = divmod(u2i, 2)
                    vu2 = U2[u2i, s]
                    for u3i in range(9):
                        if U3[u3i, l]:
                            i3, j3 = divmod(u3i, 3)
                            vu3 = U3[u3i, l]
                            U6[(i2*3+i3)*6 + (j2*3+j3), r] += vu2 * vu3
            for v2i in range(4):
                if V2[v2i, s]:
                    j2, k2 = divmod(v2i, 2)
                    vv2 = V2[v2i, s]
                    for v3i in range(9):
                        if V3[v3i, l]:
                            j3, k3 = divmod(v3i, 3)
                            vv3 = V3[v3i, l]
                            V6[(j2*3+j3)*6 + (k2*3+k3), r] += vv2 * vv3
            for w2i in range(4):
                if W2[w2i, s]:
                    k2, i2 = divmod(w2i, 2)
                    vw2 = W2[w2i, s]
                    for w3i in range(9):
                        if W3[w3i, l]:
                            k3, i3 = divmod(w3i, 3)
                            vw3 = W3[w3i, l]
                            W6[(k2*3+k3)*6 + (i2*3+i3), r] += vw2 * vw3

    return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
# EVOLVE-BLOCK-END
