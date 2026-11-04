import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = DIMENSIONS
    
    m_blocks = [2, 2, 1]
    p_blocks = [2, 2, 2]
    
    m_offsets = [0]
    for s in m_blocks[:-1]:
        m_offsets.append(m_offsets[-1] + s)
    p_offsets = [0]
    for s in p_blocks[:-1]:
        p_offsets.append(p_offsets[-1] + s)
    
    terms = []
    
    for mb_idx, ms in enumerate(m_blocks):
        for pb_idx, ps in enumerate(p_blocks):
            mo = m_offsets[mb_idx]
            po = p_offsets[pb_idx]
            
            if ms == 2 and ps == 2:
                # Standard Strassen for 2x2 x 2x2
                # A = [[a,b],[c,d]] = [[A[0,mo],A[0,mo+1]],[A[1,mo],A[1,mo+1]]]
                # B = [[e,f],[g,h]] = [[B[mo,po],B[mo,po+1]],[B[mo+1,po],B[mo+1,po+1]]]
                # C[0,po] = p1-p4+p5+p6, C[0,po+1] = p3+p4
                # C[1,po] = p2+p5, C[1,po+1] = p1-p2+p3+p7
                
                # p1 = (a+d)(e+h) = (A[0,mo]+A[1,mo+1])(B[mo,po]+B[mo+1,po+1])
                terms.append({
                    'U': {0*2+mo: 1, 1*2+mo+1: 1},
                    'V': {mo*6+po: 1, (mo+1)*6+po+1: 1},
                    'W': {po*2+0: 1, (po+1)*2+1: 1}
                })
                # p2 = (c+d)e = (A[1,mo]+A[1,mo+1])B[mo,po]
                terms.append({
                    'U': {1*2+mo: 1, 1*2+mo+1: 1},
                    'V': {mo*6+po: 1},
                    'W': {po*2+1: 1, (po+1)*2+1: 1}
                })
                # p3 = a(f-e) = A[0,mo](B[mo,po+1]-B[mo,po])
                terms.append({
                    'U': {0*2+mo: 1},
                    'V': {mo*6+po+1: 1, mo*6+po: -1},
                    'W': {(po+1)*2+0: 1, po*2+1: 1}
                })
                # p4 = d(h-g) = A[1,mo+1](B[mo+1,po+1]-B[mo+1,po])
                terms.append({
                    'U': {1*2+mo+1: 1},
                    'V': {(mo+1)*6+po+1: 1, (mo+1)*6+po: -1},
                    'W': {(po+1)*2+0: 1, (po+1)*2+1: 1}
                })
                # p5 = (a+b)h = (A[0,mo]+A[0,mo+1])B[mo+1,po+1]
                terms.append({
                    'U': {0*2+mo: 1, 0*2+mo+1: 1},
                    'V': {(mo+1)*6+po+1: 1},
                    'W': {po*2+0: 1, (po+1)*2+1: 1}
                })
                # p6 = (c-a)(e+g) = (A[1,mo]-A[0,mo])(B[mo,po]+B[mo+1,po])
                terms.append({
                    'U': {1*2+mo: 1, 0*2+mo: -1},
                    'V': {mo*6+po: 1, (mo+1)*6+po: 1},
                    'W': {po*2+0: 1, po*2+1: 1}
                })
                # p7 = (b-d)(f+h) = (A[0,mo+1]-A[1,mo+1])(B[mo,po+1]+B[mo+1,po+1])
                terms.append({
                    'U': {0*2+mo+1: 1, 1*2+mo+1: -1},
                    'V': {mo*6+po+1: 1, (mo+1)*6+po+1: 1},
                    'W': {(po+1)*2+0: 1, (po+1)*2+1: 1}
                })
            else:
                for i in range(n):
                    for j in range(ms):
                        for k in range(ps):
                            terms.append({
                                'U': {i*2 + mo + j: 1},
                                'V': {(mo + j)*6 + po + k: 1},
                                'W': {(po + k)*2 + i: 1}
                            })
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, term in enumerate(terms):
        for row, val in term['U'].items():
            U[row][c] = val
        for row, val in term['V'].items():
            V[row][c] = val
        for row, val in term['W'].items():
            W[row][c] = val
    
    # Verify
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                computed = sum(U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c] for c in range(rank))
                if abs(computed - 1.0) > 1e-9:
                    # Fallback to schoolbook
                    rank2 = n * m * p
                    U2 = [[0] * rank2 for _ in range(n * m)]
                    V2 = [[0] * rank2 for _ in range(m * p)]
                    W2 = [[0] * rank2 for _ in range(p * n)]
                    col2 = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U2[ii*m+jj][col2] = 1
                                V2[jj*p+kk][col2] = 1
                                W2[kk*n+ii][col2] = 1
                                col2 += 1
                    return {"U": U2, "V": V2, "W": W2}
    
    return {"U": U, "V": V, "W": W}
