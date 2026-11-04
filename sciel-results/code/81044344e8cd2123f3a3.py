"""Tensor decomposition for 2x3 * 3x10 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = 2, 3, 10

    # Build rank-55 decomposition:
    # Split: C = A_left(2x2) * B_top(2x10) + A_right(2x1) * B_bottom(1x10)
    # A_left uses cols 0,1 of A; A_right uses col 2 of A
    # B_top uses rows 0,1 of B; B_bottom uses row 2 of B
    #
    # For A_left * B_top: group B_top into 5 blocks of 2 cols, Strassen each: 5*7=35
    # For A_right * B_bottom: schoolbook: 2*10=20
    # Total: 55

    rank = 55
    U = [[0] * rank for _ in range(n * m)]   # 6 x 55
    V = [[0] * rank for _ in range(m * p)]   # 30 x 55
    W = [[0] * rank for _ in range(p * n)]   # 20 x 55

    r = 0

    # Standard Strassen for 2x2 x 2x2:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # m1=(a+d)(e+h), m2=(c+d)e, m3=a(f-h), m4=d(g-e), m5=(a+b)h, m6=c(f-h), m7=(b-d)(g+h)
    # C00=m1+m4-m5-m7, C01=m3+m5, C10=m2+m7, C11=m1-m2+m3+m4

    # U factors (A side): indices in A_left are (0,1) -> row0col0, row0col1, row1col0, row1col1
    # U index: i*m + j where i=row in C (0,1), j=col in A (0,1) for left part
    # So U[0]=a, U[1]=b, U[2]=c, U[3]=d
    strassen_u = [
        [(0, 1), (3, 1)],       # m1: a+d
        [(2, 1), (3, 1)],       # m2: c+d
        [(0, 1)],               # m3: a
        [(3, 1)],               # m4: d
        [(0, 1), (1, 1)],       # m5: a+b
        [(2, 1)],               # m6: c
        [(1, 1), (3, -1)],      # m7: b-d
    ]

    # V factors (B side): B_top is 2x10, rows 0,1 of B.
    # For block b, columns col0, col1 of B_top correspond to B rows 0,1 cols col0,col1
    # V index: j*p + k where j=row in B (0,1 for top), k=col in B
    # V[0*10+col0]=e, V[1*10+col0]=g, V[0*10+col1]=f, V[1*10+col1]=h
    # Wait: B is 3x10. B_top rows are 0,1. So B[0,col0]=e, B[0,col1]=f, B[1,col0]=g, B[1,col1]=h
    # V index = j*p + k = j*10 + k
    # e = B[0,col0] -> V[0*10+col0] = V[col0]
    # f = B[0,col1] -> V[0*10+col1] = V[col1]
    # g = B[1,col0] -> V[1*10+col0] = V[10+col0]
    # h = B[1,col1] -> V[1*10+col1] = V[10+col1]

    strassen_v = [
        [(0, 1), (3, 1)],       # m1: e+h -> V[col0] + V[10+col1]
        [(0, 1)],               # m2: e -> V[col0]
        [(1, 1), (3, -1)],      # m3: f-h -> V[col1] - V[10+col1]
        [(2, 1), (0, -1)],      # m4: g-e -> V[10+col0] - V[col0]
        [(3, 1)],               # m5: h -> V[10+col1]
        [(1, 1), (3, -1)],      # m6: f-h -> V[col1] - V[10+col1]
        [(2, 1), (3, 1)],       # m7: g+h -> V[10+col0] + V[10+col1]
    ]

    # W factors (C side): C is 2x10. W index = k*n + i = k*2 + i
    # For block b, output columns are col0, col1
    # C[0,col0] -> W[col0*2+0] = W[2*col0]
    # C[0,col1] -> W[col1*2+0] = W[2*col1]
    # C[1,col0] -> W[col0*2+1] = W[2*col0+1]
    # C[1,col1] -> W[col1*2+1] = W[2*col1+1]

    # C00 = m1+m4-m5-m7
    # C01 = m3+m5
    # C10 = m2+m7
    # C11 = m1-m2+m3+m4

    strassen_w = [
        [(0, 1), (3, 1)],       # m1: C00 + C11
        [(2, 1), (3, -1)],      # m2: C10 - C11
        [(1, 1), (3, 1)],       # m3: C01 + C11
        [(0, 1), (3, 1)],       # m4: C00 + C11
        [(0, -1), (1, 1)],      # m5: -C00 + C01
        [(1, 1), (3, -1)],      # m6: C01 - C11  (wait, m6 only appears in C01)
        [(2, 1), (0, -1)],      # m7: C10 - C00
    ]

    # Let me re-derive W more carefully:
    # C00 = m1 + m4 - m5 - m7
    # C01 = m3 + m5
    # C10 = m2 + m7
    # C11 = m1 - m2 + m3 + m4

    # W for each m_s:
    # m1: C00+=1, C11+=1  -> W[2*col0]+=1, W[2*col1+1]+=1
    # m2: C10+=1, C11-=1  -> W[2*col0+1]+=1, W[2*col1+1]-=1
    # m3: C01+=1, C11+=1  -> W[2*col1]+=1, W[2*col1+1]+=1
    # m4: C00+=1, C11+=1  -> W[2*col0]+=1, W[2*col1+1]+=1
    # m5: C00-=1, C01+=1  -> W[2*col0]-=1, W[2*col1]+=1
    # m6: C01+=1           -> W[2*col1]+=1
    # m7: C00-=1, C10+=1  -> W[2*col0]-=1, W[2*col0+1]+=1

    strassen_w = [
        [(0, 1), (3, 1)],       # m1: W[2*col0]+=1, W[2*col1+1]+=1
        [(1, 1), (3, -1)],      # m2: W[2*col0+1]+=1, W[2*col1+1]-=1
        [(2, 1), (3, 1)],       # m3: W[2*col1]+=1, W[2*col1+1]+=1
        [(0, 1), (3, 1)],       # m4: W[2*col0]+=1, W[2*col1+1]+=1
        [(0, -1), (2, 1)],      # m5: W[2*col0]-=1, W[2*col1]+=1
        [(2, 1)],               # m6: W[2*col1]+=1
        [(0, -1), (1, 1)],      # m7: W[2*col0]-=1, W[2*col0+1]+=1
    ]

    # Wait, I need to map w_off to actual W indices.
    # w_off 0 -> W[2*col0]   (C[0,col0])
    # w_off 1 -> W[2*col0+1] (C[1,col0])
    # w_off 2 -> W[2*col1]   (C[0,col1])
    # w_off 3 -> W[2*col1+1] (C[1,col1])

    for b in range(5):
        col0 = 2 * b
        col1 = 2 * b + 1

        for s in range(7):
            # U
            for (u_idx, coeff) in strassen_u[s]:
                U[u_idx][r] += coeff

            # V
            for (v_off, coeff) in strassen_v[s]:
                if v_off == 0:
                    v_row = col0
                elif v_off == 1:
                    v_row = col1
                elif v_off == 2:
                    v_row = 10 + col0
                elif v_off == 3:
                    v_row = 10 + col1
                else:
                    raise ValueError(f"Bad v_off {v_off}")
                V[v_row][r] += coeff

            # W
            for (w_off, coeff) in strassen_w[s]:
                if w_off == 0:
                    w_row = 2 * col0
                elif w_off == 1:
                    w_row = 2 * col0 + 1
                elif w_off == 2:
                    w_row = 2 * col1
                elif w_off == 3:
                    w_row = 2 * col1 + 1
                else:
                    raise ValueError(f"Bad w_off {w_off}")
                W[w_row][r] += coeff

            r += 1

    # Schoolbook for A_right (2x1) * B_bottom (1x10): 2*10 = 20 terms
    # A_right: col 2 of A. U index = i*3 + 2 for i=0,1
    # B_bottom: row 2 of B. V index = 2*10 + k = 20 + k for k=0..9
    # C: W index = k*2 + i
    for i in range(2):
        for k in range(10):
            U[i * 3 + 2][r] = 1
            V[20 + k][r] = 1
            W[k * 2 + i][r] = 1
            r += 1

    # Validate
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = np.dot(U_arr[i * m + j] * V_arr[j * p + k], W_arr[k * n + i])
                if not np.isclose(lhs, 1.0, atol=1e-9):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        # Fallback to trivial rank-60
        rank60 = 60
        U2 = [[0] * rank60 for _ in range(n * m)]
        V2 = [[0] * rank60 for _ in range(m * p)]
        W2 = [[0] * rank60 for _ in range(p * n)]
        for rr in range(rank60):
            ii = rr // (m * p)
            rem = rr % (m * p)
            jj = rem // p
            kk = rem % p
            U2[ii * m + jj][rr] = 1
            V2[jj * p + kk][rr] = 1
            W2[kk * n + ii][rr] = 1
        return {"U": U2, "V": V2, "W": W2}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
