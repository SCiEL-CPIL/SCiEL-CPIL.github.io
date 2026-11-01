"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-49 tensor decomposition for 2x2 * 2x14 matrix multiplication
    using Strassen's algorithm on 7 blocks of 2x2.
    
    Tensor convention: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    We need T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,...,13}.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    n, m, p = DIMENSIONS  # n=2, m=2, p=14
    nm = n * m  # 4
    mp = m * p  # 28
    pn = p * n  # 28
    num_blocks = p // 2  # 7
    rank = num_blocks * 7  # 49

    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))

    # Strassen's 7 terms for 2x2 * 2x2:
    # A = [[a, b], [c, d]] where a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # B = [[e, f], [g, h]] where e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    # C = [[i, j], [k, l]] where i=C[0,0], j=C[0,1], k=C[1,0], l=C[1,1]
    #
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # i = M1 + M4 - M5 + M7
    # j = M3 + M5
    # k = M2 + M4
    # l = M1 - M2 + M3 + M6

    # For each block t (0..6), B block columns are 2t, 2t+1
    # U index: i*m + j = i*2 + j
    # V index: j*p + k = j*14 + k
    # W index: k*n + i = k*2 + i

    for t in range(num_blocks):
        base = t * 7
        # B block: B[0, 2t], B[0, 2t+1], B[1, 2t], B[1, 2t+1]
        # V indices:
        #   e = B[0, 2t]   -> V[0*14 + 2t]   = V[2t]
        #   f = B[0, 2t+1] -> V[0*14 + 2t+1] = V[2t+1]
        #   g = B[1, 2t]   -> V[1*14 + 2t]   = V[14+2t]
        #   h = B[1, 2t+1] -> V[1*14 + 2t+1] = V[14+2t+1]
        
        ve = 2 * t
        vf = 2 * t + 1
        vg = 14 + 2 * t
        vh = 14 + 2 * t + 1

        # C block: C[0, 2t], C[0, 2t+1], C[1, 2t], C[1, 2t+1]
        # W indices (k*n + i):
        #   C[0, 2t]   -> W[2t*2 + 0]   = W[4t]
        #   C[0, 2t+1] -> W[(2t+1)*2 + 0] = W[4t+2]
        #   C[1, 2t]   -> W[2t*2 + 1]   = W[4t+1]
        #   C[1, 2t+1] -> W[(2t+1)*2 + 1] = W[4t+3]
        
        w_i = 4 * t      # C[0, 2t]
        w_j = 4 * t + 2  # C[0, 2t+1]
        w_k = 4 * t + 1  # C[1, 2t]
        w_l = 4 * t + 3  # C[1, 2t+1]

        # U indices: a=0, b=1, c=2, d=3

        # M1 = (a+d)(e+h)
        U[0, base] = 1    # a
        U[3, base] = 1    # d
        V[ve, base] = 1   # e
        V[vh, base] = 1   # h
        W[w_i, base] = 1  # i += M1
        W[w_l, base] = 1  # l += M1

        # M2 = (c+d)e
        U[2, base+1] = 1  # c
        U[3, base+1] = 1  # d
        V[ve, base+1] = 1 # e
        W[w_k, base+1] = 1  # k += M2
        W[w_l, base+1] = -1 # l -= M2

        # M3 = a(f-h)
        U[0, base+2] = 1  # a
        V[vf, base+2] = 1  # f
        V[vh, base+2] = -1 # -h
        W[w_j, base+2] = 1  # j += M3
        W[w_l, base+2] = 1  # l += M3

        # M4 = d(g-e)
        U[3, base+3] = 1  # d
        V[vg, base+3] = 1  # g
        V[ve, base+3] = -1 # -e
        W[w_i, base+3] = 1  # i += M4
        W[w_k, base+3] = 1  # k += M4

        # M5 = (a+b)h
        U[0, base+4] = 1  # a
        U[1, base+4] = 1  # b
        V[vh, base+4] = 1 # h
        W[w_i, base+4] = -1 # i -= M5
        W[w_j, base+4] = 1  # j += M5

        # M6 = (c-a)(e+f)
        U[2, base+5] = 1  # c
        U[0, base+5] = -1 # -a
        V[ve, base+5] = 1 # e
        V[vf, base+5] = 1 # f
        W[w_l, base+5] = 1  # l += M6

        # M7 = (b-d)(g+h)
        U[1, base+6] = 1  # b
        U[3, base+6] = -1 # -d
        V[vg, base+6] = 1 # g
        V[vh, base+6] = 1 # h
        W[w_i, base+6] = 1  # i += M7

    # Verify: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    def verify(Uf, Vf, Wf):
        for i in range(n):
            for j in range(m):
                u_row = Uf[i * m + j]
                for k in range(p):
                    v_row = Vf[j * p + k]
                    w_row = Wf[k * n + i]
                    s = np.dot(u_row, v_row * w_row)
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not verify(U, V, W):
        # Fallback to schoolbook rank-56
        rank_sb = n * m * p  # 56
        U = np.zeros((nm, rank_sb))
        V = np.zeros((mp, rank_sb))
        W = np.zeros((pn, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1

    # Convert to lists with integer entries where possible
    def to_int_list(arr):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in arr.tolist()]

    U_list = to_int_list(U)
    V_list = to_int_list(V)
    W_list = to_int_list(W)

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
