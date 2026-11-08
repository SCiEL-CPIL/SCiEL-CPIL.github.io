import numpy as np
import time

def construct():
    t0 = time.time()
    limit = t0 + 115.0

    n, m, p = 4, 4, 8
    rank = 98

    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    # Strassen 2x2x2 factor matrices
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))

    # P0 = (A00+A11)(B00+B11)
    U2[0,0]=1; U2[3,0]=1; V2[0,0]=1; V2[3,0]=1; W2[0,0]=1; W2[3,0]=1
    # P1 = (A10+A11)B00
    U2[2,1]=1; U2[3,1]=1; V2[0,1]=1; W2[2,1]=1; W2[3,1]=-1
    # P2 = A00(B01-B11)
    U2[0,2]=1; V2[1,2]=1; V2[3,2]=-1; W2[1,2]=1; W2[3,2]=1
    # P3 = A11(B10-B00)
    U2[3,3]=1; V2[2,3]=1; V2[0,3]=-1; W2[0,3]=1; W2[2,3]=1
    # P4 = (A00+A01)B11
    U2[0,4]=1; U2[1,4]=1; V2[3,4]=1; W2[0,4]=-1; W2[1,4]=1
    # P5 = (A10-A00)(B00+B01)
    U2[2,5]=1; U2[0,5]=-1; V2[0,5]=1; V2[1,5]=1; W2[3,5]=1
    # P6 = (A01-A11)(B10+B11)
    U2[1,6]=1; U2[3,6]=-1; V2[2,6]=1; V2[3,6]=1; W2[0,6]=1

    # Block specs for the 7 super-products in 4x4x4 Strassen
    # Block indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    A_spec = [
        [(1,0),(1,3)],
        [(1,2),(1,3)],
        [(1,0)],
        [(1,3)],
        [(1,0),(1,1)],
        [(1,2),(-1,0)],
        [(1,1),(-1,3)],
    ]
    B_spec = [
        [(1,0),(1,3)],
        [(1,0)],
        [(1,1),(-1,3)],
        [(1,2),(-1,0)],
        [(1,3)],
        [(1,0),(1,1)],
        [(1,2),(1,3)],
    ]
    C_spec = [
        [(1,0),(1,3)],
        [(1,2),(-1,3)],
        [(1,1),(1,3)],
        [(1,0),(1,2)],
        [(-1,0),(1,1)],
        [(1,3)],
        [(1,0)],
    ]

    # Precompute block offsets: block bi -> (br, bc)
    block_offsets = []
    for bi in range(4):
        br, bc = divmod(bi, 2)
        block_offsets.append((br, bc))

    for half in range(2):
        base_col = half * 49
        for kp in range(7):
            for ms in range(7):
                col = base_col + kp * 7 + ms
                if time.time() > limit:
                    break

                # --- U: Input A (4x4), index = row*4 + col ---
                for ca, bidx in A_spec[kp]:
                    br, bc = block_offsets[bidx]
                    for r in range(2):
                        for c in range(2):
                            a_eff_idx = r * 2 + c
                            coeff = U2[a_eff_idx, ms]
                            if coeff != 0:
                                U[(br*2+r)*4 + (bc*2+c), col] += ca * coeff

                # --- V: Input B (4x8), index = row*8 + col ---
                for cb, bidx in B_spec[kp]:
                    br, bc = block_offsets[bidx]
                    for r in range(2):
                        for c in range(2):
                            b_eff_idx = r * 2 + c
                            coeff = V2[b_eff_idx, ms]
                            if coeff != 0:
                                V[(br*2+r)*8 + (bc*2+c + half*4), col] += cb * coeff

                # --- W: Output C (8x4), convention W[k*n+i, r] where k=col, i=row ---
                for cc, cidx in C_spec[kp]:
                    bi, bj = divmod(cidx, 2)
                    for r in range(2):
                        for c in range(2):
                            c_eff_idx = r * 2 + c
                            coeff = W2[c_eff_idx, ms]
                            if coeff != 0:
                                C_row = bi*2 + r
                                C_col = bj*2 + c + half*4
                                W[C_col*4 + C_row, col] += cc * coeff

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
