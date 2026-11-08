import numpy as np
import time

def construct():
    # The task is to find a rank-minimal exact tensor decomposition for 
    # multiplying a 4x4 matrix by a 4x8 matrix.
    # The best known rank is 98, achieved by splitting the 4x8 multiplication
    # into two 4x4 multiplications and applying Strassen's algorithm (rank 49) to each.
    
    n, m, p = 4, 4, 8
    rank = 98

    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    # Strassen 2x2x2 factor matrices
    # U2, V2, W2 are 4x7 matrices
    U2 = np.zeros((4, 7), dtype=np.float64)
    V2 = np.zeros((4, 7), dtype=np.float64)
    W2 = np.zeros((4, 7), dtype=np.float64)

    # Define the 7 Strassen products for 2x2x2
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
    # A_spec[kp]: list of (coeff, block_idx) for A_eff
    # B_spec[kp]: list of (coeff, block_idx) for B_eff
    # C_spec[kp]: list of (coeff, block_idx) for C_eff
    
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
    # bi=0 -> (0,0), bi=1 -> (0,1), bi=2 -> (1,0), bi=3 -> (1,1)
    block_offsets = [(0,0), (0,1), (1,0), (1,1)]

    # Construct the rank 98 decomposition
    # Split B (4x8) into B_L (4x4) and B_R (4x4)
    # Split C (4x8) into C_L (4x4) and C_R (4x4)
    # A is 4x4
    
    for half in range(2):
        # half=0: Left half (cols 0-3 of B and C)
        # half=1: Right half (cols 4-7 of B and C)
        base_col = half * 49
        b_col_offset = half * 4
        c_col_offset = half * 4

        for kp in range(7):
            for ms in range(7):
                col = base_col + kp * 7 + ms

                # --- U: Input A (4x4), index = row*4 + col ---
                for ca, bidx in A_spec[kp]:
                    br, bc = block_offsets[bidx]
                    for r in range(2):
                        for c in range(2):
                            a_eff_idx = r * 2 + c
                            coeff = U2[a_eff_idx, ms]
                            if coeff != 0:
                                # Global A index
                                a_row = br * 2 + r
                                a_col = bc * 2 + c
                                U[a_row * 4 + a_col, col] += ca * coeff

                # --- V: Input B (4x8), index = row*8 + col ---
                for cb, bidx in B_spec[kp]:
                    br, bc = block_offsets[bidx]
                    for r in range(2):
                        for c in range(2):
                            b_eff_idx = r * 2 + c
                            coeff = V2[b_eff_idx, ms]
                            if coeff != 0:
                                # Global B index
                                b_row = br * 2 + r
                                b_col = bc * 2 + c + b_col_offset
                                V[b_row * 8 + b_col, col] += cb * coeff

                # --- W: Output C (4x8), convention W[k*n+i, r] where k=col, i=row ---
                # W shape is (p*n, rank) = (32, 98)
                # Index is k*4 + i
                for cc, cidx in C_spec[kp]:
                    bi, bj = divmod(cidx, 2)
                    # cidx 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                    # bi is row block, bj is col block
                    for r in range(2):
                        for c in range(2):
                            c_eff_idx = r * 2 + c
                            coeff = W2[c_eff_idx, ms]
                            if coeff != 0:
                                # Global C index
                                c_row = bi * 2 + r
                                c_col = bj * 2 + c + c_col_offset
                                # W index: c_col * 4 + c_row
                                W[c_col * 4 + c_row, col] += cc * coeff

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
