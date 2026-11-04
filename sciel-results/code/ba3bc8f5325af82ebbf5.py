import numpy as np
import time

def construct():
    n, m, p = 4, 4, 8
    start_time = time.time()
    deadline = start_time + 110.0

    # Strassen 2x2x2 decomposition (rank 7)
    # U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r]
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))

    # r=0: M1 = (A00+A11)(B00+B11)
    U2[0,0]=1; U2[3,0]=1; V2[0,0]=1; V2[3,0]=1; W2[0,0]=1; W2[3,0]=1
    # r=1: M2 = (A10+A11)B00
    U2[2,1]=1; U2[3,1]=1; V2[0,1]=1; W2[1,1]=1; W2[3,1]=-1
    # r=2: M3 = A00(B01-B11)
    U2[0,2]=1; V2[1,2]=1; V2[3,2]=-1; W2[2,2]=1; W2[3,2]=1
    # r=3: M4 = A11(B10-B00)
    U2[3,3]=1; V2[0,3]=-1; V2[2,3]=1; W2[0,3]=1; W2[1,3]=1
    # r=4: M5 = (A00+A01)B11
    U2[0,4]=1; U2[1,4]=1; V2[3,4]=1; W2[0,4]=-1; W2[2,4]=1
    # r=5: M6 = (A10-A00)(B00+B01)
    U2[0,5]=-1; U2[2,5]=1; V2[0,5]=1; V2[1,5]=1; W2[3,5]=1
    # r=6: M7 = (A01-A11)(B10+B11)
    U2[1,6]=1; U2[3,6]=-1; V2[2,6]=1; V2[3,6]=1; W2[0,6]=1

    # Build 4x4x4 decomposition (rank 49) via tensor product of Strassen
    # Use Kronecker product structure: U4 = kron(U2, U2) with appropriate reshaping
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))

    for ro in range(7):
        for ri in range(7):
            r = ro * 7 + ri
            # U4[i*4+j, r]: i=2*Io+Ii, j=2*Jo+Ji
            # U2[uo, ro] where uo = Io*2+Jo, U2[ui, ri] where ui = Ii*2+Ji
            # So U4[i*4+j, r] = U2[(Io*2+Jo), ro] * U2[(Ii*2+Ji), ri]
            # This is the Kronecker product U2[:,ro] ⊗ U2[:,ri] reshaped appropriately
            u_outer = U2[:, ro]  # shape (4,)
            u_inner = U2[:, ri]  # shape (4,)
            # u_outer[a] * u_inner[b] where a=Io*2+Jo, b=Ii*2+Ji
            # Result index: i*4+j where i=2*Io+Ii, j=2*Jo+Ji
            # So result[ (2*Io+Ii)*4 + (2*Jo+Ji) ] = u_outer[Io*2+Jo] * u_inner[Ii*2+Ji]
            kron_u = np.kron(u_outer, u_inner)  # shape (16,)
            # kron_u[a*4+b] = u_outer[a] * u_inner[b] where a=Io*2+Jo, b=Ii*2+Ji
            # But we need index (2*Io+Ii)*4 + (2*Jo+Ji)
            # Let's create a mapping
            for a in range(4):
                Io = a // 2; Jo = a % 2
                for b in range(4):
                    Ii = b // 2; Ji = b % 2
                    src_idx = a * 4 + b
                    dst_idx = (2*Io + Ii) * 4 + (2*Jo + Ji)
                    U4[dst_idx, r] = kron_u[src_idx]

            # V4[j*4+k, r]: j=2*Jo+Ji, k=2*Ko+Ki
            v_outer = V2[:, ro]
            v_inner = V2[:, ri]
            kron_v = np.kron(v_outer, v_inner)
            for a in range(4):
                Jo = a // 2; Ko = a % 2
                for b in range(4):
                    Ji = b // 2; Ki = b % 2
                    src_idx = a * 4 + b
                    dst_idx = (2*Jo + Ji) * 4 + (2*Ko + Ki)
                    V4[dst_idx, r] = kron_v[src_idx]

            # W4[k*4+i, r]: k=2*Ko+Ki, i=2*Io+Ii
            w_outer = W2[:, ro]
            w_inner = W2[:, ri]
            kron_w = np.kron(w_outer, w_inner)
            for a in range(4):
                Ko = a // 2; Io = a % 2
                for b in range(4):
                    Ki = b // 2; Ii = b % 2
                    src_idx = a * 4 + b
                    dst_idx = (2*Ko + Ki) * 4 + (2*Io + Ii)
                    W4[dst_idx, r] = kron_w[src_idx]

    # Build 4x4x8 decomposition (rank 98)
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))

    # Left 4x4 block (output columns 0-3)
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+k, 0:49] = V4[j*4+k, :]
            W[k*4+j, 0:49] = W4[k*4+j, :]

    # Right 4x4 block (output columns 4-7)
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+(4+k), 49:98] = V4[j*4+k, :]
            W[(4+k)*4+j, 49:98] = W4[k*4+j, :]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
