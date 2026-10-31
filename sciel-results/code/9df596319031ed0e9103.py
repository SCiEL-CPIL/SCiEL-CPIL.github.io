import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin

    n, m, p = 4, 4, 4
    rank = 49

    # Strassen 2x2 decomposition coefficients
    # U2[row, s]: coefficient of A[row] in s-th Strassen product
    # V2[row, s]: coefficient of B[row] in s-th Strassen product
    # W2[row, s]: coefficient of s-th Strassen product in C[row]
    # Rows: 0=00, 1=01, 2=10, 3=11

    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)

    # M0 = (A00 + A11) * (B00 + B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1

    # M1 = (A10 + A11) * B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1

    # M2 = A00 * (B01 - B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1

    # M3 = A11 * (B10 - B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1

    # M4 = (A00 + A01) * B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1

    # M5 = (A10 - A00) * (B00 + B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1

    # M6 = (A01 - A11) * (B10 + B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # Build 4x4x4 decomposition via Kronecker product of Strassen
    # A[i,j]: block (I=i//2, J=j//2), inner (ii=i%2, jj=j%2)
    # B[k,l]: block (K=k//2, L=l//2), inner (kk=k%2, ll=l%2)
    # C[i,k]: block (I=i//2, J=k//2), inner (ii=i%2, jj=k%2)
    # W row index: k*4 + i (per task convention)

    U = np.zeros((16, 49), dtype=int)
    V = np.zeros((16, 49), dtype=int)
    W = np.zeros((16, 49), dtype=int)

    # Precompute index decompositions
    I_map = np.array([i // 2 for i in range(4)])
    J_map = np.array([j // 2 for j in range(4)])
    ii_map = np.array([i % 2 for i in range(4)])
    jj_map = np.array([j % 2 for j in range(4)])

    # Vectorized Kronecker product construction
    for s in range(7):
        for t in range(7):
            r = s * 7 + t

            # U[i*4+j, r] = U2[I*2+J, s] * U2[ii*2+jj, t]
            # Block part: U2[I*2+J, s] for I in 0,1 and J in 0,1
            block_U = U2[np.array([I*2+J for I in range(2) for J in range(2)]), s]
            inner_U = U2[np.array([ii*2+jj for ii in range(2) for jj in range(2)]), t]
            # Expand to 4x4: each block element applies to 2x2 inner
            U_block = np.zeros((4, 4))
            for I in range(2):
                for J in range(2):
                    U_block[I*2:(I+1)*2, J*2:(J+1)*2] = block_U[I*2+J]
            U_inner = np.zeros((4, 4))
            for ii in range(2):
                for jj in range(2):
                    U_inner[ii*2:(ii+1)*2, jj*2:(jj+1)*2] = inner_U[ii*2+jj]
            U[:, r] = (U_block * U_inner).flatten()

            # V[k*4+l, r] = V2[K*2+L, s] * V2[kk*2+ll, t]
            block_V = V2[np.array([K*2+L for K in range(2) for L in range(2)]), s]
            inner_V = V2[np.array([kk*2+ll for kk in range(2) for ll in range(2)]), t]
            V_block = np.zeros((4, 4))
            for K in range(2):
                for L in range(2):
                    V_block[K*2:(K+1)*2, L*2:(L+1)*2] = block_V[K*2+L]
            V_inner = np.zeros((4, 4))
            for kk in range(2):
                for ll in range(2):
                    V_inner[kk*2:(kk+1)*2, ll*2:(ll+1)*2] = inner_V[kk*2+ll]
            V[:, r] = (V_block * V_inner).flatten()

            # W[k*4+i, r] = W2[I*2+J, s] * W2[ii*2+jj, t]
            # where I=i//2 (C row block), J=k//2 (C col block = B row block)
            # ii=i%2, jj=k%2
            # W row index = k*4+i
            block_W = W2[np.array([I*2+J for I in range(2) for J in range(2)]), s]
            inner_W = W2[np.array([ii*2+jj for ii in range(2) for jj in range(2)]), t]
            # W[i, k] in 2x2 block terms: W_block[I, J] for C block at (I,J)
            # W row = k*4+i, so for given (i,k): I=i//2, J=k//2
            W_block = np.zeros((4, 4))  # indexed by [i, k]
            for I in range(2):
                for J in range(2):
                    W_block[I*2:(I+1)*2, J*2:(J+1)*2] = block_W[I*2+J]
            W_inner = np.zeros((4, 4))  # indexed by [ii, jj]
            for ii in range(2):
                for jj in range(2):
                    W_inner[ii*2:(ii+1)*2, jj*2:(jj+1)*2] = inner_W[ii*2+jj]
            # W[:, r] indexed by k*4+i, so we need to transpose: W[k, i]
            W[:, r] = (W_block * W_inner).T.flatten()

            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
