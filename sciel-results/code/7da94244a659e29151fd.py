import numpy as np
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    hard_deadline = t0 + 140.0
    n, m, p = DIMENSIONS

    # Build Strassen 2x2x2 factors (rank 7)
    U2 = np.zeros((2, 2, 7))
    V2 = np.zeros((2, 2, 7))
    W2 = np.zeros((2, 2, 7))
    U2[0,0,0]=1; U2[1,1,0]=1
    V2[0,0,0]=1; V2[1,1,0]=1
    W2[0,0,0]=1; W2[1,1,0]=1
    U2[1,0,1]=1; U2[1,1,1]=1
    V2[0,0,1]=1
    W2[0,1,1]=1; W2[1,1,1]=-1
    U2[0,0,2]=1
    V2[0,1,2]=1; V2[1,1,2]=-1
    W2[1,0,2]=1; W2[1,1,2]=1
    U2[1,1,3]=1
    V2[1,0,3]=1; V2[0,0,3]=-1
    W2[0,0,3]=1; W2[0,1,3]=1
    U2[0,0,4]=1; U2[0,1,4]=1
    V2[1,1,4]=1
    W2[0,0,4]=-1; W2[1,0,4]=1
    U2[1,0,5]=1; U2[0,0,5]=-1
    V2[0,0,5]=1; V2[0,1,5]=1
    W2[1,1,5]=1
    U2[0,1,6]=1; U2[1,1,6]=-1
    V2[1,0,6]=1; V2[1,1,6]=1
    W2[0,0,6]=1

    # Build 4x4x4 factors via tensor product of Strassen (rank 49)
    U4 = np.zeros((4, 4, 49))
    V4 = np.zeros((4, 4, 49))
    W4 = np.zeros((4, 4, 49))
    for i1 in range(2):
        for j1 in range(2):
            for i0 in range(2):
                for j0 in range(2):
                    i = 2*i1 + i0
                    j = 2*j1 + j0
                    for s in range(7):
                        for r in range(7):
                            R = 7*s + r
                            U4[i, j, R] = U2[i1, j1, s] * U2[i0, j0, r]
    for j1 in range(2):
        for k1 in range(2):
            for j0 in range(2):
                for k0 in range(2):
                    j = 2*j1 + j0
                    k = 2*k1 + k0
                    for s in range(7):
                        for r in range(7):
                            R = 7*s + r
                            V4[j, k, R] = V2[j1, k1, s] * V2[j0, k0, r]
    for k1 in range(2):
        for i1 in range(2):
            for k0 in range(2):
                for i0 in range(2):
                    k = 2*k1 + k0
                    i = 2*i1 + i0
                    for s in range(7):
                        for r in range(7):
                            R = 7*s + r
                            W4[k, i, R] = W2[k1, i1, s] * W2[k0, i0, r]

    # Build 4x4x8 factors using a shared-rank approach:
    # Use the same 49 rank-1 terms for both k-blocks, but with
    # extended V and W to cover all 8 k-values.
    # This gives rank 49 if the tensor allows it, else 98.
    # We'll try rank 98 first (known correct), then attempt reduction.
    
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))

    # Block 1: k in 0..3
    for i in range(4):
        for j in range(4):
            U[i*4 + j, :49] = U4[i, j, :]
    for j in range(4):
        for k in range(4):
            V[j*8 + k, :49] = V4[j, k, :]
    for k in range(4):
        for i in range(4):
            W[k*4 + i, :49] = W4[k, i, :]

    # Block 2: k in 4..7
    for i in range(4):
        for j in range(4):
            U[i*4 + j, 49:] = U4[i, j, :]
    for j in range(4):
        for kp in range(4):
            k = 4 + kp
            V[j*8 + k, 49:] = V4[j, kp, :]
    for kp in range(4):
        k = 4 + kp
        for i in range(4):
            W[k*4 + i, 49:] = W4[kp, i, :]

    # Verify correctness via tensor contraction
    def verify_solution(Uf, Vf, Wf):
        """Check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k"""
        Uf = np.asarray(Uf)
        Vf = np.asarray(Vf)
        Wf = np.asarray(Wf)
        n, m, p = 4, 4, 8
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = Uf[i*m + j, :]
                    v = Vf[j*p + k, :]
                    w = Wf[k*n + i, :]
                    val = np.sum(u * v * w)
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    if not verify_solution(U, V, W):
        # Fallback: return schoolbook rank-128
        rank_sb = 128
        U_sb = np.zeros((16, rank_sb))
        V_sb = np.zeros((32, rank_sb))
        W_sb = np.zeros((32, rank_sb))
        for i in range(4):
            for j in range(4):
                for k in range(8):
                    r = i * 32 + j * 8 + k
                    U_sb[i*4 + j, r] = 1
                    V_sb[j*8 + k, r] = 1
                    W_sb[k*4 + i, r] = 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
