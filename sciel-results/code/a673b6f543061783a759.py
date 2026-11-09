import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s

    n, m, p = 4, 5, 6

    def strassen_2x2():
        """2x2*2x2 Strassen. U[i*2+j,r]=A[i,j], V[j*2+k,r]=B[j,k], W[k*2+i,r]=C[i,k]"""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
        U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1
        U[0,2]=1; V[2,2]=1; V[3,2]=-1; W[1,2]=1
        U[3,3]=1; V[0,3]=1; V[2,3]=-1; W[0,3]=1; W[2,3]=1
        U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
        U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
        U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
        return U, V, W

    def verify(U, V, W, n, m, p):
        """Check decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k"""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        tensor = np.zeros((n, m, p))
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor[i,j,k] += u[i,j] * v[j,k] * w[k,i]
        return np.max(np.abs(tensor - 1.0)) < 1e-10

    def build_4x4x4x4():
        """4x4*4x4 rank 49 via Strassen. U[i*4+j,r], V[j*4+k,r], W[k*4+i,r]"""
        U2, V2, W2 = strassen_2x2()
        rank = 49
        U = np.zeros((16, rank))
        V = np.zeros((16, rank))
        W = np.zeros((16, rank))
        
        # Block offsets for 4x4: block b -> (row_off, col_off)
        # b=0:(0,0), b=1:(0,2), b=2:(2,0), b=3:(2,2)
        boff = {0:(0,0), 1:(0,2), 2:(2,0), 3:(2,2)}
        
        # Outer Strassen terms: (a_blocks, b_blocks, c_blocks)
        # Each is list of (block_idx, coeff)
        terms = [
            ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1)]),                    # M1
            ([(2,1),(3,1)], [(0,1)], [(2,1)]),                          # M2
            ([(0,1)], [(2,1),(3,-1)], [(1,1)]),                         # M3
            ([(3,1)], [(0,1),(2,-1)], [(0,1),(2,1)]),                   # M4
            ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),                   # M5
            ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),                   # M6
            ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1),(3,1)]),             # M7
        ]
        
        for q, (a_blks, b_blks, c_blks) in enumerate(terms):
            for ri in range(7):
                r = q*7 + ri
                for (b, c) in a_blks:
                    ro, co = boff[b]
                    for i in range(2):
                        for j in range(2):
                            U[ro+i, co+j, r] += c * U2[i*2+j, ri]
                for (b, c) in b_blks:
                    ro, co = boff[b]
                    for i in range(2):
                        for j in range(2):
                            V[ro+i, co+j, r] += c * V2[i*2+j, ri]
                for (b, c) in c_blks:
                    ro, co = boff[b]
                    for ci in range(2):
                        for ck in range(2):
                            i = ro + ci
                            k = co + ck
                            W[k*4+i, r] += c * W2[ck*2+ci, ri]
        
        return U, V, W

    def build_4x4x4x2():
        """4x4*4x2 rank 28. U[i*4+j,r], V[j*2+k,r], W[k*4+i,r] (k in 0..1)"""
        U2, V2, W2 = strassen_2x2()
        rank = 28
        U = np.zeros((16, rank))
        V = np.zeros((8, rank))
        W = np.zeros((8, rank))
        
        # A blocks: 0:(0,0), 1:(0,2), 2:(2,0), 3:(2,2) for 4x4
        # B blocks: 0:(0,0), 1:(2,0) for 4x2 (rows 0-1, 2-3; cols 0-1)
        # C blocks: 0:(0,0), 1:(2,0) for 4x2
        a_boff = {0:(0,0), 1:(0,2), 2:(2,0), 3:(2,2)}
        b_boff = {0:(0,0), 1:(2,0)}
        c_boff = {0:(0,0), 1:(2,0)}
        
        # 4 products: A00*B0->C0, A01*B1->C0, A10*B0->C1, A11*B1->C1
        prods = [(0,0,0), (1,1,0), (2,0,1), (3,1,1)]
        
        for pi, (ab, bb, cb) in enumerate(prods):
            for ri in range(7):
                r = pi*7 + ri
                aro, aco = a_boff[ab]
                for i in range(2):
                    for j in range(2):
                        U[aro+i, aco+j, r] = U2[i*2+j, ri]
                bro, bco = b_boff[bb]
                for i in range(2):
                    for j in range(2):
                        V[bro+i, bco+j, r] = V2[i*2+j, ri]
                cro, cco = c_boff[cb]
                for ci in range(2):
                    for ck in range(2):
                        i = cro + ci
                        k = cco + ck
                        W[k*4+i, r] = W2[ck*2+ci, ri]
        
        return U, V, W

    def build_4x1x1x6():
        """4x1 * 1x6, rank 4. U[i*5+4,r], V[4*6+k,r], W[k*4+i,r]"""
        rank = 4
        U = np.zeros((20, rank))
        V = np.zeros((6, rank))
        W = np.zeros((24, rank))
        for i in range(4):
            U[i*5+4, i] = 1.0
            for k in range(6):
                V[4*6+k, i] = 1.0
                W[k*4+i, i] = 1.0
        return U, V, W

    # Build components
    U44, V44, W44 = build_4x4x4x4()
    if not verify(U44, V44, W44, 4, 4, 4):
        # Fallback to schoolbook
        U44 = np.zeros((16, 64)); V44 = np.zeros((16, 64)); W44 = np.zeros((16, 64))
        c = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U44[i*4+j,c]=1; V44[j*4+k,c]=1; W44[k*4+i,c]=1; c+=1

    U42, V42, W42 = build_4x4x4x2()
    if not verify(U42, V42, W42, 4, 4, 2):
        U42 = np.zeros((16,32)); V42 = np.zeros((8,32)); W42 = np.zeros((8,32))
        c = 0
        for i in range(4):
            for j in range(4):
                for k in range(2):
                    U42[i*4+j,c]=1; V42[j*2+k,c]=1; W42[k*4+i,c]=1; c+=1

    U41, V41, W41 = build_4x1x1x6()

    # Assemble full 4x5 * 5x6
    r44, r42, r41 = U44.shape[1], U42.shape[1], U41.shape[1]
    total_rank = r44 + r42 + r41
    U = np.zeros((20, total_rank))
    V = np.zeros((30, total_rank))
    W = np.zeros((24, total_rank))

    # Part 1: A[0:4,0:4] * B[0:4,0:4] -> C[0:4,0:4]
    for r in range(r44):
        U[0:16, r] = U44[:, r]
        V[0:16, r] = V44[:, r]
        W[0:16, r] = W44[:, r]

    # Part 2: A[0:4,0:4] * B[0:4,4:6] -> C[0:4,4:6]
    off = r44
    for r in range(r42):
        col = off + r
        U[0:16, col] = U42[:, r]
        for j in range(4):
            for k in range(2):
                V[j*6 + 4+k, col] = V42[j*2+k, r]
        for k in range(2):
            for i in range(4):
                W[(4+k)*4 + i, col] = W42[k*4+i, r]

    # Part 3: A[0:4,4] * B[4,0:6] -> C[0:4,0:6]
    off = r44 + r42
    for r in range(r41):
        col = off + r
        U[0:20, col] = U41[:, r]
        V[24:30, col] = V41[:, r]
        W[0:24, col] = W41[:, r]

    # Final verification
    if not verify(U, V, W, 4, 5, 6):
        # Ultimate fallback: schoolbook
        U = np.zeros((20, 120)); V = np.zeros((30, 120)); W = np.zeros((24, 120))
        c = 0
        for i in range(4):
            for j in range(5):
                for k in range(6):
                    U[i*5+j, c] = 1.0
                    V[j*6+k, c] = 1.0
                    W[k*4+i, c] = 1.0
                    c += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
