import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def strassen_2x2():
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a+d)(e+h)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # M2 = (c+d)e
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1
    # M3 = a(f-h)
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1
    # M4 = d(g-f)
    U[3,3]=1; V[2,3]=1; V[1,3]=-1; W[0,3]=1; W[2,3]=1
    # M5 = (a+b)h
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
    # M6 = (c-a)(e+g)
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[2,5]=1; W[3,5]=1
    # M7 = (b-d)(f+h)
    U[1,6]=1; U[3,6]=-1; V[1,6]=1; V[3,6]=1; W[0,6]=1; W[3,6]=-1
    
    return U, V, W

def strassen_4x4():
    U2, V2, W2 = strassen_2x2()
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    def block_idx(bi, bj):
        return [(bi*2+li)*4 + (bj*2+lj) for li in range(2) for lj in range(2)]
    
    blocks = {
        'A00': block_idx(0,0), 'A01': block_idx(0,1),
        'A10': block_idx(1,0), 'A11': block_idx(1,1),
        'B00': block_idx(0,0), 'B01': block_idx(0,1),
        'B10': block_idx(1,0), 'B11': block_idx(1,1),
        'C00': block_idx(0,0), 'C01': block_idx(0,1),
        'C10': block_idx(1,0), 'C11': block_idx(1,1),
    }
    
    mults = [
        ([('A00',1),('A11',1)], [('B00',1),('B11',1)], [('C00',1),('C11',1)]),
        ([('A10',1),('A11',1)], [('B00',1)], [('C10',1)]),
        ([('A00',1)], [('B10',1),('B00',-1)], [('C01',1)]),
        ([('A11',1)], [('B11',1),('B01',-1)], [('C00',1),('C10',1)]),
        ([('A00',1),('A10',1)], [('B11',1)], [('C00',-1),('C01',1)]),
        ([('A11',1),('A01',-1)], [('B00',1),('B10',1)], [('C11',1)]),
        ([('A01',1),('A11',-1)], [('B01',1),('B11',1)], [('C00',1),('C11',-1)]),
    ]
    
    for r in range(7):
        a_b, b_b, c_b = mults[r]
        for lr in range(7):
            col = r*7 + lr
            for li in range(4):
                uval = U2[li, lr]
                if uval != 0:
                    si, sj = li//2, li%2
                    for bn, s in a_b:
                        U[blocks[bn][si*2+sj], col] += s * uval
                vval = V2[li, lr]
                if vval != 0:
                    si, sj = li//2, li%2
                    for bn, s in b_b:
                        V[blocks[bn][si*2+sj], col] += s * vval
                wval = W2[li, lr]
                if wval != 0:
                    si, sj = li//2, li%2
                    for bn, s in c_b:
                        W[blocks[bn][si*2+sj], col] += s * wval
    
    return U, V, W

def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    
    U4, V4, W4 = strassen_4x4()
    rank = 98
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for r in range(49):
        U[:, r] = U4[:, r]
        for j in range(4):
            for k in range(4):
                V[j*8+k, r] = V4[j*4+k, r]
                V[j*8+k+4, 49+r] = V4[j*4+k, r]
        for k in range(4):
            for i in range(4):
                W[k*4+i, r] = W4[k*4+i, r]
                W[(k+4)*4+i, 49+r] = W4[k*4+i, r]
        U[:, 49+r] = U4[:, r]
    
    U_list = [[float(U[i,j]) for j in range(rank)] for i in range(n*m)]
    V_list = [[float(V[i,j]) for j in range(rank)] for i in range(m*p)]
    W_list = [[float(W[i,j]) for j in range(rank)] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
