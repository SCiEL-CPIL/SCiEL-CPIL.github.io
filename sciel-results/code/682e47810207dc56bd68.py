import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def _build_tensor(n, m, p):
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _validate(U, V, W, n, m, p, tol=1e-6):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
    target = _build_tensor(n, m, p)
    return np.allclose(T, target, atol=tol)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _strassen_2x2():
    """Return (U, V, W) for 2x2 x 2x2 matrix multiplication, rank 7.
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = A*B
    # Strassen:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    # C11 = p1 + p4 - p5 + p7
    # C12 = p3 + p5
    # C21 = p2 + p4
    # C22 = p1 - p2 + p3 + p6
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # p1 = (a+d)(e+h)
    # U: a=1, d=1 -> U[0,0]=1, U[3,0]=1
    # V: e=1, h=1 -> V[0,0]=1, V[3,0]=1
    # W: C11 += 1, C22 += 1 -> W[0,0]=1 (k=0,i=0), W[3,0]=1 (k=1,i=1)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # p2 = (c+d)e
    # U: c=1, d=1 -> U[2,1]=1, U[3,1]=1
    # V: e=1 -> V[0,1]=1
    # W: C21 += 1 -> W[2,1]=1 (k=0,i=1)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # p3 = a(f-h)
    # U: a=1 -> U[0,2]=1
    # V: f=1, h=-1 -> V[1,2]=1, V[3,2]=-1
    # W: C12 += 1, C22 += 1 -> W[1,2]=1 (k=1,i=0), W[3,2]=1 (k=1,i=1)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # p4 = d(g-e)
    # U: d=1 -> U[3,3]=1
    # V: g=1, e=-1 -> V[2,3]=1, V[0,3]=-1
    # W: C11 += 1, C21 += 1 -> W[0,3]=1 (k=0,i=0), W[2,3]=1 (k=0,i=1)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # p5 = (a+b)h
    # U: a=1, b=1 -> U[0,4]=1, U[1,4]=1
    # V: h=1 -> V[3,4]=1
    # W: C11 -= 1, C12 += 1 -> W[0,4]=-1 (k=0,i=0), W[1,4]=1 (k=1,i=0)
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # p6 = (c-a)(e+f)
    # U: c=1, a=-1 -> U[2,5]=1, U[0,5]=-1
    # V: e=1, f=1 -> V[0,5]=1, V[1,5]=1
    # W: C22 += 1 -> W[3,5]=1 (k=1,i=1)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # p7 = (b-d)(g+h)
    # U: b=1, d=-1 -> U[1,6]=1, U[3,6]=-1
    # V: g=1, h=1 -> V[2,6]=1, V[3,6]=1
    # W: C11 += 1 -> W[0,6]=1 (k=0,i=0)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _embed_strassen_in_4x4x4(U2, V2, W2, block_i, block_j, block_k):
    """Embed a 2x2x2 Strassen decomposition into a 4x4x4 tensor.
    block_i, block_j, block_k are the starting indices for the 2x2 blocks.
    Returns (U, V, W) of shape (4, 7) each.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    for r in range(7):
        # U: rows are i*2+j for i,j in 0..1
        for idx, val in enumerate(U2[:, r]):
            if val != 0:
                i_local, j_local = divmod(idx, 2)
                i_global = block_i + i_local
                j_global = block_j + j_local
                U[i_global * 2 + j_global, r] = val
        
        # V: rows are j*2+k for j,k in 0..1
        for idx, val in enumerate(V2[:, r]):
            if val != 0:
                j_local, k_local = divmod(idx, 2)
                j_global = block_j + j_local
                k_global = block_k + k_local
                V[j_global * 2 + k_global, r] = val
        
        # W: rows are k*2+i for k,i in 0..1
        for idx, val in enumerate(W2[:, r]):
            if val != 0:
                k_local, i_local = divmod(idx, 2)
                k_global = block_k + k_local
                i_global = block_i + i_local
                W[k_global * 2 + i_global, r] = val
    
    return U, V, W

def _build_4x5x5_decomposition():
    """Build a rank-100 decomposition for 4x5 x 5x5 matrix multiplication.
    
    Uses schoolbook: each (i,j,k) triple gives one rank-1 term.
    Total rank: 4*5*5 = 100.
    """
    n, m, p = 4, 5, 5
    
    U_list = []
    V_list = []
    W_list = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U = np.zeros(n*m)
                V = np.zeros(m*p)
                W = np.zeros(p*n)
                U[i*m + j] = 1
                V[j*p + k] = 1
                W[k*n + i] = 1
                U_list.append(U)
                V_list.append(V)
                W_list.append(W)
    
    rank = len(U_list)
    U = np.column_stack(U_list)
    V = np.column_stack(V_list)
    W = np.column_stack(W_list)
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # Direct schoolbook decomposition, rank = n*m*p = 100
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
