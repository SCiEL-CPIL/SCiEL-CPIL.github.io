import numpy as np
import time

DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Try rank 16
    if time.time() < deadline - 10:
        result = _try_rank(n, m, p, 16, deadline)
        if result is not None:
            return result
    
    # Try rank 17
    if time.time() < deadline - 5:
        result = _try_rank(n, m, p, 17, deadline)
        if result is not None:
            return result
    
    # Fallback: rank 18
    result = _rank18(n, m, p)
    if result is not None:
        return result
    
    # Schoolbook
    return _schoolbook(n, m, p)

def _try_rank(n, m, p, rank, deadline):
    for seed in range(200):
        if time.time() > deadline:
            return None
        np.random.seed(seed)
        U = np.random.randint(-2, 3, size=(n*m, rank))
        V = np.random.randint(-2, 3, size=(m*p, rank))
        W = _solve_W(n, m, p, rank, U, V, deadline)
        if W is None:
            continue
        if _verify(n, m, p, rank, U, V, W):
            # Round to half-integers
            U2 = np.round(U*2)/2
            V2 = np.round(V*2)/2
            W2 = np.round(W*2)/2
            if _verify(n, m, p, rank, U2, V2, W2):
                return _to_lists(U2, V2, W2)
    return None

def _solve_W(n, m, p, rank, U, V, deadline):
    W = np.zeros((p*n, rank))
    for i in range(n):
        for k in range(p):
            if time.time() > deadline:
                return None
            A = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
            b = np.ones(m)
            w, res, rnk, sv = np.linalg.lstsq(A, b, rcond=None)
            if res is not None and len(res) > 0 and res[0] > 1e-8:
                return None
            W[k*n+i, :] = w
    return W

def _verify(n, m, p, rank, U, V, W):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1) > 1e-6:
                    return False
    return True

def _to_lists(U, V, W):
    def conv(arr):
        out = []
        for i in range(arr.shape[0]):
            row = []
            for r in range(arr.shape[1]):
                v = arr[i, r]
                if abs(v - round(v)) < 1e-9:
                    row.append(int(round(v)))
                elif abs(v*2 - round(v*2)) < 1e-9:
                    row.append(round(v*2)/2.0)
                else:
                    row.append(float(v))
            out.append(row)
        return out
    return {"U": conv(U), "V": conv(V), "W": conv(W)}

def _rank18(n, m, p):
    if (n, m, p) != (2, 2, 5):
        return None
    rank = 18
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(10)]
    W = [[0]*rank for _ in range(10)]
    # Strassen for cols 0,1
    U[0][0]=1; U[1][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
    U[0][1]=1; U[1][1]=1; V[2][1]=1; W[1][1]=1
    U[0][2]=1; V[0][2]=1; V[1][2]=1; W[2][2]=1
    U[3][3]=1; V[2][3]=1; V[3][3]=1; W[0][3]=1; W[3][3]=1
    U[2][4]=1; U[3][4]=1; V[0][4]=1; V[2][4]=1; W[1][4]=1
    U[2][5]=1; U[3][5]=1; V[1][5]=1; W[3][5]=1
    U[0][6]=1; U[2][6]=-1; V[1][6]=1; V[3][6]=-1; W[0][6]=-1; W[2][6]=1
    # Schoolbook for cols 2,3,4
    col = 7
    for i in range(2):
        for j in range(2):
            for k in range(2,5):
                U[i*2+j][col]=1
                V[j*5+k][col]=1
                W[k*2+i][col]=1
                col+=1
    return {"U":U,"V":V,"W":W}

def _schoolbook(n, m, p):
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col]=1
                V[j*p+k][col]=1
                W[k*n+i][col]=1
                col+=1
    return {"U":U,"V":V,"W":W}
# EVOLVE-BLOCK-END
