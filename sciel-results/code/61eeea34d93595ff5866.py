import time
import numpy as np

START = time.time()
DEADLINE = START + 115.0  # safety margin under 150s kill

def _build_strassen_2x2x2():
    U = np.zeros((4, 7), dtype=np.int8)
    V = np.zeros((4, 7), dtype=np.int8)
    W = np.zeros((4, 7), dtype=np.int8)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1; W[3,1]=-1
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1; W[3,2]=1
    U[3,3]=1; V[0,3]=-1; V[2,3]=1; W[0,3]=1; W[1,3]=1
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
    U[0,5]=-1; U[2,5]=1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W

def _build_rank98():
    U2, V2, W2 = _build_strassen_2x2x2()
    rank_444 = 49
    U4 = np.zeros((16, rank_444), dtype=np.int16)
    V4 = np.zeros((16, rank_444), dtype=np.int16)
    W4 = np.zeros((16, rank_444), dtype=np.int16)
    
    for c0 in range(7):
        for c1 in range(7):
            c = c0 * 7 + c1
            u0, u1 = U2[:, c0], U2[:, c1]
            v0, v1 = V2[:, c0], V2[:, c1]
            w0, w1 = W2[:, c0], W2[:, c1]
            for i in range(4):
                for j in range(4):
                    i0, i1 = i // 2, i % 2
                    j0, j1 = j // 2, j % 2
                    val = u0[i0*2+j0] * u1[i1*2+j1]
                    if val: U4[i*4+j, c] = val
            for j in range(4):
                for k in range(4):
                    j0, j1 = j // 2, j % 2
                    k0, k1 = k // 2, k % 2
                    val = v0[j0*2+k0] * v1[j1*2+k1]
                    if val: V4[j*4+k, c] = val
            for k in range(4):
                for i in range(4):
                    k0, k1 = k // 2, k % 2
                    i0, i1 = i // 2, i % 2
                    val = w0[k0*2+i0] * w1[k1*2+i1]
                    if val: W4[k*4+i, c] = val
    
    total_rank = 98
    U = np.zeros((16, total_rank), dtype=np.int16)
    V = np.zeros((32, total_rank), dtype=np.int16)
    W = np.zeros((32, total_rank), dtype=np.int16)
    
    U[:, 0:49] = U4
    U[:, 49:98] = U4
    for j in range(4):
        for kl in range(4):
            V[j*8+kl, 0:49] = V4[j*4+kl, :]
            V[j*8+4+kl, 49:98] = V4[j*4+kl, :]
            for i in range(4):
                W[kl*4+i, 0:49] = W4[kl*4+i, :]
                W[(4+kl)*4+i, 49:98] = W4[kl*4+i, :]
    
    # Verify
    n, m, p = 4, 4, 8
    result = np.zeros((n, m, p), dtype=np.float64)
    for r in range(total_rank):
        uc = U[:, r].reshape(n, m)
        vc = V[:, r].reshape(m, p)
        wc = W[:, r].reshape(p, n)
        result += uc[:, :, None] * vc[:, :, None] * wc.T[:, :, None]
    
    if not np.allclose(result, 1.0, atol=1e-9):
        return None
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook(n=4, m=4, p=8):
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    try:
        if time.time() < DEADLINE:
            res = _build_rank98()
            if res is not None:
                return res
    except Exception:
        pass
    return _schoolbook()

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
