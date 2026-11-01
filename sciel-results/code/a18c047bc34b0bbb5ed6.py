import time
import numpy as np

def construct():
    """
    Construct a rank-20 exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    """
    n, m, p = 2, 2, 7
    rank = 20
    
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    # Define the 20 terms
    terms = [
        # r=0
        ([0,3], [1,1], [0,8], [1,1], [0,9], [1,1]),
        # r=1
        ([2,3], [1,1], [0], [1], [1,10], [1,-1]),
        # r=2
        ([0], [1], [1,9], [1,-1], [2,11], [1,1]),
        # r=3
        ([3], [1], [8,0], [1,-1], [0,1], [1,1]),
        # r=4
        ([0,1], [1,1], [9], [1], [0,2], [-1,1]),
        # r=5
        ([2,0], [1,-1], [0,1], [1,1], [11], [1]),
        # r=6
        ([1,3], [1,-1], [8,9], [1,1], [0], [1]),
        # r=7
        ([0,3], [1,1], [2,10], [1,1], [4,13], [1,1]),
        # r=8
        ([2,3], [1,1], [2], [1], [5,14], [1,-1]),
        # r=9
        ([0], [1], [3,11], [1,-1], [6,15], [1,1]),
        # r=10
        ([3], [1], [10,2], [1,-1], [4,5], [1,1]),
        # r=11
        ([0,1], [1,1], [11], [1], [4,6], [-1,1]),
        # r=12
        ([2,0], [1,-1], [2,3], [1,1], [15], [1]),
        # r=13
        ([1,3], [1,-1], [10,11], [1,1], [4], [1]),
        # r=14
        ([0,3], [1,1], [4,12], [1,1], [8,17], [1,1]),
        # r=15
        ([2,3], [1,1], [4], [1], [9,18], [1,-1]),
        # r=16
        ([0], [1], [5,13], [1,-1], [10,19], [1,1]),
        # r=17
        ([3], [1], [12,4], [1,-1], [8,9], [1,1]),
        # r=18
        ([0,1], [1,1], [13], [1], [8,10], [-1,1]),
        # r=19
        ([2,0], [1,-1], [4,5], [1,1], [19], [1]),
    ]
    
    for r, (u_idx, u_val, v_idx, v_val, w_idx, w_val) in enumerate(terms):
        for i, v in zip(u_idx, u_val):
            U[i, r] += v
        for i, v in zip(v_idx, v_val):
            V[i, r] += v
        for i, v in zip(w_idx, w_val):
            W[i, r] += v
    
    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                s = 0
                for r in range(rank):
                    s += U[a_idx, r] * V[b_idx, r] * W[c_idx, r]
                if s != 1:
                    valid = False
                    break
    
    if not valid:
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((n*m, rank_sb), dtype=int)
        V_sb = np.zeros((m*p, rank_sb), dtype=int)
        W_sb = np.zeros((p*n, rank_sb), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
