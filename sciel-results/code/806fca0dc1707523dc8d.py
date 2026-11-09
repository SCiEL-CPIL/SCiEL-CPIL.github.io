import numpy as np
import time
import random

def construct():
    start_time = time.time()
    
    def time_left():
        return 130 - (time.time() - start_time)

    def schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                    c += 1
        return U, V, W

    def check_decomp(U, V, W, n, m, p):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(rank):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(val - 1) > 1e-9:
                        return False
        return True

    def try_find_low_rank(n, m, p, target_rank, max_time=2):
        # Simple random search for low rank decomp
        # This is unlikely to succeed for 3x3x3 rank 23, but might for smaller
        for _ in range(20):
            if time_left() < max_time:
                break
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            # Normalize
            U /= np.linalg.norm(U, axis=0, keepdims=True)
            V /= np.linalg.norm(V, axis=0, keepdims=True)
            W /= np.linalg.norm(W, axis=0, keepdims=True)
            
            # Check
            if check_decomp(U, V, W, n, m, p):
                return U, V, W
        return None

    # Blocks
    # 3x3x3
    U333, V333, W333 = schoolbook(3, 3, 3)
    # Try to find rank 23? Too hard.
    
    # 3x2x3
    U323, V323, W323 = schoolbook(3, 2, 3)
    # Try to find rank 17?
    res = try_find_low_rank(3, 2, 3, 17, 5)
    if res:
        U323, V323, W323 = res

    # Assemble 3x5x6
    # Split m=5 into 3 and 2
    # Split p=6 into 3 and 3
    
    n, m, p = 3, 5, 6
    rank = U333.shape[1] * 2 + U323.shape[1] * 2
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    c = 0
    # Block 1: 3x3x3 (i=0..2, j=0..2, k=0..2)
    U[0:9, c:c+U333.shape[1]] = U333
    # V rows: j=0..2, k=0..2 -> j*6+k
    v_rows = [j*6+k for j in range(3) for k in range(3)]
    V[v_rows, c:c+V333.shape[1]] = V333
    # W rows: k=0..2, i=0..2 -> k*3+i
    w_rows = [k*3+i for k in range(3) for i in range(3)]
    W[w_rows, c:c+W333.shape[1]] = W333
    c += U333.shape[1]
    
    # Block 2: 3x3x3 (i=0..2, j=0..2, k=3..5)
    U[0:9, c:c+U333.shape[1]] = U333
    v_rows = [j*6+k for j in range(3) for k in range(3,6)]
    V[v_rows, c:c+V333.shape[1]] = V333
    w_rows = [k*3+i for k in range(3,6) for i in range(3)]
    W[w_rows, c:c+W333.shape[1]] = W333
    c += U333.shape[1]
    
    # Block 3: 3x2x3 (i=0..2, j=3..4, k=0..2)
    # U rows: i=0..2, j=3..4 -> i*5+j
    u_rows = [i*5+j for i in range(3) for j in range(3,5)]
    U[u_rows, c:c+U323.shape[1]] = U323
    # V rows: j=3..4, k=0..2 -> j*6+k
    v_rows = [j*6+k for j in range(3,5) for k in range(3)]
    V[v_rows, c:c+V323.shape[1]] = V323
    # W rows: k=0..2, i=0..2 -> k*3+i
    w_rows = [k*3+i for k in range(3) for i in range(3)]
    W[w_rows, c:c+W323.shape[1]] = W323
    c += U323.shape[1]
    
    # Block 4: 3x2x3 (i=0..2, j=3..4, k=3..5)
    U[u_rows, c:c+U323.shape[1]] = U323
    v_rows = [j*6+k for j in range(3,5) for k in range(3,6)]
    V[v_rows, c:c+V323.shape[1]] = V323
    w_rows = [k*3+i for k in range(3,6) for i in range(3)]
    W[w_rows, c:c+W323.shape[1]] = W323
    c += U323.shape[1]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
