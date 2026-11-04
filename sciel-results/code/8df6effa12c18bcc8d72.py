import numpy as np
import time
import random

DIMENSIONS = [2, 3, 5]

def construct():
    deadline = time.time() + 120
    n, m, p = DIMENSIONS
    
    # We split the multiplication (2x3) * (3x5) into (2x3) * (3x2) and (2x3) * (3x3)
    # Part 1: n=2, m=3, p1=2
    # Part 2: n=2, m=3, p2=3
    
    def find_decomp(n, m, p, target_rank, time_budget):
        """
        Find a rank-target_rank decomposition for n x m * m x p.
        Returns U (n*m x rank), V (m*p x rank), W (p*n x rank) or None.
        """
        nm = n * m
        mp = m * p
        pn = p * n
        
        triples = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    triples.append((u_idx, v_idx, w_idx))
        
        def verify(U, V, W, rank):
            for (u_idx, v_idx, w_idx) in triples:
                s = 0.0
                for c in range(rank):
                    s += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if abs(s - 1.0) > 1e-9:
                    return False
            return True

        def to_half_int(x):
            return round(x * 2) / 2

        best_result = None
        best_error = float('inf')
        
        # Try multiple random restarts
        max_restarts = 50
        start_time = time.time()
        
        for attempt in range(max_restarts):
            if time.time() > start_time + time_budget:
                break
                
            np.random.seed(attempt * 100 + 1)
            U = np.random.randn(nm, target_rank) * 0.5
            V = np.random.randn(mp, target_rank) * 0.5
            W = np.random.randn(pn, target_rank) * 0.5
            
            lr = 0.05
            # Gradient descent
            for step in range(2000):
                if time.time() > start_time + time_budget:
                    break
                
                error = 0.0
                gU = np.zeros((nm, target_rank))
                gV = np.zeros((mp, target_rank))
                gW = np.zeros((pn, target_rank))
                
                for (u_idx, v_idx, w_idx) in triples:
                    s = 0.0
                    for c in range(target_rank):
                        s += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                    err = s - 1.0
                    error += err * err
                    for c in range(target_rank):
                        gU[u_idx, c] += 2 * err * V[v_idx, c] * W[w_idx, c]
                        gV[v_idx, c] += 2 * err * U[u_idx, c] * W[w_idx, c]
                        gW[w_idx, c] += 2 * err * U[u_idx, c] * V[v_idx, c]
                
                if error < best_error:
                    best_error = error
                    # Save a copy of the best real-valued solution found so far in this restart
                    # We don't save to global best_result yet, we check rounding at the end of restart or periodically
                    if error < 1e-6:
                        U_rounded = np.round(U * 2) / 2
                        V_rounded = np.round(V * 2) / 2
                        W_rounded = np.round(W * 2) / 2
                        if verify(U_rounded, V_rounded, W_rounded, target_rank):
                            return (U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist())
                
                if error < 1e-10:
                    break
                    
                U -= lr * gU
                V -= lr * gV
                W -= lr * gW
                lr *= 0.995
                
        return None

    # Search for Part 1: 2x3 * 3x2
    # Schoolbook rank is 12. Try 11, then 12.
    U1 = V1 = W1 = None
    rank1 = 12
    for r1 in [11, 12]:
        if time.time() > deadline - 30:
            break
        res1 = find_decomp(2, 3, 2, r1, 20)
        if res1:
            U1, V1, W1 = res1
            rank1 = r1
            break
            
    if U1 is None:
        # Fallback to schoolbook for part 1
        U1 = [[0]*12 for _ in range(6)]
        V1 = [[0]*12 for _ in range(6)]
        W1 = [[0]*12 for _ in range(4)]
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    U1[i*3+j][col] = 1
                    V1[j*2+k][col] = 1
                    W1[k*2+i][col] = 1
                    col += 1
        rank1 = 12

    # Search for Part 2: 2x3 * 3x3
    # Schoolbook rank is 18. Try 16, 17, 18.
    U2 = V2 = W2 = None
    rank2 = 18
    for r2 in [16, 17, 18]:
        if time.time() > deadline - 10:
            break
        res2 = find_decomp(2, 3, 3, r2, 60)
        if res2:
            U2, V2, W2 = res2
            rank2 = r2
            break
            
    if U2 is None:
        # Fallback to schoolbook for part 2
        U2 = [[0]*18 for _ in range(6)]
        V2 = [[0]*18 for _ in range(9)]
        W2 = [[0]*18 for _ in range(6)]
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    U2[i*3+j][col] = 1
                    V2[j*3+k][col] = 1
                    W2[k*2+i][col] = 1
                    col += 1
        rank2 = 18

    total_rank = rank1 + rank2
    
    # Combine
    # U: 6 x total_rank
    U = [ [0]*total_rank for _ in range(6) ]
    for i in range(6):
        for c in range(rank1):
            U[i][c] = U1[i][c]
        for c in range(rank2):
            U[i][rank1 + c] = U2[i][c]
            
    # V: 15 x total_rank
    V = [ [0]*total_rank for _ in range(15) ]
    # Part 1 V: 6 x rank1. Indices j*2+k for k in 0,1.
    for j in range(3):
        for k in range(2):
            v_idx = j*5 + k
            for c in range(rank1):
                V[v_idx][c] = V1[j*2+k][c]
                
    # Part 2 V: 9 x rank2. Indices j*3+k_loc for k_loc in 0,1,2.
    # Global k = k_loc + 2.
    for j in range(3):
        for k_loc in range(3):
            k_global = k_loc + 2
            v_idx = j*5 + k_global
            for c in range(rank2):
                V[v_idx][rank1 + c] = V2[j*3+k_loc][c]

    # W: 10 x total_rank
    W = [ [0]*total_rank for _ in range(10) ]
    # Part 1 W: 4 x rank1. Indices k*2+i for k in 0,1.
    for k in range(2):
        for i in range(2):
            w_idx = k*2 + i
            for c in range(rank1):
                W[w_idx][c] = W1[k*2+i][c]
                
    # Part 2 W: 6 x rank2. Indices k_loc*2+i for k_loc in 0,1,2.
    # Global k = k_loc + 2.
    for k_loc in range(3):
        for i in range(2):
            k_global = k_loc + 2
            w_idx = k_global*2 + i
            for c in range(rank2):
                W[w_idx][rank1 + c] = W2[k_loc*2+i][c]

    return {"U": U, "V": V, "W": W}
