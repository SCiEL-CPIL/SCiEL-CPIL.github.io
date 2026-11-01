import numpy as np
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 10.0 # 10s is plenty for 220 vectors
    
    vectors = []
    
    # Generate D11 root system: vectors with two non-zero coordinates +/-1
    # There are C(11, 2) pairs of indices, and 4 sign combinations.
    # Total = 55 * 4 = 220 vectors.
    
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    if time.time() > deadline:
                        break
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
        if time.time() > deadline:
            break
        if time.time() > deadline:
            break
        if time.time() > deadline:
            break
        if time.time() > deadline:
            break
        if time.time() > deadline:
            break
        if time.time() > deadline:
            break
        if time.time() > deadline:
            break
        if time.time() > deadline:
            break
        if time.time() > deadline:
            break
        if time.time() > deadline:
            break
        if time.time() > deadline:
            break

    if not vectors:
        return np.array([], dtype=np.int64).reshape(0, D)

    V = np.array(vectors, dtype=np.int64)
    
    # Optional: Verification (skipped for speed, but logic is sound)
    # The D11 roots have norm sqrt(2).
    # Min distance between distinct roots is sqrt(2) (e.g., e1+e2 and e1-e2 have dot 0, dist sqrt(4)=2? No.
    # e1+e2 = (1,1,0..). e1-e2 = (1,-1,0..).
    # Diff = (0,2,0..). Norm 2.
    # e1+e2 and e1+e3.
    # Diff = (0,1,-1,0..). Norm sqrt(2).
    # So min dist is sqrt(2). Max norm is sqrt(2).
    # Condition: dist >= max_norm. sqrt(2) >= sqrt(2). Holds.
    
    return V
