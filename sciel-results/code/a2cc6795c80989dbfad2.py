import numpy as np

def construct():
    M = 10**9
    
    # Generate powerful numbers up to M
    powerful_set = set()
    max_v = int(M ** (1/3)) + 2
    
    for v in range(1, max_v + 1):
        v3 = v ** 3
        max_u = int((M // v3) ** 0.5) + 1
        for u in range(1, max_u + 1):
            p = (u * u) * v3
            if p <= M:
                powerful_set.add(p)
            else:
                break
                
    powerful_list = sorted(powerful_set)
    P = np.array(powerful_list, dtype=np.int64)
    K = len(P)
    
    if K < 2:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    max_sum = 2 * M
    counts = np.zeros(max_sum + 1, dtype=np.int64)
    
    # Chunk size to keep memory ~80MB for the sums array
    # 80MB / 8 bytes = 10 million elements
    target_chunk_elems = 10 * 10**6
    chunk_size = max(1, int(target_chunk_elems / K))
    
    for start in range(0, K, chunk_size):
        end = min(start + chunk_size, K)
        P_chunk = P[start:end]
        sums = P_chunk[:, None] + P[None, :]
        chunk_counts = np.bincount(sums.ravel(), minlength=max_sum + 1)
        counts += chunk_counts
        
    # Add D_N contribution
    # D_N = 1 if N = 2*a for some a in P
    indices = 2 * P
    counts[indices] += 1
    
    max_idx = np.argmax(counts)
    N = int(max_idx)
    
    if N < 2:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    P_set = set(powerful_list)
    pairs = []
    for a in powerful_list:
        b = N - a
        if b >= a and b in P_set:
            pairs.append([a, b])
            
    return {'N': N, 'pairs': pairs}
